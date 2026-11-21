# gpuorch: priority-aware GPU orchestrator for Kubernetes

RVCE Cloud Computing PBL (IS355TBD). A cluster-wide scheduler that places priority-tagged GPU workloads across
nodes and, when latency-critical work arrives, **throttles** (cgroup freeze) or **evicts and reschedules**
lower-priority work. Evaluated against the default kube-scheduler. Full spec: [PRD.md](PRD.md).

```
                 ┌──────────────── Control plane (dev laptop, CPU only) ────────────────┐
 Jobs w/ prio ─▶ │ K8s API ◀─▶ gpu-scheduler (schedulerName: priority-gpu-scheduler)    │
                 │               │ filter (slots, GPU mem) → score → Binding + Event     │
                 │               ▼ unschedulable gpu-high/critical: hand-off annotation  │
                 │         preemption-controller: evict (slots/mem) | throttle (compute) │
                 │         autoscaler-sim (KWOK nodes + cost)   Prometheus + Grafana     │
                 └──────────────┬─────────────── node annotation gpuorch.io/state ──────┘
                                │ HTTP /throttle /unthrottle
     ┌──────────────────────────┼──────────────────────────┐
 GPU node A (WSL2)        GPU node B (WSL2)          kind workers / KWOK nodes
 gpu-agent: NVML+cgroup   gpu-agent: NVML+cgroup     gpu-agent: mock backend + noop enforcer
```

| component | code | what it does |
|---|---|---|
| gpu-agent (DaemonSet) | `src/gpuorch/agent` | `GpuBackend` (NVML / mock scenario) + `Enforcer` (cgroup v2 / noop); `/state`, `/throttle`, `/unthrottle`, `/limits`, `/actions`, `/metrics`; writes `gpuorch.io/state` node annotation |
| gpu-scheduler | `src/gpuorch/scheduler` | highest priority first, FIFO within priority; filter on slots, GPU memory (stale state ⇒ slots only), cpu/mem, taints, selectors; score: spread for high/critical, bin-pack for normal/batch, critical-node penalty; `Binding` + `GpuScheduled` Event |
| preemption-controller | `src/gpuorch/preemption` | victims: strictly lower priority, cheapest first (priority → work lost since checkpoint → minimal set); **auto policy**: pending pod blocked by slots/memory ⇒ evict, running critical pod on saturated GPU ⇒ throttle; unthrottle when triggers finish or util low for `COOLDOWN_S` (with headroom check); `MAX_PREEMPTIONS`/`WINDOW_S` anti-thrashing per Job |
| autoscaler-sim | `src/gpuorch/autoscaler_sim` | adds KWOK GPU nodes for pending pods older than `SCALE_UP_AFTER_S`, removes idle ones, tracks $/node-hour |
| experiments | `src/gpuorch/experiments` | trace generator, offline simulator (runs the real scheduler/controller code), kind runner, charts |
| workloads | `workloads/` | fake (CPU-only), ResNet-50/BERT inference, ResNet-50 training with checkpoint/resume, load generator |

## Quickstart (no GPU, no cluster)

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                            # unit tests (GPU + integration tests skipped)
python -m gpuorch.tools.demo --offline               # critical job preempting batch work, both schedulers
python -m gpuorch.experiments.runner --trace traces/mixed-small.yaml --reps 3   # results/<run-id>/ + charts
```

`make test`, `make demo-offline` and `make experiment TRACE=traces/mixed-small.yaml` do the same if `make` is installed.

## Quickstart on kind (simulated GPUs)

Needs docker, kind, kubectl, helm (`python -m gpuorch.tools.cluster check-tools` prints install commands).

```bash
make cluster-up     # kind: 1 control-plane + 3 workers, each advertising nvidia.com/gpu=4 (1 GPU × 4 time-slices); KWOK
make images deploy  # build + kind-load images, helm install (PriorityClasses, agents, scheduler, controller, autoscaler, Prometheus, Grafana)
make demo           # 12 batch jobs fill the cluster → gpu-critical job → Running < 10 s → evicted job resumes from checkpoint
make experiment TRACE=traces/mixed-small.yaml MODE=kind
pytest -m integration   # M0–M3 acceptance tests against the live cluster
make cluster-down
```

Grafana: `kubectl -n gpuorch port-forward svc/grafana 3000` → http://localhost:3000 (dashboards: cluster
overview, per-node GPU, inference latency vs SLO, preemption events, pending queue by priority).

## Using it

```yaml
spec:
  schedulerName: priority-gpu-scheduler
  priorityClassName: gpu-critical          # gpu-critical | gpu-high | gpu-normal | gpu-batch
  containers: [{resources: {limits: {nvidia.com/gpu: 1}}}]
metadata:
  annotations:
    gpuorch.io/gpu-mem-mb: "2000"           # memory-aware packing
    gpuorch.io/latency-slo-ms: "50"         # evaluation
    gpuorch.io/preempt-policy: auto         # auto | throttle | evict (what this pod may do to others)
```

Workloads should expose the [self-reporting metrics](workloads/README.md) (`workload_gpu_mem_bytes`, ...), because
NVML per-process data is unreliable on WSL2.

## GPU node setup (Windows laptops, WSL2)

[docs/windows-gpu-node-setup.md](docs/windows-gpu-node-setup.md) covers WSL2, the driver, mirrored networking or
Tailscale, the firewall, NVIDIA Container Toolkit, k3s join and uninstalling. Run the preflight first:
`bash scripts/check_gpu_node.sh <dev-laptop-ip>`. Runbook with every `TODO(verify-on-gpu)`:
[docs/gpu-evaluation.md](docs/gpu-evaluation.md).

## Experiments

```bash
python -m gpuorch.experiments.tracegen --name my --jobs 60 --rate 0.1 --seed 1 -o traces/my.yaml
python -m gpuorch.experiments.runner --trace traces/my.yaml --mode sim --reps 3 [--nodes 3 --slots 4 --autoscale]
python -m gpuorch.experiments.analysis results/<run-id>
```

Output: `results/<run-id>/{gpuorch,default}/rep<N>/{jobs.csv,util.csv,summary.json}`, `comparison.json`, and
`charts/*.png` (latency p50/p95/p99, SLO violations, queueing delay by priority, utilisation, preemptions and
lost work, cost, OOMs).

**Simulator model** (`experiments/simulator.py`): one physical GPU per node exposed as `slots` time-sliced
replicas; pods progress at `min(1, 100/Σ util)`; frozen pods make no progress but keep memory; inference latency
scales with contention; pods that overflow GPU memory die with CUDA OOM (only the memory-unaware baseline does
that); evicted training loses work since its last checkpoint. The baseline models kube-scheduler (LeastAllocated,
native priority preemption, no GPU-memory awareness). These are modelling assumptions. The kind and real-GPU runs
are the ground truth.

Example (`traces/mixed-small.yaml`, 3 nodes × 4 slots, 3 reps):

| metric | priority-gpu-scheduler | default kube-scheduler |
|---|---|---|
| gpu-critical p95 latency | ~26 ms | ~44 ms |
| gpu-high p95 latency | ~59 ms | ~104 ms |
| preemption decision → running | ~3–11 s | ~80 s |
| training work lost to eviction | ~90 s | ~360 s |
| GPU OOM failures | 0 | ~150 |
| mean GPU utilisation | ~85 % | ~91 % (oversubscribed) |
| gpu-batch mean queueing delay | higher | lower |

The trade-off is that batch work waits longer, because gpuorch will not oversubscribe GPU memory and it freezes
batch pods while critical work needs the GPU.

## Configuration (env vars, all with defaults)

| component | variables |
|---|---|
| agent | `NODE_NAME`, `POD_IP`, `AGENT_PORT=9400`, `AGENT_INTERVAL_S=2`, `GPU_BACKEND=mock\|nvml` (nvml falls back to mock loudly), `MOCK_SCENARIO`, `ENFORCER=noop\|cgroup`, `CGROUP_ROOT=/sys/fs/cgroup`, `ANNOTATE_INTERVAL_S=5`, `STANDALONE=false` |
| scheduler | `SCHEDULER_NAME=priority-gpu-scheduler`, `SCHEDULER_INTERVAL_S=1`, `STALE_AFTER_S=15`, `SCHEDULER_CONFIG` (weights YAML from the ConfigMap), `METRICS_PORT=9401` |
| preemption | `SATURATION_UTIL=90`, `TARGET_UTIL=70`, `UNTHROTTLE_UTIL=60`, `COOLDOWN_S=20`, `MAX_PREEMPTIONS=3`, `WINDOW_S=600`, `EVICT_GRACE_S=10`, `INFLIGHT_TIMEOUT_S=60`, `METRICS_PORT=9402` |
| autoscaler-sim | `SCALE_UP_AFTER_S=30`, `SCALE_DOWN_AFTER_S=120`, `MAX_SIM_NODES=3`, `NODE_PRICE_PER_HOUR=0.35`, `SIM_GPUS_PER_NODE=4`, `SIM_GPU_MEM_MB=16384`, `METRICS_PORT=9403` |

All of these are exposed as Helm values in `deploy/helm/gpuorch/values.yaml`. Logs are JSON, one line per
decision, and every decision is also recorded as a Kubernetes Event and a Prometheus metric.

### autoscaler-sim → real Cluster Autoscaler

Scale-up on "pending pods older than N s" corresponds to CA's unschedulable-pods loop, and a KWOK node stands in
for a new VM in a GPU node group (GKE `n1-standard-8` + T4, EKS `g4dn.xlarge`). `SCALE_DOWN_AFTER_S` corresponds
to `--scale-down-unneeded-time`. On GKE: `gcloud container node-pools create gpu --accelerator type=nvidia-tesla-t4
--enable-autoscaling --min-nodes 0 --max-nodes N`. On EKS: a managed node group plus the cluster-autoscaler Helm
chart. A real cloud autoscaler is out of scope (PRD §10).

## Milestone status

| milestone | status |
|---|---|
| M0 skeleton, kind cluster | code done (`make cluster-up`); **not yet run**: docker/kind/kubectl/helm not installed on the dev laptop |
| M1 agent | done: mock + NVML backends, noop + cgroup enforcers, API, metrics, annotation; NVML verified against a fake pynvml, real GPU `TODO(verify-on-gpu)` |
| M2 priorities, workloads, scheduler | done; unit-tested; kind integration test written (`tests/integration`), not yet run |
| M3 preemption | done; every policy branch unit-tested; critical pod Running in ~8 s in simulation; kind test written, not yet run |
| M4 observability, autoscaler-sim, experiments | done; sim experiments + charts run end-to-end; kind mode `TODO(verify-on-kind)` |
| M5/M6 real GPUs | scripts, manifests, docs written; `TODO(verify-on-gpu)` (see docs/gpu-evaluation.md) |
| M7 | not started (optional) |

## Repository layout

See PRD §6. In addition: `src/gpuorch/common/cluster.py` (ClusterClient + in-memory FakeCluster),
`src/gpuorch/experiments/{simulator,baseline}.py`, `src/gpuorch/tools/{cluster,demo}.py`, `deploy/examples/`,
`docs/`.
