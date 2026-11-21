# PRD: Priority-Aware GPU Orchestrator for Kubernetes

> **For Claude Code:** Read this whole file before writing code. Build milestones in order. Anything marked **[NOW]** must run on a laptop with **no GPU**. Anything marked **[GPU]** is code you write now but only *verify* later on the GPU laptops. Never assume a GPU is present: every GPU-dependent path must have a mock fallback and tests that pass without a GPU.

---

## 1. Context

College project (RVCE, Cloud Computing PBL, course IS355TBD). We are extending a node-level, priority-aware Docker GPU controller (NVML monitoring + cgroup enforcement + container priorities) into a **cluster-wide, priority-aware GPU scheduler for Kubernetes**.

Problem: shared GPUs are usually allocated whole or first-come-first-served, so latency-critical inference jobs wait behind long, low-priority training jobs.

Goal: a central scheduler that places priority-tagged GPU workloads across nodes. When high-priority work arrives, it **throttles** or **evicts and reschedules** lower-priority work. It is evaluated against the default Kubernetes scheduler.

## 2. Hardware reality and constraints (read carefully)

| Machine | Role | Notes |
|---|---|---|
| Developer laptop (no GPU) | Development + **CPU-only control-plane node** | All of M0–M4 is built and tested here using simulated GPUs. |
| Friend laptop A (NVIDIA GPU, Windows) | GPU worker node | **No dual-boot.** Linux runs inside **WSL2**. |
| Friend laptop B (NVIDIA GPU, Windows) | GPU worker node | Same as A. |

### Known WSL2 limitations the design must work around

| Feature | Native Linux | WSL2 | Our design response |
|---|---|---|---|
| CUDA / PyTorch in containers | Yes | Yes (via Windows NVIDIA driver) | Use it. |
| NVML device-level util + memory | Yes | Yes (mostly) | Primary telemetry source. |
| NVML **per-process** GPU util/memory | Yes | **Unreliable / often empty** | Per-workload metrics come from **workload self-reporting** (see §5.4), not NVML per-process. |
| CUDA MPS | Yes | **Not supported / unreliable** | GPU sharing = **time-slicing**. MPS code path optional, only on native Linux. |
| MIG | A100/H100 only | No | **Out of scope**; documented as hardware-limited. |
| cgroup v2 (CPU/mem limits, freezer) | Yes | Yes | Use for enforcement and throttle (freeze). |
| Multi-node networking | Easy | NAT by default | Use **WSL2 mirrored networking** (Win 11) or **Tailscale**. Fallback in §7 M5. |

**Never install a Linux NVIDIA driver inside WSL2.** The Windows driver provides GPU access to WSL2.

## 3. Architecture

```
                 ┌──────────────────────── Control plane (dev laptop, CPU only) ─────────────────────────┐
                 │                                                                                        │
 Job submit ──▶  │  K8s API ◀──▶ gpu-scheduler (custom scheduler, schedulerName: priority-gpu-scheduler)  │
 (Jobs w/ prio)  │                 │  reads node GPU state from agents / Prometheus                       │
                 │                 ▼                                                                      │
                 │           preemption-controller ── decides: none | throttle | evict+reschedule        │
                 │                 │                                                                      │
                 │           autoscaler-sim (adds/removes simulated GPU nodes on demand)                  │
                 │           Prometheus + Grafana                                                         │
                 └─────────────────┼──────────────────────────────────────────────────────────────────────┘
                                   │ HTTP (agent API) + K8s API
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   GPU node A (WSL2)        GPU node B (WSL2)        Simulated nodes (kind / KWOK)
   gpu-agent DaemonSet      gpu-agent DaemonSet      gpu-agent in MOCK mode
   NVML backend             NVML backend             Mock backend (scripted util/mem)
   cgroup enforcer          cgroup enforcer          no-op enforcer (records actions)
   workloads (PyTorch)      workloads (PyTorch)      sleep/fake workloads
```

### Components

1. **gpu-agent** (DaemonSet, one per node): reports GPU state, enforces CPU/memory limits, executes throttle/unthrottle commands.
2. **gpu-scheduler**: custom Kubernetes scheduler that binds pending pods that use `schedulerName: priority-gpu-scheduler`.
3. **preemption-controller**: watches for high-priority pods that can't be placed and frees capacity.
4. **autoscaler-sim**: scales the number of simulated GPU nodes based on pending demand (stands in for a cloud cluster autoscaler).
5. **workloads**: benchmark inference (ResNet-50, BERT) and training jobs with checkpointing; also fake CPU-only versions for simulation.
6. **trace-generator + experiment runner**: generates job-arrival traces and runs experiments against both schedulers.
7. **observability**: Prometheus, Grafana dashboards, optional DCGM exporter on native Linux only.

## 4. Key design decisions

- **Language:** Python 3.11 for everything (agent, scheduler, controller, tools). Libraries: `kubernetes` (official client), `pynvml` (nvidia-ml-py), `fastapi` + `uvicorn` (agent API), `prometheus_client`, `pydantic`, `pytest`. Keep it in one language so a student team can maintain it.
- **GPU backend abstraction:** the agent depends on a `GpuBackend` interface with two implementations: `NvmlBackend` (real) and `MockBackend` (scripted from a YAML scenario file). Selected by env var `GPU_BACKEND=nvml|mock`. Auto-fallback to mock if NVML init fails, with a loud warning.
- **Enforcer abstraction:** `Enforcer` interface with `CgroupV2Enforcer` (real: writes `cpu.max`, `memory.max`, `cgroup.freeze` under the pod's cgroup) and `NoopEnforcer` (records actions in memory, exposes them via API for tests).
- **Custom scheduler, not a scheduler plugin:** a standalone Python scheduler that watches unscheduled pods with our `schedulerName`, scores nodes, and creates a `Binding`. Default-scheduler comparison runs use the same Jobs without `schedulerName`.
- **Priorities:** map onto native Kubernetes `PriorityClass` objects (§5.2). Our scheduler reads `pod.spec.priority`.
- **GPU resources:** nodes advertise `nvidia.com/gpu`. Simulated nodes advertise fake `nvidia.com/gpu` capacity (via node status patch or KWOK). Real nodes use the NVIDIA device plugin with **time-slicing** (e.g. `replicas: 4`) so several pods can share one GPU.
- **Throttle = freeze:** throttle-in-place freezes the low-priority pod's cgroup (`cgroup.freeze=1`), which releases GPU compute time but keeps GPU memory. Evict = delete pod; the owning Job recreates it and our scheduler places it elsewhere; training resumes from checkpoint.
- **Ops:** everything deployable with Helm (one umbrella chart) and runnable locally with `make` targets.

## 5. Functional requirements

### 5.1 gpu-agent
- Runs as a DaemonSet; `hostPID`, mounts `/sys/fs/cgroup` (real mode) and needs the NVIDIA runtime on GPU nodes.
- Every `AGENT_INTERVAL_S` (default 2s) collects per GPU: index, name, uuid, total/used memory, utilization %, temperature, power (if available).
- Collects per pod on the node (via the Kubernetes API + cgroup paths): CPU usage, memory usage, priority class, frozen state, plus **self-reported GPU memory/util** scraped from the workload's `/metrics` endpoint (§5.4).
- HTTP API (FastAPI):
  - `GET /healthz`
  - `GET /state` → node GPU + pod state JSON (pydantic schema `NodeGpuState`)
  - `POST /throttle {pod_uid}` / `POST /unthrottle {pod_uid}`
  - `POST /limits {pod_uid, cpu_max, memory_max}`
  - `GET /actions` (mock/noop mode: list of recorded enforcement actions)
- Exposes Prometheus `/metrics`: `gpu_util_percent{node,gpu}`, `gpu_mem_used_bytes`, `gpu_mem_total_bytes`, `pod_gpu_mem_bytes{pod,priority}`, `pod_throttled{pod}`.
- Also writes a compact summary to a node annotation `gpuorch.io/state` (JSON, rate-limited), so the scheduler can work even without reaching the agent over HTTP.
- `MockBackend` reads a scenario YAML (GPUs per node, memory, utilization curve over time, and how much each fake pod "uses"), so tests are deterministic.

### 5.2 Priorities
Create these PriorityClasses (values are suggestions):

| PriorityClass | Value | Use | Preemptible by |
|---|---|---|---|
| `gpu-critical` | 1000000 | latency-critical inference | none |
| `gpu-high` | 100000 | interactive / important inference | critical |
| `gpu-normal` | 10000 | regular training | high, critical |
| `gpu-batch` | 1000 | best-effort batch training | all above |

Pod annotations (optional, used by our scheduler):
- `gpuorch.io/gpu-mem-mb`: expected GPU memory (needed for memory-aware packing).
- `gpuorch.io/latency-slo-ms`: SLO for inference jobs (used in evaluation).
- `gpuorch.io/preempt-policy`: `throttle` | `evict` | `auto` (default `auto`).

### 5.3 gpu-scheduler
- Watches pending pods with `schedulerName: priority-gpu-scheduler`; processes them **highest priority first**, FIFO within a priority.
- Filter nodes: must have a free `nvidia.com/gpu` slot and enough **free GPU memory** (from agent state) for `gpuorch.io/gpu-mem-mb`; also respects normal CPU/memory requests and taints.
- Score nodes (configurable weights in a ConfigMap):
  - lower current GPU utilization is better (spread latency-critical work);
  - for `gpu-batch`/`gpu-normal`, prefer bin-packing onto busier nodes (keeps nodes free for critical work);
  - penalize nodes where a critical pod is already running when placing batch work.
- Bind via the `Binding` API; emit Kubernetes Events with the decision and scores (`reason: GpuScheduled`).
- If no node fits and the pod is `gpu-high` or `gpu-critical`, hand off to the preemption-controller.
- Must handle stale state: if agent data is older than `STALE_AFTER_S`, treat the node's free memory as unknown and only place there if slot-count allows (log a warning).

### 5.4 Workloads
- `workloads/inference`: a FastAPI server that loads ResNet-50 or BERT-base (PyTorch), serves `/predict`, and exposes `/metrics` with request latency histogram (p50/p95/p99), `torch.cuda.memory_allocated`, `torch.cuda.memory_reserved`. Runs on CPU if CUDA is absent (`DEVICE=auto`).
- `workloads/load-generator`: sends requests to inference services at a configured rate and records latency.
- `workloads/training`: ResNet-50 on synthetic data (no dataset download), checkpoints to a volume every N steps, resumes from the latest checkpoint on restart, reports steps/sec and GPU memory on `/metrics`.
- `workloads/fake`: tiny CPU-only containers that `sleep` and expose fake GPU metrics read from env vars. These are used on simulated nodes.
- All images are small and multi-stage. GPU images are based on an official `pytorch/pytorch` CUDA runtime image; fake images on `python:3.11-slim`.

### 5.5 preemption-controller
- Triggered when a `gpu-high`/`gpu-critical` pod can't be placed.
- Finds candidate victims: pods with strictly lower priority, cheapest first (lowest priority, then least runtime lost since last checkpoint, then smallest set that frees enough GPU memory/slots).
- Policy selection (`auto`):
  - if freeing **compute time** is enough (the critical pod's memory fits but the GPU is saturated) → **throttle** victims on the same node (freeze);
  - if **memory** or **slots** are the problem → **evict** victims (delete pod; Job recreates it; scheduler reschedules it elsewhere or it waits).
- Unthrottle: when critical/high pods on the node finish or the node's utilization drops below a threshold for `COOLDOWN_S`, unfreeze throttled pods in priority order.
- Guardrails: never preempt the same pod more than `MAX_PREEMPTIONS` times in `WINDOW_S` (avoid thrashing); log every decision as an Event and a Prometheus counter `preemptions_total{policy,victim_priority}`; record the time from decision to critical-pod-running (`preemption_latency_seconds`).

### 5.6 autoscaler-sim
- Watches pending GPU pods older than `SCALE_UP_AFTER_S`; adds a simulated GPU node (KWOK node or kind node patched with fake `nvidia.com/gpu`) up to `MAX_SIM_NODES`.
- Removes empty simulated nodes after `SCALE_DOWN_AFTER_S`.
- Tracks a fake cost model (e.g. T4 node = $0.35/hour, configurable) so the evaluation can report cost.
- Documents how this maps to Cluster Autoscaler on GKE/EKS. A real cloud autoscaler is out of scope.

### 5.7 Observability
- Prometheus (kube-prometheus-stack Helm chart or a minimal Prometheus deployment) scraping agents, scheduler, controller, workloads.
- Grafana dashboards committed as JSON: cluster GPU overview, per-node GPU util/memory, inference latency vs SLO, preemption events timeline, pending queue by priority.
- DCGM exporter: optional, native Linux only; disabled by default.

### 5.8 Experiment runner
- `trace-generator`: produces a YAML/CSV trace of job arrivals (Poisson arrivals, configurable mix of priorities and job types, durations, seed).
- `runner`: replays a trace against (a) our scheduler and (b) the default scheduler. It collects metrics from Prometheus and writes results to `results/<run-id>/` (CSV + summary JSON).
- `analysis`: a Python script or notebook that produces comparison charts (PNG): GPU utilization, critical-job p95/p99 latency, queueing delay by priority, preemption count and overhead, cost.

## 6. Repository layout

```
gpu-orchestrator/
├── PRD.md
├── README.md
├── Makefile
├── pyproject.toml              # one Python project, packages below
├── src/gpuorch/
│   ├── common/                 # pydantic models, config, logging, k8s helpers
│   ├── agent/                  # backends/ (nvml.py, mock.py), enforcer/ (cgroup.py, noop.py), api.py, main.py
│   ├── scheduler/              # filter.py, score.py, binder.py, main.py
│   ├── preemption/             # victims.py, policy.py, controller.py
│   ├── autoscaler_sim/
│   └── experiments/            # tracegen.py, runner.py, analysis.py
├── workloads/
│   ├── inference/  training/  load-generator/  fake/
├── deploy/
│   ├── helm/gpuorch/           # umbrella chart
│   ├── kind/                   # kind cluster configs (multi-node, simulated GPUs)
│   ├── k3s/                    # scripts for real WSL2 nodes
│   └── grafana/                # dashboard JSON
├── scenarios/                  # mock GPU scenario YAMLs
├── traces/
├── scripts/
│   └── check_gpu_node.sh       # preflight for friend laptops (§9)
└── tests/
    ├── unit/  integration/     # integration tests use kind + mock backend
```

## 7. Milestones

Each milestone ends with: tests passing, `README` updated, a short demo command in the Makefile.

### M0 — Project skeleton and local cluster **[NOW]**
- Repo layout, `pyproject.toml`, linting (`ruff`), `pytest`, pre-commit, GitHub Actions CI (lint + unit tests).
- `make cluster-up`: creates a **kind** cluster with 1 control-plane + 3 workers; patches workers to advertise fake `nvidia.com/gpu` (e.g. 1 GPU each, with time-slicing replicas emulated as 4).
- `make cluster-down`.
- **Accept:** `kubectl describe node` shows `nvidia.com/gpu` capacity on workers; CI is green.

### M1 — gpu-agent with mock backend **[NOW]**, NVML backend **[GPU]**
- Implement `GpuBackend`, `MockBackend`, `NvmlBackend`, `Enforcer`, `NoopEnforcer`, `CgroupV2Enforcer`, FastAPI API, Prometheus metrics, node annotation writer.
- Dockerfile + DaemonSet in the Helm chart.
- **Accept:** in kind, `GET /state` on each agent returns scenario-driven data; Prometheus metrics visible; unit tests cover both backends (NVML tests are skipped when NVML is absent, using `pytest.mark.gpu`).

### M2 — Priorities, workloads, custom scheduler **[NOW]**
- PriorityClasses in the chart; fake workloads; CPU-mode inference + training images.
- gpu-scheduler with filter/score/bind and Events.
- **Accept:** submit 10 mixed Jobs → all bound by our scheduler, higher priority first; scores visible in Events; unit tests for filter and score; an integration test on kind.

### M3 — Preemption controller **[NOW]**
- Victim selection, throttle (via agent API → NoopEnforcer in kind), evict + reschedule, unthrottle, anti-thrashing guardrails, metrics.
- **Accept:** scenario "cluster full of `gpu-batch`, then a `gpu-critical` pod arrives" → critical pod Running within a target time (e.g. < 10s in kind); correct victims chosen; evicted training Job resumes from checkpoint; tests for every policy branch.

### M4 — Observability, autoscaler-sim, experiment runner **[NOW]**
- Prometheus + Grafana dashboards; autoscaler-sim; trace generator; runner for both schedulers; analysis charts.
- **Accept:** `make experiment TRACE=traces/mixed-small.yaml` runs end-to-end on kind and produces `results/` with charts comparing our scheduler vs the default scheduler (simulated GPUs).

### M5 — Real GPU nodes on Windows laptops via WSL2 **[GPU]**
Write all scripts and docs now; run them when the laptops are available.
- `scripts/check_gpu_node.sh` preflight (see §9).
- Cluster: **k3s server on the dev laptop** (control plane; in WSL2 if it's a Windows laptop) and **k3s agents inside WSL2 on laptops A and B**.
  - Networking option 1 (preferred): Windows 11 WSL2 **mirrored networking** (`.wslconfig` → `networkingMode=mirrored`), all laptops on the same Wi-Fi/hotspot, open the k3s ports in Windows Firewall.
  - Networking option 2: **Tailscale** inside each WSL2 instance; k3s `--node-ip`/`--flannel-iface` set to the Tailscale interface.
- NVIDIA Container Toolkit inside WSL2, k3s configured with the `nvidia` container runtime, NVIDIA device plugin with **time-slicing** config.
- Switch agents on GPU nodes to `GPU_BACKEND=nvml`, `ENFORCER=cgroup`.
- **Fallback if multi-node networking fails:** run a single real GPU node (one laptop) plus simulated nodes; document this clearly in the report.
- **Accept:** `kubectl get nodes` shows both GPU laptops Ready; a pod requesting `nvidia.com/gpu: 1` runs `nvidia-smi` and a PyTorch CUDA check successfully; agent `/state` shows real GPU memory/util; throttle (freeze) visibly drops GPU util of the victim pod.

### M6 — Real-GPU evaluation **[GPU]**
- Run ResNet-50/BERT inference + ResNet-50 training traces on the real nodes, against both schedulers, 3 repetitions each.
- **Accept:** results + charts for: GPU utilization, critical-job p95/p99 latency and SLO-violation %, queueing delay by priority, preemption overhead (decision → critical running; training time lost), cost (simulated node-hours).

### M7 — Optional / stretch
- MPS sharing path on native Linux (if anyone gets access to a native Linux GPU machine, e.g. a college lab).
- DCGM exporter on native Linux.
- Cloud demo on GKE/EKS T4 nodes (only if credits/quota are available).

## 8. Evaluation metrics (definitions)

- **GPU utilization:** mean of `gpu_util_percent` across GPUs over the experiment.
- **Critical latency:** p50/p95/p99 of inference request latency for `gpu-critical` services; **SLO violation %** = share of requests above `latency-slo-ms`.
- **Queueing delay:** pod creation → Running, by priority class.
- **Preemption overhead:** count, decision-to-running time, and training progress lost (steps since last checkpoint).
- **Throughput:** completed training steps per hour (batch work should not starve).
- **Cost:** simulated node-hours × price per node-hour.
- **Baseline:** default kube-scheduler with the same PriorityClasses (native preemption only) and the same trace and seed.

## 9. GPU laptop preflight (`scripts/check_gpu_node.sh`)

The script runs inside WSL2 on a friend's laptop. It prints PASS/FAIL for each check and does not change anything:
1. Running in WSL2 (`/proc/version` contains `microsoft`), and the distro is Ubuntu 22.04+.
2. `nvidia-smi` works inside WSL2 (driver comes from Windows).
3. `python3 -c "import pynvml; ..."` → device count, memory, utilization readable. Also try per-process queries and report whether they're supported.
4. Docker or containerd present; `nvidia-container-toolkit` installed; `docker run --gpus all nvidia/cuda:<tag> nvidia-smi` works (if Docker is used).
5. cgroup v2 mounted (`/sys/fs/cgroup/cgroup.controllers` exists); freezer is usable.
6. Networking mode (mirrored or NAT), LAN IP, and whether the dev laptop's k3s port is reachable.
7. Prints GPU model, VRAM, driver and CUDA version for the report.

Also include `docs/windows-gpu-node-setup.md`: step-by-step for the friends (WSL2 install, Windows NVIDIA driver update, `.wslconfig` mirrored networking, firewall rules, NVIDIA Container Toolkit, k3s agent join command, how to uninstall cleanly afterwards).

## 10. Non-goals

- MIG (requires A100/H100).
- A production-grade replacement for kube-scheduler; no scheduler-framework Go plugins.
- A real cloud Cluster Autoscaler (simulated only, unless the M7 cloud stretch is done).
- Multi-tenant auth/quotas beyond PriorityClasses.

## 11. Non-functional requirements

- Every component runs and is testable **without a GPU** (mock backend + noop enforcer).
- Unit test coverage ≥ 70% for `scheduler`, `preemption`, `agent/backends`.
- All config via env vars/ConfigMaps with documented defaults; no hard-coded IPs.
- Structured JSON logs; every scheduling/preemption decision is traceable (Event + log + metric).
- One-command local demo: `make demo` (cluster up → deploy → run a short scenario showing a critical job preempting batch work → print summary).

## 12. Working instructions for Claude Code

1. Start at **M0**; do not skip ahead. After each milestone, run the tests and the milestone's demo command, then summarize what works and what's stubbed.
2. Check tools first: `docker`, `kind`, `kubectl`, `helm`, `python3.11`. If one is missing, print install instructions for the user's OS instead of guessing.
3. Keep GPU-only code behind the backend/enforcer interfaces, and mark GPU tests with `@pytest.mark.gpu` (skipped by default).
4. Prefer small, reviewable commits per feature with clear messages.
5. When something can't be verified without a GPU (M5/M6), write the code, scripts and docs, and add a `TODO(verify-on-gpu)` note with the exact command to run later.
6. Keep `README.md` current: architecture diagram, quickstart (no GPU), GPU node setup, how to run experiments.
