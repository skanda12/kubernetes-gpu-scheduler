# Real-GPU evaluation runbook (M5/M6)

Nothing here has been executed yet: the dev laptop has no GPU. Every step lists the exact command.
Mark each `TODO(verify-on-gpu)` done in the report once it has run.

| # | TODO(verify-on-gpu) | command | expected |
|---|---|---|---|
| 1 | preflight on laptops A and B | `bash scripts/check_gpu_node.sh <dev-ip>` | ALL CHECKS PASSED |
| 2 | NVML backend unit test | `pip install -e '.[dev,gpu]' && pytest -m gpu` | 1 passed |
| 3 | nodes joined | `kubectl get nodes -L gpuorch.io/real-gpu` | 2 GPU nodes Ready |
| 4 | time-slicing | `kubectl describe node <gpu-node> \| grep nvidia.com/gpu` | `nvidia.com/gpu: 4` |
| 5 | CUDA in pods | `bash deploy/k3s/gpu-setup.sh` (smoke test) | `cuda True ...` |
| 6 | real agent state | `kubectl -n gpuorch exec ds/gpu-agent-real -- python -c "import httpx;print(httpx.get('http://localhost:9400/state').text)"` | non-zero `memory_used_bytes` |
| 7 | freeze drops util | start `train-batch-a`, then `curl -XPOST .../throttle`, observe `nvidia-smi dmon -s u` | victim util -> ~0, memory unchanged |
| 8 | checkpoint resume | `kubectl delete pod -l gpuorch.io/job=train-batch-a`; `kubectl logs -l gpuorch.io/job=train-batch-a` | `{"event": "resumed", "step": N>0}` |
| 9 | preemption on real GPUs | `kubectl apply -f deploy/examples/real-gpu-workloads.yaml` while both GPUs are full of batch work | critical Running < 10s |

## M6 experiment

Workloads: ResNet-50 / BERT-base inference (`gpuorch/inference:dev`, `MODEL=resnet50|bert`) and ResNet-50
training (`gpuorch/training:dev`). Use the runner's kind mode with real images by passing a workload image that
honours the same env contract, or apply the trace with real workloads:

```bash
# 3 repetitions x 2 schedulers, same trace + seed
for rep in 1 2 3; do
  python -m gpuorch.experiments.runner --trace traces/mixed-small.yaml --mode kind \
      --image gpuorch/fake-workload:dev --time-scale 0.5 --prom-url http://localhost:30090 \
      --run-id m6-rep$rep
done
python -m gpuorch.experiments.analysis results/m6-rep1
```

Metrics (PRD §8) come from Prometheus (`gpu_util_percent`, `inference_request_latency_seconds`,
`preemption_latency_seconds`, `preemptions_total`) and from pod timestamps (queueing delay). Cost =
GPU node-hours × `NODE_PRICE_PER_HOUR`.

**Baseline fairness:** the default kube-scheduler run uses the same PriorityClasses (so native preemption is
active), the same trace and seed, and the same `terminationGracePeriodSeconds`.

## Known WSL2 limitations to state in the report

- No per-process NVML: per-pod GPU numbers are self-reported by the workloads.
- No MPS / MIG: sharing is time-slicing only, with no memory isolation (the scheduler's memory-aware filter
  compensates).
- Laptop GPUs throttle thermally; record `gpu_temperature_celsius` alongside results.
