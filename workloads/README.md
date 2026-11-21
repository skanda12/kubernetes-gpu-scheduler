# Workloads

All workloads expose Prometheus metrics on `:8000/metrics` and follow the **self-reporting contract** that
gpu-agent scrapes (NVML per-process data is unreliable on WSL2, PRD §2):

| metric | meaning |
|---|---|
| `workload_gpu_mem_bytes` | GPU memory this pod holds (`torch.cuda.memory_reserved`, or `FAKE_GPU_MEM_MB`) |
| `workload_gpu_util_percent` | estimated share of the GPU this pod keeps busy (busy-time fraction, or `FAKE_GPU_UTIL`) |
| `workload_seconds_since_checkpoint` | work that would be lost if evicted now (training only) |
| `workload_gpu_index` | which GPU on the node (default 0) |
| `workload_steps_total` | progress counter |

| dir | image | what |
|---|---|---|
| `fake/` | `gpuorch/fake-workload` (python:3.11-slim, stdlib only) | sleeps for `DURATION_S`, reports fake GPU usage from env, checkpoints progress to `CHECKPOINT_DIR` and resumes after eviction. Used on kind/KWOK. |
| `inference/` | `gpuorch/inference` (pytorch CUDA runtime, or CPU variant) | FastAPI `/predict` serving ResNet-50 or BERT-base (`MODEL=resnet50|bert`), latency histogram `inference_request_latency_seconds`, `DEVICE=auto|cpu|cuda`. |
| `training/` | `gpuorch/training` | ResNet-50 on synthetic data, checkpoints every `CHECKPOINT_EVERY_STEPS`, resumes from the latest checkpoint, reports steps/s. |
| `load-generator/` | `gpuorch/load-generator` | open-loop request generator at `RPS` against `TARGET_URL`, records latency. |

Build (from repo root): `make images` (CPU variants) or `make images-gpu` (CUDA variants for the WSL2 nodes).
