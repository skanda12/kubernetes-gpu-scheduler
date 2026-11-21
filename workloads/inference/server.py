"""Inference server: ResNet-50 or BERT-base on PyTorch, CPU fallback.

Env: MODEL=resnet50|bert  DEVICE=auto|cpu|cuda  BATCH=1  PORT=8000  WARMUP=3
POST /predict  (body optional; synthetic input is generated so no dataset/tokenizer download is needed)
GET  /metrics  latency histogram + torch.cuda memory + self-report contract
TODO(verify-on-gpu): `kubectl apply -f deploy/examples/inference-critical.yaml` on a WSL2 node and check
`curl <pod>:8000/metrics | grep torch_cuda_memory_reserved_bytes` is > 0.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

import torch
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

MODEL = os.environ.get("MODEL", "resnet50")
DEVICE_ENV = os.environ.get("DEVICE", "auto")
BATCH = int(os.environ.get("BATCH", "1"))
JOB = os.environ.get("JOB_NAME", os.environ.get("HOSTNAME", "inference"))

device = torch.device("cuda" if (DEVICE_ENV == "auto" and torch.cuda.is_available()) or DEVICE_ENV == "cuda" else "cpu")

LAT = Histogram("inference_request_latency_seconds", "request latency", ["job_name", "model"],
                buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 5))
REQS = Counter("inference_requests_total", "requests", ["job_name", "model"])
ALLOC = Gauge("torch_cuda_memory_allocated_bytes", "torch.cuda.memory_allocated")
RESERVED = Gauge("torch_cuda_memory_reserved_bytes", "torch.cuda.memory_reserved")
W_MEM = Gauge("workload_gpu_mem_bytes", "self-reported GPU memory")
W_UTIL = Gauge("workload_gpu_util_percent", "self-reported busy fraction %")
W_IDX = Gauge("workload_gpu_index", "GPU index")
W_SINCE = Gauge("workload_seconds_since_checkpoint", "always 0 for inference")


def build():
    if MODEL == "bert":
        from transformers import BertConfig, BertModel

        model = BertModel(BertConfig())  # bert-base architecture, random weights: no download

        def make_input():
            return {"input_ids": torch.randint(0, 30522, (BATCH, 128), device=device)}
    else:
        import torchvision

        model = torchvision.models.resnet50(weights=None)

        def make_input():
            return {"x": torch.randn(BATCH, 3, 224, 224, device=device)}
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half() if MODEL == "resnet50" else model
    return model, make_input


model, make_input = build()
lock = threading.Lock()
busy: deque[tuple[float, float]] = deque(maxlen=2000)  # (end_ts, busy_s)
app = FastAPI(title=f"inference-{MODEL}")


def infer() -> float:
    inp = make_input()
    if device.type == "cuda" and MODEL == "resnet50":
        inp = {k: v.half() for k, v in inp.items()}
    t0 = time.perf_counter()
    with lock, torch.inference_mode():
        out = model(**inp) if MODEL == "bert" else model(inp["x"])
        if device.type == "cuda":
            torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    busy.append((time.time(), dt))
    _ = out
    return dt


for _ in range(int(os.environ.get("WARMUP", "3"))):
    infer()


@app.post("/predict")
@app.get("/predict")
def predict() -> dict:
    t0 = time.perf_counter()
    infer()
    lat = time.perf_counter() - t0
    LAT.labels(JOB, MODEL).observe(lat)
    REQS.labels(JOB, MODEL).inc()
    return {"model": MODEL, "device": device.type, "latency_ms": round(lat * 1000, 2)}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model": MODEL, "device": device.type}


@app.get("/metrics")
def metrics() -> Response:
    now = time.time()
    window = 10.0
    util = 100.0 * sum(b for t, b in busy if now - t <= window) / window
    W_UTIL.set(min(100.0, util))
    W_IDX.set(0)
    W_SINCE.set(0)
    if device.type == "cuda":
        ALLOC.set(torch.cuda.memory_allocated())
        RESERVED.set(torch.cuda.memory_reserved())
        W_MEM.set(torch.cuda.memory_reserved())
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="warning")
