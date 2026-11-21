"""ResNet-50 training on synthetic data with checkpoint/resume.

Env: DEVICE=auto  BATCH=32  TOTAL_STEPS=2000  CHECKPOINT_EVERY_STEPS=100  CHECKPOINT_DIR=/ckpt/<job>
     EST_GPU_UTIL=90 (self-reported util while running)  PORT=8000  LR=0.01
Resumes from CHECKPOINT_DIR/latest.pt (model + optimizer + step). On SIGTERM it writes a final checkpoint if
CHECKPOINT_ON_SIGTERM=true, then exits 143 so the Job recreates the pod.
TODO(verify-on-gpu): delete the pod mid-run and confirm the new pod logs `resumed step=<n>` with n > 0.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torchvision
from prometheus_client import Counter, Gauge, start_http_server

DEVICE_ENV = os.environ.get("DEVICE", "auto")
device = torch.device("cuda" if (DEVICE_ENV == "auto" and torch.cuda.is_available()) or DEVICE_ENV == "cuda" else "cpu")
BATCH = int(os.environ.get("BATCH", "32" if device.type == "cuda" else "4"))
TOTAL = int(os.environ.get("TOTAL_STEPS", "2000"))
EVERY = int(os.environ.get("CHECKPOINT_EVERY_STEPS", "100"))
CKPT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/ckpt/train"))
CKPT_ON_TERM = os.environ.get("CHECKPOINT_ON_SIGTERM", "true").lower() == "true"

STEPS = Counter("workload_steps", "training steps")  # exported as workload_steps_total
SPS = Gauge("training_steps_per_second", "steps/s (EMA)")
W_MEM = Gauge("workload_gpu_mem_bytes", "self-reported GPU memory")
W_UTIL = Gauge("workload_gpu_util_percent", "self-reported GPU util")
W_SINCE = Gauge("workload_seconds_since_checkpoint", "seconds of work since last checkpoint")
W_IDX = Gauge("workload_gpu_index", "GPU index")
ALLOC = Gauge("torch_cuda_memory_allocated_bytes", "torch.cuda.memory_allocated")
RESERVED = Gauge("torch_cuda_memory_reserved_bytes", "torch.cuda.memory_reserved")


def log(**kw) -> None:
    print(json.dumps(kw), flush=True)


model = torchvision.models.resnet50(weights=None).to(device)
opt = torch.optim.SGD(model.parameters(), lr=float(os.environ.get("LR", "0.01")), momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
step = 0
last_ckpt_time = time.time()


def save() -> None:
    global last_ckpt_time
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CKPT_DIR / "latest.pt.tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
    tmp.replace(CKPT_DIR / "latest.pt")
    last_ckpt_time = time.time()
    log(event="checkpoint", step=step)


def restore() -> None:
    global step
    f = CKPT_DIR / "latest.pt"
    if f.exists():
        data = torch.load(f, map_location=device)
        model.load_state_dict(data["model"])
        opt.load_state_dict(data["opt"])
        step = int(data["step"])
        log(event="resumed", step=step)


def on_term(signum, frame) -> None:
    if CKPT_ON_TERM:
        save()
    log(event="sigterm", step=step)
    sys.exit(143)


def main() -> None:
    signal.signal(signal.SIGTERM, on_term)
    start_http_server(int(os.environ.get("PORT", "8000")))
    restore()
    W_IDX.set(0)
    x = torch.randn(BATCH, 3, 224, 224, device=device)
    y = torch.randint(0, 1000, (BATCH,), device=device)
    ema = None
    global step
    while step < TOTAL:
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
            ALLOC.set(torch.cuda.memory_allocated())
            RESERVED.set(torch.cuda.memory_reserved())
            W_MEM.set(torch.cuda.memory_reserved())
        step += 1
        STEPS.inc()
        dt = time.perf_counter() - t0
        ema = 1 / dt if ema is None else 0.9 * ema + 0.1 / dt
        SPS.set(ema)
        W_UTIL.set(float(os.environ.get("EST_GPU_UTIL", "90")))
        W_SINCE.set(time.time() - last_ckpt_time)
        if step % EVERY == 0:
            save()
        if step % 20 == 0:
            log(event="progress", step=step, loss=round(loss.item(), 4), steps_per_s=round(ema, 2), device=device.type)
    save()
    log(event="done", step=step)


if __name__ == "__main__":
    main()
