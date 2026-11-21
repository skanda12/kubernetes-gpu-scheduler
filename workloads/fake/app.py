"""Fake GPU workload (CPU only, stdlib only).

Env: DURATION_S=60  FAKE_GPU_MEM_MB=2000  FAKE_GPU_UTIL=50  WORKLOAD_TYPE=batch|training|inference
     CHECKPOINT_EVERY_S=0 (0 = never)  CHECKPOINT_DIR=  CHECKPOINT_ON_SIGTERM=false  PORT=8000
Progress advances 1 unit per wall-clock second while the process runs, so a cgroup freeze (throttle) visibly
stalls it. Evicted pods resume from the last checkpoint written to CHECKPOINT_DIR (shared hostPath on kind).
Exit 0 when done; exit 143 on SIGTERM (Job recreates the pod).
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DURATION = float(os.environ.get("DURATION_S", "60"))
MEM_MB = float(os.environ.get("FAKE_GPU_MEM_MB", "2000"))
UTIL = float(os.environ.get("FAKE_GPU_UTIL", "50"))
KIND = os.environ.get("WORKLOAD_TYPE", "batch")
CKPT_EVERY = float(os.environ.get("CHECKPOINT_EVERY_S", "0") or 0)
CKPT_DIR = os.environ.get("CHECKPOINT_DIR", "")
CKPT_ON_TERM = os.environ.get("CHECKPOINT_ON_SIGTERM", "false").lower() == "true"
PORT = int(os.environ.get("PORT", "8000"))

state = {"progress": 0.0, "ckpt": 0.0, "restored": 0.0, "started": time.time()}
lock = threading.Lock()


def ckpt_file() -> Path | None:
    return Path(CKPT_DIR) / "progress.json" if CKPT_DIR else None


def load_checkpoint() -> None:
    f = ckpt_file()
    if f and f.exists():
        data = json.loads(f.read_text())
        state["progress"] = state["ckpt"] = state["restored"] = float(data.get("progress", 0))
        print(json.dumps({"event": "resumed_from_checkpoint", "progress": state["progress"]}), flush=True)


def save_checkpoint() -> None:
    f = ckpt_file()
    if not f:
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"progress": state["progress"], "ts": time.time()}))
    tmp.replace(f)
    state["ckpt"] = state["progress"]


def metrics_text() -> str:
    with lock:
        since = state["progress"] - state["ckpt"] if KIND != "inference" else 0
        lines = [
            "# TYPE workload_gpu_mem_bytes gauge", f"workload_gpu_mem_bytes {int(MEM_MB * 1024 * 1024)}",
            "# TYPE workload_gpu_util_percent gauge", f"workload_gpu_util_percent {UTIL}",
            "# TYPE workload_seconds_since_checkpoint gauge", f"workload_seconds_since_checkpoint {since:.1f}",
            "# TYPE workload_gpu_index gauge", "workload_gpu_index 0",
            "# TYPE workload_steps_total counter", f"workload_steps_total {state['progress']:.1f}",
            "# TYPE workload_restored_progress gauge", f"workload_restored_progress {state['restored']:.1f}",
        ]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.startswith("/metrics"):
            body, ctype = metrics_text().encode(), "text/plain; version=0.0.4"
        elif self.path.startswith("/healthz"):
            body, ctype = b'{"ok":true}', "application/json"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def on_term(signum, frame):
    if CKPT_ON_TERM:
        with lock:
            save_checkpoint()
    print(json.dumps({"event": "sigterm", "progress": state["progress"], "ckpt": state["ckpt"]}), flush=True)
    sys.exit(143)


def main() -> None:
    signal.signal(signal.SIGTERM, on_term)
    load_checkpoint()
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever, daemon=True).start()
    print(json.dumps({"event": "start", "type": KIND, "duration_s": DURATION, "gpu_mem_mb": MEM_MB, "util": UTIL}),
          flush=True)
    last = time.monotonic()
    while state["progress"] < DURATION:
        time.sleep(0.5)
        now = time.monotonic()
        # cap the step so time spent frozen (cgroup freeze) doesn't count as progress
        with lock:
            state["progress"] += min(now - last, 1.0)
            if CKPT_EVERY and state["progress"] - state["ckpt"] >= CKPT_EVERY:
                save_checkpoint()
        last = now
    with lock:
        save_checkpoint()
    print(json.dumps({"event": "done", "progress": state["progress"]}), flush=True)


if __name__ == "__main__":
    main()
