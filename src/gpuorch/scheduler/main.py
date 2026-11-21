"""gpu-scheduler entrypoint.

Env: SCHEDULER_NAME=priority-gpu-scheduler, SCHEDULER_INTERVAL_S=1, STALE_AFTER_S=15,
     SCHEDULER_CONFIG=/etc/gpuorch/scheduler.yaml (weights), METRICS_PORT=9401, LOG_LEVEL=INFO
"""

from __future__ import annotations

import logging
import time

from prometheus_client import start_http_server

from gpuorch.common import config as C
from gpuorch.common.k8s import KubeCluster
from gpuorch.common.logging import log_event, setup_logging
from gpuorch.scheduler.core import GpuScheduler
from gpuorch.scheduler.score import Weights

log = logging.getLogger("gpuorch.scheduler")


def main() -> None:
    setup_logging(C.env_str("LOG_LEVEL", "INFO"))
    weights = Weights.load(C.env_str("SCHEDULER_CONFIG", "/etc/gpuorch/scheduler.yaml"))
    sched = GpuScheduler(
        KubeCluster("priority-gpu-scheduler"),
        weights,
        stale_after_s=C.env_float("STALE_AFTER_S", 15),
        scheduler_name=C.env_str("SCHEDULER_NAME", C.SCHEDULER_NAME),
    )
    start_http_server(C.env_int("METRICS_PORT", 9401))
    interval = C.env_float("SCHEDULER_INTERVAL_S", 1.0)
    log_event(log, "scheduler_start", scheduler_name=sched.scheduler_name, weights=weights.as_dict())
    while True:
        try:
            sched.run_once()
        except Exception:
            log.exception("scheduling cycle failed")
        time.sleep(interval)


if __name__ == "__main__":
    main()
