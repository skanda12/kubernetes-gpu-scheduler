"""preemption-controller entrypoint.

Env: CONTROLLER_INTERVAL_S=1, SATURATION_UTIL=90, TARGET_UTIL=70, UNTHROTTLE_UTIL=60, COOLDOWN_S=20,
     MAX_PREEMPTIONS=3, WINDOW_S=600, EVICT_GRACE_S=10, INFLIGHT_TIMEOUT_S=60, STALE_AFTER_S=15,
     METRICS_PORT=9402, LOG_LEVEL=INFO
"""

from __future__ import annotations

import logging
import time

from prometheus_client import start_http_server

from gpuorch.common import config as C
from gpuorch.common.k8s import HttpAgentClient, KubeCluster
from gpuorch.common.logging import log_event, setup_logging
from gpuorch.preemption.controller import ControllerConfig, PreemptionController

log = logging.getLogger("gpuorch.preemption")


def main() -> None:
    setup_logging(C.env_str("LOG_LEVEL", "INFO"))
    cfg = ControllerConfig.from_env()
    ctl = PreemptionController(KubeCluster("preemption-controller"), HttpAgentClient(), cfg)
    start_http_server(C.env_int("METRICS_PORT", 9402))
    interval = C.env_float("CONTROLLER_INTERVAL_S", 1.0)
    log_event(log, "controller_start", config=repr(cfg))
    while True:
        try:
            ctl.run_once()
        except Exception:
            log.exception("controller cycle failed")
        time.sleep(interval)


if __name__ == "__main__":
    main()
