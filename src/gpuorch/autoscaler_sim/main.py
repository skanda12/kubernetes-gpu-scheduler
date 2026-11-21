"""autoscaler-sim entrypoint.

Env: AUTOSCALER_INTERVAL_S=5, SCALE_UP_AFTER_S=30, SCALE_DOWN_AFTER_S=120, SCALE_UP_COOLDOWN_S=30,
     MAX_SIM_NODES=3, NODE_PRICE_PER_HOUR=0.35, SIM_GPUS_PER_NODE=4, SIM_GPU_MEM_MB=16384, SIM_KWOK=true,
     METRICS_PORT=9403
"""

from __future__ import annotations

import logging
import time

from prometheus_client import start_http_server

from gpuorch.autoscaler_sim.core import AutoscalerConfig, AutoscalerSim
from gpuorch.common import config as C
from gpuorch.common.k8s import KubeCluster
from gpuorch.common.logging import log_event, setup_logging

log = logging.getLogger("gpuorch.autoscaler")


def main() -> None:
    setup_logging(C.env_str("LOG_LEVEL", "INFO"))
    cfg = AutoscalerConfig.from_env()
    sim = AutoscalerSim(KubeCluster("autoscaler-sim"), cfg)
    start_http_server(C.env_int("METRICS_PORT", 9403))
    interval = C.env_float("AUTOSCALER_INTERVAL_S", 5)
    log_event(log, "autoscaler_start", config=repr(cfg))
    while True:
        try:
            sim.run_once()
        except Exception:
            log.exception("autoscaler cycle failed")
        time.sleep(interval)


if __name__ == "__main__":
    main()
