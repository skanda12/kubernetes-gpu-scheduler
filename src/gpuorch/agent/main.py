"""gpu-agent entrypoint (DaemonSet).

Env: NODE_NAME (required in-cluster), POD_IP, AGENT_PORT=9400, AGENT_INTERVAL_S=2, GPU_BACKEND=mock|nvml,
     MOCK_SCENARIO=/etc/gpuorch/scenario.yaml, ENFORCER=noop|cgroup, CGROUP_ROOT=/sys/fs/cgroup,
     ANNOTATE_INTERVAL_S=5, STANDALONE=false (true: no Kubernetes API, for local runs), LOG_LEVEL=INFO
"""

from __future__ import annotations

import logging
import socket
import threading
import time

import uvicorn

from gpuorch.agent.api import create_app
from gpuorch.agent.backends import make_backend
from gpuorch.agent.collector import Annotator, Collector, http_scraper
from gpuorch.agent.enforcer import make_enforcer
from gpuorch.agent.metrics import AgentMetrics
from gpuorch.common import config as C
from gpuorch.common.logging import log_event, setup_logging

log = logging.getLogger("gpuorch.agent")


def main() -> None:
    setup_logging(C.env_str("LOG_LEVEL", "INFO"))
    node = C.env_str("NODE_NAME", socket.gethostname())
    port = C.env_int("AGENT_PORT", 9400)
    interval = C.env_float("AGENT_INTERVAL_S", 2.0)
    standalone = C.env_bool("STANDALONE", False)

    backend = make_backend(C.env_str("GPU_BACKEND", "mock"), node, C.env_str("MOCK_SCENARIO", "") or None)
    enforcer = make_enforcer(C.env_str("ENFORCER", "noop"), C.env_str("CGROUP_ROOT", "/sys/fs/cgroup"))

    annotator = None
    if standalone:
        def pod_source():
            return []
    else:
        from gpuorch.common.k8s import KubeCluster

        kube = KubeCluster("gpu-agent")

        def pod_source():
            return kube.list_pods(field_selector=f"spec.nodeName={node}")

        pod_ip = C.env_str("POD_IP", "")
        agent_url = f"http://{pod_ip}:{port}" if pod_ip else None
        annotator = Annotator(kube.patch_node_annotations, node, agent_url, C.env_float("ANNOTATE_INTERVAL_S", 5))

    collector = Collector(node, backend, enforcer, pod_source, http_scraper())
    metrics = AgentMetrics()
    log_event(log, "agent_start", node=node, backend=backend.name, enforcer=enforcer.name, port=port)

    def loop() -> None:
        while True:
            try:
                state = collector.collect()
                metrics.update(state)
                if annotator:
                    annotator.maybe_write(state)
            except Exception:
                log.exception("collect loop error")
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True, name="collector").start()
    uvicorn.run(create_app(collector, metrics), host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
