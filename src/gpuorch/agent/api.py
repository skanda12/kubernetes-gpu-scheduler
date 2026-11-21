"""gpu-agent HTTP API (PRD §5.1)."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from gpuorch.agent.collector import Collector
from gpuorch.agent.enforcer import EnforcementError
from gpuorch.agent.metrics import AgentMetrics
from gpuorch.common.logging import log_event
from gpuorch.common.models import EnforcementAction, LimitsRequest, NodeGpuState, PodRef

log = logging.getLogger(__name__)


def create_app(collector: Collector, metrics: AgentMetrics) -> FastAPI:
    app = FastAPI(title="gpuorch gpu-agent", version="0.1.0")
    enforcer = collector.enforcer

    def _check_pod(uid: str) -> None:
        # Only act on pods we know are on this node (a typo'd uid must not freeze something random).
        if collector.known_pod(uid) is None:
            collector.collect()
            if collector.known_pod(uid) is None:
                raise HTTPException(404, f"pod {uid} not on node {collector.node_name}")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "node": collector.node_name, "backend": collector.backend.name,
                "enforcer": enforcer.name}

    @app.get("/state", response_model=NodeGpuState)
    def state() -> NodeGpuState:
        return collector.state

    @app.post("/throttle")
    def throttle(ref: PodRef) -> dict:
        _check_pod(ref.pod_uid)
        try:
            enforcer.freeze(ref.pod_uid)
        except EnforcementError as exc:
            raise HTTPException(500, str(exc)) from exc
        log_event(log, "pod_throttled", pod_uid=ref.pod_uid, node=collector.node_name)
        metrics.update(collector.collect())
        return {"ok": True, "pod_uid": ref.pod_uid, "frozen": True}

    @app.post("/unthrottle")
    def unthrottle(ref: PodRef) -> dict:
        _check_pod(ref.pod_uid)
        try:
            enforcer.unfreeze(ref.pod_uid)
        except EnforcementError as exc:
            raise HTTPException(500, str(exc)) from exc
        log_event(log, "pod_unthrottled", pod_uid=ref.pod_uid, node=collector.node_name)
        metrics.update(collector.collect())
        return {"ok": True, "pod_uid": ref.pod_uid, "frozen": False}

    @app.post("/limits")
    def limits(req: LimitsRequest) -> dict:
        _check_pod(req.pod_uid)
        try:
            enforcer.set_limits(req.pod_uid, req.cpu_max, req.memory_max)
        except EnforcementError as exc:
            raise HTTPException(500, str(exc)) from exc
        log_event(log, "pod_limits_set", pod_uid=req.pod_uid, cpu_max=req.cpu_max, memory_max=req.memory_max)
        return {"ok": True}

    @app.get("/actions", response_model=list[EnforcementAction])
    def actions() -> list[EnforcementAction]:
        return enforcer.actions()

    @app.get("/metrics")
    def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
