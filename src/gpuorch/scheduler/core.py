"""gpu-scheduler scheduling cycle: queue -> filter -> score -> bind (or hand off to preemption)."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from gpuorch.common import config as C
from gpuorch.common import metrics as M
from gpuorch.common.cluster import BindConflict, ClusterClient
from gpuorch.common.logging import log_event
from gpuorch.common.priorities import class_for_value, is_latency_critical
from gpuorch.common.types import PodInfo
from gpuorch.scheduler.binder import bind_with_event
from gpuorch.scheduler.filter import filter_node
from gpuorch.scheduler.score import Weights, score_node
from gpuorch.scheduler.state import NodeView, build_views

log = logging.getLogger("gpuorch.scheduler")


@dataclass
class ScheduleResult:
    pod: str
    node: str | None
    scores: dict[str, dict] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    handed_off: bool = False


def queue_order(pods: list[PodInfo]) -> list[PodInfo]:
    """Highest priority first, FIFO within a priority."""
    return sorted(pods, key=lambda p: (-p.priority, p.created, p.name))


def is_ours(pod: PodInfo, scheduler_name: str) -> bool:
    return pod.scheduler_name == scheduler_name and pod.node_name is None and pod.phase == "Pending" and not pod.deleting


class GpuScheduler:
    def __init__(
        self,
        cluster: ClusterClient,
        weights: Weights | None = None,
        clock: Callable[[], float] = time.time,
        stale_after_s: float = 15.0,
        scheduler_name: str = C.SCHEDULER_NAME,
    ) -> None:
        self.cluster = cluster
        self.weights = weights or Weights()
        self.clock = clock
        self.stale_after_s = stale_after_s
        self.scheduler_name = scheduler_name
        self._last_reason: dict[str, str] = {}

    def run_once(self) -> list[ScheduleResult]:
        now = self.clock()
        nodes = self.cluster.list_nodes()
        pods = self.cluster.list_pods()
        views = build_views(nodes, pods, now, self.stale_after_s)
        M.STALE_NODES.set(sum(1 for v in views.values() if v.stale and v.gpu_slots_total > 0))
        pending = queue_order([p for p in pods if is_ours(p, self.scheduler_name)])
        counts = Counter(class_for_value(p.priority) for p in pending)
        for pc in ("gpu-critical", "gpu-high", "gpu-normal", "gpu-batch", "none"):
            M.PENDING.labels(pc).set(counts.get(pc, 0))
        results = [self.schedule_pod(p, views, now) for p in pending]
        live = {p.uid for p in pending}
        self._last_reason = {k: v for k, v in self._last_reason.items() if k in live}
        return results

    def schedule_pod(self, pod: PodInfo, views: dict[str, NodeView], now: float) -> ScheduleResult:
        pclass = class_for_value(pod.priority)
        nominated = pod.annotations.get(C.ANN_NOMINATED_NODE)
        scores: dict[str, dict] = {}
        reasons: dict[str, str] = {}
        blockers: dict[str, str] = {}
        for name, view in views.items():
            fr = filter_node(pod, view)
            if not fr.fits:
                reasons[name] = fr.reason
                if fr.blocker:
                    blockers[name] = fr.blocker
                continue
            for w in fr.warnings:
                log_event(log, "stale_gpu_state", logging.WARNING, pod=pod.key, node=name, detail=w)
            total, parts = score_node(pod, view, self.weights, nominated)
            scores[name] = {"total": total, **parts}

        if not scores:
            return self._unschedulable(pod, pclass, reasons, blockers, now)

        best = max(scores, key=lambda n: (scores[n]["total"], n == nominated, -len(views[n].pods), n))
        try:
            bind_with_event(self.cluster, pod, best, scores)
        except BindConflict as exc:
            log_event(log, "bind_conflict", logging.WARNING, pod=pod.key, node=best, error=str(exc))
            return ScheduleResult(pod.key, None, scores, reasons)
        views[best].assume(pod)
        M.BINDINGS.labels(pclass).inc()
        M.QUEUE_DELAY.labels(pclass).observe(max(0.0, now - pod.created))
        if pod.annotations.get(C.ANN_PREEMPTION_REQUESTED) or nominated:
            self.cluster.patch_pod_annotations(pod, {C.ANN_PREEMPTION_REQUESTED: None})
        log_event(log, "pod_scheduled", pod=pod.key, node=best, priority=pod.priority, priority_class=pclass,
                  scores=scores)
        return ScheduleResult(pod.key, best, scores, reasons)

    def _unschedulable(
        self, pod: PodInfo, pclass: str, reasons: dict[str, str], blockers: dict[str, str], now: float
    ) -> ScheduleResult:
        M.UNSCHEDULABLE.labels(pclass).inc()
        msg = "0/{} nodes fit: {}".format(len(reasons), "; ".join(f"{n}: {r}" for n, r in sorted(reasons.items())))
        if self._last_reason.get(pod.uid) != msg:  # don't spam identical events every cycle
            self._last_reason[pod.uid] = msg
            self.cluster.emit_event(pod, "FailedScheduling", msg, "Warning")
            log_event(log, "pod_unschedulable", pod=pod.key, priority_class=pclass, reasons=reasons)
        handed = False
        if is_latency_critical(pod.priority) and pod.gpu > 0:
            handed = True
            if not pod.annotations.get(C.ANN_PREEMPTION_REQUESTED):
                payload = json.dumps({"ts": now, "blockers": blockers})
                self.cluster.patch_pod_annotations(pod, {C.ANN_PREEMPTION_REQUESTED: payload})
                log_event(log, "preemption_handoff", pod=pod.key, blockers=blockers)
        return ScheduleResult(pod.key, None, {}, reasons, handed_off=handed)
