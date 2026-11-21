"""preemption-controller loop: pending hand-offs -> evict; saturated GPUs -> throttle; unthrottle on cooldown."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from gpuorch.common import config as C
from gpuorch.common import metrics as M
from gpuorch.common.cluster import AgentClient, ClusterClient
from gpuorch.common.logging import log_event
from gpuorch.common.priorities import class_for_value, is_latency_critical
from gpuorch.common.types import PodInfo
from gpuorch.preemption.policy import Decision, PolicyConfig, decide_pending, decide_saturation
from gpuorch.preemption.victims import AntiThrash
from gpuorch.scheduler.state import NodeView, build_views

log = logging.getLogger("gpuorch.preemption")


@dataclass
class ControllerConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    cooldown_s: float = 20.0
    max_preemptions: int = 3
    window_s: float = 600.0
    evict_grace_s: int = 10
    inflight_timeout_s: float = 60.0
    stale_after_s: float = 15.0

    @classmethod
    def from_env(cls) -> ControllerConfig:
        return cls(
            policy=PolicyConfig(
                saturation_util=C.env_float("SATURATION_UTIL", 90),
                target_util=C.env_float("TARGET_UTIL", 70),
                unthrottle_util=C.env_float("UNTHROTTLE_UTIL", 60),
                default_pod_util=C.env_float("DEFAULT_POD_UTIL", 50),
            ),
            cooldown_s=C.env_float("COOLDOWN_S", 20),
            max_preemptions=C.env_int("MAX_PREEMPTIONS", 3),
            window_s=C.env_float("WINDOW_S", 600),
            evict_grace_s=C.env_int("EVICT_GRACE_S", 10),
            inflight_timeout_s=C.env_float("INFLIGHT_TIMEOUT_S", 60),
            stale_after_s=C.env_float("STALE_AFTER_S", 15),
        )


@dataclass
class Inflight:
    ts: float
    policy: str
    node: str
    victim_uids: set[str]
    pod_key: str


@dataclass
class ThrottleRecord:
    node: str
    ts: float
    priority: int
    triggers: set[str]
    pod_key: str


class PreemptionController:
    def __init__(
        self,
        cluster: ClusterClient,
        agent: AgentClient,
        cfg: ControllerConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cluster = cluster
        self.agent = agent
        self.cfg = cfg or ControllerConfig()
        self.clock = clock
        self.guard = AntiThrash(self.cfg.max_preemptions, self.cfg.window_s)
        self.inflight: dict[str, Inflight] = {}
        self.throttled: dict[str, ThrottleRecord] = {}
        self.decisions: list[tuple[float, str, Decision]] = []  # (ts, preemptor key, decision) for tests/reports
        self._low_since: dict[str, float] = {}
        self._last_unthrottle: dict[str, float] = {}
        self._node_action_ts: dict[str, float] = {}
        self._skipped_msg: dict[str, str] = {}

    # ------------------------------------------------------------------------------------------
    def run_once(self) -> None:
        now = self.clock()
        nodes = {n.name: n for n in self.cluster.list_nodes()}
        pods = self.cluster.list_pods()
        by_uid = {p.uid: p for p in pods}
        views = build_views(list(nodes.values()), pods, now, self.cfg.stale_after_s)
        self._track_inflight(by_uid, now)
        self._handle_pending(pods, views, now)
        self._handle_saturation(views, now)
        self._handle_unthrottle(views, by_uid, now)
        M.THROTTLED_PODS.set(len(self.throttled))

    # -- latency tracking ----------------------------------------------------------------------
    def _track_inflight(self, by_uid: dict[str, PodInfo], now: float) -> None:
        for uid, inf in list(self.inflight.items()):
            pod = by_uid.get(uid)
            if pod is None:
                del self.inflight[uid]
            elif pod.phase == "Running" and pod.node_name:
                M.PREEMPTION_LATENCY.labels(inf.policy).observe(now - inf.ts)
                log_event(log, "preemptor_running", pod=pod.key, node=pod.node_name, policy=inf.policy,
                          latency_s=round(now - inf.ts, 3))
                del self.inflight[uid]
            elif now - inf.ts > self.cfg.inflight_timeout_s:
                log_event(log, "preemption_timeout", logging.WARNING, pod=pod.key, node=inf.node)
                self.cluster.patch_pod_annotations(pod, {C.ANN_NOMINATED_NODE: None})
                del self.inflight[uid]

    # -- pending high/critical pods -> evict -------------------------------------------------
    def _handle_pending(self, pods: list[PodInfo], views: dict[str, NodeView], now: float) -> None:
        waiting = [
            p for p in pods
            if p.annotations.get(C.ANN_PREEMPTION_REQUESTED) and p.node_name is None and not p.deleting
            and is_latency_critical(p.priority) and p.uid not in self.inflight
        ]
        for pod in sorted(waiting, key=lambda p: (-p.priority, p.created)):
            decision = decide_pending(pod, views, self.guard, now)
            if decision.policy == "none":
                self._skip(pod, decision.reason)
                continue
            self._execute(pod, decision, views, now)

    # -- running high/critical pods on saturated GPUs -> throttle ----------------------------
    def _handle_saturation(self, views: dict[str, NodeView], now: float) -> None:
        for name, view in views.items():
            if view.stale or view.summary is None:
                continue
            # wait for telemetry newer than our last action on this node (avoid over-throttling)
            if view.summary.ts <= self._node_action_ts.get(name, float("-inf")):
                continue
            frozen = {uid for uid, r in self.throttled.items() if r.node == name}
            preemptors = [
                p for p in view.pods
                if is_latency_critical(p.priority) and p.gpu > 0 and p.phase == "Running" and p.uid not in frozen
            ]
            if not preemptors:
                continue
            top = max(preemptors, key=lambda p: (p.priority, -p.created))
            decision = decide_saturation(top, view, self.guard, now, self.cfg.policy, frozen)
            if decision.policy != "none":
                self._execute(top, decision, views, now)

    # -- unthrottle ---------------------------------------------------------------------------
    def _handle_unthrottle(self, views: dict[str, NodeView], by_uid: dict[str, PodInfo], now: float) -> None:
        # adopt pods frozen by a previous controller instance (restart safety)
        for view in views.values():
            if view.summary is None:
                continue
            for uid, ps in view.summary.pods.items():
                pod = by_uid.get(uid)
                if ps.frozen and uid not in self.throttled and pod is not None and pod.active:
                    self.throttled[uid] = ThrottleRecord(view.name, now, pod.priority, set(), pod.key)

        for uid in list(self.throttled):
            pod = by_uid.get(uid)
            if pod is None or not pod.active:
                del self.throttled[uid]

        by_node: dict[str, list[str]] = {}
        for uid, rec in self.throttled.items():
            by_node.setdefault(rec.node, []).append(uid)

        for node, uids in by_node.items():
            view = views.get(node)
            if view is None:
                continue
            uids.sort(key=lambda u: -self.throttled[u].priority)  # unfreeze highest priority first
            triggers = set().union(*(self.throttled[u].triggers for u in uids))
            if triggers:
                alive = any(by_uid.get(t) is not None and by_uid[t].active for t in triggers)
            else:  # adopted: any running latency-critical pod keeps them frozen
                alive = any(is_latency_critical(p.priority) and p.phase == "Running" for p in view.pods)
            if not alive:
                for uid in uids:
                    self._unfreeze(view, by_uid[uid], "triggers_finished", now)
                continue
            util = view.gpu_util
            if util is None or util >= self.cfg.policy.unthrottle_util:
                self._low_since.pop(node, None)
                continue
            since = self._low_since.setdefault(node, now)
            last = self._last_unthrottle.get(node, float("-inf"))
            if now - since < self.cfg.cooldown_s or now - last < self.cfg.cooldown_s:
                continue
            cand = by_uid[uids[0]]
            est = view.pod_util(cand.uid)
            est = est if est is not None else self.cfg.policy.default_pod_util
            if util + est < self.cfg.policy.saturation_util:  # headroom check: don't re-saturate
                self._unfreeze(view, cand, "low_util", now)
                self._low_since.pop(node, None)

    def _unfreeze(self, view: NodeView, pod: PodInfo, reason: str, now: float) -> None:
        if not self.agent.unthrottle(view.node, pod.uid):
            log_event(log, "unthrottle_failed", logging.WARNING, pod=pod.key, node=view.name)
            return
        self.throttled.pop(pod.uid, None)
        self._last_unthrottle[view.name] = now
        self._node_action_ts[view.name] = now
        M.UNTHROTTLES.labels(reason).inc()
        self.cluster.emit_event(pod, "GpuUnthrottled", f"unfrozen on {view.name} ({reason})")
        log_event(log, "pod_unthrottled", pod=pod.key, node=view.name, reason=reason)

    # -- execution ----------------------------------------------------------------------------
    def _skip(self, pod: PodInfo, reason: str) -> None:
        if self._skipped_msg.get(pod.uid) != reason:
            self._skipped_msg[pod.uid] = reason
            M.PREEMPTION_SKIPPED.labels("no_victims").inc()
            self.cluster.emit_event(pod, "PreemptionNotPossible", reason, "Warning")
            log_event(log, "preemption_skipped", pod=pod.key, reason=reason)

    def _execute(self, preemptor: PodInfo, d: Decision, views: dict[str, NodeView], now: float) -> None:
        view = views[d.node]
        done = []
        for v in d.victims:
            vclass = class_for_value(v.priority)
            if d.policy == "throttle":
                if not self.agent.throttle(view.node, v.uid):
                    M.PREEMPTION_SKIPPED.labels("agent_unreachable").inc()
                    log_event(log, "throttle_failed", logging.WARNING, victim=v.key, node=d.node)
                    continue
                self.throttled[v.uid] = ThrottleRecord(d.node, now, v.priority, {preemptor.uid}, v.key)
            else:
                self.cluster.delete_pod(v, self.cfg.evict_grace_s)
                v.deleting = True  # later preemptors in this cycle must not pick it again
            self.guard.record(v, now)
            done.append(v)
            M.PREEMPTIONS.labels(d.policy, vclass).inc()
            self.cluster.emit_event(
                v, "GpuPreempted", f"{d.policy} by {preemptor.key} (prio {preemptor.priority}): {d.reason}", "Warning"
            )
        if not done:
            return
        self._node_action_ts[d.node] = now
        self.decisions.append((now, preemptor.key, d))
        self.cluster.emit_event(preemptor, "PreemptionTriggered", d.reason)
        log_event(log, "preemption_decision", preemptor=preemptor.key, policy=d.policy, node=d.node,
                  victims=[v.key for v in done], reason=d.reason)
        if d.policy == "evict" and preemptor.node_name is None:
            self.cluster.patch_pod_annotations(preemptor, {C.ANN_NOMINATED_NODE: d.node})
            self.inflight[preemptor.uid] = Inflight(now, d.policy, d.node, {v.uid for v in done}, preemptor.key)
        else:  # preemptor already running: latency = decision -> action acknowledged
            M.PREEMPTION_LATENCY.labels(d.policy).observe(max(0.0, self.clock() - now))
