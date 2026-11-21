"""Victim selection (PRD §5.5): strictly lower priority, cheapest first.

Cost order: (1) lowest priority, (2) least work lost since last checkpoint, (3) smallest set that frees
enough slots / GPU memory (greedy add, then prune unnecessary victims).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from gpuorch.common.priorities import can_preempt
from gpuorch.common.types import PodInfo
from gpuorch.scheduler.filter import filter_node
from gpuorch.scheduler.state import NodeView

RESOURCE_BLOCKERS = {"slots", "gpu_mem", "cpu", "memory"}


class AntiThrash:
    """Never preempt the same workload (Job owner) more than `max_n` times within `window_s`."""

    def __init__(self, max_n: int = 3, window_s: float = 600.0) -> None:
        self.max_n = max_n
        self.window_s = window_s
        self._hist: dict[str, deque[float]] = defaultdict(deque)

    def _trim(self, key: str, now: float) -> deque[float]:
        q = self._hist[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        return q

    def allowed(self, pod: PodInfo, now: float) -> bool:
        return len(self._trim(pod.owner_key, now)) < self.max_n

    def record(self, pod: PodInfo, now: float) -> None:
        self._trim(pod.owner_key, now).append(now)

    def count(self, pod: PodInfo, now: float) -> int:
        return len(self._trim(pod.owner_key, now))


def lost_work_s(pod: PodInfo, view: NodeView, now: float) -> float:
    """Seconds of progress lost if evicted: since last checkpoint if reported, else whole runtime."""
    since = view.since_checkpoint(pod.uid)
    if since is not None:
        return max(0.0, since)
    return max(0.0, now - pod.start_time) if pod.start_time else 0.0


def cost_key(pod: PodInfo, view: NodeView, now: float) -> tuple:
    return (pod.priority, lost_work_s(pod, view, now), -pod.gpu_mem_mb, pod.created)


def candidates(preemptor: PodInfo, view: NodeView, guard: AntiThrash, now: float, frozen_ok: bool = True) -> list[PodInfo]:
    out = [
        p for p in view.pods
        if p.gpu > 0 and not p.deleting and can_preempt(preemptor.priority, p.priority) and guard.allowed(p, now)
        and (frozen_ok or not view.frozen(p.uid))
    ]
    return sorted(out, key=lambda p: cost_key(p, view, now))


def select_eviction_victims(
    preemptor: PodInfo, view: NodeView, guard: AntiThrash, now: float
) -> list[PodInfo] | None:
    """Smallest cheap set on this node whose removal lets the preemptor pass the filter, or None."""
    base = filter_node(preemptor, view)
    if base.fits:
        return []
    if base.blocker not in RESOURCE_BLOCKERS:
        return None  # taints / selectors / readiness can't be fixed by evicting
    chosen: list[PodInfo] = []
    for cand in candidates(preemptor, view, guard, now):
        chosen.append(cand)
        if filter_node(preemptor, view.without(chosen)).fits:
            break
    else:
        return None
    # prune: drop the most expensive victims that turn out to be unnecessary
    for v in sorted(chosen, key=lambda p: cost_key(p, view, now), reverse=True):
        trial = [c for c in chosen if c.uid != v.uid]
        if filter_node(preemptor, view.without(trial)).fits:
            chosen = trial
    return chosen


@dataclass
class EvictionPlan:
    node: str
    victims: list[PodInfo]
    lost_s: float


def best_eviction_plan(
    preemptor: PodInfo, views: dict[str, NodeView], guard: AntiThrash, now: float
) -> EvictionPlan | None:
    plans = []
    for name, view in views.items():
        victims = select_eviction_victims(preemptor, view, guard, now)
        if victims is None:
            continue
        if not victims:  # fits already (state changed since scheduler gave up) - nothing to do
            return EvictionPlan(name, [], 0.0)
        lost = sum(lost_work_s(v, view, now) for v in victims)
        plans.append((max(v.priority for v in victims), lost, len(victims), name, EvictionPlan(name, victims, lost)))
    if not plans:
        return None
    return min(plans, key=lambda t: t[:4])[-1]


def select_throttle_victims(
    preemptor: PodInfo,
    view: NodeView,
    guard: AntiThrash,
    now: float,
    target_util: float,
    already_frozen: set[str],
    default_pod_util: float = 50.0,
) -> list[PodInfo]:
    """Cheapest running lower-priority pods to freeze so that estimated utilisation <= target."""
    util = view.gpu_util
    if util is None:
        return []
    running = [p for p in view.pods if p.gpu > 0 and p.phase == "Running" and p.uid not in already_frozen
               and not view.frozen(p.uid)]
    share = util / len(running) if running else default_pod_util

    def est(p: PodInfo) -> float:
        u = view.pod_util(p.uid)
        return u if u is not None else share

    chosen: list[PodInfo] = []
    for cand in candidates(preemptor, view, guard, now, frozen_ok=False):
        if cand.uid in already_frozen or cand.phase != "Running":
            continue
        if util <= target_util:
            break
        chosen.append(cand)
        util -= est(cand)
    return chosen
