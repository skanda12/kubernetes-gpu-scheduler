"""Policy selection (PRD §5.5).

auto:
  * pending pod blocked by slots / GPU memory  -> EVICT (freezing keeps slots + memory, so it can't help)
  * running high/critical pod on a saturated GPU (memory fit, compute is the problem) -> THROTTLE co-located
    lower-priority pods (cgroup freeze)
The preemptor's `gpuorch.io/preempt-policy` annotation can force `evict` or `throttle`. A forced `throttle`
for a *pending* pod can't free slots/memory, so we fall back to evict and say so in the Event.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gpuorch.common.types import PodInfo
from gpuorch.preemption.victims import AntiThrash, best_eviction_plan, select_throttle_victims
from gpuorch.scheduler.state import NodeView


@dataclass
class PolicyConfig:
    saturation_util: float = 90.0    # GPU util % at which a critical pod's node counts as saturated
    target_util: float = 70.0        # throttle until estimated util <= this
    unthrottle_util: float = 60.0    # util must be below this (for COOLDOWN_S) before unfreezing
    default_pod_util: float = 50.0   # estimate when a pod doesn't self-report util


@dataclass
class Decision:
    policy: str  # evict | throttle | none
    node: str | None = None
    victims: list[PodInfo] = field(default_factory=list)
    reason: str = ""


def decide_pending(preemptor: PodInfo, views: dict[str, NodeView], guard: AntiThrash, now: float) -> Decision:
    note = ""
    if preemptor.preempt_policy == "throttle":
        note = "preempt-policy=throttle cannot free slots/GPU memory for a pending pod; falling back to evict. "
    plan = best_eviction_plan(preemptor, views, guard, now)
    if plan is None:
        return Decision("none", reason=note + "no node can be freed by evicting lower-priority pods")
    if not plan.victims:
        return Decision("none", node=plan.node, reason="pod already fits; waiting for scheduler")
    names = ", ".join(v.key for v in plan.victims)
    return Decision(
        "evict", plan.node, plan.victims,
        note + f"slots/GPU memory blocked; evicting {names} on {plan.node} (lost work ~{plan.lost_s:.0f}s)",
    )


def decide_saturation(
    preemptor: PodInfo,
    view: NodeView,
    guard: AntiThrash,
    now: float,
    cfg: PolicyConfig,
    already_frozen: set[str],
) -> Decision:
    util = view.gpu_util
    if util is None or util < cfg.saturation_util:
        return Decision("none", reason="not saturated")
    victims = select_throttle_victims(
        preemptor, view, guard, now, cfg.target_util, already_frozen, cfg.default_pod_util
    )
    if not victims:
        return Decision("none", node=view.name, reason="saturated but no preemptible co-located pods")
    policy = "evict" if preemptor.preempt_policy == "evict" else "throttle"
    names = ", ".join(v.key for v in victims)
    return Decision(policy, view.name, victims, f"GPU saturated ({util:.0f}%); {policy} {names}")
