"""Node filtering (PRD §5.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from gpuorch.common.types import PodInfo
from gpuorch.scheduler.state import NodeView


@dataclass
class FilterResult:
    fits: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    # Which resource blocked the pod: used by the preemption policy (slots/memory -> evict).
    blocker: str | None = None


def tolerates_all(pod: PodInfo, view: NodeView) -> bool:
    for taint in view.node.taints:
        if taint.effect not in ("NoSchedule", "NoExecute"):
            continue
        if not any(t.tolerates(taint) for t in pod.tolerations):
            return False
    return True


def filter_node(pod: PodInfo, view: NodeView, gpu_mem_aware: bool = True) -> FilterResult:
    """gpu_mem_aware=False mimics kube-scheduler: it only counts nvidia.com/gpu slots."""
    n = view.node
    if not n.ready:
        return FilterResult(False, "node not ready", blocker="node")
    if n.unschedulable:
        return FilterResult(False, "node cordoned", blocker="node")
    if not tolerates_all(pod, view):
        return FilterResult(False, "untolerated taint", blocker="node")
    for k, v in pod.node_selector.items():
        if n.labels.get(k) != v:
            return FilterResult(False, f"nodeSelector {k}={v} mismatch", blocker="node")
    if pod.cpu_m > view.free_cpu_m:
        return FilterResult(False, f"insufficient cpu ({view.free_cpu_m}m free)", blocker="cpu")
    if pod.mem_bytes > view.free_mem_bytes:
        return FilterResult(False, "insufficient memory", blocker="memory")
    if pod.gpu <= 0:
        return FilterResult(True)
    if pod.gpu > view.free_slots:
        return FilterResult(False, f"no free {pod.gpu} nvidia.com/gpu slot ({view.free_slots} free)", blocker="slots")
    if not gpu_mem_aware:
        return FilterResult(True)
    warnings = []
    need = pod.gpu_mem_mb
    free = view.free_gpu_mem_mb
    if free is None:
        warnings.append(f"agent state stale/unknown on {view.name}; placing by slot count only")
    elif need > free:
        return FilterResult(False, f"insufficient GPU memory (need {need}MiB, free {free}MiB)", blocker="gpu_mem")
    return FilterResult(True, warnings=warnings)
