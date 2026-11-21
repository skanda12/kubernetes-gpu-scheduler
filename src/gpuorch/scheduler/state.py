"""Per-node view combining Kubernetes allocations with agent GPU telemetry."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from gpuorch.common import config as C
from gpuorch.common.models import NodeGpuSummary
from gpuorch.common.priorities import GPU_CRITICAL
from gpuorch.common.types import NodeInfo, PodInfo


@dataclass
class NodeView:
    node: NodeInfo
    pods: list[PodInfo] = field(default_factory=list)  # active pods on the node
    summary: NodeGpuSummary | None = None
    stale: bool = True

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def gpu_slots_total(self) -> int:
        return self.node.alloc_gpu

    @property
    def gpu_slots_used(self) -> int:
        return sum(p.gpu for p in self.pods)

    @property
    def free_slots(self) -> int:
        return self.gpu_slots_total - self.gpu_slots_used

    @property
    def free_cpu_m(self) -> int:
        return self.node.alloc_cpu_m - sum(p.cpu_m for p in self.pods)

    @property
    def free_mem_bytes(self) -> int:
        return self.node.alloc_mem_bytes - sum(p.mem_bytes for p in self.pods)

    @property
    def gpu_mem_total_mb(self) -> int | None:
        return self.summary.mem_total_mb if self.summary and not self.stale else None

    @property
    def gpu_mem_used_mb(self) -> int | None:
        """Observed usage, corrected so each pod counts as max(reported, requested).

        Pods that were just bound haven't allocated yet, so their annotation reserves the memory."""
        if self.summary is None or self.stale:
            return None
        reported = self.summary.pods
        used = self.summary.mem_used_mb - sum(ps.mem_mb for ps in reported.values())
        for p in self.pods:
            if p.gpu <= 0:
                continue
            r = reported.get(p.uid)
            used += max(r.mem_mb if r else 0, p.gpu_mem_mb)
        return max(0, used)

    @property
    def free_gpu_mem_mb(self) -> int | None:
        total, used = self.gpu_mem_total_mb, self.gpu_mem_used_mb
        if total is None or used is None:
            return None
        return total - used

    @property
    def gpu_util(self) -> float | None:
        return self.summary.util if self.summary and not self.stale else None

    @property
    def has_critical(self) -> bool:
        return any(p.priority >= GPU_CRITICAL.value and p.gpu > 0 for p in self.pods)

    def frozen(self, uid: str) -> bool:
        if not self.summary:
            return False
        ps = self.summary.pods.get(uid)
        return bool(ps and ps.frozen)

    def pod_util(self, uid: str) -> float | None:
        if not self.summary:
            return None
        ps = self.summary.pods.get(uid)
        return ps.util if ps else None

    def since_checkpoint(self, uid: str) -> float | None:
        if not self.summary:
            return None
        ps = self.summary.pods.get(uid)
        return ps.since_ckpt if ps else None

    def assume(self, pod: PodInfo) -> None:
        """Account for a pod we just bound, before the next telemetry update."""
        pod.node_name = self.name
        self.pods.append(pod)

    def without(self, victims: list[PodInfo]) -> NodeView:
        """Hypothetical view with victims removed (used for preemption planning)."""
        uids = {v.uid for v in victims}
        return NodeView(self.node, [p for p in self.pods if p.uid not in uids], self.summary, self.stale)


def build_views(
    nodes: list[NodeInfo], pods: list[PodInfo], now: float, stale_after_s: float
) -> dict[str, NodeView]:
    by_node: dict[str, list[PodInfo]] = defaultdict(list)
    for p in pods:
        if p.active:
            by_node[p.node_name].append(p)
    views = {}
    for n in nodes:
        summary = NodeGpuSummary.from_annotation(n.annotations.get(C.ANN_NODE_STATE))
        stale = summary is None or (now - summary.ts) > stale_after_s
        views[n.name] = NodeView(n, by_node.get(n.name, []), summary, stale)
    return views
