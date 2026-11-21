"""A model of the default kube-scheduler, used as the baseline in the offline simulator.

What it does (mirrors upstream defaults, simplified):
  * PrioritySort queue (priority desc, FIFO)
  * Filter: NodeResourcesFit on cpu/memory/nvidia.com/gpu *counts* + taints/selectors. No GPU-memory awareness.
  * Score: LeastAllocated (cpu, memory) -> spreads pods
  * Native preemption (PostFilter/DefaultPreemption): any higher-priority pod may evict lower-priority pods on a
    single node; victims = minimal set after "reprieving" the most important pods (priority desc, start asc);
    node choice minimises (highest victim priority, sum of victim priorities, #victims). No throttling, no
    checkpoint awareness.
On a real cluster the baseline *is* kube-scheduler (Jobs without schedulerName); this class is only for simulation.
"""

from __future__ import annotations

from collections.abc import Callable

from gpuorch.common import config as C
from gpuorch.common.cluster import ClusterClient
from gpuorch.common.types import PodInfo
from gpuorch.scheduler.core import queue_order
from gpuorch.scheduler.filter import filter_node
from gpuorch.scheduler.state import NodeView, build_views


def _least_allocated(v: NodeView) -> float:
    cpu = v.free_cpu_m / max(1, v.node.alloc_cpu_m)
    mem = v.free_mem_bytes / max(1, v.node.alloc_mem_bytes)
    return (cpu + mem) / 2


class DefaultSchedulerSim:
    scheduler_name = "default-scheduler"

    def __init__(self, cluster: ClusterClient, clock: Callable[[], float], grace_s: int = 10) -> None:
        self.cluster = cluster
        self.clock = clock
        self.grace_s = grace_s
        self.preemptions: list[tuple[float, str, list[str]]] = []

    def _fits(self, pod: PodInfo, view: NodeView) -> bool:
        return filter_node(pod, view, gpu_mem_aware=False).fits

    def run_once(self) -> None:
        now = self.clock()
        pods = self.cluster.list_pods()
        views = build_views(self.cluster.list_nodes(), pods, now, stale_after_s=1e9)
        pending = [p for p in pods if p.scheduler_name == self.scheduler_name and p.node_name is None
                   and p.phase == "Pending" and not p.deleting]
        for pod in queue_order(pending):
            fits = [v for v in views.values() if self._fits(pod, v)]
            nominated = pod.annotations.get(C.ANN_NOMINATED_NODE)
            if fits:
                best = max(fits, key=lambda v, nom=nominated: (v.name == nom, _least_allocated(v), v.name))
                self.cluster.bind(pod, best.name)
                best.assume(pod)
                continue
            if nominated in views and any(p.deleting for p in views[nominated].pods):
                continue  # victims still terminating: wait (like nominatedNodeName)
            self._preempt(pod, views, now)

    def _preempt(self, pod: PodInfo, views: dict[str, NodeView], now: float) -> None:
        best = None
        for view in views.values():
            lower = [p for p in view.pods if p.priority < pod.priority and not p.deleting]
            if not lower or not self._fits(pod, view.without(lower)):
                continue
            victims = list(lower)
            for cand in sorted(lower, key=lambda p: (-p.priority, p.start_time or 0)):  # reprieve important first
                trial = [v for v in victims if v.uid != cand.uid]
                if self._fits(pod, view.without(trial)):
                    victims = trial
            key = (max(v.priority for v in victims), sum(v.priority for v in victims), len(victims), view.name)
            if best is None or key < best[0]:
                best = (key, view, victims)
        if best is None:
            return
        _, view, victims = best
        for v in victims:
            self.cluster.delete_pod(v, self.grace_s)
            v.deleting = True  # still holds its slot until termination completes
        self.cluster.patch_pod_annotations(pod, {C.ANN_NOMINATED_NODE: view.name})
        self.preemptions.append((now, pod.key, [v.key for v in victims]))
