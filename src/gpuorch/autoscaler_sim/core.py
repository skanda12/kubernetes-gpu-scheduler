"""autoscaler-sim (PRD §5.6): adds/removes simulated GPU nodes and tracks a fake cost model.

Mapping to a real cloud: this plays the role of Cluster Autoscaler + a GPU node pool.
  * scale-up trigger  = CA's "unschedulable pods" loop (here: pending GPU pods older than SCALE_UP_AFTER_S)
  * new KWOK node     = a new VM in the GKE/EKS node group (e.g. n1-standard-8 + T4, g4dn.xlarge)
  * scale-down        = CA's `scale-down-unneeded-time` (here: SCALE_DOWN_AFTER_S with no pods)
  * price             = on-demand node-hour price (NODE_PRICE_PER_HOUR, default T4 ~ $0.35/h)
On GKE use `--enable-autoscaling --min-nodes 0 --max-nodes N` on a GPU node pool; on EKS use a managed node
group with the cluster-autoscaler Helm chart and `k8s.io/cluster-autoscaler/enabled` tags.

Since KWOK nodes have no kubelet and no gpu-agent, this component also
  * writes the `gpuorch.io/state` annotation for its nodes (acting as a mock agent), and
  * completes pods on its nodes after their `gpuorch.io/duration-s` (KWOK pods never exit by themselves).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from gpuorch.common import config as C
from gpuorch.common import metrics as M
from gpuorch.common.cluster import ClusterClient
from gpuorch.common.logging import log_event
from gpuorch.common.models import NodeGpuSummary, PodSummary
from gpuorch.common.types import NodeInfo, PodInfo, Taint

log = logging.getLogger("gpuorch.autoscaler")


@dataclass
class AutoscalerConfig:
    scale_up_after_s: float = 30.0
    scale_down_after_s: float = 120.0
    scale_up_cooldown_s: float = 30.0
    max_sim_nodes: int = 3
    price_per_hour: float = 0.35
    gpus_per_node: int = 4  # time-slicing replicas of one physical GPU
    gpu_mem_mb: int = 16384
    cpu_m: int = 8000
    mem_bytes: int = 32 * 1024**3
    node_prefix: str = "gpuorch-sim"
    kwok: bool = True
    default_pod_util: float = 50.0

    @classmethod
    def from_env(cls) -> AutoscalerConfig:
        return cls(
            scale_up_after_s=C.env_float("SCALE_UP_AFTER_S", 30),
            scale_down_after_s=C.env_float("SCALE_DOWN_AFTER_S", 120),
            scale_up_cooldown_s=C.env_float("SCALE_UP_COOLDOWN_S", 30),
            max_sim_nodes=C.env_int("MAX_SIM_NODES", 3),
            price_per_hour=C.env_float("NODE_PRICE_PER_HOUR", 0.35),
            gpus_per_node=C.env_int("SIM_GPUS_PER_NODE", 4),
            gpu_mem_mb=C.env_int("SIM_GPU_MEM_MB", 16384),
            kwok=C.env_bool("SIM_KWOK", True),
        )


@dataclass
class AutoscalerSim:
    cluster: ClusterClient
    cfg: AutoscalerConfig = field(default_factory=AutoscalerConfig)
    clock: Callable[[], float] = time.time
    cost_dollars: float = 0.0
    node_seconds: float = 0.0
    _last_tick: float | None = None
    _last_scale_up: float = float("-inf")
    _empty_since: dict[str, float] = field(default_factory=dict)
    _seq: int = 0

    def make_node(self) -> NodeInfo:
        self._seq += 1
        labels = {C.LABEL_SIMULATED: "true", "type": "kwok", "kubernetes.io/role": "agent",
                  "kubernetes.io/hostname": f"{self.cfg.node_prefix}-{self._seq}"}
        annotations = {"kwok.x-k8s.io/node": "fake", "node.alpha.kubernetes.io/ttl": "0"} if self.cfg.kwok else {}
        taints = [Taint("kwok.x-k8s.io/node", "fake", "NoSchedule")] if self.cfg.kwok else []
        return NodeInfo(
            name=f"{self.cfg.node_prefix}-{int(self.clock())}-{self._seq}",
            alloc_cpu_m=self.cfg.cpu_m, alloc_mem_bytes=self.cfg.mem_bytes, alloc_gpu=self.cfg.gpus_per_node,
            labels=labels, annotations=annotations, taints=taints,
        )

    def run_once(self) -> None:
        now = self.clock()
        nodes = self.cluster.list_nodes()
        pods = self.cluster.list_pods()
        gpu_nodes = [n for n in nodes if n.alloc_gpu > 0]
        sim = [n for n in nodes if n.simulated]
        if self._last_tick is not None:
            dt = max(0.0, now - self._last_tick)
            self.node_seconds += dt * len(gpu_nodes)
            self.cost_dollars += dt * len(gpu_nodes) * self.cfg.price_per_hour / 3600.0
        self._last_tick = now
        M.SIM_NODES.set(len(sim))
        M.SIM_COST.set(self.cost_dollars)

        self._reap_and_report(sim, pods, now)

        waiting = [
            p for p in pods
            if p.gpu > 0 and p.node_name is None and p.phase == "Pending" and not p.deleting
            and now - p.created >= self.cfg.scale_up_after_s and not p.annotations.get(C.ANN_NOMINATED_NODE)
        ]
        if waiting and len(sim) < self.cfg.max_sim_nodes and now - self._last_scale_up >= self.cfg.scale_up_cooldown_s:
            node = self.make_node()
            self.cluster.create_node(node)
            self._last_scale_up = now
            M.SIM_SCALE.labels("up").inc()
            log_event(log, "scale_up", node=node.name, pending=len(waiting), sim_nodes=len(sim) + 1)

        for n in sim:
            busy = any(p.active and p.node_name == n.name for p in pods)
            if busy:
                self._empty_since.pop(n.name, None)
                continue
            since = self._empty_since.setdefault(n.name, now)
            if now - since >= self.cfg.scale_down_after_s:
                self.cluster.delete_node(n.name)
                self._empty_since.pop(n.name, None)
                M.SIM_SCALE.labels("down").inc()
                log_event(log, "scale_down", node=n.name, idle_s=round(now - since))

    def _reap_and_report(self, sim: list[NodeInfo], pods: list[PodInfo], now: float) -> None:
        for n in sim:
            on_node = [p for p in pods if p.active and p.node_name == n.name]
            for p in on_node:
                dur = p.annotations.get(C.ANN_DURATION_S)
                started = p.start_time or p.created
                if dur and p.phase == "Running" and now - started >= float(dur):
                    self.cluster.set_pod_phase(p, "Succeeded")
            gpu_pods = [p for p in on_node if p.gpu > 0]
            util = 0.0
            psum = {}
            for p in gpu_pods:
                u = float(p.annotations.get(C.ANN_SIM_GPU_UTIL, self.cfg.default_pod_util))
                util += u
                psum[p.uid] = PodSummary(mem_mb=p.gpu_mem_mb, util=u)
            summary = NodeGpuSummary(
                ts=now, gpus=1, mem_total_mb=self.cfg.gpu_mem_mb,
                mem_used_mb=min(self.cfg.gpu_mem_mb, 300 + sum(p.gpu_mem_mb for p in gpu_pods)),
                util=min(100.0, util), pods=psum,
            )
            self.cluster.patch_node_annotations(n.name, {C.ANN_NODE_STATE: summary.to_annotation()})
