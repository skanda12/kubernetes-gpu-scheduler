from __future__ import annotations

import pytest

from gpuorch.common import config as C
from gpuorch.common.cluster import FakeAgentClient, FakeCluster
from gpuorch.common.models import NodeGpuSummary, PodSummary
from gpuorch.common.priorities import BY_NAME
from gpuorch.common.types import NodeInfo, PodInfo


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def cluster(clock) -> FakeCluster:
    return FakeCluster(clock=clock)


@pytest.fixture
def agent() -> FakeAgentClient:
    return FakeAgentClient()


def gpu_node(name: str, slots: int = 4, mem_total_mb: int = 16384, mem_used_mb: int = 300, util: float = 0.0,
             ts: float | None = 1000.0, pods: dict[str, PodSummary] | None = None, **kw) -> NodeInfo:
    ann = {}
    if ts is not None:
        ann[C.ANN_NODE_STATE] = NodeGpuSummary(
            ts=ts, gpus=1, mem_total_mb=mem_total_mb, mem_used_mb=mem_used_mb, util=util, pods=pods or {},
            agent_url=f"http://{name}:9400",
        ).to_annotation()
    return NodeInfo(name=name, alloc_gpu=slots, annotations=ann, **kw)


def gpu_pod(name: str, pclass: str = "gpu-batch", mem_mb: int = 2000, gpu: int = 1, node: str | None = None,
            phase: str | None = None, created: float = 0.0, ours: bool = True, **kw) -> PodInfo:
    ann = kw.pop("annotations", {})
    ann.setdefault(C.ANN_GPU_MEM_MB, str(mem_mb))
    return PodInfo(
        name=name, uid=kw.pop("uid", f"uid-{name}"), node_name=node,
        phase=phase or ("Running" if node else "Pending"),
        scheduler_name=C.SCHEDULER_NAME if ours else "default-scheduler",
        priority=BY_NAME[pclass].value, priority_class=pclass, gpu=gpu, cpu_m=kw.pop("cpu_m", 500),
        mem_bytes=kw.pop("mem_bytes", 1024**3), annotations=ann, created=created,
        start_time=kw.pop("start_time", 900.0 if node else None), owner=kw.pop("owner", f"Job/{name}"), **kw,
    )


def set_state(cluster: FakeCluster, node: str, **kw) -> None:
    """Rewrite a node's gpuorch.io/state annotation."""
    n = cluster.nodes[node]
    cur = NodeGpuSummary.from_annotation(n.annotations.get(C.ANN_NODE_STATE))
    data = cur.model_dump() if cur else {"ts": 0, "gpus": 1, "mem_total_mb": 16384, "mem_used_mb": 0, "util": 0}
    data.update(kw)
    n.annotations[C.ANN_NODE_STATE] = NodeGpuSummary.model_validate(data).to_annotation()
