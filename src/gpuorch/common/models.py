"""Pydantic schemas shared by the agent API, the node annotation and the scheduler."""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, Field


class GpuInfo(BaseModel):
    index: int
    name: str
    uuid: str
    memory_total_bytes: int
    memory_used_bytes: int
    utilization_percent: float
    temperature_c: float | None = None
    power_w: float | None = None

    @property
    def memory_free_bytes(self) -> int:
        return max(0, self.memory_total_bytes - self.memory_used_bytes)


class PodGpuState(BaseModel):
    uid: str
    name: str
    namespace: str
    priority: int = 0
    priority_class: str | None = None
    cpu_usage_usec: int | None = None
    memory_usage_bytes: int | None = None
    frozen: bool = False
    # Self-reported by the workload's /metrics (NVML per-process is unreliable on WSL2).
    gpu_mem_bytes: int | None = None
    gpu_util_percent: float | None = None
    seconds_since_checkpoint: float | None = None


class NodeGpuState(BaseModel):
    node: str
    timestamp: float = Field(default_factory=time.time)
    backend: str
    enforcer: str
    gpus: list[GpuInfo] = []
    pods: list[PodGpuState] = []

    def summary(self, agent_url: str | None = None) -> NodeGpuSummary:
        total = sum(g.memory_total_bytes for g in self.gpus)
        used = sum(g.memory_used_bytes for g in self.gpus)
        util = sum(g.utilization_percent for g in self.gpus) / len(self.gpus) if self.gpus else 0.0
        return NodeGpuSummary(
            ts=self.timestamp,
            gpus=len(self.gpus),
            mem_total_mb=total // (1024 * 1024),
            mem_used_mb=used // (1024 * 1024),
            util=round(util, 1),
            pods={
                p.uid: PodSummary(
                    mem_mb=(p.gpu_mem_bytes or 0) // (1024 * 1024),
                    util=p.gpu_util_percent,
                    frozen=p.frozen,
                    since_ckpt=p.seconds_since_checkpoint,
                )
                for p in self.pods
            },
            agent_url=agent_url,
        )


class PodSummary(BaseModel):
    mem_mb: int = 0
    util: float | None = None
    frozen: bool = False
    since_ckpt: float | None = None


class NodeGpuSummary(BaseModel):
    """Compact form written to the node annotation `gpuorch.io/state`."""

    ts: float
    gpus: int
    mem_total_mb: int
    mem_used_mb: int
    util: float
    pods: dict[str, PodSummary] = {}
    agent_url: str | None = None

    def to_annotation(self) -> str:
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def from_annotation(cls, raw: str | None) -> NodeGpuSummary | None:
        if not raw:
            return None
        try:
            return cls.model_validate(json.loads(raw))
        except (ValueError, TypeError):
            return None


class PodRef(BaseModel):
    pod_uid: str


class LimitsRequest(BaseModel):
    pod_uid: str
    cpu_max: str | None = None  # cgroup v2 format: "<quota> <period>" or "max"
    memory_max: str | None = None  # bytes or "max"


class EnforcementAction(BaseModel):
    ts: float
    action: str  # freeze | unfreeze | limits
    pod_uid: str
    detail: dict = {}
    ok: bool = True
