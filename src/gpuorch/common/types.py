"""Lightweight, client-library-independent views of Nodes and Pods.

The scheduler, preemption controller and autoscaler operate on these types only, which keeps them unit-testable
against `FakeCluster` and lets the offline simulator drive the exact same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gpuorch.common import config as C


@dataclass
class Taint:
    key: str
    value: str = ""
    effect: str = "NoSchedule"


@dataclass
class Toleration:
    key: str | None = None
    operator: str = "Equal"
    value: str | None = None
    effect: str | None = None

    def tolerates(self, taint: Taint) -> bool:
        if self.effect and self.effect != taint.effect:
            return False
        if self.key is None:  # empty key + Exists tolerates everything
            return self.operator == "Exists"
        if self.key != taint.key:
            return False
        if self.operator == "Exists":
            return True
        return (self.value or "") == taint.value


@dataclass
class NodeInfo:
    name: str
    alloc_cpu_m: int = 8000
    alloc_mem_bytes: int = 32 * 1024**3
    alloc_gpu: int = 0
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    taints: list[Taint] = field(default_factory=list)
    unschedulable: bool = False
    ready: bool = True
    created: float = 0.0

    @property
    def simulated(self) -> bool:
        return self.labels.get(C.LABEL_SIMULATED) == "true"


TERMINAL_PHASES = {"Succeeded", "Failed"}


@dataclass
class PodInfo:
    name: str
    namespace: str = "default"
    uid: str = ""
    node_name: str | None = None
    phase: str = "Pending"
    scheduler_name: str = "default-scheduler"
    priority: int = 0
    priority_class: str | None = None
    cpu_m: int = 0
    mem_bytes: int = 0
    gpu: int = 0
    annotations: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    tolerations: list[Toleration] = field(default_factory=list)
    node_selector: dict[str, str] = field(default_factory=dict)
    created: float = 0.0
    start_time: float | None = None
    deleting: bool = False
    owner: str | None = None  # "<Kind>/<name>" of the controller owner, e.g. Job/train-3
    ip: str | None = None

    @property
    def key(self) -> str:
        return f"{self.namespace}/{self.name}"

    @property
    def owner_key(self) -> str:
        """Stable identity across Job-recreated pods (used by anti-thrashing guardrail)."""
        return f"{self.namespace}/{self.owner}" if self.owner else self.key

    @property
    def active(self) -> bool:
        """Holds node resources (bound and not terminal)."""
        return self.node_name is not None and self.phase not in TERMINAL_PHASES

    @property
    def gpu_mem_mb(self) -> int:
        try:
            return int(float(self.annotations.get(C.ANN_GPU_MEM_MB, "0")))
        except ValueError:
            return 0

    @property
    def preempt_policy(self) -> str:
        policy = self.annotations.get(C.ANN_PREEMPT_POLICY, "auto")
        return policy if policy in {"auto", "throttle", "evict"} else "auto"

    @property
    def slo_ms(self) -> float | None:
        raw = self.annotations.get(C.ANN_LATENCY_SLO_MS)
        try:
            return float(raw) if raw else None
        except ValueError:
            return None
