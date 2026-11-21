from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from gpuorch.common.models import GpuInfo


@dataclass
class PodLoad:
    """What a pod is (self-reportedly) doing on a GPU. Only the mock backend consumes this."""

    uid: str
    name: str
    gpu_index: int
    mem_bytes: int
    util_percent: float
    frozen: bool


class GpuBackend(ABC):
    name: str = "base"

    @abstractmethod
    def init(self) -> None:
        """Initialise the backend. Raise on failure (caller may fall back to mock)."""

    @abstractmethod
    def gpus(self) -> list[GpuInfo]: ...

    def update_pod_loads(self, loads: list[PodLoad]) -> None:  # noqa: B027 - optional hook
        """Real backends observe the hardware directly; the mock uses this to simulate load."""

    def per_process_supported(self) -> bool:
        return False

    def close(self) -> None:  # noqa: B027
        pass
