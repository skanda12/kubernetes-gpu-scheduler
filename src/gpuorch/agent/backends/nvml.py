"""Real GPUs via NVML (nvidia-ml-py). Works on native Linux and WSL2 (device-level metrics).

Per-process queries are often empty/unsupported on WSL2, so per-pod GPU numbers come from workload
self-reporting instead (PRD §5.4). TODO(verify-on-gpu): `pytest -m gpu tests/unit/test_backends.py`
inside WSL2 on a GPU laptop.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from gpuorch.agent.backends.base import GpuBackend
from gpuorch.common.models import GpuInfo

log = logging.getLogger(__name__)


class NvmlBackend(GpuBackend):
    name = "nvml"

    def __init__(self) -> None:
        self._nvml: Any = None
        self._handles: list[Any] = []

    def init(self) -> None:
        import pynvml  # provided by nvidia-ml-py

        pynvml.nvmlInit()
        self._nvml = pynvml
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            raise RuntimeError("NVML reports 0 GPUs")
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]

    @staticmethod
    def _str(v: Any) -> str:
        return v.decode() if isinstance(v, bytes) else str(v)

    def _try(self, fn, *args) -> Any:
        try:
            return fn(*args)
        except self._nvml.NVMLError:
            return None

    def gpus(self) -> list[GpuInfo]:
        n = self._nvml
        out = []
        for i, h in enumerate(self._handles):
            mem = n.nvmlDeviceGetMemoryInfo(h)
            util = self._try(n.nvmlDeviceGetUtilizationRates, h)
            temp = self._try(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU)
            power = self._try(n.nvmlDeviceGetPowerUsage, h)  # milliwatts
            out.append(
                GpuInfo(
                    index=i,
                    name=self._str(n.nvmlDeviceGetName(h)),
                    uuid=self._str(n.nvmlDeviceGetUUID(h)),
                    memory_total_bytes=int(mem.total),
                    memory_used_bytes=int(mem.used),
                    utilization_percent=float(util.gpu) if util is not None else 0.0,
                    temperature_c=float(temp) if temp is not None else None,
                    power_w=power / 1000.0 if power is not None else None,
                )
            )
        return out

    def per_process_supported(self) -> bool:
        n = self._nvml
        try:
            procs = n.nvmlDeviceGetComputeRunningProcesses(self._handles[0])
            return len(procs) > 0
        except Exception:
            return False

    def close(self) -> None:
        if self._nvml is not None:
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()
