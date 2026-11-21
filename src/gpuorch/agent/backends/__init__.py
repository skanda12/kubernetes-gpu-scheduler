from __future__ import annotations

import logging

from gpuorch.agent.backends.base import GpuBackend, PodLoad
from gpuorch.agent.backends.mock import MockBackend
from gpuorch.agent.backends.nvml import NvmlBackend
from gpuorch.common.logging import log_event

log = logging.getLogger(__name__)

__all__ = ["GpuBackend", "PodLoad", "MockBackend", "NvmlBackend", "make_backend"]


def make_backend(kind: str, node_name: str, scenario: str | None = None) -> GpuBackend:
    """GPU_BACKEND=nvml|mock. NVML failures fall back to mock with a loud warning."""
    if kind == "nvml":
        backend: GpuBackend = NvmlBackend()
        try:
            backend.init()
            return backend
        except Exception as exc:
            log_event(
                log, "nvml_init_failed_falling_back_to_mock", logging.WARNING, error=repr(exc),
                hint="Is the Windows NVIDIA driver installed and is the pod using the nvidia runtime?",
            )
            print(f"!!! WARNING: NVML init failed ({exc!r}); using MOCK GPU backend !!!", flush=True)
    elif kind != "mock":
        raise ValueError(f"unknown GPU_BACKEND {kind!r} (expected nvml|mock)")
    backend = MockBackend(node_name, scenario)
    backend.init()
    return backend
