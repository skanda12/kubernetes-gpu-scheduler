"""Node scoring (PRD §5.3). Weights come from the scheduler ConfigMap (SCHEDULER_CONFIG file)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from gpuorch.common.priorities import is_latency_critical
from gpuorch.common.types import PodInfo
from gpuorch.scheduler.state import NodeView


@dataclass
class Weights:
    # latency-critical pods (gpu-high / gpu-critical)
    spread: float = 1.0          # prefer low GPU utilisation
    free_gpu_mem: float = 0.5    # prefer nodes with more free GPU memory
    # gpu-normal / gpu-batch
    binpack: float = 1.0         # prefer busier nodes (slots + GPU memory)
    critical_penalty: float = 0.8  # avoid nodes running gpu-critical work
    # all
    nominated: float = 2.0       # strong preference for the node preemption freed for this pod
    stale_util: float = 100.0    # utilisation assumed when telemetry is stale (pessimistic)

    @classmethod
    def load(cls, path: str | Path | None) -> Weights:
        if not path or not Path(path).exists():
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        data = data.get("weights", data)
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in known})

    def as_dict(self) -> dict:
        return asdict(self)


def score_node(pod: PodInfo, view: NodeView, w: Weights, nominated: str | None = None) -> tuple[float, dict]:
    util = view.gpu_util if view.gpu_util is not None else w.stale_util
    total_mem, free_mem = view.gpu_mem_total_mb, view.free_gpu_mem_mb
    free_mem_frac = (free_mem / total_mem) if (total_mem and free_mem is not None) else 0.0
    slot_used_frac = view.gpu_slots_used / view.gpu_slots_total if view.gpu_slots_total else 0.0
    parts: dict[str, float] = {}

    if pod.gpu <= 0:  # non-GPU pod: least allocated CPU
        parts["cpu_free"] = 100.0 * view.free_cpu_m / max(1, view.node.alloc_cpu_m)
    elif is_latency_critical(pod.priority):
        parts["spread"] = w.spread * (100.0 - util)
        parts["free_gpu_mem"] = w.free_gpu_mem * 100.0 * free_mem_frac
    else:
        mem_used_frac = 1.0 - free_mem_frac if total_mem else slot_used_frac
        parts["binpack"] = w.binpack * 100.0 * (slot_used_frac + mem_used_frac) / 2
        if view.has_critical:
            parts["critical_penalty"] = -w.critical_penalty * 100.0
    if nominated and nominated == view.name:
        parts["nominated"] = w.nominated * 100.0
    total = round(sum(parts.values()), 2)
    return total, {k: round(v, 2) for k, v in parts.items()}
