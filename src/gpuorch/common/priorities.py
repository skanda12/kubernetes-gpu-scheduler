"""PriorityClass definitions (PRD §5.2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriorityLevel:
    name: str
    value: int
    description: str


GPU_CRITICAL = PriorityLevel("gpu-critical", 1_000_000, "latency-critical inference")
GPU_HIGH = PriorityLevel("gpu-high", 100_000, "interactive / important inference")
GPU_NORMAL = PriorityLevel("gpu-normal", 10_000, "regular training")
GPU_BATCH = PriorityLevel("gpu-batch", 1_000, "best-effort batch training")

PRIORITY_LEVELS = [GPU_CRITICAL, GPU_HIGH, GPU_NORMAL, GPU_BATCH]
BY_NAME = {p.name: p for p in PRIORITY_LEVELS}


def value_of(name: str | None, default: int = 0) -> int:
    if name and name in BY_NAME:
        return BY_NAME[name].value
    return default


def class_for_value(value: int) -> str:
    """Map a raw priority value to the highest gpuorch class it reaches (or 'none')."""
    for level in PRIORITY_LEVELS:
        if value >= level.value:
            return level.name
    return "none"


def is_latency_critical(value: int) -> bool:
    """gpu-high and gpu-critical: may trigger preemption, prefer spreading."""
    return value >= GPU_HIGH.value


def can_preempt(preemptor: int, victim: int) -> bool:
    """Strictly-lower priority only; gpu-critical is never preemptible."""
    return victim < preemptor and victim < GPU_CRITICAL.value
