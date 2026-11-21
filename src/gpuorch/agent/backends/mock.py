"""Scenario-driven fake GPUs (deterministic for tests).

Scenario YAML (see scenarios/*.yaml):

    nodes:
      default:                 # fallback for any node name not listed
        gpus:
          - name: "Tesla T4 (mock)"
            memory_mb: 16384
            base_mem_mb: 300           # driver/context overhead
            util_curve: [[0, 5], [60, 40], [120, 5]]   # (t_seconds, util%) piecewise linear
            loop_s: 120                # optional: repeat curve every loop_s seconds
      kind-worker2:
        gpus: [...]
    pods:                      # optional: fake usage for pods that don't self-report
      batch-train-0: {gpu_mem_mb: 4000, gpu_util: 60}

Device utilisation = curve(t) + sum(util of non-frozen pods), capped at 100.
Device memory = base + sum(pod mem) (frozen pods keep their memory, like a real cgroup freeze).
"""

from __future__ import annotations

import time
from bisect import bisect_right
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from gpuorch.agent.backends.base import GpuBackend, PodLoad
from gpuorch.common.models import GpuInfo

MB = 1024 * 1024

DEFAULT_SCENARIO: dict[str, Any] = {
    "nodes": {
        "default": {
            "gpus": [{"name": "Tesla T4 (mock)", "memory_mb": 16384, "base_mem_mb": 300, "util_curve": [[0, 0]]}]
        }
    }
}


def interpolate(curve: list[list[float]], t: float, loop_s: float | None = None) -> float:
    if not curve:
        return 0.0
    if loop_s:
        t = t % loop_s
    xs = [p[0] for p in curve]
    if t <= xs[0]:
        return float(curve[0][1])
    if t >= xs[-1]:
        return float(curve[-1][1])
    i = bisect_right(xs, t)
    (x0, y0), (x1, y1) = curve[i - 1], curve[i]
    return float(y0 + (y1 - y0) * (t - x0) / (x1 - x0))


def load_scenario(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return DEFAULT_SCENARIO
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or DEFAULT_SCENARIO


class MockBackend(GpuBackend):
    name = "mock"

    def __init__(
        self,
        node_name: str,
        scenario: dict[str, Any] | str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.node_name = node_name
        self.scenario = scenario if isinstance(scenario, dict) else load_scenario(scenario)
        self.clock = clock
        self._t0 = clock()
        self._loads: list[PodLoad] = []

    def init(self) -> None:
        nodes = self.scenario.get("nodes", {})
        self.spec = nodes.get(self.node_name) or nodes.get("default") or DEFAULT_SCENARIO["nodes"]["default"]
        if not self.spec.get("gpus"):
            raise ValueError(f"scenario has no gpus for node {self.node_name}")

    def scenario_pod_usage(self, pod_name: str) -> tuple[int, float] | None:
        """Scripted (mem_bytes, util%) for a pod that does not self-report."""
        entry = (self.scenario.get("pods") or {}).get(pod_name)
        if not entry:
            return None
        return int(entry.get("gpu_mem_mb", 0) * MB), float(entry.get("gpu_util", 0))

    def update_pod_loads(self, loads: list[PodLoad]) -> None:
        self._loads = list(loads)

    def gpus(self) -> list[GpuInfo]:
        t = self.clock() - self._t0
        out = []
        for idx, g in enumerate(self.spec["gpus"]):
            loads = [ld for ld in self._loads if ld.gpu_index == idx]
            util = interpolate(g.get("util_curve", [[0, 0]]), t, g.get("loop_s"))
            util += sum(ld.util_percent for ld in loads if not ld.frozen)
            total = int(g.get("memory_mb", 16384) * MB)
            used = int(g.get("base_mem_mb", 0) * MB) + sum(ld.mem_bytes for ld in loads)
            out.append(
                GpuInfo(
                    index=idx,
                    name=g.get("name", "Mock GPU"),
                    uuid=g.get("uuid", f"GPU-mock-{self.node_name}-{idx}"),
                    memory_total_bytes=total,
                    memory_used_bytes=min(used, total),
                    utilization_percent=round(min(100.0, util), 1),
                    temperature_c=float(g.get("temperature_c", 45 + util * 0.3)),
                    power_w=float(g.get("power_w", 20 + util * 0.5)),
                )
            )
        return out
