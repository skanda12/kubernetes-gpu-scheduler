from __future__ import annotations

import contextlib

from prometheus_client import CollectorRegistry, Gauge

from gpuorch.common.models import NodeGpuState


class AgentMetrics:
    """Per-agent registry so tests can create many agents without duplicate-metric errors."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        r = self.registry
        self.util = Gauge("gpu_util_percent", "GPU utilisation %", ["node", "gpu"], registry=r)
        self.mem_used = Gauge("gpu_mem_used_bytes", "GPU memory used", ["node", "gpu"], registry=r)
        self.mem_total = Gauge("gpu_mem_total_bytes", "GPU memory total", ["node", "gpu"], registry=r)
        self.temp = Gauge("gpu_temperature_celsius", "GPU temperature", ["node", "gpu"], registry=r)
        self.pod_mem = Gauge("pod_gpu_mem_bytes", "Self-reported pod GPU memory", ["pod", "priority"], registry=r)
        self.pod_util = Gauge("pod_gpu_util_percent", "Self-reported pod GPU util", ["pod", "priority"], registry=r)
        self.throttled = Gauge("pod_throttled", "1 if the pod cgroup is frozen", ["pod"], registry=r)
        self._pods: set[tuple[str, str]] = set()

    def update(self, state: NodeGpuState) -> None:
        for g in state.gpus:
            labels = (state.node, str(g.index))
            self.util.labels(*labels).set(g.utilization_percent)
            self.mem_used.labels(*labels).set(g.memory_used_bytes)
            self.mem_total.labels(*labels).set(g.memory_total_bytes)
            if g.temperature_c is not None:
                self.temp.labels(*labels).set(g.temperature_c)
        current = set()
        for p in state.pods:
            prio = p.priority_class or str(p.priority)
            key = (f"{p.namespace}/{p.name}", prio)
            current.add(key)
            self.pod_mem.labels(*key).set(p.gpu_mem_bytes or 0)
            self.pod_util.labels(*key).set(p.gpu_util_percent or 0)
            self.throttled.labels(key[0]).set(1 if p.frozen else 0)
        for gone in self._pods - current:  # drop series of finished pods
            for g in (self.pod_mem, self.pod_util):
                with contextlib.suppress(KeyError):
                    g.remove(*gone)
            with contextlib.suppress(KeyError):
                self.throttled.remove(gone[0])
        self._pods = current
