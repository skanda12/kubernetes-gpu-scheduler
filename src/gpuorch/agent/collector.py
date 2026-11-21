"""Collects node GPU + pod state every AGENT_INTERVAL_S."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from prometheus_client.parser import text_string_to_metric_families

from gpuorch.agent.backends import GpuBackend, MockBackend, PodLoad
from gpuorch.agent.enforcer import Enforcer
from gpuorch.common import config as C
from gpuorch.common.models import NodeGpuState, PodGpuState
from gpuorch.common.types import PodInfo

log = logging.getLogger(__name__)
MB = 1024 * 1024


@dataclass
class SelfReport:
    gpu_mem_bytes: int | None = None
    gpu_util_percent: float | None = None
    seconds_since_checkpoint: float | None = None
    gpu_index: int = 0


def parse_self_report(text: str) -> SelfReport:
    """Parse the workload /metrics contract (see workloads/README.md)."""
    rep = SelfReport()
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if s.name == "workload_gpu_mem_bytes":
                rep.gpu_mem_bytes = int(s.value)
            elif s.name == "workload_gpu_util_percent":
                rep.gpu_util_percent = float(s.value)
            elif s.name == "workload_seconds_since_checkpoint":
                rep.seconds_since_checkpoint = float(s.value)
            elif s.name == "workload_gpu_index":
                rep.gpu_index = int(s.value)
    return rep


def http_scraper(timeout_s: float = 0.5, default_port: int = 8000) -> Callable[[PodInfo], SelfReport | None]:
    def scrape(pod: PodInfo) -> SelfReport | None:
        if not pod.ip:
            return None
        port = pod.annotations.get(C.ANN_METRICS_PORT, str(default_port))
        try:
            r = httpx.get(f"http://{pod.ip}:{port}/metrics", timeout=timeout_s)
            r.raise_for_status()
            return parse_self_report(r.text)
        except (httpx.HTTPError, ValueError):
            return None

    return scrape


class Collector:
    def __init__(
        self,
        node_name: str,
        backend: GpuBackend,
        enforcer: Enforcer,
        pod_source: Callable[[], list[PodInfo]],
        scraper: Callable[[PodInfo], SelfReport | None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.node_name = node_name
        self.backend = backend
        self.enforcer = enforcer
        self.pod_source = pod_source
        self.scraper = scraper or (lambda p: None)
        self.clock = clock
        self._last_report: dict[str, SelfReport] = {}
        self._pods: dict[str, PodInfo] = {}
        self.state = NodeGpuState(node=node_name, backend=backend.name, enforcer=enforcer.name, timestamp=clock())

    def known_pod(self, uid: str) -> PodInfo | None:
        return self._pods.get(uid)

    def _report_for(self, pod: PodInfo, frozen: bool) -> SelfReport | None:
        # A frozen pod cannot answer HTTP; reuse its last report (memory stays allocated while frozen).
        rep = None if frozen else self.scraper(pod)
        if rep is None:
            rep = self._last_report.get(pod.uid)
        if rep is None and isinstance(self.backend, MockBackend):
            scripted = self.backend.scenario_pod_usage(pod.name)
            if scripted:
                rep = SelfReport(gpu_mem_bytes=scripted[0], gpu_util_percent=scripted[1])
            elif pod.gpu_mem_mb:
                rep = SelfReport(gpu_mem_bytes=pod.gpu_mem_mb * MB, gpu_util_percent=0.0)
        if rep is not None:
            self._last_report[pod.uid] = rep
        return rep

    def collect(self) -> NodeGpuState:
        try:
            pods = [p for p in self.pod_source() if p.active and p.node_name == self.node_name]
        except Exception as exc:
            log.warning("pod listing failed: %s", exc)
            pods = list(self._pods.values())
        self._pods = {p.uid: p for p in pods}
        pod_states, loads = [], []
        for p in pods:
            frozen = self.enforcer.is_frozen(p.uid)
            rep = self._report_for(p, frozen) if p.gpu > 0 else None
            cpu, mem = self.enforcer.usage(p.uid)
            pod_states.append(
                PodGpuState(
                    uid=p.uid, name=p.name, namespace=p.namespace, priority=p.priority,
                    priority_class=p.priority_class, cpu_usage_usec=cpu, memory_usage_bytes=mem, frozen=frozen,
                    gpu_mem_bytes=rep.gpu_mem_bytes if rep else None,
                    gpu_util_percent=rep.gpu_util_percent if rep else None,
                    seconds_since_checkpoint=rep.seconds_since_checkpoint if rep else None,
                )
            )
            if rep:
                loads.append(
                    PodLoad(p.uid, p.name, rep.gpu_index, rep.gpu_mem_bytes or 0, rep.gpu_util_percent or 0.0, frozen)
                )
        for uid in list(self._last_report):
            if uid not in self._pods:
                del self._last_report[uid]
        self.backend.update_pod_loads(loads)
        self.state = NodeGpuState(
            node=self.node_name, timestamp=self.clock(), backend=self.backend.name, enforcer=self.enforcer.name,
            gpus=self.backend.gpus(), pods=pod_states,
        )
        return self.state


class Annotator:
    """Writes the compact summary to node annotation `gpuorch.io/state`, rate-limited."""

    def __init__(
        self,
        patch: Callable[[str, dict[str, str | None]], None],
        node_name: str,
        agent_url: str | None,
        min_interval_s: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.patch = patch
        self.node_name = node_name
        self.agent_url = agent_url
        self.min_interval_s = min_interval_s
        self.clock = clock
        self._last = float("-inf")

    def maybe_write(self, state: NodeGpuState, force: bool = False) -> bool:
        now = self.clock()
        if not force and now - self._last < self.min_interval_s:
            return False
        ann = {C.ANN_NODE_STATE: state.summary(self.agent_url).to_annotation()}
        if self.agent_url:
            ann[C.ANN_AGENT_URL] = self.agent_url
        try:
            self.patch(self.node_name, ann)
        except Exception as exc:
            log.warning("node annotation write failed: %s", exc)
            return False
        self._last = now
        return True
