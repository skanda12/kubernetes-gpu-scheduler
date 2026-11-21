"""Real Kubernetes implementation of ClusterClient + HTTP agent client + quantity parsing."""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

import httpx

from gpuorch.common import config as C
from gpuorch.common.cluster import BindConflict
from gpuorch.common.logging import log_event
from gpuorch.common.types import NodeInfo, PodInfo, Taint, Toleration

log = logging.getLogger(__name__)

_SUFFIX = {
    "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4, "Pi": 1024**5, "Ei": 1024**6,
    "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18,
}
_QTY = re.compile(r"^([+-]?[0-9.]+(?:[eE][+-]?[0-9]+)?)([a-zA-Z]*)$")


def parse_quantity(q: str | int | float | None) -> float:
    """Parse a Kubernetes resource quantity ('500m', '2Gi', '1') into a float of base units."""
    if q is None:
        return 0.0
    if isinstance(q, int | float):
        return float(q)
    m = _QTY.match(str(q).strip())
    if not m or m.group(2) not in _SUFFIX:
        raise ValueError(f"bad quantity {q!r}")
    return float(m.group(1)) * _SUFFIX[m.group(2)]


def cpu_millis(q: Any) -> int:
    return int(round(parse_quantity(q) * 1000))


def _ts(value: dt.datetime | None) -> float | None:
    return value.timestamp() if value else None


def node_from_k8s(n: Any) -> NodeInfo:
    alloc = (n.status.allocatable or {}) if n.status else {}
    ready = any(c.type == "Ready" and c.status == "True" for c in (n.status.conditions or []))
    return NodeInfo(
        name=n.metadata.name,
        alloc_cpu_m=cpu_millis(alloc.get("cpu", "0")),
        alloc_mem_bytes=int(parse_quantity(alloc.get("memory", "0"))),
        alloc_gpu=int(parse_quantity(alloc.get(C.GPU_RESOURCE, "0"))),
        labels=dict(n.metadata.labels or {}),
        annotations=dict(n.metadata.annotations or {}),
        taints=[Taint(t.key, t.value or "", t.effect) for t in (n.spec.taints or [])],
        unschedulable=bool(n.spec.unschedulable),
        ready=ready,
        created=_ts(n.metadata.creation_timestamp) or 0.0,
    )


def pod_from_k8s(p: Any) -> PodInfo:
    cpu = mem = gpu = 0.0
    for c in p.spec.containers or []:
        req = (c.resources.requests or {}) if c.resources else {}
        lim = (c.resources.limits or {}) if c.resources else {}
        cpu += parse_quantity(req.get("cpu", "0"))
        mem += parse_quantity(req.get("memory", "0"))
        gpu += parse_quantity(lim.get(C.GPU_RESOURCE, req.get(C.GPU_RESOURCE, "0")))
    owner = None
    for ref in p.metadata.owner_references or []:
        if ref.controller:
            owner = f"{ref.kind}/{ref.name}"
    return PodInfo(
        name=p.metadata.name,
        namespace=p.metadata.namespace,
        uid=p.metadata.uid,
        node_name=p.spec.node_name,
        phase=(p.status.phase if p.status else None) or "Pending",
        scheduler_name=p.spec.scheduler_name or "default-scheduler",
        priority=p.spec.priority or 0,
        priority_class=p.spec.priority_class_name,
        cpu_m=int(round(cpu * 1000)),
        mem_bytes=int(mem),
        gpu=int(gpu),
        annotations=dict(p.metadata.annotations or {}),
        labels=dict(p.metadata.labels or {}),
        tolerations=[
            Toleration(t.key, t.operator or "Equal", t.value, t.effect) for t in (p.spec.tolerations or [])
        ],
        node_selector=dict(p.spec.node_selector or {}),
        created=_ts(p.metadata.creation_timestamp) or 0.0,
        start_time=_ts(p.status.start_time) if p.status else None,
        deleting=p.metadata.deletion_timestamp is not None,
        owner=owner,
        ip=p.status.pod_ip if p.status else None,
    )


def load_kube_config() -> None:
    from kubernetes import config as kconfig

    try:
        kconfig.load_incluster_config()
    except kconfig.ConfigException:
        kconfig.load_kube_config()


class KubeCluster:
    def __init__(self, component: str = "gpuorch") -> None:
        from kubernetes import client

        load_kube_config()
        self.v1 = client.CoreV1Api()
        self.client = client
        self.component = component

    def list_nodes(self) -> list[NodeInfo]:
        return [node_from_k8s(n) for n in self.v1.list_node().items]

    def list_pods(self, field_selector: str | None = None) -> list[PodInfo]:
        kwargs = {"field_selector": field_selector} if field_selector else {}
        return [pod_from_k8s(p) for p in self.v1.list_pod_for_all_namespaces(**kwargs).items]

    def bind(self, pod: PodInfo, node_name: str) -> None:
        c = self.client
        body = c.V1Binding(
            metadata=c.V1ObjectMeta(name=pod.name, namespace=pod.namespace),
            target=c.V1ObjectReference(api_version="v1", kind="Node", name=node_name),
        )
        try:
            # _preload_content=False avoids the client's known Binding-response deserialization bug.
            self.v1.create_namespaced_binding(pod.namespace, body, _preload_content=False)
        except c.ApiException as exc:
            if exc.status in (404, 409):
                raise BindConflict(f"{pod.key}: {exc.reason}") from exc
            raise

    def delete_pod(self, pod: PodInfo, grace_period_s: int = 10) -> None:
        try:
            self.v1.delete_namespaced_pod(pod.name, pod.namespace, grace_period_seconds=grace_period_s)
        except self.client.ApiException as exc:
            if exc.status != 404:
                raise

    def patch_pod_annotations(self, pod: PodInfo, annotations: dict[str, str | None]) -> None:
        self.v1.patch_namespaced_pod(pod.name, pod.namespace, {"metadata": {"annotations": annotations}})

    def patch_node_annotations(self, node_name: str, annotations: dict[str, str | None]) -> None:
        self.v1.patch_node(node_name, {"metadata": {"annotations": annotations}})

    def emit_event(self, pod: PodInfo, reason: str, message: str, type: str = "Normal") -> None:
        c = self.client
        now = dt.datetime.now(dt.UTC)
        ev = c.CoreV1Event(
            metadata=c.V1ObjectMeta(generate_name=f"{pod.name}.", namespace=pod.namespace),
            involved_object=c.V1ObjectReference(
                api_version="v1", kind="Pod", name=pod.name, namespace=pod.namespace, uid=pod.uid
            ),
            reason=reason,
            message=message[:1000],
            type=type,
            source=c.V1EventSource(component=self.component),
            first_timestamp=now,
            last_timestamp=now,
            count=1,
        )
        try:
            self.v1.create_namespaced_event(pod.namespace, ev)
        except Exception as exc:  # events are best effort
            log.warning("event emit failed: %s", exc)

    def create_node(self, node: NodeInfo) -> None:
        c = self.client
        body = c.V1Node(
            metadata=c.V1ObjectMeta(name=node.name, labels=node.labels, annotations=node.annotations),
            spec=c.V1NodeSpec(
                taints=[c.V1Taint(key=t.key, value=t.value, effect=t.effect) for t in node.taints]
            ),
            status=c.V1NodeStatus(
                allocatable=_node_resources(node),
                capacity=_node_resources(node),
            ),
        )
        self.v1.create_node(body)

    def delete_node(self, node_name: str) -> None:
        try:
            self.v1.delete_node(node_name)
        except self.client.ApiException as exc:
            if exc.status != 404:
                raise

    def set_pod_phase(self, pod: PodInfo, phase: str) -> None:
        """Used for pods on KWOK nodes (no kubelet): mark simulated work as finished."""
        self.v1.patch_namespaced_pod_status(pod.name, pod.namespace, {"status": {"phase": phase}})

    def patch_node_gpu_capacity(self, node_name: str, gpus: int) -> None:
        """Advertise fake nvidia.com/gpu capacity (extended resource via status patch)."""
        body = [{"op": "add", "path": "/status/capacity/nvidia.com~1gpu", "value": str(gpus)}]
        self.v1.api_client.call_api(
            f"/api/v1/nodes/{node_name}/status", "PATCH", body=body,
            header_params={"Content-Type": "application/json-patch+json"}, auth_settings=["BearerToken"],
        )


def _node_resources(node: NodeInfo) -> dict[str, str]:
    return {
        "cpu": f"{node.alloc_cpu_m}m",
        "memory": str(node.alloc_mem_bytes),
        "pods": "110",
        C.GPU_RESOURCE: str(node.alloc_gpu),
    }


class HttpAgentClient:
    """Calls gpu-agent's throttle API. Agent URL comes from the node annotation the agent writes."""

    def __init__(self, timeout_s: float = 3.0, port: int = 9400) -> None:
        self.timeout_s = timeout_s
        self.port = port

    def _url(self, node: NodeInfo) -> str | None:
        from gpuorch.common.models import NodeGpuSummary

        url = node.annotations.get(C.ANN_AGENT_URL)
        if not url:
            summary = NodeGpuSummary.from_annotation(node.annotations.get(C.ANN_NODE_STATE))
            url = summary.agent_url if summary else None
        return url.rstrip("/") if url else None

    def _post(self, node: NodeInfo, path: str, pod_uid: str) -> bool:
        url = self._url(node)
        if not url:
            log_event(log, "agent_url_missing", logging.WARNING, node=node.name)
            return False
        try:
            r = httpx.post(f"{url}{path}", json={"pod_uid": pod_uid}, timeout=self.timeout_s)
            return r.status_code == 200
        except httpx.HTTPError as exc:
            log_event(log, "agent_call_failed", logging.WARNING, node=node.name, path=path, error=str(exc))
            return False

    def throttle(self, node: NodeInfo, pod_uid: str) -> bool:
        return self._post(node, "/throttle", pod_uid)

    def unthrottle(self, node: NodeInfo, pod_uid: str) -> bool:
        return self._post(node, "/unthrottle", pod_uid)
