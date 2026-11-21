"""Cluster access abstraction.

`ClusterClient` is what the scheduler / preemption controller / autoscaler-sim talk to.
- `KubeCluster` (gpuorch.common.k8s) is the real implementation over the official `kubernetes` client.
- `FakeCluster` (here) is an in-memory implementation used by unit tests and the offline simulator.
"""

from __future__ import annotations

import copy
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from gpuorch.common.types import NodeInfo, PodInfo


@dataclass
class Event:
    ts: float
    namespace: str
    object_name: str
    kind: str
    reason: str
    message: str
    type: str = "Normal"


class ClusterClient(Protocol):
    def list_nodes(self) -> list[NodeInfo]: ...
    def list_pods(self) -> list[PodInfo]: ...
    def bind(self, pod: PodInfo, node_name: str) -> None: ...
    def delete_pod(self, pod: PodInfo, grace_period_s: int = 10) -> None: ...
    def patch_pod_annotations(self, pod: PodInfo, annotations: dict[str, str | None]) -> None: ...
    def patch_node_annotations(self, node_name: str, annotations: dict[str, str | None]) -> None: ...
    def emit_event(self, pod: PodInfo, reason: str, message: str, type: str = "Normal") -> None: ...
    def create_node(self, node: NodeInfo) -> None: ...
    def delete_node(self, node_name: str) -> None: ...
    def set_pod_phase(self, pod: PodInfo, phase: str) -> None: ...


class AgentClient(Protocol):
    """Talks to gpu-agent HTTP APIs (throttle/unthrottle)."""

    def throttle(self, node: NodeInfo, pod_uid: str) -> bool: ...
    def unthrottle(self, node: NodeInfo, pod_uid: str) -> bool: ...


class BindConflict(Exception):
    """Pod already bound / gone."""


@dataclass
class FakeCluster:
    """In-memory cluster. `on_delete` lets the simulator emulate Job controllers recreating pods."""

    clock: Callable[[], float] = time.time
    nodes: dict[str, NodeInfo] = field(default_factory=dict)
    pods: dict[str, PodInfo] = field(default_factory=dict)  # key: uid
    events: list[Event] = field(default_factory=list)
    on_delete: Callable[[PodInfo], None] | None = None
    deleted: list[PodInfo] = field(default_factory=list)
    _uids: itertools.count = field(default_factory=lambda: itertools.count(1))

    # -- helpers for tests / simulator ---------------------------------------------------------
    def add_node(self, node: NodeInfo) -> NodeInfo:
        if not node.created:
            node.created = self.clock()
        self.nodes[node.name] = node
        return node

    def add_pod(self, pod: PodInfo) -> PodInfo:
        if not pod.uid:
            pod.uid = f"uid-{next(self._uids)}"
        if not pod.created:
            pod.created = self.clock()
        self.pods[pod.uid] = pod
        return pod

    def get_pod(self, uid: str) -> PodInfo | None:
        return self.pods.get(uid)

    def pod_by_name(self, name: str, namespace: str = "default") -> PodInfo | None:
        for p in self.pods.values():
            if p.name == name and p.namespace == namespace:
                return p
        return None

    def events_for(self, name: str, reason: str | None = None) -> list[Event]:
        return [e for e in self.events if e.object_name == name and (reason is None or e.reason == reason)]

    # -- ClusterClient -----------------------------------------------------------------------
    def list_nodes(self) -> list[NodeInfo]:
        return [copy.deepcopy(n) for n in self.nodes.values()]

    def list_pods(self) -> list[PodInfo]:
        return [copy.deepcopy(p) for p in self.pods.values()]

    def bind(self, pod: PodInfo, node_name: str) -> None:
        live = self.pods.get(pod.uid)
        if live is None or live.node_name is not None:
            raise BindConflict(pod.key)
        if node_name not in self.nodes:
            raise BindConflict(f"node {node_name} not found")
        live.node_name = node_name

    def delete_pod(self, pod: PodInfo, grace_period_s: int = 10) -> None:
        live = self.pods.pop(pod.uid, None)
        if live is None:
            return
        live.deleting = True
        self.deleted.append(live)
        if self.on_delete:
            self.on_delete(live)

    def patch_pod_annotations(self, pod: PodInfo, annotations: dict[str, str | None]) -> None:
        live = self.pods.get(pod.uid)
        if live is None:
            return
        _apply(live.annotations, annotations)

    def patch_node_annotations(self, node_name: str, annotations: dict[str, str | None]) -> None:
        node = self.nodes.get(node_name)
        if node is not None:
            _apply(node.annotations, annotations)

    def emit_event(self, pod: PodInfo, reason: str, message: str, type: str = "Normal") -> None:
        self.events.append(Event(self.clock(), pod.namespace, pod.name, "Pod", reason, message, type))

    def create_node(self, node: NodeInfo) -> None:
        self.add_node(node)

    def delete_node(self, node_name: str) -> None:
        self.nodes.pop(node_name, None)

    def set_pod_phase(self, pod: PodInfo, phase: str) -> None:
        live = self.pods.get(pod.uid)
        if live is not None:
            live.phase = phase


def _apply(target: dict[str, str], patch: dict[str, str | None]) -> None:
    for k, v in patch.items():
        if v is None:
            target.pop(k, None)
        else:
            target[k] = v


@dataclass
class FakeAgentClient:
    """Records throttle calls; optionally mirrors them onto a FakeCluster node annotation."""

    calls: list[tuple[str, str, str]] = field(default_factory=list)  # (action, node, uid)
    fail: bool = False
    on_call: Callable[[str, str, str], None] | None = None

    def throttle(self, node: NodeInfo, pod_uid: str) -> bool:
        return self._call("throttle", node, pod_uid)

    def unthrottle(self, node: NodeInfo, pod_uid: str) -> bool:
        return self._call("unthrottle", node, pod_uid)

    def _call(self, action: str, node: NodeInfo, uid: str) -> bool:
        self.calls.append((action, node.name, uid))
        if self.fail:
            return False
        if self.on_call:
            self.on_call(action, node.name, uid)
        return True
