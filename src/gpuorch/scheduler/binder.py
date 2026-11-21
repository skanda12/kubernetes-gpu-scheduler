from __future__ import annotations

import json

from gpuorch.common.cluster import ClusterClient
from gpuorch.common.types import PodInfo


def bind_with_event(cluster: ClusterClient, pod: PodInfo, node: str, scores: dict[str, dict]) -> None:
    """Bind via the Binding API and record the decision as an Event (reason: GpuScheduled)."""
    cluster.bind(pod, node)
    summary = {n: s["total"] for n, s in scores.items()}
    msg = f"bound to {node}; scores={json.dumps(summary)}; breakdown={json.dumps(scores.get(node, {}))}"
    cluster.emit_event(pod, "GpuScheduled", msg)
