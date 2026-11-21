from __future__ import annotations

import datetime as dt

import pytest
from kubernetes import client as k

from gpuorch.common import config as C
from gpuorch.common.k8s import HttpAgentClient, cpu_millis, node_from_k8s, parse_quantity, pod_from_k8s
from gpuorch.common.logging import JsonFormatter, log_event, setup_logging
from gpuorch.common.priorities import can_preempt, class_for_value, value_of
from gpuorch.common.types import NodeInfo


def test_parse_quantity():
    assert parse_quantity("500m") == 0.5
    assert parse_quantity("2Gi") == 2 * 1024**3
    assert parse_quantity("1k") == 1000
    assert parse_quantity(3) == 3.0 and parse_quantity(None) == 0
    assert cpu_millis("1.5") == 1500
    with pytest.raises(ValueError):
        parse_quantity("12XB")


def test_node_and_pod_conversion():
    ts = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    node = k.V1Node(
        metadata=k.V1ObjectMeta(name="w1", labels={"a": "b"}, annotations={C.ANN_AGENT_URL: "http://x"},
                                creation_timestamp=ts),
        spec=k.V1NodeSpec(taints=[k.V1Taint(key="t", effect="NoSchedule")]),
        status=k.V1NodeStatus(
            allocatable={"cpu": "4", "memory": "8Gi", C.GPU_RESOURCE: "4"},
            conditions=[k.V1NodeCondition(type="Ready", status="True")],
        ),
    )
    n = node_from_k8s(node)
    assert (n.alloc_cpu_m, n.alloc_gpu, n.ready, n.taints[0].key) == (4000, 4, True, "t")

    pod = k.V1Pod(
        metadata=k.V1ObjectMeta(
            name="p", namespace="ns", uid="u", annotations={C.ANN_GPU_MEM_MB: "3000"}, creation_timestamp=ts,
            owner_references=[k.V1OwnerReference(api_version="batch/v1", kind="Job", name="j", uid="x", controller=True)],
        ),
        spec=k.V1PodSpec(
            scheduler_name=C.SCHEDULER_NAME, priority=1000, priority_class_name="gpu-batch",
            tolerations=[k.V1Toleration(key="t", operator="Exists")],
            containers=[k.V1Container(name="c", resources=k.V1ResourceRequirements(
                requests={"cpu": "250m", "memory": "512Mi"}, limits={C.GPU_RESOURCE: "1"}))],
        ),
        status=k.V1PodStatus(phase="Running", pod_ip="10.0.0.9", start_time=ts),
    )
    p = pod_from_k8s(pod)
    assert (p.cpu_m, p.mem_bytes, p.gpu, p.gpu_mem_mb, p.owner_key) == (250, 512 * 1024**2, 1, 3000, "ns/Job/j")
    assert p.tolerations[0].tolerates(n.taints[0]) and p.start_time == ts.timestamp()


def test_agent_client_without_url_returns_false():
    assert HttpAgentClient().throttle(NodeInfo("n"), "u") is False
    assert HttpAgentClient(timeout_s=0.2).unthrottle(NodeInfo("n", annotations={C.ANN_AGENT_URL: "http://127.0.0.1:1"}), "u") is False


def test_priorities_helpers():
    assert value_of("gpu-high") == 100000 and value_of("nope", 7) == 7
    assert class_for_value(5) == "none" and class_for_value(2_000_000) == "gpu-critical"
    assert can_preempt(1_000_000, 100_000) and not can_preempt(2_000_000, 1_000_000)


def test_json_logging(capsys):
    import json
    import logging

    setup_logging("INFO")
    log_event(logging.getLogger("t"), "decision", pod="a/b")
    rec = logging.LogRecord("t", logging.INFO, "", 0, "m", (), None)
    rec.event = "x"
    assert json.loads(JsonFormatter().format(rec))["event"] == "x"
