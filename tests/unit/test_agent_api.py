from __future__ import annotations

from fastapi.testclient import TestClient

from gpuorch.agent.api import create_app
from gpuorch.agent.backends import MockBackend
from gpuorch.agent.collector import Annotator, Collector, SelfReport, parse_self_report
from gpuorch.agent.enforcer import NoopEnforcer
from gpuorch.agent.metrics import AgentMetrics
from gpuorch.common import config as C
from gpuorch.common.models import NodeGpuSummary
from tests.conftest import gpu_pod

MB = 1024 * 1024
SCEN = {"nodes": {"default": {"gpus": [{"memory_mb": 16384, "base_mem_mb": 300, "util_curve": [[0, 5]]}]}},
        "pods": {"scripted": {"gpu_mem_mb": 1000, "gpu_util": 10}}}


def make(pods, reports=None):
    backend = MockBackend("w1", SCEN, clock=lambda: 0.0)
    backend.init()
    enforcer = NoopEnforcer()
    reports = reports or {}
    col = Collector("w1", backend, enforcer, lambda: pods, lambda p: reports.get(p.name), clock=lambda: 50.0)
    metrics = AgentMetrics()
    return col, enforcer, metrics, TestClient(create_app(col, metrics))


def test_parse_self_report():
    text = (
        "# TYPE workload_gpu_mem_bytes gauge\nworkload_gpu_mem_bytes 2.097152e+09\n"
        "# TYPE workload_gpu_util_percent gauge\nworkload_gpu_util_percent 55\n"
        "# TYPE workload_seconds_since_checkpoint gauge\nworkload_seconds_since_checkpoint 12.5\n"
        "workload_gpu_index 0\n"
    )
    r = parse_self_report(text)
    assert r.gpu_mem_bytes == 2097152000 and r.gpu_util_percent == 55 and r.seconds_since_checkpoint == 12.5


def test_state_throttle_actions_metrics():
    pods = [gpu_pod("train", node="w1", mem_mb=4000), gpu_pod("scripted", node="w1"),
            gpu_pod("other-node", node="w2"), gpu_pod("cpu-only", node="w1", gpu=0)]
    col, enforcer, metrics, client = make(pods, {"train": SelfReport(4000 * MB, 70.0, 30.0)})
    metrics.update(col.collect())

    assert client.get("/healthz").json()["backend"] == "mock"
    st = client.get("/state").json()
    assert st["node"] == "w1" and len(st["pods"]) == 3
    gpu = st["gpus"][0]
    assert gpu["utilization_percent"] == 85  # 5 base + 70 + 10 scripted
    assert gpu["memory_used_bytes"] == (300 + 4000 + 1000) * MB

    r = client.post("/throttle", json={"pod_uid": "uid-train"})
    assert r.status_code == 200 and enforcer.is_frozen("uid-train")
    st = client.get("/state").json()
    assert st["gpus"][0]["utilization_percent"] == 15  # frozen: util gone
    assert st["gpus"][0]["memory_used_bytes"] == (300 + 4000 + 1000) * MB  # memory kept
    train = next(p for p in st["pods"] if p["name"] == "train")
    assert train["frozen"] and train["seconds_since_checkpoint"] == 30.0

    assert client.post("/throttle", json={"pod_uid": "uid-other-node"}).status_code == 404
    assert client.post("/unthrottle", json={"pod_uid": "uid-train"}).status_code == 200
    assert client.post("/limits", json={"pod_uid": "uid-train", "cpu_max": "50000 100000"}).status_code == 200
    assert [a["action"] for a in client.get("/actions").json()] == ["freeze", "unfreeze", "limits"]

    prom = client.get("/metrics").text
    assert 'gpu_util_percent{gpu="0",node="w1"}' in prom
    assert "gpu_mem_total_bytes" in prom and "pod_throttled" in prom and "pod_gpu_mem_bytes" in prom


def test_frozen_pod_reuses_last_report():
    pods = [gpu_pod("train", node="w1")]
    reports = {"train": SelfReport(2000 * MB, 50.0)}
    col, enforcer, _, _ = make(pods, reports)
    col.collect()
    enforcer.freeze("uid-train")
    reports.clear()  # frozen process can't answer /metrics
    st = col.collect()
    assert st.pods[0].gpu_mem_bytes == 2000 * MB and st.pods[0].frozen


def test_annotation_fallback_and_finished_pods_dropped():
    pods = [gpu_pod("annot-only", node="w1", mem_mb=1234)]
    col, _, metrics, _ = make(pods)
    st = col.collect()
    assert st.pods[0].gpu_mem_bytes == 1234 * MB
    metrics.update(st)
    pods.clear()
    metrics.update(col.collect())
    assert metrics.registry.get_sample_value("pod_throttled", {"pod": "default/annot-only"}) is None


def test_annotator_rate_limited():
    writes = []
    t = [0.0]
    ann = Annotator(lambda n, a: writes.append((n, a)), "w1", "http://10.0.0.5:9400", 5.0, clock=lambda: t[0])
    col, *_ = make([gpu_pod("x", node="w1")])
    st = col.collect()
    assert ann.maybe_write(st)
    assert not ann.maybe_write(st)
    t[0] = 6.0
    assert ann.maybe_write(st)
    assert len(writes) == 2
    summary = NodeGpuSummary.from_annotation(writes[0][1][C.ANN_NODE_STATE])
    assert summary.agent_url == "http://10.0.0.5:9400" and summary.mem_total_mb == 16384
    assert "uid-x" in summary.pods


def test_annotator_swallows_errors():
    def boom(n, a):
        raise RuntimeError("api down")

    col, *_ = make([])
    assert Annotator(boom, "w1", None).maybe_write(col.collect()) is False
    assert NodeGpuSummary.from_annotation("{not json") is None
