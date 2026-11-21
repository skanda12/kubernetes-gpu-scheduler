from __future__ import annotations

import json

from gpuorch.common import config as C
from gpuorch.common.models import PodSummary
from gpuorch.common.types import Taint, Toleration
from gpuorch.scheduler.core import GpuScheduler, queue_order
from gpuorch.scheduler.filter import filter_node
from gpuorch.scheduler.score import Weights, score_node
from gpuorch.scheduler.state import build_views
from tests.conftest import gpu_node, gpu_pod

NOW = 1000.0


def views_of(nodes, pods=(), now=NOW, stale=15):
    return build_views(list(nodes), list(pods), now, stale)


# ---------------------------------------------------------------- filter
def test_filter_slots_and_gpu_memory():
    node = gpu_node("n1", slots=2, mem_total_mb=8000, mem_used_mb=300)
    running = [gpu_pod("a", node="n1", mem_mb=3000)]
    v = views_of([node], running)["n1"]
    assert filter_node(gpu_pod("p", mem_mb=4000), v).fits
    r = filter_node(gpu_pod("p", mem_mb=5000), v)
    assert not r.fits and r.blocker == "gpu_mem"
    v.assume(gpu_pod("b", mem_mb=100))
    r = filter_node(gpu_pod("p", mem_mb=10), v)
    assert not r.fits and r.blocker == "slots"


def test_gpu_mem_used_counts_max_of_reported_and_requested():
    node = gpu_node("n1", mem_used_mb=300 + 1000, pods={"uid-a": PodSummary(mem_mb=1000)})
    pods = [gpu_pod("a", node="n1", mem_mb=3000), gpu_pod("b", node="n1", mem_mb=2000)]  # b: just bound
    v = views_of([node], pods)["n1"]
    assert v.gpu_mem_used_mb == 300 + 3000 + 2000


def test_filter_stale_state_slot_only_with_warning():
    node = gpu_node("n1", mem_total_mb=1000, ts=NOW - 100)
    v = views_of([node])["n1"]
    assert v.stale and v.free_gpu_mem_mb is None
    r = filter_node(gpu_pod("p", mem_mb=50000), v)
    assert r.fits and r.warnings
    assert views_of([gpu_node("n2", ts=None)])["n2"].stale


def test_filter_taints_selectors_cpu_ready():
    tainted = gpu_node("t", taints=[Taint("dedicated", "x", "NoSchedule")])
    v = views_of([tainted])["t"]
    assert filter_node(gpu_pod("p"), v).reason == "untolerated taint"
    tol = gpu_pod("p", tolerations=[Toleration("dedicated", "Equal", "x", "NoSchedule")])
    assert filter_node(tol, v).fits
    assert filter_node(gpu_pod("p", tolerations=[Toleration(None, "Exists")]), v).fits
    assert filter_node(gpu_pod("p", node_selector={"gpu": "t4"}), v).blocker == "node"
    plain = views_of([gpu_node("plain")])["plain"]
    assert filter_node(gpu_pod("p", cpu_m=99999), plain).blocker == "cpu"
    assert filter_node(gpu_pod("p", mem_bytes=10**15), plain).blocker == "memory"
    assert not filter_node(gpu_pod("p"), views_of([gpu_node("r", ready=False)])["r"]).fits
    assert not filter_node(gpu_pod("p"), views_of([gpu_node("c", unschedulable=True)])["c"]).fits
    pref = gpu_node("pref", taints=[Taint("soft", "", "PreferNoSchedule")])
    assert filter_node(gpu_pod("p"), views_of([pref])["pref"]).fits


# ---------------------------------------------------------------- score
def test_score_critical_spreads_to_idle_node():
    vs = views_of([gpu_node("busy", util=90), gpu_node("idle", util=10)])
    pod = gpu_pod("inf", "gpu-critical")
    w = Weights()
    assert score_node(pod, vs["idle"], w)[0] > score_node(pod, vs["busy"], w)[0]


def test_score_batch_binpacks_and_avoids_critical():
    nodes = [gpu_node("full", util=50), gpu_node("empty"), gpu_node("crit")]
    pods = [gpu_pod("a", node="full"), gpu_pod("b", node="full"), gpu_pod("c", "gpu-critical", node="crit"),
            gpu_pod("d", node="crit")]
    vs = views_of(nodes, pods)
    batch = gpu_pod("batch")
    s = {n: score_node(batch, vs[n], Weights())[0] for n in vs}
    assert s["full"] > s["empty"]
    assert s["full"] > s["crit"]
    _, parts = score_node(batch, vs["crit"], Weights())
    assert parts["critical_penalty"] < 0


def test_score_nominated_and_cpu_pod_and_weights_file(tmp_path):
    vs = views_of([gpu_node("a"), gpu_node("b")])
    pod = gpu_pod("x", "gpu-high")
    assert score_node(pod, vs["b"], Weights(), nominated="b")[0] > score_node(pod, vs["a"], Weights(), "b")[0]
    assert "cpu_free" in score_node(gpu_pod("c", gpu=0), vs["a"], Weights())[1]
    f = tmp_path / "w.yaml"
    f.write_text("weights:\n  spread: 3\n  bogus: 1\n")
    assert Weights.load(f).spread == 3.0
    assert Weights.load(tmp_path / "missing.yaml") == Weights()


# ---------------------------------------------------------------- queue + cycle
def test_queue_order_priority_then_fifo():
    pods = [gpu_pod("b1", created=1), gpu_pod("c1", "gpu-critical", created=5), gpu_pod("n1", "gpu-normal", created=2),
            gpu_pod("b0", created=0), gpu_pod("c0", "gpu-critical", created=3)]
    assert [p.name for p in queue_order(pods)] == ["c0", "c1", "n1", "b0", "b1"]


def test_cycle_binds_ten_mixed_jobs_highest_priority_first(cluster, clock):
    for i in range(3):
        cluster.add_node(gpu_node(f"w{i}", ts=clock()))
    classes = ["gpu-batch", "gpu-normal", "gpu-critical", "gpu-high", "gpu-batch",
               "gpu-normal", "gpu-batch", "gpu-high", "gpu-critical", "gpu-batch"]
    for i, pc in enumerate(classes):
        cluster.add_pod(gpu_pod(f"j{i}", pc, created=float(i), uid=f"u{i}"))
    cluster.add_pod(gpu_pod("not-ours", ours=False, uid="nx"))
    results = GpuScheduler(cluster, clock=clock).run_once()
    assert all(r.node for r in results) and len(results) == 10
    order = [r.pod.split("/")[1] for r in results]
    assert order[:2] == ["j2", "j8"] and order[2:4] == ["j3", "j7"]
    assert cluster.pod_by_name("not-ours").node_name is None
    ev = cluster.events_for("j2", "GpuScheduled")
    assert ev and "scores=" in ev[0].message
    # critical pods spread across different nodes
    assert cluster.pod_by_name("j2").node_name != cluster.pod_by_name("j8").node_name


def test_unschedulable_critical_hands_off_once(cluster, clock):
    cluster.add_node(gpu_node("w0", slots=1, ts=clock()))
    cluster.add_pod(gpu_pod("batch", node="w0", uid="ub"))
    cluster.add_pod(gpu_pod("crit", "gpu-critical", uid="uc"))
    cluster.add_pod(gpu_pod("lowpend", "gpu-batch", uid="ul"))
    s = GpuScheduler(cluster, clock=clock)
    res = {r.pod: r for r in s.run_once()}
    assert res["default/crit"].handed_off and not res["default/lowpend"].handed_off
    ann = json.loads(cluster.get_pod("uc").annotations[C.ANN_PREEMPTION_REQUESTED])
    assert ann["blockers"] == {"w0": "slots"}
    s.run_once()
    assert len(cluster.events_for("crit", "FailedScheduling")) == 1  # no event spam
    # once capacity frees up it binds and clears the hand-off annotation
    cluster.set_pod_phase(cluster.get_pod("ub"), "Succeeded")
    s.run_once()
    pod = cluster.get_pod("uc")
    assert pod.node_name == "w0" and C.ANN_PREEMPTION_REQUESTED not in pod.annotations
