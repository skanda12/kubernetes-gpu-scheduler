from __future__ import annotations

from gpuorch.common import config as C
from gpuorch.common.models import PodSummary
from gpuorch.preemption.controller import ControllerConfig, PreemptionController
from gpuorch.preemption.policy import PolicyConfig, decide_pending, decide_saturation
from gpuorch.preemption.victims import (
    AntiThrash,
    best_eviction_plan,
    lost_work_s,
    select_eviction_victims,
    select_throttle_victims,
)
from gpuorch.scheduler.core import GpuScheduler
from gpuorch.scheduler.state import build_views
from tests.conftest import gpu_node, gpu_pod, set_state

NOW = 1000.0


def views_of(nodes, pods):
    return build_views(nodes, pods, NOW, 15)


# ---------------------------------------------------------------- victims
def test_antithrash_window():
    g = AntiThrash(max_n=2, window_s=10)
    p = gpu_pod("v")
    assert g.allowed(p, 0)
    g.record(p, 0)
    g.record(p, 1)
    assert not g.allowed(p, 5)
    # Job-recreated pod (new uid, same owner) is the same workload
    assert g.count(gpu_pod("v", uid="new-uid"), 5) == 2
    assert g.allowed(p, 11.5)  # both records aged out


def test_lost_work_prefers_checkpoint_report():
    node = gpu_node("n", pods={"uid-a": PodSummary(since_ckpt=12)})
    vs = views_of([node], [gpu_pod("a", node="n"), gpu_pod("b", node="n", start_time=NOW - 300)])
    assert lost_work_s(vs["n"].pods[0], vs["n"], NOW) == 12
    assert lost_work_s(vs["n"].pods[1], vs["n"], NOW) == 300


def test_victims_lowest_priority_then_least_lost_work_minimal_set():
    node = gpu_node("n", slots=3, pods={"uid-b1": PodSummary(since_ckpt=500), "uid-b2": PodSummary(since_ckpt=5)})
    pods = [gpu_pod("norm", "gpu-normal", node="n"), gpu_pod("b1", node="n"), gpu_pod("b2", node="n"),
            gpu_pod("crit", "gpu-critical", node="n")]
    pods = pods[:3]
    vs = views_of([node], pods)
    victims = select_eviction_victims(gpu_pod("c", "gpu-critical"), vs["n"], AntiThrash(), NOW)
    assert [v.name for v in victims] == ["b2"]  # one slot needed; batch with least lost work


def test_victims_need_memory_picks_enough():
    node = gpu_node("n", slots=4, mem_total_mb=10000, mem_used_mb=300)
    pods = [gpu_pod("small", node="n", mem_mb=1000), gpu_pod("big", node="n", mem_mb=6000)]
    vs = views_of([node], pods)
    victims = select_eviction_victims(gpu_pod("c", "gpu-critical", mem_mb=5000), vs["n"], AntiThrash(), NOW)
    assert [v.name for v in victims] == ["big"]  # pruning drops the unnecessary small one


def test_victims_never_equal_or_higher_or_critical_and_respect_guard():
    node = gpu_node("n", slots=2)
    pods = [gpu_pod("h1", "gpu-high", node="n"), gpu_pod("c1", "gpu-critical", node="n")]
    vs = views_of([node], pods)
    assert select_eviction_victims(gpu_pod("h2", "gpu-high"), vs["n"], AntiThrash(), NOW) is None
    assert select_eviction_victims(gpu_pod("c2", "gpu-critical"), vs["n"], AntiThrash(), NOW)[0].name == "h1"
    g = AntiThrash(max_n=1)
    g.record(pods[0], NOW)
    assert select_eviction_victims(gpu_pod("c2", "gpu-critical"), vs["n"], g, NOW) is None


def test_victims_not_for_taint_blocked_nodes():
    from gpuorch.common.types import Taint

    node = gpu_node("n", slots=1, taints=[Taint("x", "", "NoSchedule")])
    vs = views_of([node], [gpu_pod("b", node="n")])
    assert select_eviction_victims(gpu_pod("c", "gpu-critical"), vs["n"], AntiThrash(), NOW) is None


def test_best_plan_prefers_lower_priority_victims_across_nodes():
    nodes = [gpu_node("a", slots=1), gpu_node("b", slots=1)]
    pods = [gpu_pod("norm", "gpu-normal", node="a"), gpu_pod("batch", node="b")]
    plan = best_eviction_plan(gpu_pod("c", "gpu-critical"), views_of(nodes, pods), AntiThrash(), NOW)
    assert plan.node == "b" and [v.name for v in plan.victims] == ["batch"]


def test_throttle_victims_until_target():
    node = gpu_node("n", util=100, pods={"uid-b1": PodSummary(util=40), "uid-b2": PodSummary(util=30),
                                         "uid-c": PodSummary(util=30)})
    pods = [gpu_pod("b1", node="n"), gpu_pod("b2", node="n", start_time=NOW - 1), gpu_pod("c", "gpu-critical", node="n")]
    vs = views_of([node], pods)
    vic = select_throttle_victims(pods[2], vs["n"], AntiThrash(), NOW, 70, set())
    assert [v.name for v in vic] == ["b2"]  # 100-30=70 <= target; b2 cheaper (less runtime lost)
    vic = select_throttle_victims(pods[2], vs["n"], AntiThrash(), NOW, 40, set())
    assert {v.name for v in vic} == {"b1", "b2"}


# ---------------------------------------------------------------- policy branches
def test_policy_pending_evicts_and_forced_throttle_falls_back():
    nodes = [gpu_node("n", slots=1)]
    pods = [gpu_pod("b", node="n")]
    d = decide_pending(gpu_pod("c", "gpu-critical"), views_of(nodes, pods), AntiThrash(), NOW)
    assert d.policy == "evict" and d.node == "n"
    forced = gpu_pod("c", "gpu-critical", annotations={C.ANN_PREEMPT_POLICY: "throttle"})
    d = decide_pending(forced, views_of(nodes, pods), AntiThrash(), NOW)
    assert d.policy == "evict" and "falling back" in d.reason


def test_policy_pending_none_cases():
    nodes = [gpu_node("n", slots=1)]
    d = decide_pending(gpu_pod("c", "gpu-critical"), views_of(nodes, [gpu_pod("x", "gpu-critical", node="n")]),
                       AntiThrash(), NOW)
    assert d.policy == "none" and "no node" in d.reason
    d = decide_pending(gpu_pod("c", "gpu-critical"), views_of(nodes, []), AntiThrash(), NOW)
    assert d.policy == "none" and "already fits" in d.reason


def test_policy_saturation_throttle_evict_and_none():
    node = gpu_node("n", util=98, pods={"uid-b": PodSummary(util=60)})
    pods = [gpu_pod("b", node="n"), gpu_pod("c", "gpu-critical", node="n")]
    vs = views_of([node], pods)
    cfg = PolicyConfig()
    assert decide_saturation(pods[1], vs["n"], AntiThrash(), NOW, cfg, set()).policy == "throttle"
    forced = gpu_pod("c", "gpu-critical", node="n", annotations={C.ANN_PREEMPT_POLICY: "evict"})
    assert decide_saturation(forced, vs["n"], AntiThrash(), NOW, cfg, set()).policy == "evict"
    calm = views_of([gpu_node("n", util=50)], pods)
    assert decide_saturation(pods[1], calm["n"], AntiThrash(), NOW, cfg, set()).policy == "none"
    only_crit = views_of([node], [pods[1]])
    d = decide_saturation(pods[1], only_crit["n"], AntiThrash(), NOW, cfg, set())
    assert d.policy == "none" and "no preemptible" in d.reason


# ---------------------------------------------------------------- controller end-to-end on FakeCluster
def full_cluster(cluster, clock, nodes=3, slots=4):
    for i in range(nodes):
        cluster.add_node(gpu_node(f"w{i}", slots=slots, ts=clock()))
        for j in range(slots):
            cluster.add_pod(gpu_pod(f"batch-{i}-{j}", node=f"w{i}", uid=f"b{i}{j}", start_time=clock() - 10 * (j + 1)))


def test_critical_arrives_on_full_cluster_evicts_and_runs(cluster, clock, agent):
    full_cluster(cluster, clock)
    sched = GpuScheduler(cluster, clock=clock)
    ctl = PreemptionController(cluster, agent, ControllerConfig(), clock=clock)
    cluster.add_pod(gpu_pod("crit", "gpu-critical", uid="crit", created=clock()))

    sched.run_once()  # unschedulable -> hand-off
    ctl.run_once()    # evicts one batch pod
    assert len(cluster.deleted) == 1
    victim = cluster.deleted[0]
    assert victim.name.endswith("-0")  # least runtime lost on its node
    crit = cluster.get_pod("crit")
    assert crit.annotations[C.ANN_NOMINATED_NODE] == victim.node_name
    assert cluster.events_for(victim.name, "GpuPreempted")

    clock.advance(1)
    sched.run_once()
    crit = cluster.get_pod("crit")
    assert crit.node_name == victim.node_name
    cluster.set_pod_phase(crit, "Running")
    clock.advance(1)
    ctl.run_once()
    assert "crit" not in ctl.inflight  # latency observed


def test_inflight_prevents_double_eviction_and_times_out(cluster, clock, agent):
    full_cluster(cluster, clock, nodes=1, slots=2)
    cluster.add_pod(gpu_pod("crit", "gpu-critical", uid="crit",
                            annotations={C.ANN_PREEMPTION_REQUESTED: "{}"}))
    ctl = PreemptionController(cluster, agent, ControllerConfig(inflight_timeout_s=5), clock=clock)
    ctl.run_once()
    ctl.run_once()
    assert len(cluster.deleted) == 1
    clock.advance(6)
    ctl.run_once()  # timeout: nominated cleared, may re-plan next cycle
    assert C.ANN_NOMINATED_NODE not in cluster.get_pod("crit").annotations


def test_no_victims_emits_event_once(cluster, clock, agent):
    cluster.add_node(gpu_node("w0", slots=1, ts=clock()))
    cluster.add_pod(gpu_pod("c0", "gpu-critical", node="w0", uid="c0"))
    cluster.add_pod(gpu_pod("c1", "gpu-critical", uid="c1", annotations={C.ANN_PREEMPTION_REQUESTED: "{}"}))
    ctl = PreemptionController(cluster, agent, clock=clock)
    ctl.run_once()
    ctl.run_once()
    assert len(cluster.events_for("c1", "PreemptionNotPossible")) == 1 and not cluster.deleted


def test_saturation_throttle_then_unthrottle_when_critical_finishes(cluster, clock, agent):
    cluster.add_node(gpu_node("w0", util=100, ts=clock(),
                              pods={"ub": PodSummary(util=70), "uc": PodSummary(util=30)}))
    cluster.add_pod(gpu_pod("batch", node="w0", uid="ub"))
    cluster.add_pod(gpu_pod("crit", "gpu-critical", node="w0", uid="uc"))
    ctl = PreemptionController(cluster, agent, clock=clock)
    ctl.run_once()
    assert agent.calls == [("throttle", "w0", "ub")]
    assert "ub" in ctl.throttled
    ctl.run_once()  # no fresh telemetry yet -> no more throttling
    assert len(agent.calls) == 1
    cluster.set_pod_phase(cluster.get_pod("uc"), "Succeeded")
    clock.advance(1)
    ctl.run_once()
    assert agent.calls[-1] == ("unthrottle", "w0", "ub") and not ctl.throttled


def test_unthrottle_on_low_util_after_cooldown_with_headroom(cluster, clock, agent):
    cluster.add_node(gpu_node("w0", util=100, ts=clock(), pods={"ub": PodSummary(util=30), "uc": PodSummary(util=20)}))
    cluster.add_pod(gpu_pod("batch", node="w0", uid="ub"))
    cluster.add_pod(gpu_pod("crit", "gpu-critical", node="w0", uid="uc"))
    ctl = PreemptionController(cluster, agent, ControllerConfig(cooldown_s=10), clock=clock)
    ctl.run_once()
    assert ("throttle", "w0", "ub") in agent.calls
    clock.advance(1)
    set_state(cluster, "w0", ts=clock(), util=20, pods={"ub": {"util": 30, "frozen": True}, "uc": {"util": 20}})
    ctl.run_once()  # low util starts cooldown
    clock.advance(5)
    ctl.run_once()
    assert agent.calls[-1][0] == "throttle"
    clock.advance(6)
    ctl.run_once()
    assert agent.calls[-1] == ("unthrottle", "w0", "ub")


def test_unthrottle_blocked_without_headroom(cluster, clock, agent):
    cluster.add_node(gpu_node("w0", util=100, ts=clock(), pods={"ub": PodSummary(util=80), "uc": PodSummary(util=50)}))
    cluster.add_pod(gpu_pod("batch", node="w0", uid="ub"))
    cluster.add_pod(gpu_pod("crit", "gpu-critical", node="w0", uid="uc"))
    ctl = PreemptionController(cluster, agent, ControllerConfig(cooldown_s=1), clock=clock)
    ctl.run_once()
    set_state(cluster, "w0", ts=clock() + 1, util=50, pods={"ub": {"util": 80, "frozen": True}, "uc": {"util": 50}})
    for _ in range(5):
        clock.advance(2)
        ctl.run_once()
    assert all(c[0] == "throttle" for c in agent.calls)  # 50 + 80 would re-saturate


def test_adopts_frozen_pods_after_restart_and_agent_failure(cluster, clock, agent):
    cluster.add_node(gpu_node("w0", util=10, ts=clock(), pods={"ub": PodSummary(util=50, frozen=True)}))
    cluster.add_pod(gpu_pod("batch", node="w0", uid="ub"))
    agent.fail = True
    ctl = PreemptionController(cluster, agent, clock=clock)
    ctl.run_once()  # no critical pods running -> try to unfreeze, agent fails
    assert ("unthrottle", "w0", "ub") in agent.calls and "ub" in ctl.throttled
    agent.fail = False
    ctl.run_once()
    assert not ctl.throttled


def test_throttle_agent_failure_counts_nothing(cluster, clock, agent):
    cluster.add_node(gpu_node("w0", util=100, ts=clock(), pods={"ub": PodSummary(util=70)}))
    cluster.add_pod(gpu_pod("batch", node="w0", uid="ub"))
    cluster.add_pod(gpu_pod("crit", "gpu-critical", node="w0", uid="uc"))
    agent.fail = True
    ctl = PreemptionController(cluster, agent, clock=clock)
    ctl.run_once()
    assert not ctl.throttled and not ctl.decisions


def test_ignores_pods_of_other_schedulers(cluster, clock, agent):
    """Baseline isolation: default-scheduler pods keep native k8s behaviour (no throttle, no evict)."""
    cluster.add_node(gpu_node("w0", slots=1, util=100, ts=clock(), pods={"ub": PodSummary(util=70)}))
    cluster.add_pod(gpu_pod("batch", node="w0", uid="ub", ours=False))
    cluster.add_pod(gpu_pod("crit", "gpu-critical", node="w0", uid="uc", ours=False))
    cluster.add_pod(gpu_pod("crit2", "gpu-critical", uid="uc2", ours=False,
                            annotations={C.ANN_PREEMPTION_REQUESTED: "{}"}))
    ctl = PreemptionController(cluster, agent, clock=clock)
    ctl.run_once()
    assert not agent.calls and not cluster.deleted and not ctl.decisions


def test_guardrail_stops_thrashing(cluster, clock, agent):
    cfg = ControllerConfig(max_preemptions=2, window_s=1000)
    cluster.add_node(gpu_node("w0", slots=1, ts=clock()))
    ctl = PreemptionController(cluster, agent, cfg, clock=clock)
    for i in range(4):
        cluster.add_pod(gpu_pod(f"train-{i}", node="w0", uid=f"t{i}", owner="Job/train"))
        cluster.add_pod(gpu_pod(f"crit-{i}", "gpu-critical", uid=f"c{i}", annotations={C.ANN_PREEMPTION_REQUESTED: "{}"}))
        ctl.run_once()
        cluster.pods.pop(f"c{i}", None)  # critical finished / gone
        ctl.inflight.clear()
        clock.advance(1)
    assert len(cluster.deleted) == 2  # Job/train preempted at most twice in the window
