"""Acceptance tests on a live kind cluster with gpuorch deployed (mock backend + noop enforcer).

    make cluster-up images deploy && pytest -m integration
TODO(verify-on-kind): these have not been run yet on the dev laptop (docker/kind/kubectl/helm not installed).
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from gpuorch.common import config as C

pytestmark = pytest.mark.integration
NS = "gpuorch"


@pytest.fixture(scope="module")
def kube():
    try:
        from kubernetes import client

        from gpuorch.common.k8s import load_kube_config

        load_kube_config()
        v1 = client.CoreV1Api()
        v1.list_node()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no cluster: {exc}")
    return client


@pytest.fixture
def ns(kube):
    v1 = kube.CoreV1Api()
    name = f"it-{uuid.uuid4().hex[:6]}"
    v1.create_namespace({"metadata": {"name": name}})
    yield name
    v1.delete_namespace(name)


def wait(pred, timeout=120, interval=1.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return time.time() - t0
        time.sleep(interval)
    raise AssertionError("timeout")


def fake_job(name, pclass, mem=2000, util=30, duration=600, ours=True, ns="default"):
    from gpuorch.experiments.runner import job_manifest

    job = {"name": name, "type": "batch", "priority_class": pclass, "duration_s": duration, "gpu_mem_mb": mem,
           "gpu_util": util, "checkpoint_every_s": 5}
    m = job_manifest(job, "gpuorch" if ours else "default", ns, "gpuorch/fake-workload:dev", 1.0, "it")
    m["spec"]["template"]["spec"]["tolerations"] = []
    return m


def test_m0_workers_advertise_fake_gpus(kube):
    nodes = kube.CoreV1Api().list_node(label_selector="gpuorch.io/gpu-node=true").items
    assert len(nodes) == 3
    assert all(int(n.status.capacity.get(C.GPU_RESOURCE, 0)) >= 1 for n in nodes)


def test_m1_agents_report_scenario_state(kube):
    v1 = kube.CoreV1Api()
    pods = v1.list_namespaced_pod(NS, label_selector="app=gpu-agent").items
    assert len(pods) == 3
    def proxy(pod, path):  # raw body: the client's deserializer mangles JSON into a repr string
        r = v1.connect_get_namespaced_pod_proxy_with_path(f"{pod}:9400", NS, path, _preload_content=False)
        return r.data.decode()

    for p in pods:
        state = json.loads(proxy(p.metadata.name, "state"))
        assert state["backend"] == "mock" and state["gpus"][0]["memory_total_bytes"] == 16384 * 1024 * 1024
        assert state["node"] == p.spec.node_name
        assert 'gpu_util_percent{gpu="0",node="' in proxy(p.metadata.name, "metrics")
    nodes = v1.list_node(label_selector="gpuorch.io/gpu-node=true").items
    assert all(C.ANN_NODE_STATE in (n.metadata.annotations or {}) for n in nodes)


def test_m2_ten_mixed_jobs_bound_by_our_scheduler(kube, ns):
    batch, v1 = kube.BatchV1Api(), kube.CoreV1Api()
    classes = ["gpu-batch", "gpu-normal", "gpu-critical", "gpu-high"] * 3
    for i, pc in enumerate(classes[:10]):
        batch.create_namespaced_job(ns, fake_job(f"mix-{i}", pc, mem=1000, duration=120, ns=ns))
    wait(lambda: len([p for p in v1.list_namespaced_pod(ns).items if p.spec.node_name]) == 10, 90)
    events = [e for e in v1.list_namespaced_event(ns).items if e.reason == "GpuScheduled"]
    assert len(events) >= 10 and all("scores=" in e.message for e in events)
    # bind order (from the scheduler's structured decision log) must be highest priority first
    sched_pod = v1.list_namespaced_pod(NS, label_selector="app=gpu-scheduler").items[0].metadata.name
    order = []
    for line in v1.read_namespaced_pod_log(sched_pod, NS, tail_lines=500).splitlines():
        rec = json.loads(line) if line.startswith("{") else {}
        if rec.get("event") == "pod_scheduled" and rec["pod"].startswith(f"{ns}/"):
            order.append(rec["priority"])
    assert len(order) == 10 and order == sorted(order, reverse=True)


def _agent_actions(v1, node):
    pod = next(p for p in v1.list_namespaced_pod(NS, label_selector="app=gpu-agent").items if p.spec.node_name == node)
    r = v1.connect_get_namespaced_pod_proxy_with_path(f"{pod.metadata.name}:9400", NS, "actions", _preload_content=False)
    return json.loads(r.data.decode())


def test_m3_throttle_on_saturated_gpu_then_unthrottle(kube, ns):
    """Compute (not memory/slots) is the problem -> freeze co-located batch pods; unfreeze when critical ends."""
    batch, v1 = kube.BatchV1Api(), kube.CoreV1Api()
    node = "gpuorch-worker3"

    def pinned(m):
        m["spec"]["template"]["spec"]["nodeSelector"] = {"kubernetes.io/hostname": node}
        return m

    for i in range(3):
        batch.create_namespaced_job(ns, pinned(fake_job(f"t-{i}", "gpu-batch", mem=1000, util=35, ns=ns)))
    wait(lambda: sum(p.status.phase == "Running" for p in v1.list_namespaced_pod(ns).items) == 3, 120)
    batch.create_namespaced_job(ns, pinned(fake_job("tcrit", "gpu-critical", mem=1000, util=30, duration=40, ns=ns)))

    def throttled():
        return [e for e in v1.list_namespaced_event(ns).items
                if e.reason == "GpuPreempted" and e.message.startswith("throttle")]

    wait(lambda: throttled(), 60)
    frozen_uids = {a["pod_uid"] for a in _agent_actions(v1, node) if a["action"] == "freeze"}
    ns_uids = {p.metadata.uid for p in v1.list_namespaced_pod(ns).items}
    assert frozen_uids & ns_uids, "agent on the node recorded no freeze for our pods"
    assert not any(e.reason == "GpuPreempted" and e.message.startswith("evict")
                   for e in v1.list_namespaced_event(ns).items)  # compute problem -> no eviction

    # critical job ends after ~40s -> controller unfreezes (reason triggers_finished)
    wait(lambda: any(e.reason == "GpuUnthrottled" for e in v1.list_namespaced_event(ns).items), 120, 2)
    unfrozen = {a["pod_uid"] for a in _agent_actions(v1, node) if a["action"] == "unfreeze"}
    assert (frozen_uids & ns_uids) <= unfrozen


def test_m3_critical_preempts_full_cluster_of_batch(kube, ns):
    batch, v1 = kube.BatchV1Api(), kube.CoreV1Api()
    for i in range(12):
        batch.create_namespaced_job(ns, fake_job(f"b-{i}", "gpu-batch", mem=3000, ns=ns))
    wait(lambda: sum(p.status.phase == "Running" for p in v1.list_namespaced_pod(ns).items) == 12, 180)
    time.sleep(8)  # let them checkpoint
    batch.create_namespaced_job(ns, fake_job("crit", "gpu-critical", mem=2000, duration=30, ns=ns))

    def crit_running():
        return any(p.status.phase == "Running" for p in v1.list_namespaced_pod(ns, label_selector="gpuorch.io/job=crit").items)

    took = wait(crit_running, 60, 0.5)
    assert took < 10, f"critical pod took {took:.1f}s"
    pre = [e for e in v1.list_namespaced_event(ns).items if e.reason == "GpuPreempted"]
    assert pre and all(e.involved_object.name.startswith("b-") for e in pre)
    victim_job = pre[0].involved_object.name.rsplit("-", 1)[0]

    def resumed():
        for p in v1.list_namespaced_pod(ns, label_selector=f"gpuorch.io/job={victim_job}").items:
            if (p.status.phase == "Running" and p.metadata.name != pre[0].involved_object.name
                    and "resumed_from_checkpoint" in v1.read_namespaced_pod_log(p.metadata.name, ns)):
                return True
        return False

    wait(resumed, 120, 3)
