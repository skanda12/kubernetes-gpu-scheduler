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


def fake_job(name, pclass, mem=2000, util=30, duration=600, ours=True):
    from gpuorch.experiments.runner import job_manifest

    job = {"name": name, "type": "batch", "priority_class": pclass, "duration_s": duration, "gpu_mem_mb": mem,
           "gpu_util": util, "checkpoint_every_s": 5}
    m = job_manifest(job, "gpuorch" if ours else "default", "x", "gpuorch/fake-workload:dev", 1.0, "it")
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
    for p in pods:
        raw = v1.connect_get_namespaced_pod_proxy_with_path(f"{p.metadata.name}:9400", NS, "state")
        state = json.loads(raw.replace("'", '"')) if isinstance(raw, str) else raw
        assert state["backend"] == "mock" and state["gpus"][0]["memory_total_bytes"] == 16384 * 1024 * 1024
        metrics = v1.connect_get_namespaced_pod_proxy_with_path(f"{p.metadata.name}:9400", NS, "metrics")
        assert "gpu_util_percent" in metrics
    nodes = v1.list_node(label_selector="gpuorch.io/gpu-node=true").items
    assert all(C.ANN_NODE_STATE in (n.metadata.annotations or {}) for n in nodes)


def test_m2_ten_mixed_jobs_bound_by_our_scheduler(kube, ns):
    batch, v1 = kube.BatchV1Api(), kube.CoreV1Api()
    classes = ["gpu-batch", "gpu-normal", "gpu-critical", "gpu-high"] * 3
    for i, pc in enumerate(classes[:10]):
        batch.create_namespaced_job(ns, fake_job(f"mix-{i}", pc, mem=1000, duration=120))
    wait(lambda: len([p for p in v1.list_namespaced_pod(ns).items if p.spec.node_name]) == 10, 90)
    events = [e for e in v1.list_namespaced_event(ns).items if e.reason == "GpuScheduled"]
    assert len(events) >= 10 and all("scores=" in e.message for e in events)


def test_m3_critical_preempts_full_cluster_of_batch(kube, ns):
    batch, v1 = kube.BatchV1Api(), kube.CoreV1Api()
    for i in range(12):
        batch.create_namespaced_job(ns, fake_job(f"b-{i}", "gpu-batch", mem=3000))
    wait(lambda: sum(p.status.phase == "Running" for p in v1.list_namespaced_pod(ns).items) == 12, 180)
    time.sleep(8)  # let them checkpoint
    batch.create_namespaced_job(ns, fake_job("crit", "gpu-critical", mem=2000, duration=30))

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
