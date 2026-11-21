"""`make demo`: a critical job preempting batch work.

    python -m gpuorch.tools.demo --offline      # no cluster: real scheduler/controller code in the simulator
    python -m gpuorch.tools.demo                # kind: cluster up -> images -> deploy -> scenario -> summary
    python -m gpuorch.tools.demo --skip-setup   # reuse an already-deployed cluster
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from gpuorch.common import config as C

BATCH_JOBS = 12  # 3 workers x 4 slots -> cluster full


def demo_trace() -> dict:
    jobs = [
        {"name": f"batch-{i:02d}", "arrival_s": float(i % 3), "kind": "batch", "type": "batch",
         "priority_class": "gpu-batch", "duration_s": 400.0, "gpu_mem_mb": 3000, "gpu_util": 30.0,
         "checkpoint_every_s": 30.0}
        for i in range(BATCH_JOBS)
    ]
    jobs.append({"name": "critical-inference", "arrival_s": 60.0, "kind": "inference-critical", "type": "inference",
                 "priority_class": "gpu-critical", "duration_s": 120.0, "gpu_mem_mb": 2500, "gpu_util": 35.0,
                 "slo_ms": 50.0, "base_latency_ms": 18.0, "rps": 20.0})
    jobs.append({"name": "high-inference", "arrival_s": 90.0, "kind": "inference-high", "type": "inference",
                 "priority_class": "gpu-high", "duration_s": 90.0, "gpu_mem_mb": 2000, "gpu_util": 30.0,
                 "slo_ms": 120.0, "base_latency_ms": 40.0, "rps": 10.0})
    return {"name": "demo", "seed": 0, "jobs": jobs}


def offline() -> None:
    from gpuorch.experiments.simulator import SimConfig, Simulation, summarize

    print("=== SIMULATION (offline simulator, not a real cluster) ===")
    print("=== offline demo: 12 gpu-batch jobs fill 3 GPUs (4 slots each); gpu-critical arrives at t=60s ===\n")
    for policy in ("gpuorch", "default"):
        sim = Simulation(demo_trace(), policy, SimConfig(seed=1))
        res = sim.run()
        s = summarize(res)
        rows = {r["job"]: r for r in res["rows"]}
        crit = rows["critical-inference"]
        print(f"--- {policy} ---")
        if policy == "gpuorch":
            for e in sim.cluster.events:
                if e.reason in ("GpuPreempted", "PreemptionTriggered", "GpuUnthrottled"):
                    print(f"  t={e.ts:6.1f}s  {e.reason:20s} {e.object_name:22s} {e.message[:90]}")
        print(f"  critical: arrival->running {crit['queue_delay_s']:.1f}s, p95 {crit['p95_ms']} ms, "
              f"p99 {crit['p99_ms']} ms (SLO 50 ms)")
        print(f"  preemptions: {s['preemptions']}  training work lost: {s['training_lost_work_s']}s  "
              f"crashed jobs: {s['crashed_jobs']} (crash events: {s['crash_events']})  makespan: {s['makespan_s']:.0f}s\n")


def kind(skip_setup: bool, timeout: float) -> None:  # pragma: no cover - needs a live cluster
    """TODO(verify-on-kind): `make demo` once docker/kind/kubectl/helm are installed."""
    from kubernetes import client

    from gpuorch.common.k8s import load_kube_config
    from gpuorch.experiments.runner import job_manifest
    from gpuorch.tools import cluster as cl

    if not skip_setup:
        cl.up()
        cl.images()
        cl.deploy()
    load_kube_config()
    v1, batch = client.CoreV1Api(), client.BatchV1Api()
    ns = "gpuorch-demo"
    try:
        v1.delete_namespace(ns)
        cl.wait_for(lambda: ns not in [n.metadata.name for n in v1.list_namespace().items], 120, 2, "old demo ns")
    except client.ApiException:
        pass
    v1.create_namespace({"metadata": {"name": ns}})
    trace = demo_trace()
    run_id = f"demo{int(time.time())}"

    def submit(job, duration):
        m = job_manifest({**job, "duration_s": duration}, "gpuorch", ns, cl.FAKE_IMAGE, 1.0, run_id)
        m["spec"]["template"]["spec"]["tolerations"] = []  # keep demo off KWOK nodes so resume is observable
        if job["type"] != "inference":
            env = m["spec"]["template"]["spec"]["containers"][0]["env"]
            env[[e["name"] for e in env].index("CHECKPOINT_EVERY_S")]["value"] = "10"
        batch.create_namespaced_job(ns, m)
        return m["metadata"]["name"]

    def running(label_job: str) -> bool:
        pods = v1.list_namespaced_pod(ns, label_selector=f"gpuorch.io/job={label_job}").items
        return any(p.status.phase == "Running" for p in pods)

    print(f"\n[1/4] submitting {BATCH_JOBS} gpu-batch jobs (fill every slot)")
    names = [submit(j, 600) for j in trace["jobs"][:BATCH_JOBS]]
    cl.wait_for(lambda: all(running(n) for n in names), timeout, 2, "batch jobs running")
    print("      all batch jobs Running; waiting 15s so they write checkpoints")
    time.sleep(15)

    print("[2/4] submitting gpu-critical inference job")
    t0 = time.time()
    crit = submit(trace["jobs"][BATCH_JOBS], 60)
    took = cl.wait_for(lambda: running(crit), timeout, 0.5, "critical job running")
    print(f"      critical pod Running after {took:.1f}s (target < 10s)")

    print("[3/4] preemption decisions")
    events = v1.list_namespaced_event(ns).items
    victims = []
    for e in sorted(events, key=lambda e: e.last_timestamp or e.event_time or 0):
        if e.reason in ("GpuPreempted", "PreemptionTriggered", "GpuScheduled") and (
            e.reason != "GpuScheduled" or e.involved_object.name.startswith(crit)):
            print(f"      {e.reason:20s} {e.involved_object.name:28s} {e.message[:100]}")
            if e.reason == "GpuPreempted":
                victims.append(e.involved_object.name)

    print("[4/4] waiting for an evicted batch job to resume from checkpoint (after the critical job ends)")
    resumed = None
    deadline = time.time() + timeout + 120
    while time.time() < deadline and victims and not resumed:
        job = victims[0].rsplit("-", 1)[0]
        for p in v1.list_namespaced_pod(ns, label_selector=f"gpuorch.io/job={job}").items:
            if p.metadata.name == victims[0] or p.status.phase != "Running":
                continue
            log = v1.read_namespaced_pod_log(p.metadata.name, ns)
            for line in log.splitlines():
                if "resumed_from_checkpoint" in line:
                    resumed = (p.metadata.name, json.loads(line)["progress"])
        time.sleep(3)
    summary = {"critical_time_to_running_s": round(took, 2), "victims": victims,
               "resumed_from_checkpoint": resumed, "elapsed_s": round(time.time() - t0, 1)}
    print("\nSUMMARY " + json.dumps(summary, indent=2))
    if took > 10:
        print("WARNING: critical pod took longer than the 10s target", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-setup", action="store_true")
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args(argv)
    if a.offline:
        offline()
    else:
        kind(a.skip_setup, a.timeout)
    _ = C


if __name__ == "__main__":
    main()
