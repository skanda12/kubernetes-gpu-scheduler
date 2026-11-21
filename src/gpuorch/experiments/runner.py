"""Experiment runner (PRD Â§5.8): replay a trace against our scheduler and the default scheduler.

    # offline (no cluster needed): drives the real scheduler/controller code in the simulator
    python -m gpuorch.experiments.runner --trace traces/mixed-small.yaml --mode sim --reps 3
    # on kind / k3s (gpuorch Helm chart deployed): submits real Jobs with the fake (or GPU) workload image
    python -m gpuorch.experiments.runner --trace traces/mixed-small.yaml --mode kind --time-scale 0.25

Output: results/<run-id>/<scheduler>/rep<N>/{jobs.csv,util.csv,summary.json}, results/<run-id>/comparison.json,
and charts in results/<run-id>/charts/ (via analysis.py).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import time
from pathlib import Path

from gpuorch.common import config as C
from gpuorch.experiments.simulator import SimConfig, Simulation, summarize
from gpuorch.experiments.tracegen import load_trace

log = logging.getLogger("gpuorch.runner")
SCHEDULERS = ("gpuorch", "default")


def write_run(out: Path, res: dict, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in res["rows"]]
    if rows:
        with (out / "jobs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with (out / "util.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "node", "gpu_util_pct"])
        w.writerows(res["util"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def aggregate(summaries: list[dict]) -> dict:
    """Mean (and min/max) of scalar metrics across repetitions; nested dicts recurse."""
    def agg(values: list):
        vals = [v for v in values if v is not None]
        if not vals:
            return None
        if isinstance(vals[0], dict):
            keys = {k for v in vals for k in v}
            return {k: agg([v.get(k) for v in vals]) for k in sorted(keys)}
        if isinstance(vals[0], int | float) and not isinstance(vals[0], bool):
            mean = sum(vals) / len(vals)
            return round(mean, 3) if len(vals) == 1 else {"mean": round(mean, 3), "min": min(vals), "max": max(vals)}
        return vals[0]
    keys = {k for s in summaries for k in s}
    return {k: agg([s.get(k) for s in summaries]) for k in sorted(keys)}


def run_sim(trace: dict, out: Path, reps: int, schedulers: list[str], cfg_kwargs: dict) -> dict:
    comparison = {}
    for sched in schedulers:
        sums = []
        for rep in range(reps):
            cfg = SimConfig(seed=1000 + rep, **cfg_kwargs)
            t0 = time.time()
            res = Simulation(trace, sched, cfg).run()
            s = summarize(res)
            s["wallclock_s"] = round(time.time() - t0, 2)
            write_run(out / sched / f"rep{rep}", res, s)
            sums.append(s)
            print(f"  [{sched} rep{rep}] critical p95={s['latency_ms']['gpu-critical']['p95']}ms "
                  f"util={s['gpu_util_mean_pct']}% preemptions={s['preemption_count']} ooms={s['oom_failures']}")
        comparison[sched] = aggregate(sums)
    return comparison


# ------------------------------------------------------------------------------------------- kind mode
def job_manifest(job: dict, scheduler: str, namespace: str, image: str, time_scale: float, run_id: str) -> dict:
    """Kubernetes Job for one trace entry, using the fake workload (CPU-only, self-reports fake GPU metrics)."""
    dur = max(5.0, job["duration_s"] * time_scale)
    ann = {C.ANN_GPU_MEM_MB: str(job["gpu_mem_mb"]), C.ANN_DURATION_S: str(int(dur)),
           C.ANN_SIM_GPU_UTIL: str(job["gpu_util"]), C.ANN_METRICS_PORT: "8000",
           "prometheus.io/scrape": "true", "prometheus.io/port": "8000"}
    if job.get("slo_ms"):
        ann[C.ANN_LATENCY_SLO_MS] = str(job["slo_ms"])
    env = [
        {"name": "DURATION_S", "value": str(dur)},
        {"name": "FAKE_GPU_MEM_MB", "value": str(job["gpu_mem_mb"])},
        {"name": "FAKE_GPU_UTIL", "value": str(job["gpu_util"])},
        {"name": "WORKLOAD_TYPE", "value": job["type"]},
        {"name": "CHECKPOINT_EVERY_S", "value": str((job.get("checkpoint_every_s") or 0) * time_scale)},
        {"name": "CHECKPOINT_DIR", "value": f"/ckpt/{run_id}/{scheduler}/{job['name']}"},
        {"name": "BASE_LATENCY_MS", "value": str(job.get("base_latency_ms") or 0)},
    ]
    spec = {
        "priorityClassName": job["priority_class"],
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 10,
        "tolerations": [{"key": "kwok.x-k8s.io/node", "operator": "Exists", "effect": "NoSchedule"}],
        "containers": [{
            "name": "work", "image": image, "imagePullPolicy": "IfNotPresent", "env": env,
            "ports": [{"containerPort": 8000, "name": "metrics"}],
            "resources": {"requests": {"cpu": "50m", "memory": "64Mi"}, "limits": {C.GPU_RESOURCE: "1"}},
            "volumeMounts": [{"name": "ckpt", "mountPath": "/ckpt"}],
        }],
        "volumes": [{"name": "ckpt", "hostPath": {"path": "/ckpt", "type": "DirectoryOrCreate"}}],
    }
    if scheduler == "gpuorch":
        spec["schedulerName"] = C.SCHEDULER_NAME
    name = f"{job['name']}".lower().replace("_", "-")[:50]
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": namespace, "labels": {"gpuorch.io/run": run_id}},
        "spec": {"backoffLimit": 20, "ttlSecondsAfterFinished": 3600, "template": {
            "metadata": {"annotations": ann, "labels": {"gpuorch.io/run": run_id, "gpuorch.io/job": name,
                                                        "gpuorch.io/priority": job["priority_class"]}},
            "spec": spec}},
    }


def run_kind(trace: dict, out: Path, schedulers: list[str], image: str, time_scale: float, timeout_s: float,
             run_id: str, prom_url: str | None) -> dict:  # pragma: no cover - needs a live cluster
    """TODO(verify-on-kind): `make cluster-up deploy experiment-kind TRACE=traces/mixed-small.yaml`."""
    from kubernetes import client

    from gpuorch.common.k8s import load_kube_config

    load_kube_config()
    v1, batch = client.CoreV1Api(), client.BatchV1Api()
    comparison = {}
    for sched in schedulers:
        ns = f"exp-{run_id}-{sched}"[:63]
        v1.create_namespace({"metadata": {"name": ns}})
        t0 = time.time()
        pending = sorted(trace["jobs"], key=lambda j: j["arrival_s"])
        submitted: dict[str, dict] = {}
        util: list[tuple[float, str, float]] = []
        while time.time() - t0 < timeout_s:
            now = time.time() - t0
            while pending and pending[0]["arrival_s"] * time_scale <= now:
                job = pending.pop(0)
                m = job_manifest(job, sched, ns, image, time_scale, run_id)
                batch.create_namespaced_job(ns, m)
                submitted[m["metadata"]["name"]] = {**job, "submitted_at": time.time()}
            for n in v1.list_node().items:
                st = (n.metadata.annotations or {}).get(C.ANN_NODE_STATE)
                if st:
                    util.append((round(now, 1), n.metadata.name, json.loads(st).get("util", 0)))
            jobs = batch.list_namespaced_job(ns).items
            if not pending and jobs and all((j.status.succeeded or 0) >= 1 for j in jobs):
                break
            time.sleep(2)
        res = _collect_kind(v1, batch, ns, submitted, util, time.time() - t0, prom_url)
        s = summarize(res)
        write_run(out / sched / "rep0", res, s)
        comparison[sched] = s
        v1.delete_namespace(ns)
    return comparison


def _collect_kind(v1, batch, ns, submitted, util, elapsed, prom_url):  # pragma: no cover - needs a live cluster
    rows = []
    pods = v1.list_namespaced_pod(ns).items
    events = v1.list_namespaced_event(ns).items
    by_job: dict[str, list] = {}
    for p in pods:
        by_job.setdefault((p.metadata.labels or {}).get("gpuorch.io/job", "?"), []).append(p)
    preempted = _preempted_by_job(events)
    for name, job in submitted.items():
        jp = by_job.get(name, [])
        starts = [p.status.start_time.timestamp() for p in jp if p.status and p.status.start_time]
        first = min(starts) if starts else None
        restarts = max(0, len(jp) - 1)
        rows.append({
            "job": name, "type": job["type"], "priority_class": job["priority_class"],
            "arrival_s": job["arrival_s"], "first_start_s": first,
            "queue_delay_s": (first - job["submitted_at"]) if first else elapsed,
            "started": first is not None, "completed": any(p.status.phase == "Succeeded" for p in jp),
            "done_at_s": None, "duration_s": job["duration_s"], "restarts": restarts,
            "evictions": preempted.get(name, 0), "ooms": 0, "lost_work_s": 0.0, "throttled_s": 0,
            "requests": 0, "slo_ms": job.get("slo_ms"), "slo_violations": 0, "unavailable_requests": 0,
            "p50_ms": None, "p95_ms": None, "p99_ms": None, "preemption_wait_s": None, "_latencies": [],
        })
    pre = {}
    for e in events:
        if e.reason == "GpuPreempted":
            policy = "throttle" if e.message.startswith("throttle") else "evict"
            pre[(policy, "?")] = pre.get((policy, "?"), 0) + 1
        elif e.reason == "Preempted":  # kube-scheduler native preemption
            pre[("evict", "?")] = pre.get(("evict", "?"), 0) + 1
    if prom_url:
        _enrich_from_prometheus(rows, prom_url)
    return {"policy": ns, "end_s": elapsed, "rows": rows, "util": util, "node_seconds": elapsed * sum(1 for n in v1.list_node().items if int((n.status.allocatable or {}).get(C.GPU_RESOURCE, "0")) > 0),
            "preemptions": pre, "price_per_hour": 0.35}


def _preempted_by_job(events) -> dict[str, int]:  # pragma: no cover
    out: dict[str, int] = {}
    for e in events:
        if e.reason in ("GpuPreempted", "Preempted"):
            job = e.involved_object.name.rsplit("-", 1)[0]
            out[job] = out.get(job, 0) + 1
    return out


def _enrich_from_prometheus(rows: list[dict], prom_url: str) -> None:  # pragma: no cover
    """Pull inference latency quantiles per job from Prometheus (load-generator / inference metrics)."""
    import httpx

    for r in rows:
        if r["type"] != "inference":
            continue
        for q, key in ((0.5, "p50_ms"), (0.95, "p95_ms"), (0.99, "p99_ms")):
            expr = (f'1000*histogram_quantile({q}, sum by (le) (rate(inference_request_latency_seconds_bucket'
                    f'{{job_name="{r["job"]}"}}[5m])))')
            try:
                data = httpx.get(f"{prom_url}/api/v1/query", params={"query": expr}, timeout=5).json()
                res = data["data"]["result"]
                r[key] = round(float(res[0]["value"][1]), 2) if res else None
            except Exception:
                r[key] = None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--mode", choices=["sim", "kind"], default="sim")
    ap.add_argument("--schedulers", default=",".join(SCHEDULERS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--nodes", type=int, default=3, help="sim: GPU nodes")
    ap.add_argument("--slots", type=int, default=4, help="sim: time-slicing replicas per GPU")
    ap.add_argument("--autoscale", action="store_true", help="sim: enable autoscaler-sim")
    ap.add_argument("--image", default="gpuorch/fake-workload:dev", help="kind: workload image")
    ap.add_argument("--time-scale", type=float, default=0.25, help="kind: compress trace time")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--prom-url", default=None)
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)

    trace = load_trace(a.trace)
    run_id = a.run_id or f"{trace.get('name', 'trace')}-{a.mode}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    out = Path(a.out) / run_id
    scheds = [s.strip() for s in a.schedulers.split(",") if s.strip()]
    print(f"run {run_id}: {len(trace['jobs'])} jobs, mode={a.mode}, schedulers={scheds}, reps={a.reps}")
    if a.mode == "sim":
        comparison = run_sim(trace, out, a.reps, scheds, {"nodes": a.nodes, "slots": a.slots, "autoscale": a.autoscale})
    else:
        comparison = run_kind(trace, out, scheds, a.image, a.time_scale, a.timeout, run_id, a.prom_url)
    meta = {"run_id": run_id, "trace": a.trace, "mode": a.mode, "reps": a.reps, "nodes": a.nodes, "slots": a.slots}
    (out / "comparison.json").write_text(json.dumps({"meta": meta, "results": comparison}, indent=2))
    print(f"wrote {out / 'comparison.json'}")
    if not a.no_charts:
        from gpuorch.experiments.analysis import make_charts

        for p in make_charts(out):
            print(f"chart: {p}")


if __name__ == "__main__":
    main()
