"""Offline discrete-time cluster simulator.

Runs the *real* GpuScheduler + PreemptionController (+ optional AutoscalerSim) against an in-memory FakeCluster,
or the DefaultSchedulerSim baseline, on the same trace + seed. Lets the team iterate on policies and produce
comparison charts without Docker/kind/GPUs; kind (M4) and real-GPU (M6) runs use `runner.py --mode kind`.

GPU model (time-slicing): each node = one physical GPU exposed as `slots` nvidia.com/gpu replicas.
  * demand D = sum of gpu_util of running, non-frozen pods; each pod progresses at share = min(1, 100/D)
  * memory: if a starting pod would push actual GPU memory over capacity it dies with CUDA OOM (only possible
    when the scheduler ignores GPU memory, i.e. the baseline) and its Job restarts after a backoff
  * frozen (throttled) pods: progress 0, demand 0, memory kept
  * inference latency = base_latency / share * lognormal noise; requests that arrive while the service is not yet
    running count as unavailable (SLO violations)
  * evicted training/batch pods lose progress since their last checkpoint and are recreated by the "Job"
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from gpuorch.autoscaler_sim.core import AutoscalerConfig, AutoscalerSim
from gpuorch.common import config as C
from gpuorch.common.cluster import FakeAgentClient, FakeCluster
from gpuorch.common.models import NodeGpuSummary, PodSummary
from gpuorch.common.priorities import value_of
from gpuorch.common.types import NodeInfo, PodInfo
from gpuorch.experiments.baseline import DefaultSchedulerSim
from gpuorch.preemption.controller import ControllerConfig, PreemptionController
from gpuorch.scheduler.core import GpuScheduler

POLICIES = ("gpuorch", "default")


@dataclass
class SimConfig:
    nodes: int = 3
    slots: int = 4
    gpu_mem_mb: int = 16384
    base_mem_mb: int = 300
    tick_s: float = 1.0
    startup_s: float = 2.0
    startup_jitter_s: float = 3.0
    grace_s: int = 10           # terminationGracePeriodSeconds given to evicted pods (upper bound)
    termination_s: float = 2.0  # how long workloads actually take to exit on SIGTERM
    restart_backoff_s: float = 10.0
    job_recreate_s: float = 1.0
    max_time_s: float = 4 * 3600
    price_per_hour: float = 0.35
    autoscale: bool = False
    max_sim_nodes: int = 2
    seed: int = 0
    max_samples_per_tick: int = 20
    controller: ControllerConfig = field(default_factory=lambda: ControllerConfig(cooldown_s=20))


class SimClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@dataclass
class JobState:
    spec: dict
    progress: float = 0.0
    ckpt: float = 0.0
    restarts: int = 0
    evictions: int = 0
    ooms: int = 0
    lost_s: float = 0.0
    throttled_s: float = 0.0
    pod_uid: str | None = None
    pod_seq: int = 0
    submitted: bool = False
    next_pod_at: float | None = None
    first_start: float | None = None
    done_at: float | None = None
    requests: float = 0.0
    unavailable: float = 0.0
    slow: float = 0.0
    latencies: list[float] = field(default_factory=list)
    nominated_at: float | None = None
    preempt_latencies: list[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def is_inference(self) -> bool:
        return self.spec["type"] == "inference"


class Simulation:
    def __init__(self, trace: dict, policy: str, cfg: SimConfig | None = None) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.cfg = cfg or SimConfig()
        self.policy = policy
        self.rng = random.Random(self.cfg.seed)
        self.clock = SimClock()
        self.cluster = FakeCluster(clock=self.clock, on_delete=self._on_delete)
        for i in range(self.cfg.nodes):
            self.cluster.add_node(self._node(f"gpu-node-{i}"))
        self.jobs = [JobState(dict(j)) for j in sorted(trace["jobs"], key=lambda j: j["arrival_s"])]
        self.by_uid: dict[str, JobState] = {}
        self.frozen: set[str] = set()
        self.terminating: dict[str, float] = {}
        self.bound_at: dict[str, float] = {}
        self.preemptions: Counter = Counter()
        self.util_samples: list[tuple[float, str, float]] = []
        self.node_seconds = 0.0
        self.agent = FakeAgentClient(on_call=self._agent_call)
        if policy == "gpuorch":
            self.scheduler = GpuScheduler(self.cluster, clock=self.clock, stale_after_s=15)
            ccfg = self.cfg.controller
            ccfg.evict_grace_s = self.cfg.grace_s
            self.controller: PreemptionController | None = PreemptionController(self.cluster, self.agent, ccfg, self.clock)
            self.sched_name = C.SCHEDULER_NAME
        else:
            self.scheduler = DefaultSchedulerSim(self.cluster, self.clock, self.cfg.grace_s)
            self.controller = None
            self.sched_name = "default-scheduler"
        self.autoscaler = None
        if self.cfg.autoscale:
            acfg = AutoscalerConfig(scale_up_after_s=30, scale_down_after_s=120, max_sim_nodes=self.cfg.max_sim_nodes,
                                    price_per_hour=self.cfg.price_per_hour, gpus_per_node=self.cfg.slots,
                                    gpu_mem_mb=self.cfg.gpu_mem_mb, kwok=False)
            self.autoscaler = AutoscalerSim(self.cluster, acfg, self.clock)

    # -- cluster plumbing -------------------------------------------------------------------------
    def _node(self, name: str) -> NodeInfo:
        return NodeInfo(name, alloc_cpu_m=16000, alloc_mem_bytes=64 * 1024**3, alloc_gpu=self.cfg.slots)

    def _agent_call(self, action: str, node: str, uid: str) -> None:
        if action == "throttle":
            self.frozen.add(uid)
            pod = self.cluster.get_pod(uid)
            self.preemptions[("throttle", pod.priority_class if pod else "?")] += 1
        else:
            self.frozen.discard(uid)

    def _on_delete(self, pod: PodInfo) -> None:
        now = self.clock()
        pod.deleting = True
        self.cluster.pods[pod.uid] = pod  # keeps holding resources while terminating
        self.terminating[pod.uid] = now + min(self.cfg.grace_s, self.cfg.termination_s)
        self.frozen.discard(pod.uid)
        self.preemptions[("evict", pod.priority_class)] += 1
        job = self.by_uid.get(pod.uid)
        if job and job.pod_uid == pod.uid:
            self._lose_progress(job)
            job.evictions += 1
            job.pod_uid = None
            job.next_pod_at = now + self.cfg.job_recreate_s

    def _lose_progress(self, job: JobState) -> None:
        if not job.is_inference:
            job.lost_s += job.progress - job.ckpt
            job.progress = job.ckpt

    def _submit_pod(self, job: JobState) -> None:
        s = job.spec
        job.pod_seq += 1
        ann = {C.ANN_GPU_MEM_MB: str(s["gpu_mem_mb"]), C.ANN_SIM_GPU_UTIL: str(s["gpu_util"])}
        if s.get("slo_ms"):
            ann[C.ANN_LATENCY_SLO_MS] = str(s["slo_ms"])
        pod = PodInfo(
            name=f"{s['name']}-{job.pod_seq}", scheduler_name=self.sched_name, priority=value_of(s["priority_class"]),
            priority_class=s["priority_class"], cpu_m=1000, mem_bytes=2 * 1024**3, gpu=1, annotations=ann,
            owner=f"Job/{s['name']}",
        )
        self.cluster.add_pod(pod)
        job.pod_uid = pod.uid
        job.submitted = True
        job.next_pod_at = None
        self.by_uid[pod.uid] = job

    # -- main loop --------------------------------------------------------------------------------
    def run(self) -> dict:
        dt = self.cfg.tick_s
        while self.clock.t < self.cfg.max_time_s and any(j.done_at is None for j in self.jobs):
            t = self.clock.t
            for job in self.jobs:
                if job.done_at is not None:
                    continue
                if not job.submitted and job.spec["arrival_s"] <= t or job.next_pod_at is not None and job.next_pod_at <= t:
                    self._submit_pod(job)
            self._publish_state()
            self.scheduler.run_once()
            if self.controller:
                self.controller.run_once()
            if self.autoscaler:
                self.autoscaler.run_once()
            self._lifecycle(t)
            self._progress(t, dt)
            self.node_seconds += dt * sum(1 for n in self.cluster.nodes.values() if n.alloc_gpu > 0)
            self.clock.t += dt
        return self.results()

    def _publish_state(self) -> None:
        """Play the gpu-agent: write fresh gpuorch.io/state annotations."""
        now = self.clock()
        by_node: dict[str, list[PodInfo]] = defaultdict(list)
        for p in self.cluster.pods.values():
            if p.node_name and p.phase == "Running":
                by_node[p.node_name].append(p)
        for name in self.cluster.nodes:
            pods = by_node.get(name, [])
            util = sum(float(p.annotations[C.ANN_SIM_GPU_UTIL]) for p in pods if p.uid not in self.frozen and not p.deleting)
            mem = self.cfg.base_mem_mb + sum(p.gpu_mem_mb for p in pods)
            psum = {}
            for p in pods:
                job = self.by_uid.get(p.uid)
                since = (job.progress - job.ckpt) if job and not job.is_inference else None
                psum[p.uid] = PodSummary(mem_mb=p.gpu_mem_mb, util=float(p.annotations[C.ANN_SIM_GPU_UTIL]),
                                         frozen=p.uid in self.frozen, since_ckpt=since)
            summary = NodeGpuSummary(ts=now, gpus=1, mem_total_mb=self.cfg.gpu_mem_mb, mem_used_mb=min(mem, self.cfg.gpu_mem_mb),
                                     util=min(100.0, util), pods=psum)
            self.cluster.nodes[name].annotations[C.ANN_NODE_STATE] = summary.to_annotation()

    def _lifecycle(self, t: float) -> None:
        for uid, until in list(self.terminating.items()):
            if until <= t:
                self.cluster.pods.pop(uid, None)
                del self.terminating[uid]
        for pod in list(self.cluster.pods.values()):
            job = self.by_uid.get(pod.uid)
            if job and pod.annotations.get(C.ANN_NOMINATED_NODE) and job.nominated_at is None:
                job.nominated_at = t
            if pod.node_name is None or pod.phase != "Pending" or pod.deleting:
                continue
            if pod.uid not in self.bound_at:  # container start time varies (image pull, CUDA init)
                self.bound_at[pod.uid] = t + self.rng.uniform(0, self.cfg.startup_jitter_s)
            if t - self.bound_at[pod.uid] < self.cfg.startup_s:
                continue
            used = self.cfg.base_mem_mb + sum(
                p.gpu_mem_mb for p in self.cluster.pods.values() if p.node_name == pod.node_name and p.phase == "Running"
            )
            if used + pod.gpu_mem_mb > self.cfg.gpu_mem_mb:  # CUDA OOM at startup
                pod.phase = "Failed"
                if job:
                    job.ooms += 1
                    job.restarts += 1
                    job.pod_uid = None
                    job.next_pod_at = t + self.cfg.restart_backoff_s
                continue
            pod.phase = "Running"
            pod.start_time = t
            if job:
                if job.first_start is None:
                    job.first_start = t
                if job.nominated_at is not None:
                    job.preempt_latencies.append(t - job.nominated_at)
                    job.nominated_at = None

    def _progress(self, t: float, dt: float) -> None:
        running: dict[str, list[PodInfo]] = defaultdict(list)
        for p in self.cluster.pods.values():
            if p.node_name and p.phase == "Running" and not p.deleting:
                running[p.node_name].append(p)
        for name in self.cluster.nodes:
            pods = running.get(name, [])
            demand = sum(float(p.annotations[C.ANN_SIM_GPU_UTIL]) for p in pods if p.uid not in self.frozen)
            share = min(1.0, 100.0 / demand) if demand > 0 else 1.0
            self.util_samples.append((t, name, min(100.0, demand)))
            for p in pods:
                job = self.by_uid.get(p.uid)
                if job is None or job.pod_uid != p.uid:
                    continue
                if p.uid in self.frozen:
                    job.throttled_s += dt
                    continue
                if job.is_inference:
                    self._serve(job, share, dt)
                    job.progress += dt
                else:
                    job.progress += dt * share
                    every = job.spec.get("checkpoint_every_s")
                    if every and job.progress - job.ckpt >= every:
                        job.ckpt = math.floor(job.progress / every) * every
                if job.progress >= job.spec["duration_s"]:
                    p.phase = "Succeeded"
                    job.done_at = t + dt
                    job.pod_uid = None
        for job in self.jobs:  # requests that hit a service that isn't running yet
            if job.is_inference and job.submitted and job.done_at is None:
                pod = self.cluster.get_pod(job.pod_uid) if job.pod_uid else None
                if pod is None or pod.phase != "Running":
                    n = (job.spec.get("rps") or 10) * dt
                    job.requests += n
                    job.unavailable += n

    def _serve(self, job: JobState, share: float, dt: float) -> None:
        rps = job.spec.get("rps") or 10
        n = rps * dt
        k = max(1, min(self.cfg.max_samples_per_tick, int(n)))
        base = job.spec.get("base_latency_ms") or 20
        slo = job.spec.get("slo_ms")
        slow = 0
        for _ in range(k):
            lat = base / share * self.rng.lognormvariate(0, 0.2)
            job.latencies.append(lat)
            if slo and lat > slo:
                slow += 1
        job.requests += n
        job.slow += n * slow / k

    # -- results ----------------------------------------------------------------------------------
    def results(self) -> dict:
        end = self.clock.t
        rows = []
        for j in self.jobs:
            lat = sorted(j.latencies)
            rows.append({
                "job": j.name, "type": j.spec["type"], "priority_class": j.spec["priority_class"],
                "arrival_s": j.spec["arrival_s"], "first_start_s": j.first_start,
                "queue_delay_s": (j.first_start - j.spec["arrival_s"]) if j.first_start is not None else end - j.spec["arrival_s"],
                "started": j.first_start is not None, "completed": j.done_at is not None, "done_at_s": j.done_at,
                "duration_s": j.spec["duration_s"], "restarts": j.restarts, "evictions": j.evictions, "ooms": j.ooms,
                "lost_work_s": round(j.lost_s, 2), "throttled_s": j.throttled_s,
                "requests": round(j.requests), "slo_ms": j.spec.get("slo_ms"),
                "slo_violations": round(j.slow + j.unavailable), "unavailable_requests": round(j.unavailable),
                "p50_ms": _pct(lat, 50), "p95_ms": _pct(lat, 95), "p99_ms": _pct(lat, 99),
                "preemption_wait_s": round(sum(j.preempt_latencies) / len(j.preempt_latencies), 2) if j.preempt_latencies else None,
                "_latencies": lat,
            })
        return {"policy": self.policy, "end_s": end, "rows": rows, "util": self.util_samples,
                "node_seconds": self.node_seconds, "preemptions": dict(self.preemptions),
                "price_per_hour": self.cfg.price_per_hour}


def _pct(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(q / 100 * len(sorted_vals)) - 1))
    return round(sorted_vals[idx], 2)


def summarize(res: dict) -> dict:
    rows = res["rows"]
    out: dict = {"policy": res["policy"], "label": res.get("label", "simulation"), "makespan_s": res["end_s"]}
    by_pc: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pc[r["priority_class"]].append(r)
    out["queue_delay_s"] = {
        pc: {"mean": round(sum(r["queue_delay_s"] for r in rs) / len(rs), 2),
             "p95": _pct(sorted(r["queue_delay_s"] for r in rs), 95), "n": len(rs)}
        for pc, rs in sorted(by_pc.items())
    }
    out["latency_ms"] = {}
    for pc in ("gpu-critical", "gpu-high"):
        lats = sorted(x for r in by_pc.get(pc, []) for x in r["_latencies"])
        reqs = sum(r["requests"] for r in by_pc.get(pc, []))
        viol = sum(r["slo_violations"] for r in by_pc.get(pc, []))
        out["latency_ms"][pc] = {"p50": _pct(lats, 50), "p95": _pct(lats, 95), "p99": _pct(lats, 99),
                                 "slo_violation_pct": round(100 * viol / reqs, 2) if reqs else None}
    utils = [u for _, _, u in res["util"]]
    out["gpu_util_mean_pct"] = round(sum(utils) / len(utils), 2) if utils else 0.0
    train = [r for r in rows if r["type"] != "inference"]
    out["preemptions"] = {f"{k[0]}/{k[1]}" if isinstance(k, tuple) else k: v for k, v in res["preemptions"].items()}
    out["preemption_count"] = sum(res["preemptions"].values())
    waits = [r["preemption_wait_s"] for r in rows if r["preemption_wait_s"] is not None]
    out["preemption_decision_to_running_s"] = round(sum(waits) / len(waits), 2) if waits else None
    out["training_lost_work_s"] = round(sum(r["lost_work_s"] for r in train), 1)
    # crash_events counts every pod crash (a job can crash, restart, and crash again);
    # crashed_jobs counts distinct jobs that crashed at least once.
    out["crash_events"] = sum(r["ooms"] for r in rows)
    out["crashed_jobs"] = sum(1 for r in rows if r["ooms"] > 0)
    hours = res["end_s"] / 3600 or 1
    done_work = sum(r["duration_s"] for r in train if r["completed"])
    out["training_work_s_per_hour"] = round(done_work / hours, 1)
    out["jobs_completed"] = sum(1 for r in rows if r["completed"])
    out["jobs_total"] = len(rows)
    out["node_hours"] = round(res["node_seconds"] / 3600, 3)
    out["cost_dollars"] = round(out["node_hours"] * res["price_per_hour"], 3)
    return out
