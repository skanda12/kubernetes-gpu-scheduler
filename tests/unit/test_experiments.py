from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gpuorch.common import config as C
from gpuorch.experiments import runner
from gpuorch.experiments.simulator import SimConfig, Simulation, summarize
from gpuorch.experiments.tracegen import generate, load_trace, write_trace
from gpuorch.tools.demo import demo_trace

ROOT = Path(__file__).resolve().parents[2]


def test_tracegen_deterministic_and_roundtrip(tmp_path):
    a, b = generate(30, 0.2, 5), generate(30, 0.2, 5)
    assert a == b and len(a["jobs"]) == 30
    assert generate(30, 0.2, 6) != a
    arrivals = [j["arrival_s"] for j in a["jobs"]]
    assert arrivals == sorted(arrivals)
    assert {j["priority_class"] for j in a["jobs"]} <= {"gpu-critical", "gpu-high", "gpu-normal", "gpu-batch"}
    for ext in ("yaml", "csv"):
        p = tmp_path / f"t.{ext}"
        write_trace(a, p)
        back = load_trace(p)
        assert back["jobs"][3]["gpu_mem_mb"] == a["jobs"][3]["gpu_mem_mb"]
        assert back["jobs"][3]["duration_s"] == a["jobs"][3]["duration_s"]


def test_tracegen_cli(tmp_path):
    from gpuorch.experiments.tracegen import main

    mix = tmp_path / "mix.yaml"
    mix.write_text("only-batch: {weight: 1, type: batch, priority_class: gpu-batch, duration_s: [10, 20],"
                   " gpu_mem_mb: [1000, 2000], gpu_util: [50, 60]}\n")
    out = tmp_path / "t.yaml"
    main(["--jobs", "5", "--mix", str(mix), "-o", str(out)])
    assert {j["kind"] for j in load_trace(out)["jobs"]} == {"only-batch"}


def test_committed_traces_load():
    for name in ("mixed-small.yaml", "mixed-large.yaml", "smoke.csv"):
        assert load_trace(ROOT / "traces" / name)["jobs"]


@pytest.mark.parametrize("policy", ["gpuorch", "default"])
def test_simulation_completes_demo(policy):
    res = Simulation(demo_trace(), policy, SimConfig(seed=1)).run()
    s = summarize(res)
    assert s["jobs_completed"] == s["jobs_total"] == 14
    crit = next(r for r in res["rows"] if r["job"] == "critical-inference")
    assert crit["queue_delay_s"] < 10  # PRD M3 target
    assert s["preemptions"].get("evict/gpu-batch", 0) >= 1


def test_gpuorch_beats_default_on_critical_latency_and_ooms():
    trace = generate(30, 0.12, 3)
    ours = summarize(Simulation(trace, "gpuorch", SimConfig(seed=2)).run())
    base = summarize(Simulation(trace, "default", SimConfig(seed=2)).run())
    assert ours["crash_events"] == 0 and ours["crashed_jobs"] == 0
    assert base["crashed_jobs"] <= base["crash_events"]
    assert ours["label"] == "simulation"
    assert ours["latency_ms"]["gpu-critical"]["p95"] <= base["latency_ms"]["gpu-critical"]["p95"]


def test_simulation_with_autoscaler_adds_nodes():
    trace = generate(25, 0.5, 9)  # bursty
    sim = Simulation(trace, "gpuorch", SimConfig(nodes=1, autoscale=True, max_sim_nodes=2, seed=1))
    s = summarize(sim.run())
    assert s["jobs_completed"] == s["jobs_total"]
    assert sim.autoscaler is not None and sim.autoscaler._seq >= 1
    with pytest.raises(ValueError):
        Simulation(trace, "fifo")


def test_runner_sim_and_charts(tmp_path):
    trace = tmp_path / "t.yaml"
    write_trace(generate(10, 0.3, 1, name="tiny"), trace)
    runner.main(["--trace", str(trace), "--reps", "2", "--out", str(tmp_path / "res"), "--run-id", "r1"])
    run = tmp_path / "res" / "r1"
    comp = json.loads((run / "comparison.json").read_text())
    assert set(comp["results"]) == {"gpuorch", "default"}
    assert isinstance(comp["results"]["gpuorch"]["gpu_util_mean_pct"], dict)  # mean/min/max over reps
    for f in ("jobs.csv", "util.csv", "summary.json"):
        assert (run / "default" / "rep1" / f).exists()
    assert {p.name for p in (run / "charts").glob("*.png")} >= {"latency.png", "overview.png", "queueing_delay.png"}


def test_aggregate():
    agg = runner.aggregate([{"a": 1, "b": {"x": 2}, "s": "p"}, {"a": 3, "b": {"x": 4}, "s": "p"}])
    assert agg["a"] == {"mean": 2.0, "min": 1, "max": 3} and agg["b"]["x"]["mean"] == 3.0 and agg["s"] == "p"


def test_job_manifest():
    job = generate(1, 1, 1)["jobs"][0]
    m = runner.job_manifest(job, "gpuorch", "ns", "img:1", 0.5, "run")
    spec = m["spec"]["template"]["spec"]
    assert spec["schedulerName"] == C.SCHEDULER_NAME and spec["priorityClassName"] == job["priority_class"]
    assert spec["containers"][0]["resources"]["limits"][C.GPU_RESOURCE] == "1"
    assert m["spec"]["template"]["metadata"]["annotations"][C.ANN_GPU_MEM_MB] == str(job["gpu_mem_mb"])
    assert "schedulerName" not in runner.job_manifest(job, "default", "ns", "img", 1, "r")["spec"]["template"]["spec"]
    env = {e["name"]: e["value"] for e in spec["containers"][0]["env"]}
    assert env["CHECKPOINT_DIR"] == f"/ckpt/run/gpuorch/{job['name']}"  # run token isolates checkpoints


def test_offline_demo_runs(capsys):
    from gpuorch.tools.demo import main

    main(["--offline"])
    out = capsys.readouterr().out
    assert "GpuPreempted" in out and "critical: arrival->running" in out


def test_check_tools_reports_missing(capsys, monkeypatch):
    from gpuorch.tools import cluster

    monkeypatch.setattr(cluster.shutil, "which", lambda t: None)
    assert cluster.check_tools() is False
    assert "install:" in capsys.readouterr().out


def test_fake_workload_checkpoint_and_resume(tmp_path):
    app = ROOT / "workloads" / "fake" / "app.py"
    env = {**os.environ, "DURATION_S": "1.2", "CHECKPOINT_EVERY_S": "0.5", "CHECKPOINT_DIR": str(tmp_path),
           "PORT": "0", "FAKE_GPU_MEM_MB": "100"}
    r = subprocess.run([sys.executable, str(app)], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    progress = json.loads((tmp_path / "progress.json").read_text())["progress"]
    assert progress >= 1.2
    env["DURATION_S"] = str(progress + 0.5)
    r = subprocess.run([sys.executable, str(app)], env=env, capture_output=True, text=True, timeout=30)
    assert "resumed_from_checkpoint" in r.stdout
