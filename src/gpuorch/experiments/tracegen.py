"""Job-arrival trace generator (PRD §5.8): Poisson arrivals, configurable mix, seeded.

    python -m gpuorch.experiments.tracegen --name mixed-small --jobs 40 --rate 0.1 --seed 42 -o traces/mixed-small.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class JobType:
    weight: float
    type: str  # inference | training | batch
    priority_class: str
    duration_s: tuple[float, float]
    gpu_mem_mb: tuple[int, int]
    gpu_util: tuple[float, float]
    slo_ms: float | None = None
    base_latency_ms: float | None = None
    rps: float | None = None
    checkpoint_every_s: float | None = None


DEFAULT_MIX: dict[str, JobType] = {
    "inference-critical": JobType(0.15, "inference", "gpu-critical", (90, 240), (1500, 3000), (25, 40),
                                  slo_ms=50, base_latency_ms=18, rps=20),
    "inference-high": JobType(0.15, "inference", "gpu-high", (90, 240), (1500, 4000), (20, 35),
                              slo_ms=120, base_latency_ms=40, rps=10),
    "training-normal": JobType(0.30, "training", "gpu-normal", (150, 420), (3000, 6000), (50, 80),
                               checkpoint_every_s=30),
    "batch": JobType(0.40, "batch", "gpu-batch", (120, 480), (2000, 6000), (50, 90), checkpoint_every_s=60),
}


@dataclass
class TraceJob:
    name: str
    arrival_s: float
    kind: str
    type: str
    priority_class: str
    duration_s: float
    gpu_mem_mb: int
    gpu_util: float
    slo_ms: float | None = None
    base_latency_ms: float | None = None
    rps: float | None = None
    checkpoint_every_s: float | None = None


def generate(n_jobs: int, rate: float, seed: int, mix: dict[str, JobType] | None = None,
             name: str = "trace") -> dict:
    """rate = mean arrivals per second (Poisson process => exponential inter-arrival times)."""
    mix = mix or DEFAULT_MIX
    rng = random.Random(seed)
    kinds = list(mix)
    weights = [mix[k].weight for k in kinds]
    t = 0.0
    jobs = []
    for i in range(n_jobs):
        t += rng.expovariate(rate)
        kind = rng.choices(kinds, weights)[0]
        jt = mix[kind]
        jobs.append(TraceJob(
            name=f"j{i:03d}-{kind}", arrival_s=round(t, 2), kind=kind, type=jt.type,
            priority_class=jt.priority_class,
            duration_s=round(rng.uniform(*jt.duration_s), 1),
            gpu_mem_mb=int(rng.uniform(*jt.gpu_mem_mb) // 100 * 100),
            gpu_util=round(rng.uniform(*jt.gpu_util), 1),
            slo_ms=jt.slo_ms, base_latency_ms=jt.base_latency_ms, rps=jt.rps,
            checkpoint_every_s=jt.checkpoint_every_s,
        ))
    return {"name": name, "seed": seed, "rate": rate, "jobs": [asdict(j) for j in jobs]}


def load_trace(path: str | Path) -> dict:
    p = Path(path)
    if p.suffix == ".csv":
        with p.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        jobs = []
        for r in rows:
            for k, v in list(r.items()):
                if v == "":
                    r[k] = None
                elif k in {"arrival_s", "duration_s", "gpu_util", "slo_ms", "base_latency_ms", "rps",
                           "checkpoint_every_s"}:
                    r[k] = float(v)
                elif k == "gpu_mem_mb":
                    r[k] = int(float(v))
            jobs.append(r)
        return {"name": p.stem, "seed": None, "jobs": jobs}
    return yaml.safe_load(p.read_text())


def write_trace(trace: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".csv":
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(TraceJob.__dataclass_fields__))
            w.writeheader()
            w.writerows(trace["jobs"])
    else:
        p.write_text(yaml.safe_dump(trace, sort_keys=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="trace")
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--rate", type=float, default=0.1, help="mean arrivals per second")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mix", help="YAML file overriding the job mix (same fields as DEFAULT_MIX)")
    ap.add_argument("-o", "--out", required=True, help=".yaml or .csv")
    a = ap.parse_args(argv)
    mix = None
    if a.mix:
        raw = yaml.safe_load(Path(a.mix).read_text())
        mix = {k: JobType(**{**v, **{f: tuple(v[f]) for f in ("duration_s", "gpu_mem_mb", "gpu_util") if f in v}})
               for k, v in raw.items()}
    trace = generate(a.jobs, a.rate, a.seed, mix, a.name)
    write_trace(trace, a.out)
    print(f"wrote {len(trace['jobs'])} jobs to {a.out}")


if __name__ == "__main__":
    main()
