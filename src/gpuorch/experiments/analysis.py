"""Comparison charts (PNG) from a results/<run-id>/ directory.

    python -m gpuorch.experiments.analysis results/<run-id>
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

LABELS = {"gpuorch": "priority-gpu-scheduler", "default": "default kube-scheduler"}
COLORS = {"gpuorch": "#2563eb", "default": "#9ca3af"}
CLASSES = ["gpu-critical", "gpu-high", "gpu-normal", "gpu-batch"]


def _mean(v):
    if isinstance(v, dict) and "mean" in v:
        return v["mean"]
    return v if v is not None else 0


def _err(v):
    if isinstance(v, dict) and "mean" in v:
        return [[v["mean"] - v["min"]], [v["max"] - v["mean"]]]
    return None


def make_charts(run_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    comp = json.loads((run_dir / "comparison.json").read_text())["results"]
    scheds = [s for s in ("gpuorch", "default") if s in comp]
    charts = run_dir / "charts"
    charts.mkdir(exist_ok=True)
    out: list[Path] = []

    def bar_group(ax, groups, getter, ylabel, title):
        width = 0.8 / len(scheds)
        for i, s in enumerate(scheds):
            vals = [getter(comp[s], g) for g in groups]
            xs = [x + i * width for x in range(len(groups))]
            bars = ax.bar(xs, [_mean(v) for v in vals], width, label=LABELS[s], color=COLORS[s])
            for b, v in zip(bars, vals, strict=True):
                ax.annotate(f"{_mean(v):.3g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks([x + width * (len(scheds) - 1) / 2 for x in range(len(groups))], groups)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    def save(fig, name):
        p = charts / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)

    # 1. critical/high latency
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pc in zip(axes, ("gpu-critical", "gpu-high"), strict=True):
        bar_group(ax, ["p50", "p95", "p99"], lambda c, q, pc=pc: (c["latency_ms"].get(pc) or {}).get(q),
                  "latency (ms)", f"{pc} inference latency")
    save(fig, "latency.png")

    # 2. SLO violations
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_group(ax, ["gpu-critical", "gpu-high"], lambda c, pc: (c["latency_ms"].get(pc) or {}).get("slo_violation_pct"),
              "% requests over SLO (incl. unavailable)", "SLO violations")
    save(fig, "slo_violations.png")

    # 3. queueing delay by priority
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_group(ax, CLASSES, lambda c, pc: (c["queue_delay_s"].get(pc) or {}).get("mean"),
              "mean seconds (arrival -> running)", "Queueing delay by priority class")
    save(fig, "queueing_delay.png")

    # 4. utilisation, preemption overhead, cost, OOMs
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    metrics = [("gpu_util_mean_pct", "mean GPU util (%)"), ("training_lost_work_s", "training work lost (s)"),
               ("cost_dollars", "simulated cost ($)"), ("oom_failures", "GPU OOM failures")]
    for ax, (key, title) in zip(axes, metrics, strict=True):
        for i, s in enumerate(scheds):
            v = comp[s].get(key)
            ax.bar(i, _mean(v), color=COLORS[s], yerr=_err(v), capsize=4)
            ax.annotate(f"{_mean(v):.3g}", (i, _mean(v)), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(scheds)), [LABELS[s].split()[0] for s in scheds], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    save(fig, "overview.png")

    # 5. preemptions by policy
    fig, ax = plt.subplots(figsize=(7, 4))
    kinds = sorted({k for s in scheds for k in (comp[s].get("preemptions") or {})})
    if kinds:
        bar_group(ax, kinds, lambda c, k: (c.get("preemptions") or {}).get(k, 0), "count",
                  "Preemptions (decision->running: "
                  + ", ".join(f"{s}={_mean(comp[s].get('preemption_decision_to_running_s')):.1f}s" for s in scheds) + ")")
        ax.tick_params(axis="x", labelsize=7)
    save(fig, "preemptions.png")

    # 6. utilisation over time (rep0)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for s in scheds:
        f = run_dir / s / "rep0" / "util.csv"
        if not f.exists():
            continue
        per_t: dict[float, list[float]] = defaultdict(list)
        with f.open() as fh:
            for row in csv.DictReader(fh):
                per_t[float(row["t_s"])].append(float(row["gpu_util_pct"]))
        ts = sorted(per_t)
        ax.plot(ts, [sum(per_t[t]) / len(per_t[t]) for t in ts], label=LABELS[s], color=COLORS[s], lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean GPU util (%)")
    ax.set_title("Cluster GPU utilisation over time")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "util_timeline.png")
    return out


if __name__ == "__main__":
    for path in make_charts(sys.argv[1]):
        print(path)
