"""Generate the Grafana dashboards in deploy/grafana/ (committed JSON). Re-run after editing: python scripts/gen_dashboards.py"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "deploy" / "grafana"
DS = {"type": "prometheus", "uid": "prometheus"}


def panel(pid, title, exprs, x, y, w=12, h=8, kind="timeseries", unit=None, legend=None, thresholds=None):
    p = {
        "id": pid, "type": kind, "title": title, "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": (legend or [None] * len(exprs))[i] or ""}
                    for i, e in enumerate(exprs)],
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {},
    }
    if unit:
        p["fieldConfig"]["defaults"]["unit"] = unit
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": thresholds}
        if kind == "timeseries":
            p["fieldConfig"]["defaults"]["custom"] = {"thresholdsStyle": {"mode": "line"}}
    return p


def dashboard(uid, title, panels, variables=None):
    return {
        "uid": uid, "title": title, "tags": ["gpuorch"], "timezone": "browser", "schemaVersion": 39,
        "refresh": "5s", "time": {"from": "now-30m", "to": "now"}, "panels": panels,
        "templating": {"list": variables or []}, "editable": True,
    }


def node_var():
    return {"name": "node", "type": "query", "datasource": DS, "query": "label_values(gpu_util_percent, node)",
            "includeAll": True, "multi": True, "current": {"text": "All", "value": "$__all"}, "refresh": 2}


DASHBOARDS = {
    "cluster-overview": dashboard("gpuorch-overview", "GPU cluster overview", [
        panel(1, "Mean GPU utilisation", ["avg(gpu_util_percent)"], 0, 0, 6, 5, "stat", "percent"),
        panel(2, "GPU memory used", ["sum(gpu_mem_used_bytes) / sum(gpu_mem_total_bytes)"], 6, 0, 6, 5, "gauge", "percentunit"),
        panel(3, "Pending GPU pods", ["sum(gpuorch_pending_pods)"], 12, 0, 4, 5, "stat"),
        panel(4, "Throttled pods", ["sum(pod_throttled)"], 16, 0, 4, 5, "stat"),
        panel(5, "Simulated cost ($)", ["max(autoscaler_sim_cost_dollars)"], 20, 0, 4, 5, "stat", "currencyUSD"),
        panel(6, "GPU utilisation by node", ["gpu_util_percent"], 0, 5, 12, 8, unit="percent", legend=["{{node}}/{{gpu}}"]),
        panel(7, "Preemptions (rate, 5m)", ["sum by (policy) (increase(preemptions_total[5m]))"], 12, 5, 12, 8,
              legend=["{{policy}}"]),
        panel(8, "GPU nodes (simulated)", ["autoscaler_sim_nodes"], 0, 13, 12, 6, legend=["sim nodes"]),
        panel(9, "Pod GPU memory by priority", ["sum by (priority) (pod_gpu_mem_bytes)"], 12, 13, 12, 6, unit="bytes",
              legend=["{{priority}}"]),
    ]),
    "node-gpu": dashboard("gpuorch-node", "Per-node GPU", [
        panel(1, "Utilisation", ['gpu_util_percent{node=~"$node"}'], 0, 0, 24, 8, unit="percent", legend=["{{node}}/{{gpu}}"]),
        panel(2, "Memory used vs total", ['gpu_mem_used_bytes{node=~"$node"}', 'gpu_mem_total_bytes{node=~"$node"}'],
              0, 8, 12, 8, unit="bytes", legend=["used {{node}}", "total {{node}}"]),
        panel(3, "Temperature", ['gpu_temperature_celsius{node=~"$node"}'], 12, 8, 12, 8, unit="celsius",
              legend=["{{node}}"]),
        panel(4, "Per-pod self-reported GPU util", ["pod_gpu_util_percent"], 0, 16, 12, 8, unit="percent",
              legend=["{{pod}}"]),
        panel(5, "Per-pod throttle state", ["pod_throttled"], 12, 16, 12, 8, legend=["{{pod}}"]),
    ], [node_var()]),
    "inference-latency": dashboard("gpuorch-latency", "Inference latency vs SLO", [
        panel(1, "p50 / p95 / p99 (server side)", [
            "1000*histogram_quantile(0.50, sum by (le, job_name) (rate(inference_request_latency_seconds_bucket[1m])))",
            "1000*histogram_quantile(0.95, sum by (le, job_name) (rate(inference_request_latency_seconds_bucket[1m])))",
            "1000*histogram_quantile(0.99, sum by (le, job_name) (rate(inference_request_latency_seconds_bucket[1m])))",
        ], 0, 0, 24, 9, unit="ms", legend=["p50 {{job_name}}", "p95 {{job_name}}", "p99 {{job_name}}"],
            thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 50}]),
        panel(2, "Client-side p95 (load generator)", [
            "1000*histogram_quantile(0.95, sum by (le, target) (rate(loadgen_request_latency_seconds_bucket[1m])))"],
            0, 9, 12, 8, unit="ms", legend=["{{target}}"],
            thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 50}]),
        panel(3, "Requests/s", ["sum by (job_name) (rate(inference_requests_total[1m]))"], 12, 9, 12, 8, unit="reqps",
              legend=["{{job_name}}"]),
    ]),
    "preemptions": dashboard("gpuorch-preemption", "Preemption events", [
        panel(1, "Preemptions by policy & victim priority",
              ["sum by (policy, victim_priority) (increase(preemptions_total[1m]))"], 0, 0, 24, 8,
              legend=["{{policy}} / {{victim_priority}}"]),
        panel(2, "Decision -> preemptor running (p50/p95)", [
            "histogram_quantile(0.5, sum by (le, policy) (rate(preemption_latency_seconds_bucket[5m])))",
            "histogram_quantile(0.95, sum by (le, policy) (rate(preemption_latency_seconds_bucket[5m])))"],
            0, 8, 12, 8, unit="s", legend=["p50 {{policy}}", "p95 {{policy}}"],
            thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 10}]),
        panel(3, "Unthrottles", ["sum by (reason) (increase(unthrottles_total[1m]))"], 12, 8, 12, 8, legend=["{{reason}}"]),
        panel(4, "Currently throttled", ["gpuorch_throttled_pods"], 0, 16, 12, 6),
        panel(5, "Preemption not possible", ["sum by (reason) (increase(preemption_skipped_total[5m]))"], 12, 16, 12, 6,
              legend=["{{reason}}"]),
    ]),
    "pending-queue": dashboard("gpuorch-queue", "Pending queue by priority", [
        panel(1, "Pending pods (priority-gpu-scheduler)", ["gpuorch_pending_pods"], 0, 0, 24, 8, legend=["{{priority_class}}"]),
        panel(2, "Scheduling delay p95 (creation -> bind)", [
            "histogram_quantile(0.95, sum by (le, priority_class) (rate(gpuorch_scheduling_delay_seconds_bucket[5m])))"],
            0, 8, 12, 8, unit="s", legend=["{{priority_class}}"]),
        panel(3, "Bindings/min", ["sum by (priority_class) (rate(gpuorch_scheduler_bindings_total[1m])) * 60"], 12, 8, 12, 8,
              legend=["{{priority_class}}"]),
        panel(4, "Unschedulable attempts/min", ["sum by (priority_class) (rate(gpuorch_scheduler_unschedulable_total[1m])) * 60"],
              0, 16, 12, 6, legend=["{{priority_class}}"]),
        panel(5, "Stale GPU nodes", ["gpuorch_stale_nodes"], 12, 16, 12, 6),
    ]),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, d in DASHBOARDS.items():
        (OUT / f"{name}.json").write_text(json.dumps(d, indent=2) + "\n")
        print(OUT / f"{name}.json")


if __name__ == "__main__":
    main()
