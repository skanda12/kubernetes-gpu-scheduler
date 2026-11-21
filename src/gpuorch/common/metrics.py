"""Prometheus metrics for scheduler / preemption-controller / autoscaler-sim (default registry)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# scheduler
BINDINGS = Counter("gpuorch_scheduler_bindings_total", "Pods bound by gpu-scheduler", ["priority_class"])
UNSCHEDULABLE = Counter(
    "gpuorch_scheduler_unschedulable_total", "Scheduling attempts with no fitting node", ["priority_class"]
)
PENDING = Gauge("gpuorch_pending_pods", "Pending pods for our scheduler", ["priority_class"])
QUEUE_DELAY = Histogram(
    "gpuorch_scheduling_delay_seconds", "Pod creation -> bind", ["priority_class"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
)
STALE_NODES = Gauge("gpuorch_stale_nodes", "GPU nodes whose agent state is stale")

# preemption
PREEMPTIONS = Counter("preemptions_total", "Preemption actions", ["policy", "victim_priority"])
PREEMPTION_LATENCY = Histogram(
    "preemption_latency_seconds", "Preemption decision -> preemptor running", ["policy"],
    buckets=(0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 60),
)
UNTHROTTLES = Counter("unthrottles_total", "Pods unfrozen", ["reason"])
PREEMPTION_SKIPPED = Counter("preemption_skipped_total", "Preemption not possible", ["reason"])
THROTTLED_PODS = Gauge("gpuorch_throttled_pods", "Pods currently throttled by the controller")

# autoscaler-sim
SIM_NODES = Gauge("autoscaler_sim_nodes", "Simulated GPU nodes currently provisioned")
SIM_COST = Gauge("autoscaler_sim_cost_dollars", "Accumulated simulated cost (all GPU nodes)")
SIM_SCALE = Counter("autoscaler_sim_scale_events_total", "Scale events", ["direction"])
