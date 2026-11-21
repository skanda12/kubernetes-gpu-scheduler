"""Env-var configuration helpers. Every setting has a documented default (see README 'Configuration')."""

from __future__ import annotations

import os

SCHEDULER_NAME = "priority-gpu-scheduler"
GPU_RESOURCE = "nvidia.com/gpu"

# Annotation / label keys
ANN_PREFIX = "gpuorch.io/"
ANN_GPU_MEM_MB = "gpuorch.io/gpu-mem-mb"
ANN_LATENCY_SLO_MS = "gpuorch.io/latency-slo-ms"
ANN_PREEMPT_POLICY = "gpuorch.io/preempt-policy"
ANN_NODE_STATE = "gpuorch.io/state"
ANN_AGENT_URL = "gpuorch.io/agent-url"
ANN_PREEMPTION_REQUESTED = "gpuorch.io/preemption-requested"
ANN_NOMINATED_NODE = "gpuorch.io/nominated-node"
ANN_METRICS_PORT = "gpuorch.io/metrics-port"
ANN_DURATION_S = "gpuorch.io/duration-s"
ANN_SIM_GPU_UTIL = "gpuorch.io/sim-gpu-util"  # util% a pod "uses" on a simulated (KWOK) node
LABEL_SIMULATED = "gpuorch.io/simulated"


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"env {name}={raw!r} is not a number") from exc


def env_int(name: str, default: int) -> int:
    return int(env_float(name, default))


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
