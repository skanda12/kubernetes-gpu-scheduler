from __future__ import annotations

import logging

from gpuorch.agent.enforcer.base import EnforcementError, Enforcer
from gpuorch.agent.enforcer.cgroup import CgroupV2Enforcer
from gpuorch.agent.enforcer.noop import NoopEnforcer
from gpuorch.common.logging import log_event

log = logging.getLogger(__name__)

__all__ = ["Enforcer", "EnforcementError", "CgroupV2Enforcer", "NoopEnforcer", "make_enforcer"]


def make_enforcer(kind: str, cgroup_root: str = "/sys/fs/cgroup") -> Enforcer:
    """ENFORCER=cgroup|noop. cgroup failures fall back to noop with a warning."""
    if kind == "cgroup":
        try:
            return CgroupV2Enforcer(cgroup_root)
        except EnforcementError as exc:
            log_event(log, "cgroup_enforcer_unavailable_falling_back_to_noop", logging.WARNING, error=str(exc))
            return NoopEnforcer()
    if kind == "noop":
        return NoopEnforcer()
    raise ValueError(f"unknown ENFORCER {kind!r} (expected cgroup|noop)")
