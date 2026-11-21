from __future__ import annotations

import time
from abc import ABC, abstractmethod

from gpuorch.common.models import EnforcementAction


class EnforcementError(Exception):
    pass


class Enforcer(ABC):
    name = "base"

    def __init__(self) -> None:
        self._actions: list[EnforcementAction] = []

    def _record(self, action: str, pod_uid: str, ok: bool = True, **detail) -> None:
        self._actions.append(EnforcementAction(ts=time.time(), action=action, pod_uid=pod_uid, detail=detail, ok=ok))
        del self._actions[:-1000]  # bounded history

    def actions(self) -> list[EnforcementAction]:
        return list(self._actions)

    @abstractmethod
    def freeze(self, pod_uid: str) -> None: ...

    @abstractmethod
    def unfreeze(self, pod_uid: str) -> None: ...

    @abstractmethod
    def is_frozen(self, pod_uid: str) -> bool: ...

    @abstractmethod
    def set_limits(self, pod_uid: str, cpu_max: str | None, memory_max: str | None) -> None: ...

    def usage(self, pod_uid: str) -> tuple[int | None, int | None]:
        """(cpu_usage_usec, memory_bytes) of the pod cgroup, if known."""
        return None, None
