from __future__ import annotations

from gpuorch.agent.enforcer.base import Enforcer


class NoopEnforcer(Enforcer):
    """Records actions in memory (exposed via GET /actions). Used in kind / mock mode and tests."""

    name = "noop"

    def __init__(self) -> None:
        super().__init__()
        self._frozen: set[str] = set()
        self._limits: dict[str, dict[str, str | None]] = {}

    def freeze(self, pod_uid: str) -> None:
        self._frozen.add(pod_uid)
        self._record("freeze", pod_uid)

    def unfreeze(self, pod_uid: str) -> None:
        self._frozen.discard(pod_uid)
        self._record("unfreeze", pod_uid)

    def is_frozen(self, pod_uid: str) -> bool:
        return pod_uid in self._frozen

    def set_limits(self, pod_uid: str, cpu_max: str | None, memory_max: str | None) -> None:
        self._limits[pod_uid] = {"cpu_max": cpu_max, "memory_max": memory_max}
        self._record("limits", pod_uid, cpu_max=cpu_max, memory_max=memory_max)

    def limits(self, pod_uid: str) -> dict[str, str | None] | None:
        return self._limits.get(pod_uid)
