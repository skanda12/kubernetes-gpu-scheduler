"""cgroup v2 enforcement on the pod-level cgroup.

Handles both kubelet cgroup drivers:
  systemd : <root>/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod<uid_with_underscores>.slice
  cgroupfs: <root>/kubepods/burstable/pod<uid>
Freezing the pod cgroup stops all its processes: CUDA kernels stop being submitted, so the GPU time-slice
goes to other pods, but GPU memory stays allocated (PRD §4 "Throttle = freeze").

TODO(verify-on-gpu): on a WSL2 GPU node run
  curl -XPOST localhost:9400/throttle -d '{"pod_uid":"<uid>"}' -H 'content-type: application/json'
and watch `nvidia-smi dmon -s u` — the victim's utilisation should drop to ~0.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from gpuorch.agent.enforcer.base import EnforcementError, Enforcer


class CgroupV2Enforcer(Enforcer):
    name = "cgroup"

    def __init__(self, root: str | Path = "/sys/fs/cgroup", max_depth: int = 6) -> None:
        super().__init__()
        self.root = Path(root)
        self.max_depth = max_depth
        self._cache: dict[str, Path] = {}
        if not (self.root / "cgroup.controllers").exists():
            raise EnforcementError(f"{self.root} is not a cgroup v2 mount (cgroup.controllers missing)")

    def pod_cgroup(self, pod_uid: str) -> Path:
        cached = self._cache.get(pod_uid)
        if cached and cached.exists():
            return cached
        needles = {f"pod{pod_uid}", f"pod{pod_uid.replace('-', '_')}"}
        base_depth = len(self.root.parts)
        for dirpath, dirnames, _ in os.walk(self.root):
            depth = len(Path(dirpath).parts) - base_depth
            if depth >= self.max_depth:
                dirnames[:] = []
                continue
            for d in dirnames:
                stem = d.removesuffix(".slice")
                if any(stem == n or stem.endswith("-" + n) for n in needles):
                    path = Path(dirpath) / d
                    self._cache[pod_uid] = path
                    return path
        raise EnforcementError(f"cgroup for pod {pod_uid} not found under {self.root}")

    def _write(self, pod_uid: str, filename: str, value: str, action: str) -> None:
        try:
            target = self.pod_cgroup(pod_uid) / filename
            target.write_text(value)
        except (OSError, EnforcementError) as exc:
            self._record(action, pod_uid, ok=False, error=str(exc))
            raise EnforcementError(str(exc)) from exc
        self._record(action, pod_uid, file=str(target), value=value)

    def freeze(self, pod_uid: str) -> None:
        self._write(pod_uid, "cgroup.freeze", "1", "freeze")

    def unfreeze(self, pod_uid: str) -> None:
        self._write(pod_uid, "cgroup.freeze", "0", "unfreeze")

    def is_frozen(self, pod_uid: str) -> bool:
        try:
            path = self.pod_cgroup(pod_uid)
        except EnforcementError:
            return False
        events = path / "cgroup.events"
        if events.exists():
            for line in events.read_text().splitlines():
                if line.startswith("frozen "):
                    return line.split()[1] == "1"
        freeze = path / "cgroup.freeze"
        return freeze.exists() and freeze.read_text().strip() == "1"

    def set_limits(self, pod_uid: str, cpu_max: str | None, memory_max: str | None) -> None:
        if cpu_max:
            self._write(pod_uid, "cpu.max", cpu_max, "limits")
        if memory_max:
            self._write(pod_uid, "memory.max", memory_max, "limits")

    def usage(self, pod_uid: str) -> tuple[int | None, int | None]:
        try:
            path = self.pod_cgroup(pod_uid)
        except EnforcementError:
            return None, None
        cpu = mem = None
        try:
            for line in (path / "cpu.stat").read_text().splitlines():
                if line.startswith("usage_usec "):
                    cpu = int(line.split()[1])
        except OSError:
            pass
        with contextlib.suppress(OSError, ValueError):
            mem = int((path / "memory.current").read_text().strip())
        return cpu, mem
