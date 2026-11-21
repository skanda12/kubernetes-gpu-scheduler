"""`gpuorch <command> [args]` - thin dispatcher over the component entrypoints."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "agent": "gpuorch.agent.main",
    "scheduler": "gpuorch.scheduler.main",
    "preemption": "gpuorch.preemption.main",
    "autoscaler": "gpuorch.autoscaler_sim.main",
    "tracegen": "gpuorch.experiments.tracegen",
    "run": "gpuorch.experiments.runner",
    "analyze": "gpuorch.experiments.analysis",
    "cluster": "gpuorch.tools.cluster",
    "demo": "gpuorch.tools.demo",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: gpuorch {" + "|".join(COMMANDS) + "} [args...]")
        sys.exit(2)
    mod = importlib.import_module(COMMANDS[sys.argv[1]])
    sys.argv = [f"gpuorch {sys.argv[1]}", *sys.argv[2:]]
    if sys.argv[0].endswith("analyze"):
        for p in mod.make_charts(sys.argv[1]):
            print(p)
    else:
        mod.main()


if __name__ == "__main__":
    main()
