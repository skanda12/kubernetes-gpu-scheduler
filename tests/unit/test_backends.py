from __future__ import annotations

import pytest

from gpuorch.agent.backends import MockBackend, NvmlBackend, PodLoad, make_backend
from gpuorch.agent.backends.mock import interpolate, load_scenario

MB = 1024 * 1024

SCENARIO = {
    "nodes": {
        "default": {"gpus": [{"name": "T4", "memory_mb": 16000, "base_mem_mb": 500, "util_curve": [[0, 10], [10, 30]]}]},
        "big": {"gpus": [{"memory_mb": 1000}, {"memory_mb": 2000, "util_curve": [[0, 0], [10, 100]], "loop_s": 10}]},
    },
    "pods": {"train-0": {"gpu_mem_mb": 4000, "gpu_util": 60}},
}


class T:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_interpolate():
    curve = [[0, 0], [10, 100], [20, 50]]
    assert interpolate(curve, -1) == 0
    assert interpolate(curve, 5) == 50
    assert interpolate(curve, 15) == 75
    assert interpolate(curve, 99) == 50
    assert interpolate(curve, 25, loop_s=20) == 50
    assert interpolate([], 3) == 0


def test_mock_curve_and_default_node():
    t = T()
    b = MockBackend("unknown-node", SCENARIO, clock=t)
    b.init()
    g = b.gpus()[0]
    assert g.name == "T4" and g.utilization_percent == 10
    assert g.memory_total_bytes == 16000 * MB and g.memory_used_bytes == 500 * MB
    t.t = 5
    assert b.gpus()[0].utilization_percent == 20


def test_mock_pod_loads_frozen_keeps_memory_drops_util():
    t = T()
    b = MockBackend("n", SCENARIO, clock=t)
    b.init()
    b.update_pod_loads([PodLoad("u1", "p", 0, 2000 * MB, 50, False), PodLoad("u2", "q", 0, 1000 * MB, 80, True)])
    g = b.gpus()[0]
    assert g.utilization_percent == 60  # 10 + 50 (frozen pod contributes nothing)
    assert g.memory_used_bytes == 3500 * MB  # frozen pod keeps its memory
    b.update_pod_loads([PodLoad("u1", "p", 0, 2000 * MB, 99, False)])
    assert b.gpus()[0].utilization_percent == 100  # capped


def test_mock_multi_gpu_and_scenario_pods():
    t = T()
    b = MockBackend("big", SCENARIO, clock=t)
    b.init()
    assert len(b.gpus()) == 2
    t.t = 15  # loop_s=10 -> t=5
    assert b.gpus()[1].utilization_percent == 50
    assert b.scenario_pod_usage("train-0") == (4000 * MB, 60.0)
    assert b.scenario_pod_usage("nope") is None


def test_mock_bad_scenario():
    b = MockBackend("n", {"nodes": {"default": {"gpus": []}}})
    with pytest.raises(ValueError):
        b.init()


def test_load_scenario_file(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("nodes:\n  default:\n    gpus:\n      - memory_mb: 8000\n")
    assert load_scenario(p)["nodes"]["default"]["gpus"][0]["memory_mb"] == 8000
    assert "nodes" in load_scenario(None)


def test_make_backend_falls_back_to_mock(monkeypatch):
    def boom(self):
        raise RuntimeError("no nvml")

    monkeypatch.setattr(NvmlBackend, "init", boom)
    b = make_backend("nvml", "n")
    assert b.name == "mock"
    assert make_backend("mock", "n").gpus()
    with pytest.raises(ValueError):
        make_backend("cuda", "n")


def test_nvml_backend_with_fake_pynvml(monkeypatch):
    """Exercise NvmlBackend logic against a fake pynvml module (no GPU needed)."""
    import sys
    import types

    fake = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    fake.NVMLError = NVMLError
    fake.NVML_TEMPERATURE_GPU = 0
    fake.nvmlInit = lambda: None
    fake.nvmlShutdown = lambda: None
    fake.nvmlDeviceGetCount = lambda: 1
    fake.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"
    fake.nvmlDeviceGetName = lambda h: b"NVIDIA GeForce RTX 3060 Laptop GPU"
    fake.nvmlDeviceGetUUID = lambda h: "GPU-123"
    fake.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(total=6 * 1024**3, used=1024**3)
    fake.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=42)
    fake.nvmlDeviceGetTemperature = lambda h, s: 55

    def no_power(h):
        raise NVMLError("not supported")

    fake.nvmlDeviceGetPowerUsage = no_power
    fake.nvmlDeviceGetComputeRunningProcesses = lambda h: []
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    b = NvmlBackend()
    b.init()
    g = b.gpus()[0]
    assert g.name.startswith("NVIDIA") and g.utilization_percent == 42 and g.power_w is None
    assert g.memory_free_bytes == 5 * 1024**3
    assert b.per_process_supported() is False  # WSL2-like: empty process list
    b.close()


@pytest.mark.gpu
def test_nvml_real_gpu():
    """TODO(verify-on-gpu): run inside WSL2 on a GPU laptop:  pytest -m gpu tests/unit/test_backends.py"""
    b = NvmlBackend()
    b.init()
    gpus = b.gpus()
    assert gpus and gpus[0].memory_total_bytes > 0
    print("per-process supported:", b.per_process_supported())
    b.close()
