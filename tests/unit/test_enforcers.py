from __future__ import annotations

import pytest

from gpuorch.agent.enforcer import CgroupV2Enforcer, EnforcementError, NoopEnforcer, make_enforcer

UID = "1b2c3d4e-0000-1111-2222-333344445555"


def test_noop_records_actions():
    e = NoopEnforcer()
    e.freeze("a")
    assert e.is_frozen("a")
    e.set_limits("a", "50000 100000", "1073741824")
    e.unfreeze("a")
    assert not e.is_frozen("a")
    assert [a.action for a in e.actions()] == ["freeze", "limits", "unfreeze"]
    assert e.limits("a") == {"cpu_max": "50000 100000", "memory_max": "1073741824"}
    assert e.usage("a") == (None, None)


def _fake_cgroupfs(root, driver):
    (root / "cgroup.controllers").write_text("cpu memory")
    if driver == "systemd":
        d = root / "kubepods.slice" / "kubepods-burstable.slice" / f"kubepods-burstable-pod{UID.replace('-', '_')}.slice"
    else:
        d = root / "kubepods" / "besteffort" / f"pod{UID}"
    d.mkdir(parents=True)
    (d / "cgroup.freeze").write_text("0")
    (d / "cgroup.events").write_text("populated 1\nfrozen 0\n")
    (d / "cpu.stat").write_text("usage_usec 12345\nuser_usec 1\n")
    (d / "memory.current").write_text("4096\n")
    (d / "cpu.max").write_text("max 100000")
    (d / "memory.max").write_text("max")
    return d


@pytest.mark.parametrize("driver", ["systemd", "cgroupfs"])
def test_cgroup_freeze_and_limits(tmp_path, driver):
    d = _fake_cgroupfs(tmp_path, driver)
    e = CgroupV2Enforcer(tmp_path)
    assert e.pod_cgroup(UID) == d
    e.freeze(UID)
    assert (d / "cgroup.freeze").read_text() == "1"
    (d / "cgroup.events").write_text("frozen 1\n")  # kernel would update this
    assert e.is_frozen(UID)
    e.unfreeze(UID)
    assert (d / "cgroup.freeze").read_text() == "0"
    e.set_limits(UID, "20000 100000", "536870912")
    assert (d / "cpu.max").read_text() == "20000 100000"
    assert (d / "memory.max").read_text() == "536870912"
    assert e.usage(UID) == (12345, 4096)


def test_cgroup_errors(tmp_path):
    with pytest.raises(EnforcementError):
        CgroupV2Enforcer(tmp_path)  # not a cgroup2 mount
    (tmp_path / "cgroup.controllers").write_text("")
    e = CgroupV2Enforcer(tmp_path)
    with pytest.raises(EnforcementError):
        e.freeze("missing")
    assert e.actions()[-1].ok is False
    assert e.is_frozen("missing") is False
    assert e.usage("missing") == (None, None)


def test_make_enforcer(tmp_path):
    assert make_enforcer("noop").name == "noop"
    assert make_enforcer("cgroup", str(tmp_path)).name == "noop"  # falls back
    (tmp_path / "cgroup.controllers").write_text("")
    assert make_enforcer("cgroup", str(tmp_path)).name == "cgroup"
    with pytest.raises(ValueError):
        make_enforcer("bogus")
