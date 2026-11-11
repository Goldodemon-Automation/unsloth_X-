# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vendor matrix for NPU detection over fake sysfs trees and fake platform answers.

No hardware and no runtime: every probe is pointed at a tmp_path inventory or at a stubbed
platform module, so the tests say what a machine reports without needing to be that machine.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from utils.hardware import npu

_AMD_KEYS = {
    "present",
    "family",
    "name",
    "driver",
    "driver_version",
    "firmware_version",
    "device_node",
    "enabled",
    "supported",
}


def _device(
    root: Path,
    name: str,
    vendor: str,
    device: str,
    *,
    driver: str | None = None,
    device_class: str | None = None,
) -> Path:
    """One fake PCI function under a fake sysfs root.

    The driver is a real directory holding a directory named after the driver rather than the
    sysfs symlink: creating a symlink needs a privilege Windows does not grant by default, and
    ``resolve().name`` reads the same either way.
    """
    # Windows forbids ":" in names; the probe never reads the directory name.
    path = root / "devices" / (name.replace(":", "_") if os.name == "nt" else name)
    path.mkdir(parents = True)
    (path / "vendor").write_text(f"0x{vendor}\n")
    (path / "device").write_text(f"0x{device}\n")
    if device_class:
        (path / "class").write_text(f"0x{device_class}\n")
    if driver:
        _link_driver(path, root, driver)
    return path


def _link_driver(device: Path, root: Path, driver: str) -> None:
    """Emulate ``device/driver -> drivers/<name>`` the way sysfs has it.

    A symlink needs a privilege Windows does not grant by default, so a directory junction
    stands in there; ``resolve().name`` is the driver name either way.
    """
    target = root / "drivers" / driver
    target.mkdir(parents = True, exist_ok = True)
    link = device / "driver"
    try:
        link.symlink_to(target, target_is_directory = True)
    except OSError:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output = True,
            timeout = 30,
            check = False,
        )


def _fake_sysfs(monkeypatch, tmp_path: Path) -> None:
    """Redirect every Linux vendor scan at the fake tree instead of the real /sys."""
    devices = tmp_path / "devices"
    module = tmp_path / "module"
    scan = npu._linux_vendor

    def fake(vendor, table, name, unknown = None, *roots):
        return scan(vendor, table, name, unknown, devices, module)

    monkeypatch.setattr(npu.sys, "platform", "linux")
    monkeypatch.setattr(npu, "_linux_vendor", fake)


class _Completed:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def _sysctl(answer: str):
    def run(argv, **kwargs):
        assert argv[0] == "sysctl" and argv[-1] == npu._APPLE_ANE_SYSCTL
        assert kwargs.get("timeout") and kwargs.get("check") is False
        return _Completed(answer)

    return run


def _apple(monkeypatch, machine: str = "arm64") -> None:
    monkeypatch.setattr(npu.sys, "platform", "darwin")
    monkeypatch.setattr(npu.platform, "machine", lambda: machine)


def test_amd_xdna2_is_present_and_supported(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    npu_dir = _device(tmp_path, "0000:c4:00.1", "1022", "17f0", driver = "amdxdna")
    (npu_dir / "fw_version").write_text("1.1.2.65\n")
    (npu_dir / "vbnv").write_text("NPU Strix Halo\n")
    (tmp_path / "module" / "amdxdna").mkdir(parents = True)
    (tmp_path / "module" / "amdxdna" / "version").write_text("2.25.260102\n")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    device = npu.detect_npu_device("amd")
    assert device["present"] is True
    assert device["vendor"] == "amd"
    assert device["pci_id"] == "1022:17f0"
    assert device["family"] == "XDNA2"
    assert device["supported"] is True
    assert device["name"] == "NPU Strix Halo"
    assert device["driver"] == "amdxdna"
    assert device["driver_version"] == "2.25.260102"
    assert device["firmware_version"] == "1.1.2.65"


def test_amd_xdna1_is_present_but_unsupported(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:c4:00.1", "1022", "1502", driver = "amdxdna")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    device = npu.detect_npu_device("amd")
    assert device["present"] is True
    assert device["family"] == "XDNA1"
    assert device["supported"] is False

    summary = npu.detect_npu()
    assert summary["present"] is True and summary["supported"] is False
    assert summary["reason"] == "unsupported_family"
    assert summary["primary"]["family"] == "XDNA1"


def test_detect_amd_npu_keeps_its_published_key_set(tmp_path, monkeypatch):
    """The compatibility contract: existing callers read exactly these keys."""
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:c4:00.1", "1022", "17f0", driver = "amdxdna")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    info = npu.detect_amd_npu()
    assert set(info) == _AMD_KEYS
    assert info["present"] is True and info["supported"] is True


@pytest.mark.parametrize(
    "product, family",
    [("7d5d", "NPU3"), ("ad1d", "NPU3"), ("fd3e", "NPU3"), ("b03e", "NPU4"), ("643e", "NPU")],
)
def test_intel_known_device_ids(tmp_path, monkeypatch, product, family):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:00:0b.0", "8086", product, driver = "intel_vpu")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    device = npu.detect_npu_device("intel")
    assert device["present"] is True
    assert device["vendor"] == "intel"
    assert device["pci_id"] == f"8086:{product}"
    assert device["family"] == family
    assert device["supported"] is True


def test_intel_class_fallback_catches_an_unlisted_accelerator(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:00:0b.0", "8086", "1234", device_class = "118000")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    device = npu.detect_npu_device("intel")
    assert device["present"] is True
    assert device["family"] == "NPU"
    assert device["pci_id"] == "8086:1234"
    # Present, but no runtime here claims an unknown generation.
    assert device["supported"] is False


def test_intel_device_outside_the_accelerator_class_is_ignored(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:00:02.0", "8086", "1234", device_class = "030000")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    assert npu.detect_npu_device("intel")["present"] is False


def test_qualcomm_hexagon_snapdragon_x_elite(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:01:00.0", "17cb", "1100", driver = "qcom_npu")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    device = npu.detect_npu_device("qualcomm")
    assert device["present"] is True
    assert device["vendor"] == "qualcomm"
    assert device["pci_id"] == "17cb:1100"
    assert device["family"] == "Hexagon"
    assert device["supported"] is True


def test_qualcomm_class_fallback_catches_an_unlisted_hexagon(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:01:00.0", "17cb", "1102", device_class = "118000")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    device = npu.detect_npu_device("qualcomm")
    assert device["present"] is True
    assert device["family"] == "Hexagon"
    assert device["supported"] is False


def test_apple_ane_is_found_when_sysctl_reports_it(monkeypatch):
    _apple(monkeypatch)
    monkeypatch.setattr(npu.subprocess, "run", _sysctl("1\n"))
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    device = npu.detect_npu_device("apple")
    assert device["present"] is True
    assert device["vendor"] == "apple"
    assert device["pci_id"] is None
    assert device["family"] == "ANE"
    assert device["supported"] is True


def test_apple_is_absent_off_apple_silicon(monkeypatch):
    monkeypatch.setattr(npu.sys, "platform", "linux")
    assert npu.detect_npu_device("apple")["present"] is False


def test_apple_is_absent_on_a_darwin_host_without_apple_silicon(monkeypatch):
    _apple(monkeypatch, "x86_64")
    assert npu.detect_npu_device("apple")["present"] is False


def test_apple_ane_is_present_but_unprobed_when_sysctl_is_missing(monkeypatch):
    _apple(monkeypatch)

    def missing(argv, **kwargs):
        raise FileNotFoundError("sysctl")

    monkeypatch.setattr(npu.subprocess, "run", missing)
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    device = npu.detect_npu_device("apple")
    assert device["present"] is True
    assert device["supported"] is False
    assert device["reason"] == "ane_unprobed"


def test_microsoft_has_no_pci_device_of_its_own():
    assert npu.detect_npu_device("microsoft") == {
        "present": False,
        "vendor": "microsoft",
        "reason": "not_pci_visible",
    }


def test_an_unknown_vendor_is_reported_not_raised():
    assert npu.detect_npu_device("riscv") == {
        "present": False,
        "vendor": "riscv",
        "reason": "unknown_vendor",
    }


def test_detect_npu_without_the_flag_reports_disabled(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:c4:00.1", "1022", "17f0", driver = "amdxdna")
    monkeypatch.delenv(npu.ENABLE_ENV, raising = False)

    summary = npu.detect_npu()
    assert summary["enabled"] is False
    assert summary["supported"] is False
    assert summary["reason"] == "disabled"
    assert len(summary["devices"]) == len(npu.VENDORS)

    capability = npu.npu_training_capability()
    assert capability["mode"] == "unavailable"
    assert capability["reasons"] == ["disabled"]
    assert capability["tasks"] == []
    assert capability["backward_pass"] is False
    assert capability["providers"] == [] or isinstance(capability["providers"], list)
    assert npu.ENABLE_ENV in capability["notes"]


def test_detect_npu_with_nothing_present_reports_absent(monkeypatch):
    monkeypatch.setattr(npu.sys, "platform", "linux")
    monkeypatch.setattr(npu, "_linux_probe", lambda: {"present": False})
    monkeypatch.setattr(npu, "_linux_vendor", lambda *args, **kwargs: ({"present": False}, None))
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    summary = npu.detect_npu()
    assert summary["present"] is False and summary["primary"] is None
    assert summary["reason"] == "absent"


def test_a_raising_vendor_probe_leaves_the_others_intact(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:c4:00.1", "1022", "17f0", driver = "amdxdna")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    def boom():
        raise RuntimeError("registry unreadable")

    monkeypatch.setitem(npu._PROBES, "intel", boom)

    summary = npu.detect_npu()
    assert [device["vendor"] for device in summary["devices"]] == list(npu.VENDORS)
    assert summary["supported"] is True
    assert summary["primary"]["vendor"] == "amd"
    intel = summary["devices"][1]
    assert intel["present"] is False and intel["reason"] == "probe_failed"


def test_every_probe_raising_still_yields_a_usable_aggregate(monkeypatch):
    def boom():
        raise PermissionError("no hardware access")

    for vendor in npu.VENDORS:
        monkeypatch.setitem(npu._PROBES, vendor, boom)
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    summary = npu.detect_npu()
    assert summary["present"] is False
    assert summary["supported"] is False
    assert summary["reason"] == "probe_failed"
    assert summary["primary"] is None
    assert len(summary["devices"]) == len(npu.VENDORS)


def test_primary_prefers_a_supported_device_over_the_first_present_one(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:c4:00.1", "1022", "1502", driver = "amdxdna")
    _device(tmp_path, "0000:00:0b.0", "8086", "7d5d", driver = "intel_vpu")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    summary = npu.detect_npu()
    assert summary["supported"] is True
    assert summary["primary"]["vendor"] == "intel"
    assert summary["primary"]["family"] == "NPU3"


def _provider(installed: bool, vendors: tuple[str, ...], supports: tuple[str, ...]) -> dict:
    return {
        "name": "fake",
        "installed": installed,
        "version": "1.0",
        "detail": None,
        "vendors": vendors,
        "supports": supports,
    }


def test_capability_is_assist_when_a_provider_can_run_inference(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:c4:00.1", "1022", "17f0", driver = "amdxdna")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")
    fake = [_provider(True, ("amd",), ("inference", "eval"))]
    monkeypatch.setattr(npu, "npu_runtime_providers", lambda: fake)

    capability = npu.npu_training_capability()
    assert capability["mode"] == "assist"
    assert capability["backward_pass"] is False
    assert capability["tasks"] == ["eval", "inference"]
    assert capability["reasons"] == []
    assert capability["providers"] == fake
    assert capability["device"]["vendor"] == "amd"
    assert "backward pass" in capability["notes"]


def test_capability_is_unavailable_without_a_runtime(tmp_path, monkeypatch):
    _fake_sysfs(monkeypatch, tmp_path)
    _device(tmp_path, "0000:c4:00.1", "1022", "17f0", driver = "amdxdna")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")
    fake = [
        _provider(False, ("amd",), ("inference", "eval")),
        _provider(True, ("intel",), ("inference",)),
        # Installed for this vendor but unable to run inference, so it cannot make it usable.
        _provider(True, ("amd",), ("qat_calibrate",)),
    ]
    monkeypatch.setattr(npu, "npu_runtime_providers", lambda: fake)

    capability = npu.npu_training_capability()
    assert capability["mode"] == "unavailable"
    assert capability["reasons"] == ["no_runtime"]
    assert capability["tasks"] == []
    assert "no installed NPU runtime" in capability["notes"]
