# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The inference backend reads the same NPU matrix the Studio reports.

Both answers are faked -- ``utils.hardware.npu.detect_npu`` for the matrix and
``npu_backend.detect_amd_npu`` for the legacy probe -- so these tests say what a machine reports
without needing to be that machine.
"""

from __future__ import annotations

import pytest

from core.inference import npu_backend as nb
from utils.hardware import npu

# What ``status()`` and ``_require_supported()`` read off ``hardware()``. ``enabled`` is not part
# of the legacy AMD contract, but the refusal message names the flag, so the helper carries it too.
_BACKEND_KEYS = {
    "present",
    "supported",
    "enabled",
    "family",
    "name",
    "driver",
    "driver_version",
    "firmware_version",
    "device_node",
}


def _absent() -> dict:
    """What the AMD-only probe answers on a machine with no AMD NPU in it."""
    return {"present": False}


def _never_called() -> dict:
    pytest.fail("the AMD-only probe must not answer when the matrix has a usable NPU")


def _device(vendor: str, family: str, *, supported: bool = True) -> dict:
    """One device entry as ``detect_npu()`` publishes it."""
    return {
        "present": True,
        "family": family,
        "name": f"{vendor} NPU",
        "driver": f"{vendor}_npu",
        "driver_version": "1.0",
        "firmware_version": "2.0",
        "device_node": "/dev/accel/accel0",
        "vendor": vendor,
        "pci_id": None,
        "enabled": True,
        "supported": supported,
    }


def _summary(
    primary: dict | None,
    *,
    present: bool = True,
    enabled: bool = True,
    supported: bool | None = None,
) -> dict:
    """The aggregate ``detect_npu()`` returns, around one primary device."""
    if supported is None:
        supported = enabled and present
    return {
        "present": present,
        "enabled": enabled,
        "supported": supported,
        "reason": None if supported else "unsupported_family",
        "devices": [primary] if primary is not None else [],
        "primary": primary,
    }


def _amd_xdna2() -> dict:
    """The old single-vendor answer, key for key as ``detect_amd_npu()`` returns it."""
    return {
        "present": True,
        "family": "XDNA2",
        "name": "NPU Strix Halo",
        "driver": "amdxdna",
        "driver_version": "2.25.260102",
        "firmware_version": "1.1.2.65",
        "device_node": "/dev/accel/accel0",
        "enabled": True,
        "supported": True,
    }


def _matrix(monkeypatch, summary: dict, legacy = _absent) -> None:
    """Point the helper at a fake matrix and a fake AMD-only probe."""
    monkeypatch.setattr(npu, "detect_npu", lambda: summary, raising = False)
    monkeypatch.setattr(nb, "detect_amd_npu", legacy)


def test_an_intel_npu_reaches_the_inference_backend(monkeypatch):
    # The AMD-only probe fails the test if consulted: a usable matrix decision comes first.
    _matrix(monkeypatch, _summary(_device("intel", "NPU3")), _never_called)

    hardware = nb._detect_npu_hardware()
    assert set(hardware) == _BACKEND_KEYS
    assert hardware["present"] is True
    assert hardware["supported"] is True
    assert hardware["family"] == "NPU3"
    assert hardware["name"] == "intel NPU"
    assert hardware["driver"] == "intel_npu"
    assert hardware["device_node"] == "/dev/accel/accel0"


def test_a_qualcomm_npu_is_usable_without_an_amd_one(tmp_path, monkeypatch):
    _matrix(monkeypatch, _summary(_device("qualcomm", "Hexagon")), _never_called)
    backend = nb.LemonadeNpuBackend(root = tmp_path)

    assert backend.hardware()["family"] == "Hexagon"
    # The point of the matrix: this backend used to refuse the machine outright.
    backend._require_supported()


def test_supported_follows_the_matrix_not_the_device_entry(monkeypatch):
    # The device itself is a supported generation; the flag is off, so the aggregate says no.
    _matrix(
        monkeypatch,
        _summary(_device("intel", "NPU3", supported = True), enabled = False, supported = False),
    )

    hardware = nb._detect_npu_hardware()
    assert hardware["present"] is True
    assert hardware["supported"] is False
    assert hardware["enabled"] is False


def test_a_present_but_unsupported_npu_is_not_reported_as_absent(monkeypatch):
    _matrix(
        monkeypatch,
        _summary(_device("intel", "NPU", supported = False), supported = False),
    )

    hardware = nb._detect_npu_hardware()
    # A caller can explain an unusable NPU; it must not read the same as a machine without one.
    assert hardware["present"] is True
    assert hardware["supported"] is False
    assert hardware["family"] == "NPU"


def test_a_raising_matrix_falls_back_to_the_amd_probe(monkeypatch):
    monkeypatch.setattr(nb, "detect_amd_npu", lambda: _amd_xdna2())

    def boom():
        raise RuntimeError("matrix unavailable")

    monkeypatch.setattr(npu, "detect_npu", boom, raising = False)

    assert nb._detect_npu_hardware() == _amd_xdna2()


def test_a_checkout_without_the_matrix_falls_back_to_the_amd_probe(monkeypatch):
    monkeypatch.setattr(nb, "detect_amd_npu", lambda: _amd_xdna2())
    monkeypatch.delattr(npu, "detect_npu")

    assert nb._detect_npu_hardware() == _amd_xdna2()


def test_a_matrix_with_nothing_present_falls_back_to_the_amd_probe(monkeypatch):
    _matrix(monkeypatch, _summary(None, present = False), lambda: _amd_xdna2())

    assert nb._detect_npu_hardware() == _amd_xdna2()


def test_the_amd_probe_answers_when_the_matrix_has_nothing_usable(monkeypatch):
    # AMD hardware the matrix reports as unusable: the legacy answer is the one published callers
    # read, and it says the same thing about the same PCI function.
    _matrix(
        monkeypatch,
        _summary(_device("amd", "XDNA1", supported = False), supported = False),
        lambda: dict(_amd_xdna2(), supported = False, family = "XDNA1"),
    )

    hardware = nb._detect_npu_hardware()
    assert hardware["present"] is True
    assert hardware["supported"] is False
    assert hardware["family"] == "XDNA1"


def test_the_amd_path_still_matches_where_the_keys_overlap(monkeypatch):
    monkeypatch.setattr(nb, "detect_amd_npu", lambda: _amd_xdna2())
    monkeypatch.delattr(npu, "detect_npu")

    hardware = nb._detect_npu_hardware()
    for key, value in _amd_xdna2().items():
        assert hardware[key] == value
    assert set(hardware) == _BACKEND_KEYS


# The device keys plus the four ``_device()`` derives, straight from the module's own constant.
_ENTRY_KEYS = set(npu._DEVICE_KEYS) | {"vendor", "pci_id", "enabled", "supported"}


def test_every_matrix_entry_has_the_same_device_keys():
    for entry in npu.detect_npu()["devices"]:
        assert _ENTRY_KEYS <= set(entry), entry["vendor"]
        # Only the two keys that explain themselves may be added.
        assert set(entry) - _ENTRY_KEYS <= {"reason", "error"}


def test_the_microsoft_entry_matches_a_vendor_device_entry():
    devices = {entry["vendor"]: entry for entry in npu.detect_npu()["devices"]}
    microsoft = devices["microsoft"]

    assert microsoft["present"] is False
    assert microsoft["reason"] == "not_pci_visible"
    assert set(microsoft) - {"reason"} == set(devices["amd"]) - {"reason"}
    # detect_npu_device("microsoft") is a published answer callers compare key for key, so the
    # shape is completed in the matrix entry instead of in the probe.
    assert npu._probe_microsoft() == {
        "present": False,
        "vendor": "microsoft",
        "reason": "not_pci_visible",
    }

def _refusal(tmp_path, monkeypatch, hardware: dict) -> str:
    """The message `_require_supported()` raises for a given `hardware()` answer."""
    backend = nb.LemonadeNpuBackend(root = tmp_path)
    monkeypatch.setattr(backend, "hardware", lambda: hardware)
    with pytest.raises(nb.NpuError) as caught:
        backend._require_supported()
    return str(caught.value)


def test_a_non_amd_npu_is_told_what_the_runtime_needs(tmp_path, monkeypatch):
    """The refusal read "No AMD Ryzen AI NPU was found" on an Intel machine: a wrong diagnosis rather
    than a limitation, and one a user would act on by suspecting their AMD driver."""
    message = _refusal(
        tmp_path,
        monkeypatch,
        {
            "present": True,
            "supported": False,
            "enabled": True,
            "vendor": "intel",
            "family": "NPU4",
            "name": "Intel NPU",
        },
    )

    assert "AMD" in message
    assert "NPU4" in message
    assert "Ryzen AI NPU was found" not in message


def test_no_npu_at_all_is_reported_without_naming_a_vendor(tmp_path, monkeypatch):
    message = _refusal(tmp_path, monkeypatch, {"present": False, "supported": False, "enabled": True})

    assert "AMD" not in message
    assert "No NPU was found" in message
