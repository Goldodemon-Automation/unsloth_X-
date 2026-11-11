# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detect NPUs from any vendor without starting a runtime.

XDNA 2 (1022:17f0) is supported, and only with ``UNSLOTH_STUDIO_ENABLE_NPU=1``; XDNA 1
(1022:1502) is present but unsupported. Runtime readiness is checked separately by ``flm
validate``. Intel, Qualcomm and Apple devices come off the same PCI/registry walk and report the
same device shape, so the hardware section, the NPU loads and the CLI's NPU ids can name whichever
accelerator a machine actually has, and ``npu_training_capability()`` can say plainly what the
installed runtimes can and cannot do with it. ``detect_amd_npu()`` keeps the keys its existing
callers already read.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from .npu_providers import npu_runtime_providers

_PCI_ROOT = Path("/sys/bus/pci/devices")
_MODULE_ROOT = Path("/sys/module")
# PCI class for a processing accelerator. This is the generic fallback: a vendor's NPU whose id we
# have never seen still identifies itself here, which beats guessing an id that may not exist.
_ACCELERATOR_CLASS = "118000"
# Off by default: the NPU section, NPU loads and the CLI's NPU ids all wait on this.
ENABLE_ENV = "UNSLOTH_STUDIO_ENABLE_NPU"

VENDORS: tuple[str, ...] = ("amd", "intel", "qualcomm", "apple", "microsoft")

_AMD_VENDOR = "1022"
_INTEL_VENDOR = "8086"
_QUALCOMM_VENDOR = "17cb"

_APPLE_ANE_SYSCTL = "hw.optional.ane"
_APPLE_ANE_TIMEOUT_S = 2

# Device id -> (family, whether the runtimes we know of ship kernels for that generation). Only
# ids seen in the field: a made-up id would report hardware that may not exist.
_AMD_DEVICES: dict[str, tuple[str, bool]] = {
    "17f0": ("XDNA2", True),
    "1502": ("XDNA1", False),
}
_INTEL_DEVICES: dict[str, tuple[str, bool]] = {
    "7d5d": ("NPU3", True),  # Meteor Lake
    "ad1d": ("NPU3", True),  # Arrow Lake / Lunar Lake
    "fd3e": ("NPU3", True),  # Arrow Lake-H / Core Ultra 200H
    "b03e": ("NPU4", True),  # Panther Lake
    "643e": ("NPU", True),  # Keem Bay
}
_QUALCOMM_DEVICES: dict[str, tuple[str, bool]] = {
    "1100": ("Hexagon", True),  # Snapdragon X Elite
    "1101": ("Hexagon", True),  # Snapdragon X Plus
}

# The keys every device dict carries; kept in one place so the vendor matrix cannot drift from the
# legacy AMD shape that ``detect_amd_npu()`` callers already depend on.
_DEVICE_KEYS = (
    "present",
    "family",
    "name",
    "driver",
    "driver_version",
    "firmware_version",
    "device_node",
)


def npu_enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding = "utf-8", errors = "replace").strip()
    except OSError:
        return None


def _bare(text: Optional[str]) -> str:
    """A sysfs attribute value as it is compared here: lowercase, without the 0x prefix."""
    return (text or "").lower().removeprefix("0x")


def _scan_linux_pci(pci_root: Path) -> list[dict[str, Any]]:
    """Every PCI function under sysfs with its vendor, device and class, normalised.

    Unreadable entries are dropped rather than raised on: sysfs has files an unprivileged process
    cannot read, and losing one function must not lose the rest.
    """
    try:
        entries = sorted(pci_root.iterdir())
    except OSError:
        return []
    found: list[dict[str, Any]] = []
    for entry in entries:
        vendor = _bare(_read(entry / "vendor"))
        if not vendor:
            continue
        found.append(
            {
                "path": entry,
                "vendor": vendor,
                "device": _bare(_read(entry / "device")),
                "class": _bare(_read(entry / "class")),
            }
        )
    return found


def _linux_details(entry: dict[str, Any], module_root: Path, default_name: str) -> dict[str, Any]:
    """The sysfs detail that is the same for every vendor: name, driver, firmware, accel node."""
    path = entry["path"]
    driver_link = path / "driver"
    driver = driver_link.resolve().name if driver_link.exists() else None
    accel_dir = path / "accel"
    accel = sorted(accel_dir.glob("accel*")) if accel_dir.is_dir() else []
    return {
        "present": True,
        "name": _read(path / "vbnv") or default_name,
        "driver": driver,
        "driver_version": _read(module_root / driver / "version") if driver else None,
        "firmware_version": _read(path / "fw_version"),
        "device_node": f"/dev/accel/{accel[0].name}" if accel else None,
    }


def _linux_vendor(
    vendor: str,
    devices: dict[str, tuple[str, bool]],
    default_name: str,
    unknown_family: Optional[str],
    pci_root: Path = _PCI_ROOT,
    module_root: Path = _MODULE_ROOT,
) -> tuple[dict[str, Any], Optional[str]]:
    """First NPU of one PCI vendor in sysfs, plus the PCI id it was found under.

    ``unknown_family`` is the family to report for an accelerator-class device whose id is not in
    the table; None disables that fallback for a vendor whose ids we require exactly.
    """
    for entry in _scan_linux_pci(pci_root):
        if entry["vendor"] != vendor:
            continue
        match = devices.get(entry["device"])
        if match is None:
            if unknown_family is None or entry["class"] != _ACCELERATOR_CLASS:
                continue
            # Unlisted id, but the class says processing accelerator: report it as present and
            # unsupported instead of hiding a device the machine really has.
            match = (unknown_family, False)
        info = _linux_details(entry, module_root, default_name)
        info["family"] = match[0]
        info["generation_supported"] = match[1]
        return info, f"{vendor}:{entry['device']}"
    return {"present": False}, None


def _linux_amd(
    pci_root: Path = _PCI_ROOT, module_root: Path = _MODULE_ROOT
) -> tuple[dict[str, Any], Optional[str]]:
    """AMD NPU on sysfs, plus the PCI id of the function that matched."""
    return _linux_vendor(_AMD_VENDOR, _AMD_DEVICES, "AMD NPU", None, pci_root, module_root)


def _linux_probe(pci_root: Path = _PCI_ROOT, module_root: Path = _MODULE_ROOT) -> dict[str, Any]:
    """AMD NPU exactly as ``detect_amd_npu()`` has always reported it, key for key."""
    info, _pci_id = _linux_amd(pci_root, module_root)
    info.pop("generation_supported", None)
    return info


def _windows_pci_ids(vendor: str) -> list[str]:
    """Hardware ids the PCI enum key holds for one vendor; empty when the key is unreadable."""
    base = r"SYSTEM\CurrentControlSet\Enum\PCI"
    prefix = f"VEN_{vendor.upper()}&DEV_"
    try:
        import winreg
        pci = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except (ImportError, OSError):
        return []
    ids: list[str] = []
    with pci:
        index = 0
        while True:
            try:
                hardware_id = winreg.EnumKey(pci, index)
            except OSError:
                return ids
            index += 1
            if hardware_id.upper().startswith(prefix):
                ids.append(hardware_id)
    return ids


def _reg_value(key, name: str) -> Optional[str]:
    import winreg
    try:
        value, _kind = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return str(value) if value is not None else None


def _windows_details(hardware_id: str) -> dict[str, Any]:
    """Friendly name and driver version for one PCI instance; blanks when unreadable."""
    import winreg

    driver_version = None
    name = None
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, rf"SYSTEM\CurrentControlSet\Enum\PCI\{hardware_id}"
        ) as by_id:
            instance = winreg.EnumKey(by_id, 0)
            with winreg.OpenKey(by_id, instance) as device:
                name = _reg_value(device, "FriendlyName") or _reg_value(device, "DeviceDesc")
                driver_key = _reg_value(device, "Driver")
        if driver_key:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                rf"SYSTEM\CurrentControlSet\Control\Class\{driver_key}",
            ) as driver:
                driver_version = _reg_value(driver, "DriverVersion")
    except OSError:
        pass
    if name and ";" in name:
        # DeviceDesc is stored as "@oemNN.inf,%key%;Readable name".
        name = name.rsplit(";", 1)[1]
    return {
        "present": True,
        "name": name,
        "driver": "installed" if driver_version else None,
        "driver_version": driver_version,
        "firmware_version": None,
        "device_node": None,
    }


def _windows_vendor(
    vendor: str,
    devices: dict[str, tuple[str, bool]],
    default_name: str,
) -> tuple[dict[str, Any], Optional[str]]:
    """First NPU of one PCI vendor in the registry, plus the PCI id it was found under."""
    for hardware_id in _windows_pci_ids(vendor):
        product = hardware_id.upper().split("&DEV_", 1)[1][:4].lower()
        match = devices.get(product)
        if match is None:
            continue
        info = _windows_details(hardware_id)
        info["name"] = info["name"] or default_name
        info["family"] = match[0]
        info["generation_supported"] = match[1]
        return info, f"{vendor}:{product}"
    return {"present": False}, None


def _windows_amd() -> tuple[dict[str, Any], Optional[str]]:
    """AMD NPU in the registry, plus the PCI id of the instance that matched."""
    return _windows_vendor(_AMD_VENDOR, _AMD_DEVICES, "AMD NPU")


def _windows_probe() -> dict[str, Any]:
    """AMD NPU exactly as ``detect_amd_npu()`` has always reported it, key for key."""
    info, _pci_id = _windows_amd()
    info.pop("generation_supported", None)
    return info


def _vendor_scan(
    vendor: str,
    devices: dict[str, tuple[str, bool]],
    default_name: str,
    unknown_family: Optional[str],
) -> tuple[dict[str, Any], Optional[str]]:
    """Dispatch one vendor's PCI scan to the platform that owns an inventory."""
    if sys.platform.startswith("linux"):
        return _linux_vendor(vendor, devices, default_name, unknown_family)
    if sys.platform == "win32":
        # The registry exposes no device class, so the accelerator-class fallback stays Linux only.
        return _windows_vendor(vendor, devices, default_name)
    return {"present": False}, None


def _device(vendor: str, pci_id: Optional[str], info: dict[str, Any]) -> dict[str, Any]:
    """One vendor's device in the shape every caller reads.

    A probe only fills the keys it could read, so the shared shape is completed here: a consumer
    looping over devices must not need a guard per key.
    """
    device: dict[str, Any] = {"present": bool(info.get("present"))}
    for key in _DEVICE_KEYS[1:]:
        device[key] = info.get(key)
    device["vendor"] = vendor
    device["pci_id"] = pci_id
    device["enabled"] = npu_enabled()
    # `supported` is the runtime question, not the hardware one: the flag is off by default, so a
    # perfectly good NPU reads as present but unsupported until the operator enables it.
    device["supported"] = (
        device["present"] and device["enabled"] and bool(info.get("generation_supported"))
    )
    for key in ("reason", "error"):
        if info.get(key):
            device[key] = info[key]
    return device


def _probe_amd() -> dict[str, Any]:
    info, pci_id = _vendor_scan(_AMD_VENDOR, _AMD_DEVICES, "AMD NPU", None)
    return _device("amd", pci_id, info)


def _probe_intel() -> dict[str, Any]:
    # Any other Intel accelerator-class function counts too, so a generation newer than the table
    # still shows up in the matrix rather than disappearing.
    info, pci_id = _vendor_scan(_INTEL_VENDOR, _INTEL_DEVICES, "Intel NPU", "NPU")
    return _device("intel", pci_id, info)


def _probe_qualcomm() -> dict[str, Any]:
    info, pci_id = _vendor_scan(
        _QUALCOMM_VENDOR, _QUALCOMM_DEVICES, "Qualcomm Hexagon NPU", "Hexagon"
    )
    return _device("qualcomm", pci_id, info)


def _apple_device() -> dict[str, Any]:
    """Apple Neural Engine. ANE is not a PCI function, so it is probed through its sysctl.

    A sysctl that cannot be run is not evidence of absence on Apple Silicon, so the device stays
    present and unsupported instead of vanishing from the matrix.
    """
    info: dict[str, Any] = {"present": False}
    if sys.platform != "darwin":
        return info
    try:
        machine = platform.machine()
    except Exception:  # noqa: BLE001 -- detection must never break startup
        machine = ""
    if machine != "arm64":
        return info
    info = {
        "present": True,
        "family": "ANE",
        "name": "Apple Neural Engine",
        "driver": None,
        "driver_version": None,
        "firmware_version": None,
        "device_node": None,
        "generation_supported": False,
    }
    try:
        result = subprocess.run(
            ["sysctl", "-n", _APPLE_ANE_SYSCTL],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = _APPLE_ANE_TIMEOUT_S,
            check = False,
        )
    except (OSError, subprocess.SubprocessError):
        info["reason"] = "ane_unprobed"
        return info
    answer = (result.stdout or "").strip()
    if answer == "1":
        info["generation_supported"] = True
    elif answer == "0":
        # The sysctl answered: this chip reports no ANE, which is a real absence.
        info = {"present": False, "reason": "ane_absent"}
    else:
        info["reason"] = "ane_unprobed"
    return info


def _apple_probe() -> dict[str, Any]:
    return _device("apple", None, _apple_device())


def _probe_microsoft() -> dict[str, Any]:
    """Windows ML is a provider, not a device: it drives NPUs other vendors' PCI entries own.

    A bare marker rather than a full device entry because this is also the published answer to
    ``detect_npu_device("microsoft")``, which callers compare key for key, so it cannot grow the
    device keys itself. ``_matrix_entry()`` completes it for the aggregate.
    """
    return {"present": False, "vendor": "microsoft", "reason": "not_pci_visible"}


def _matrix_entry(vendor: str, device: dict[str, Any]) -> dict[str, Any]:
    """One ``detect_npu()`` entry in the shape ``_device()`` promises.

    ``_probe_microsoft()`` answers with a marker (see above) and the aggregate's probe-failure
    path builds its own dict, so the shared shape is filled here through ``_device()``: nothing
    looping over ``devices`` should have to special-case one vendor.
    """
    if "supported" in device:
        return device
    return _device(vendor, device.get("pci_id"), device)


_PROBES: dict[str, Callable[[], dict[str, Any]]] = {
    "amd": _probe_amd,
    "intel": _probe_intel,
    "qualcomm": _probe_qualcomm,
    "apple": _apple_probe,
    "microsoft": _probe_microsoft,
}


def detect_amd_npu() -> dict[str, Any]:
    """Return NPU hardware details and whether FastFlowLM supports this platform."""
    try:
        if sys.platform.startswith("linux"):
            info = _linux_probe()
        elif sys.platform == "win32":
            info = _windows_probe()
        else:
            info = {"present": False}
    except Exception as exc:  # noqa: BLE001 -- detection must never break startup or /status
        info = {"present": False, "error": str(exc)}
    info["enabled"] = npu_enabled()
    info["supported"] = (
        info["enabled"] and bool(info.get("present")) and info.get("family") == "XDNA2"
    )
    return info


def detect_npu_device(vendor: str) -> dict[str, Any]:
    """Probe one NPU vendor; never raises, so an unknown or broken vendor reads as absent."""
    name = (vendor or "").strip().lower()
    if name not in VENDORS:
        return {"present": False, "vendor": vendor, "reason": "unknown_vendor"}
    probe = _PROBES[name]
    try:
        return probe()
    except Exception as exc:  # noqa: BLE001 -- one vendor raising must not blank the others
        return _device(name, None, {"present": False, "reason": "probe_failed", "error": str(exc)})


def detect_npu() -> dict[str, Any]:
    """Every vendor's NPU plus the single device a UI should lead with."""
    devices: list[dict[str, Any]] = []
    failed = False
    for vendor in VENDORS:
        try:
            device = detect_npu_device(vendor)
        except Exception as exc:  # noqa: BLE001 -- a caller may have swapped in a raw probe
            device = {
                "present": False,
                "vendor": vendor,
                "reason": "probe_failed",
                "error": str(exc),
            }
        device = _matrix_entry(vendor, device)
        if device.get("reason") == "probe_failed":
            failed = True
        devices.append(device)
    enabled = npu_enabled()
    supported_devices = [device for device in devices if device.get("supported")]
    present_devices = [device for device in devices if device.get("present")]
    primary = (supported_devices or present_devices or [None])[0]
    supported = enabled and bool(supported_devices)
    if supported:
        reason: Optional[str] = None
    elif not enabled:
        reason = "disabled"
    elif not present_devices:
        # Distinguish "this machine has none" from "we could not look", because only the second
        # is worth a bug report.
        reason = "probe_failed" if failed else "absent"
    else:
        reason = "unsupported_family"
    return {
        "present": bool(present_devices),
        "enabled": enabled,
        "supported": supported,
        "reason": reason,
        "devices": devices,
        "primary": primary,
    }


def _provider_rows() -> list[dict[str, Any]]:
    """Provider probes, isolated so a broken runtime install cannot blank the matrix."""
    try:
        return npu_runtime_providers()
    except Exception:  # noqa: BLE001 -- availability reporting must never break /status
        return []


# Order the tasks are listed in, so a UI shows the same order for every provider set.
_CAPABILITY_TASKS = ("tokenize", "eval", "qat_calibrate", "inference")
# No runtime drives an NPU backward pass today, so "full" is unreachable until a provider row
# claims one of these. Keeping the claim a provider-side fact is what stops the mode inflating.
_BACKWARD_TOKENS = ("backward", "backward_pass", "training")


def npu_training_capability() -> dict[str, Any]:
    """What training can actually use the NPU for, in terms a UI can show unedited."""
    summary = detect_npu()
    providers = _provider_rows()
    primary = summary.get("primary")
    result: dict[str, Any] = {
        "mode": "unavailable",
        "backward_pass": False,
        "device": primary,
        "tasks": [],
        "providers": providers,
        "reasons": [],
        "notes": "",
    }
    name = (primary or {}).get("name") or (primary or {}).get("family") or "An NPU"
    if not summary.get("enabled"):
        result["reasons"] = ["disabled"]
        result["notes"] = f"NPU support is off; set {ENABLE_ENV}=1 to let Studio use the NPU."
        return result
    if not summary.get("present"):
        result["reasons"] = ["absent"]
        result["notes"] = (
            "No NPU was detected on this machine, so training runs entirely on the primary "
            "compute device."
        )
        return result
    if not summary.get("supported"):
        result["reasons"] = ["unsupported_family"]
        result["notes"] = (
            f"{name} is not an NPU generation any runtime here supports, so training runs "
            "entirely on the primary compute device."
        )
        return result
    vendor = (primary or {}).get("vendor")
    installed = [provider for provider in providers if provider.get("installed")]
    usable = [
        provider
        for provider in installed
        if "inference" in provider.get("supports", ()) and vendor in provider.get("vendors", ())
    ]
    if not usable:
        result["reasons"] = ["no_runtime"]
        result["notes"] = (
            f"{name} is present, but no installed NPU runtime can drive it, so training runs "
            "entirely on the primary compute device."
        )
        return result
    supported_tasks = {task for provider in usable for task in provider.get("supports", ())}
    tasks = [task for task in _CAPABILITY_TASKS if task in supported_tasks]
    backward = any(
        token in provider.get("supports", ())
        for provider in usable
        for token in _BACKWARD_TOKENS
    )
    result["mode"] = "full" if backward else "assist"
    result["backward_pass"] = backward
    result["tasks"] = tasks
    if backward:
        result["notes"] = f"{name} can run the backward pass, so Studio can train on it directly."
        return result
    listed = ", ".join(tasks) if tasks else "no tasks"
    result["notes"] = (
        f"{name} has no backward pass, so Studio uses it for {listed} only and keeps the "
        "backward pass on the primary compute device."
    )
    return result
