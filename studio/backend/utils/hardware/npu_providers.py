# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Report which NPU runtimes are installed and what each one can do.

Nothing here imports a runtime: every row is answered from distribution metadata, module specs
and PATH lookups, so the hardware section can list providers on a machine where loading a runtime
would take seconds, fail on a missing driver, or not exist at all. Every provider is always
returned, installed or not, because the UI shows the alternatives next to the working one.

The answer changes when someone installs a runtime, which is minutes apart from the requests that
ask for it, so the rows are cached for a short TTL: six PATH walks per call is tens of
milliseconds of stat calls that a request path cannot afford to repeat.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

PROVIDERS: tuple[str, ...] = (
    "openvino",
    "onnxruntime_qnn",
    "vitis_ai",
    "fastflowlm",
    "lemonade",
    "windows_ml",
)

# Vendors each provider can drive and the tasks it can do on them, in one table: a failed probe
# and a successful one must describe the provider the same way, or the UI would change its story
# depending on whether the probe threw.
_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "openvino": (("intel",), ("inference", "tokenize", "eval", "qat_calibrate")),
    "onnxruntime_qnn": (("qualcomm",), ("inference", "eval")),
    "vitis_ai": (("amd",), ("inference", "eval", "qat_calibrate")),
    "fastflowlm": (("amd",), ("inference", "eval")),
    "lemonade": (("amd", "intel", "qualcomm"), ("inference", "eval")),
    "windows_ml": (("microsoft", "qualcomm", "intel", "amd"), ("inference",)),
}

_QNN_HTP_LIBRARIES = ("QnnHtp.dll", "libQnnHtp.so")
# Where the Qualcomm SDK keeps the HTP backend under QNN_SDK_ROOT; the runtime refuses to load
# without it, so its absence makes the onnxruntime QNN row "not installed" even when the wheel is.
_QNN_SDK_LIB_DIRS = (
    "",
    "lib/aarch64-windows-msvc",
    "lib/aarch64-ubuntu-gcc7.5",
    "lib/aarch64-android",
)
_WINDOWS_ML_PACKAGES = ("winrt-Windows.AI.MachineLearning", "windows_ml")


def _entry(
    name: str,
    installed: bool = False,
    version: Optional[str] = None,
    detail: Optional[str] = None,
) -> dict[str, Any]:
    vendors, supports = _SPEC[name]
    return {
        "name": name,
        "installed": bool(installed),
        "version": version,
        "detail": detail,
        "vendors": tuple(vendors),
        "supports": tuple(supports),
    }


def _distribution(package: str) -> tuple[bool, Optional[str]]:
    """(installed, version) from distribution metadata alone; the runtime is never imported."""
    try:
        return True, importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False, None
    except Exception:  # noqa: BLE001 -- unreadable metadata reads as "not installed"
        return False, None


def _module_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:  # noqa: BLE001 -- a broken parent package raises here
        return False


def _spec_origin(module: str) -> Optional[str]:
    """Where an installed module would be loaded from, or None. Nothing is imported."""
    try:
        spec = importlib.util.find_spec(module)
    except Exception:  # noqa: BLE001 -- a broken install must not raise here
        return None
    if spec is None:
        return None
    return getattr(spec, "origin", None)


def _qnn_htp_libraries() -> list[Path]:
    """The QnnHtp backend beside the onnxruntime package, or anywhere under QNN_SDK_ROOT."""
    roots: list[Path] = []
    origin = _spec_origin("onnxruntime")
    if origin:
        package = Path(origin).resolve().parent
        # The wheel puts the DLL/SO at the package root, under capi/, or in the site-packages
        # sibling the QNN build copies it into.
        roots.extend([package, package / "capi", package.parent])
    sdk_root = os.environ.get("QNN_SDK_ROOT")
    if sdk_root:
        roots.extend(Path(sdk_root) / part for part in _QNN_SDK_LIB_DIRS)
    found: list[Path] = []
    for root in roots:
        for library in _QNN_HTP_LIBRARIES:
            candidate = root / library
            if candidate.is_file():
                found.append(candidate)
    return found


def _probe_openvino() -> dict[str, Any]:
    installed, version = _distribution("openvino")
    return _entry("openvino", installed, version)


def _probe_onnxruntime_qnn() -> dict[str, Any]:
    installed, version = _distribution("onnxruntime")
    if not installed:
        return _entry("onnxruntime_qnn", False, None, "onnxruntime is not installed")
    libraries = _qnn_htp_libraries()
    if not libraries:
        # The x86 wheel ships no HTP backend: the row would claim a runtime that cannot run.
        return _entry(
            "onnxruntime_qnn",
            False,
            version,
            "QnnHtp.dll/libQnnHtp.so not found beside onnxruntime or under QNN_SDK_ROOT",
        )
    return _entry("onnxruntime_qnn", True, version, str(libraries[0]))


def _probe_vitis_ai() -> dict[str, Any]:
    runtime_installed, runtime_version = _distribution("vitis_ai_runtime")
    qat_installed, qat_version = _distribution("vitis_ai_quantization_pytorch")
    if not runtime_installed and not qat_installed:
        return _entry(
            "vitis_ai", False, None, "neither vitis_ai_runtime nor the QAT package is installed"
        )
    detail = None
    if runtime_installed and qat_installed:
        detail = "runtime and QAT packages installed"
    elif qat_installed:
        # Calibration alone is still useful on an XDNA part, so a QAT-only install counts.
        detail = "vitis_ai_quantization_pytorch installed without vitis_ai_runtime"
    return _entry("vitis_ai", True, runtime_version or qat_version, detail)


def _probe_fastflowlm() -> dict[str, Any]:
    binary = shutil.which("flm")
    return _entry("fastflowlm", bool(binary), None, binary)


def _probe_lemonade() -> dict[str, Any]:
    binary = shutil.which("lemonade")
    return _entry("lemonade", bool(binary), None, binary)


def _probe_windows_ml() -> dict[str, Any]:
    if sys.platform != "win32":
        return _entry("windows_ml", False, None, "Windows ML only exists on Windows")
    for package in _WINDOWS_ML_PACKAGES:
        installed, version = _distribution(package)
        if installed:
            return _entry("windows_ml", True, version, package)
    if _module_present("winrt.Windows.AI.MachineLearning"):
        return _entry("windows_ml", True, None, "winrt.Windows.AI.MachineLearning")
    return _entry("windows_ml", False, None, "no Windows ML package or winrt projection found")


_PROBES = {
    "openvino": _probe_openvino,
    "onnxruntime_qnn": _probe_onnxruntime_qnn,
    "vitis_ai": _probe_vitis_ai,
    "fastflowlm": _probe_fastflowlm,
    "lemonade": _probe_lemonade,
    "windows_ml": _probe_windows_ml,
}


# Rows are cached for a minute: long enough that a page load or a status poll costs nothing,
# short enough that a runtime installed under the running server is noticed without a restart.
_CACHE_TTL_SECONDS = 60.0

# One lock, held across the probe, so a burst of concurrent requests pays for one probe rather
# than one each; the probes are local metadata reads, so the hold is measured in milliseconds.
_cache_lock = threading.Lock()
# (fingerprint, expires_at, rows) of the last complete probe, or None when nothing is cached.
_cache: Optional[tuple[tuple[Any, ...], float, tuple[dict[str, Any], ...]]] = None


def _environment_fingerprint() -> tuple[Any, ...]:
    """A cheap description of everything the probes read that can change between two calls.

    The probe and helper functions go into the stamp as objects rather than as addresses, which
    also keeps them alive: swapping one out -- a test stub, a plugin -- must never be answered
    from a cache filled before the swap. Distribution metadata is the one input left out, since
    reading it is most of what a probe costs; the TTL is what eventually notices an install.
    """
    return (
        sys.platform,
        os.environ.get("PATH"),
        os.environ.get("PATHEXT"),
        os.environ.get("QNN_SDK_ROOT"),
        _distribution,
        _module_present,
        _spec_origin,
        _qnn_htp_libraries,
        shutil.which,
        tuple(_PROBES.get(name) for name in PROVIDERS),
    )


def _probe_providers() -> tuple[list[dict[str, Any]], bool]:
    """Probe every provider; the flag says whether every probe answered without raising."""
    rows: list[dict[str, Any]] = []
    complete = True
    for name in PROVIDERS:
        try:
            rows.append(_PROBES[name]())
        except Exception as exc:  # noqa: BLE001 -- one broken install must not blank the matrix
            complete = False
            rows.append(_entry(name, False, None, f"probe failed: {exc}"))
    return rows, complete


def npu_runtime_providers() -> list[dict[str, Any]]:
    """Every provider row, installed or not. A probe that raises reports just its own row.

    Rows are served from a short-lived cache, but never from a stale one: a caller whose
    environment, stubs or platform differ from the cached probe gets a fresh answer, and rows
    come back as fresh dicts every time so editing them cannot edit the cache. A probe that
    raised is not cached at all, so a transient failure cannot stick for the whole TTL.
    """
    global _cache
    fingerprint = _environment_fingerprint()
    with _cache_lock:
        cached = _cache
        if cached is not None:
            cached_fingerprint, expires_at, cached_rows = cached
            if cached_fingerprint == fingerprint and time.monotonic() < expires_at:
                return [dict(row) for row in cached_rows]
        rows, complete = _probe_providers()
        if complete:
            _cache = (fingerprint, time.monotonic() + _CACHE_TTL_SECONDS, tuple(rows))
        else:
            _cache = None
    return [dict(row) for row in rows]


def invalidate() -> None:
    """Forget the cached rows, so the next call probes the host again."""
    global _cache
    with _cache_lock:
        _cache = None


def provider_for_vendor(
    vendor: str, providers: Optional[list[dict[str, Any]]] = None
) -> Optional[dict[str, Any]]:
    """The first installed provider that lists this vendor, or None when nothing can drive it."""
    rows = providers if providers is not None else npu_runtime_providers()
    for provider in rows:
        if provider.get("installed") and vendor in provider.get("vendors", ()):
            return provider
    return None


def providers_supporting(
    task: str, providers: Optional[list[dict[str, Any]]] = None
) -> list[dict[str, Any]]:
    """Installed providers that can do this task, in matrix order."""
    rows = providers if providers is not None else npu_runtime_providers()
    return [
        provider
        for provider in rows
        if provider.get("installed") and task in provider.get("supports", ())
    ]
