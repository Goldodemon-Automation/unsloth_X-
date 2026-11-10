# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether the installed PyTorch can train on an Intel GPU at all, and which card a run lands on.

The device registry (core/training/device_targets.py) lists the integrated GPU and sizes its
budget, but it cannot say *why* an iGPU cannot train: from there, "the wheel is not an XPU
build" and "the run picked the other card" are both just a GPU that is not training. This module
answers the first question -- :func:`xpu_torch_status` is the one place that says, in a sentence a
UI can show verbatim, whether the PyTorch installed here is an XPU build -- and offers the pin
(:func:`xpu_device_selector`, :func:`apply_xpu_selection`) that keeps a hybrid machine's run on the
card the user picked.

torch is imported only inside the functions and every read of it is guarded: a planning-only
install, the registry's own tests and a machine with no PyTorch must all get an answer instead of
an ImportError. The guidance this module implements is written down for users in docs/igpu.md.
"""

from __future__ import annotations

import os
from typing import MutableMapping, Optional

# The oneAPI runtime reads this to decide which devices SYCL will enumerate. It is the namespaced
# sibling of the bare ZE_AFFINITY_MASK masks the Studio's GPU visibility layer already manages.
XPU_DEVICE_SELECTOR_KEY = "ONEAPI_DEVICE_SELECTOR"

# One plain-English sentence each, shown verbatim by a UI, so they are constants rather than
# f-strings: a typo in a user-facing sentence is a bug, and the tests pin the exact text. The
# wrong-build sentences name the build that IS installed -- calling a ROCm wheel "a CUDA build"
# would send its owner to the wrong page -- and end with the same remedy, because the remedy
# really is the same: install the XPU build.
_NOTE_XPU_BUILD = "PyTorch's XPU build is installed, so the integrated GPU can train."
_NOTE_CUDA_BUILD = (
    "PyTorch here is a CUDA build, so this Intel GPU cannot train "
    "until the XPU build is installed."
)
_NOTE_ROCM_BUILD = (
    "PyTorch here is a ROCm build, so this Intel GPU cannot train "
    "until the XPU build is installed."
)
_NOTE_CPU_BUILD = (
    "PyTorch here is a CPU build, so this Intel GPU cannot train "
    "until the XPU build is installed."
)
# The same sentence the registry's torch_missing reason uses (device_targets._reason_hint): the two
# answer the same question and must not disagree in front of the user.
_NOTE_NO_TORCH = "PyTorch is not installed in this environment."

_BUILD_NOTES = {
    "xpu": _NOTE_XPU_BUILD,
    "cuda": _NOTE_CUDA_BUILD,
    "rocm": _NOTE_ROCM_BUILD,
    "cpu": _NOTE_CPU_BUILD,
    "unknown": _NOTE_NO_TORCH,
}


def xpu_torch_status() -> dict:
    """What the installed PyTorch means for training on an Intel GPU, as plain facts.

    Keys: ``available`` (torch.xpu can see a device right now), ``build`` -- which GPU runtime
    the wheel carries, one of ``xpu``/``cuda``/``rocm``/``cpu``, or ``unknown`` when torch cannot
    be read at all -- ``device_count``, ``driver`` (the graphics-driver version when the stack
    reports one, else None) and ``note``: one plain-English sentence for a UI to show verbatim.

    Never raises. A machine with no PyTorch gets its own answer rather than an exception, because
    that answer is the whole point: it is the difference between "the integrated GPU trains" and
    "the integrated GPU is a row that cannot".
    """
    try:
        import torch
    except Exception:
        # Absent and unimportable answer the same here. This reports what a run would find, and
        # both mean the same remedy -- install the wheel the device needs.
        return {
            "available": False,
            "build": "unknown",
            "device_count": 0,
            "driver": None,
            "note": _NOTE_NO_TORCH,
        }
    xpu = getattr(torch, "xpu", None)
    available = _probe_bool(getattr(xpu, "is_available", None))
    build = _build_name(torch, xpu_available = available)
    return {
        "available": available,
        "build": build,
        "device_count": _probe_count(getattr(xpu, "device_count", None)),
        "driver": _driver_string(xpu),
        "note": _BUILD_NOTES[build],
    }


def xpu_device_selector(index: int) -> str:
    """The ONEAPI_DEVICE_SELECTOR value that pins a run to one Intel GPU.

    ``level_zero:<index>`` -- the Level-Zero backend namespace, not a bare number. A bare mask
    names an ordinal with no backend attached, which is ambiguous as soon as more than one
    Level-Zero platform is visible. The ordinal is Level-Zero's own enumeration order: it need
    not follow PCI order, and once the filter is set it re-indexes torch's devices, so the pinned
    card is what the run then reports as ``xpu:0``.
    """
    return f"level_zero:{int(index)}"


def apply_xpu_selection(
    index: int,
    environ: Optional[MutableMapping[str, str]] = None,
) -> dict:
    """Pin one Intel GPU through :data:`XPU_DEVICE_SELECTOR_KEY`; report only what it set.

    A selector the user already set wins -- this only fills in a missing one and never
    overwrites an existing value -- so the return value is exactly what this call wrote and is
    what a job's setup line should log. Like every environment default in this package, the
    value must be in place before the training process imports torch: torch, SYCL and the
    allocator read it while loading and setting it later changes nothing.
    """
    target_environ = os.environ if environ is None else environ
    value = xpu_device_selector(index)
    applied: dict[str, str] = {}
    # An empty value is not a choice somebody made, so it counts as unset and gets filled.
    if not target_environ.get(XPU_DEVICE_SELECTOR_KEY):
        target_environ[XPU_DEVICE_SELECTOR_KEY] = value
        applied[XPU_DEVICE_SELECTOR_KEY] = value
    return applied


def _build_name(torch_module: object, *, xpu_available: bool) -> str:
    """Which GPU runtime the wheel carries: ``xpu``, ``rocm``, ``cuda`` or ``cpu``.

    The evidence is the same hardware.py reads: the ``torch.version.{cuda,hip,xpu}`` markers
    plus the wheel's local version tag. Neither alone is the whole story -- a conda or vendor
    wheel is untagged and records its runtime in torch.version, an AMD SDK/Radeon wheel is
    tagged "+rocm" and leaves version.hip unset -- so both are consulted. hip is read before
    cuda because a ROCm build records both, and a runtime that answers beats every tag: an
    untagged build with a working XPU backend is an XPU build.
    """
    if xpu_available:
        return "xpu"
    version = getattr(torch_module, "version", None)
    label = str(getattr(torch_module, "__version__", "") or "").lower()
    if getattr(version, "hip", None) or "+rocm" in label:
        return "rocm"
    if getattr(version, "xpu", None) or "+xpu" in label:
        return "xpu"
    # "+cu" never matches the CPU tag: "2.6.0+cpu" contains "+cp", not "+cu".
    if getattr(version, "cuda", None) or "+cu" in label:
        return "cuda"
    # Untagged and built against no GPU runtime: the PyPI CPU wheel shape. The CPU local tag
    # ("+cpu", "+cpu.cxx11.abi") lands here too, because no runtime marker is set on it.
    return "cpu"


def _probe_bool(probe: object) -> bool:
    """A guarded torch probe: a wedged driver reports "not available", never an exception."""
    try:
        return bool(probe()) if callable(probe) else False
    except Exception:
        return False


def _probe_count(probe: object) -> int:
    """The device count as reported, clamped to zero: a negative or failing probe is no device."""
    try:
        return max(0, int(probe())) if callable(probe) else 0
    except Exception:
        return 0


def _driver_string(xpu: object) -> Optional[str]:
    """The graphics-driver version when the installed stack reports one, else None.

    Best-effort on purpose: torch exposes no stable driver-version API across XPU builds, so
    None means "not reported" and never "no driver". Only what torch itself answers is read --
    a status line must not spawn a vendor tool to render.
    """
    probe = getattr(xpu, "driver_version", None)
    if not callable(probe):
        return None
    try:
        value = probe()
    except Exception:
        return None
    return str(value) if value else None
