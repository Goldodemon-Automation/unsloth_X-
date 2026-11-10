# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Intel XPU status, the Level-Zero selector and the pin, driven by stand-in torch modules.

Every branch runs against a fake ``torch`` in ``sys.modules`` and no real torch is imported
anywhere in this file, because the interesting answers are exactly the ones a machine without the
right wheel gives -- a CUDA build on an Intel host, a ROCm wheel that mislabels itself, a CPU
build, no PyTorch at all -- and each of those must answer instead of raising, with a note a UI
shows verbatim. The exact note strings are pinned here for the same reason: they are user-facing
text, not formatting.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.hardware import xpu  # noqa: E402

NOTE_XPU = "PyTorch's XPU build is installed, so the integrated GPU can train."
NOTE_CUDA = (
    "PyTorch here is a CUDA build, so this Intel GPU cannot train "
    "until the XPU build is installed."
)
NOTE_ROCM = (
    "PyTorch here is a ROCm build, so this Intel GPU cannot train "
    "until the XPU build is installed."
)
NOTE_CPU = (
    "PyTorch here is a CPU build, so this Intel GPU cannot train "
    "until the XPU build is installed."
)
NOTE_NO_TORCH = "PyTorch is not installed in this environment."


def _fake_torch(
    label = "2.6.0",
    *,
    cuda = None,
    hip = None,
    xpu_build = None,
    available = False,
    count = 0,
    driver = None,
):
    """A stand-in torch carrying only what xpu_torch_status reads: torch.__version__, torch.version
    and the two torch.xpu probes (plus the optional driver_version). No real torch is involved."""
    torch = types.ModuleType("torch")
    torch.__version__ = label
    torch.version = types.SimpleNamespace(cuda = cuda, hip = hip, xpu = xpu_build)
    device = types.SimpleNamespace(is_available = lambda: available, device_count = lambda: count)
    if driver is not None:
        device.driver_version = lambda: driver
    torch.xpu = device
    return torch


def _all_branches():
    """Yield (name, stand-in) for every branch xpu_torch_status has; None stands for no torch."""
    yield "no torch", None
    yield "xpu build", _fake_torch(xpu_build = "20250101", available = True, count = 1)
    yield "xpu build, runtime down", _fake_torch(xpu_build = "20250101")
    yield "xpu label, untagged markers", _fake_torch("2.9.1+xpu")
    yield "cuda build", _fake_torch("2.6.0+cu124", cuda = "12.4")
    yield "rocm build", _fake_torch("2.6.0+rocm6.3", hip = "6.3.40091")
    yield "cpu build", _fake_torch("2.6.0+cpu")
    yield "untagged cpu wheel", _fake_torch("2.11.0")


def test_a_machine_with_no_torch_answers_the_documented_shape(monkeypatch):
    # None in sys.modules is what an import of a missing package looks like from inside a
    # function; the status must answer, not raise, and "unknown" must not pretend to name a build.
    monkeypatch.setitem(sys.modules, "torch", None)

    assert xpu.xpu_torch_status() == {
        "available": False,
        "build": "unknown",
        "device_count": 0,
        "driver": None,
        "note": NOTE_NO_TORCH,
    }


def test_an_xpu_build_with_a_live_runtime_is_available(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(xpu_build = "20250101", available = True, count = 1),
    )

    status = xpu.xpu_torch_status()

    assert status == {
        "available": True,
        "build": "xpu",
        "device_count": 1,
        "driver": None,
        "note": NOTE_XPU,
    }


def test_an_untagged_xpu_label_still_reads_as_an_xpu_build(monkeypatch):
    # A source or vendor wheel carries no local tag but "2.9.1+xpu" in __version__; the label is
    # evidence too (hardware._torch_reports_an_xpu_runtime reads it the same way).
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("2.9.1+xpu"))

    status = xpu.xpu_torch_status()

    assert status["build"] == "xpu"
    assert status["available"] is False


def test_a_down_xpu_runtime_is_still_an_xpu_build(monkeypatch):
    # The build question and the runtime question are different ones: a wedged Level-Zero stack
    # makes the GPU unavailable but does not turn the wheel into a CPU wheel.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(xpu_build = "20250101"))

    status = xpu.xpu_torch_status()

    assert status["build"] == "xpu"
    assert status["available"] is False
    assert status["device_count"] == 0


def test_a_cuda_build_says_it_cannot_train_the_intel_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("2.6.0+cu124", cuda = "12.4"))

    status = xpu.xpu_torch_status()

    assert status["build"] == "cuda"
    assert status["note"] == NOTE_CUDA


def test_a_rocm_build_is_not_labeled_a_cuda_build(monkeypatch):
    # hip settles it even though a ROCm build records cuda too: a note that calls a ROCm wheel
    # "a CUDA build" would send its owner to the wrong page.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch("2.6.0+cu118", hip = "6.3.40091", cuda = "11.8"),
    )

    status = xpu.xpu_torch_status()

    assert status["build"] == "rocm"
    assert status["note"] == NOTE_ROCM


def test_an_amd_sdk_wheel_with_no_hip_marker_still_reads_as_rocm(monkeypatch):
    # The AMD SDK / Radeon wheels leave torch.version.hip unset and carry "+rocm" in __version__
    # (hardware._torch_reports_a_hip_runtime reads it the same way).
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("2.6.0+rocm6.3"))

    assert xpu.xpu_torch_status()["build"] == "rocm"


def test_a_cpu_wheel_is_reported_as_a_cpu_build(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("2.6.0+cpu"))

    status = xpu.xpu_torch_status()

    assert status["build"] == "cpu"
    assert status["note"] == NOTE_CPU


def test_an_untagged_wheel_with_no_runtime_marker_is_a_cpu_build(monkeypatch):
    # The PyPI CPU wheel shape: no local tag and none of version.{cuda,hip,xpu} set.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("2.11.0"))

    assert xpu.xpu_torch_status()["build"] == "cpu"


def test_a_torch_without_an_xpu_namespace_still_answers(monkeypatch):
    # A CPU-only build has no torch.xpu at all; the status must read that as "no device", not
    # as an attribute error.
    torch = types.ModuleType("torch")
    torch.__version__ = "2.6.0+cpu"
    torch.version = types.SimpleNamespace(cuda = None, hip = None, xpu = None)
    monkeypatch.setitem(sys.modules, "torch", torch)

    status = xpu.xpu_torch_status()

    assert status["available"] is False
    assert status["device_count"] == 0
    assert status["build"] == "cpu"


def test_the_driver_is_read_when_the_stack_reports_one(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(xpu_build = "20250101", available = True, count = 1, driver = "32.0.101.5768"),
    )

    assert xpu.xpu_torch_status()["driver"] == "32.0.101.5768"
    # And None -- never an exception -- when the stack exposes nothing to read.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(xpu_build = "20250101", available = True))
    assert xpu.xpu_torch_status()["driver"] is None


def test_the_status_carries_the_five_documented_keys(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(xpu_build = "20250101", available = True))

    assert set(xpu.xpu_torch_status()) == {"available", "build", "device_count", "driver", "note"}


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, "level_zero:0"), (1, "level_zero:1"), (12, "level_zero:12")],
)
def test_the_selector_names_the_level_zero_namespace(index, expected):
    # The backend namespace is the point: a bare number is ambiguous the moment more than one
    # Level-Zero platform is visible.
    assert xpu.xpu_device_selector(index) == expected


def test_apply_xpu_selection_sets_the_pin_and_reports_only_it():
    environ: dict = {}

    applied = xpu.apply_xpu_selection(1, environ)

    assert applied == {"ONEAPI_DEVICE_SELECTOR": "level_zero:1"}
    assert environ == {"ONEAPI_DEVICE_SELECTOR": "level_zero:1"}
    # Already set -- by us or anyone -- means it set nothing and reports nothing.
    assert xpu.apply_xpu_selection(2, environ) == {}
    assert environ["ONEAPI_DEVICE_SELECTOR"] == "level_zero:1"


def test_a_user_supplied_selector_wins():
    # A user who typed a selector has a reason; the pin only ever fills the gap.
    environ = {"ONEAPI_DEVICE_SELECTOR": "level_zero:7"}

    assert xpu.apply_xpu_selection(0, environ) == {}
    assert environ == {"ONEAPI_DEVICE_SELECTOR": "level_zero:7"}


def test_an_empty_selector_counts_as_unset():
    environ = {"ONEAPI_DEVICE_SELECTOR": ""}

    applied = xpu.apply_xpu_selection(0, environ)

    assert applied == {"ONEAPI_DEVICE_SELECTOR": "level_zero:0"}
    assert environ["ONEAPI_DEVICE_SELECTOR"] == "level_zero:0"


def test_apply_xpu_selection_defaults_to_the_process_environment(monkeypatch):
    monkeypatch.delenv("ONEAPI_DEVICE_SELECTOR", raising = False)

    applied = xpu.apply_xpu_selection(0)

    assert applied == {"ONEAPI_DEVICE_SELECTOR": "level_zero:0"}
    assert xpu.apply_xpu_selection(0) == {}


def test_every_note_is_one_plain_sentence(monkeypatch):
    # The notes are shown verbatim by a UI: a sentence, with real words in it and no identifier
    # spelling leaking through.
    for name, fake in _all_branches():
        monkeypatch.setitem(sys.modules, "torch", fake)
        note = xpu.xpu_torch_status()["note"]
        assert isinstance(note, str) and note, name
        assert note.endswith("."), name
        assert " " in note, name
        assert "_" not in note, name
