# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`gpu_runtime`: the sentence that explains an Intel GPU row the registry cannot explain.

`core.training.device_targets` marks a GPU it cannot see training with one token, `torch_missing`, and
the only sentence that token has is "PyTorch is not installed in this environment." On the common
hybrid host that is wrong -- PyTorch is installed, it simply is not an XPU build -- and the UI has no
way to say better, because the registry deliberately never imports torch. `utils.hardware.xpu` does
answer the question, so the route now carries its answer in `gpu_runtime`.

The registry runs for real here, so the row under test is the row the page actually renders, and the
route's own cache and invalidator are exercised as written. What is stubbed is the host: the probe is
hand-built and the runtime read is replaced, because the interesting answers are exactly the ones this
machine cannot give on demand -- a CUDA wheel on an Intel host, a driver that raises.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth.authentication import get_current_subject  # noqa: E402
from core.training.device_targets import TargetProbe  # noqa: E402

GB = 1024**3

# The five keys `utils.hardware.xpu.xpu_torch_status` publishes. The route's contract is that
# `gpu_runtime` is either None or a dict carrying all five, so a UI can render `note` unconditionally.
GPU_RUNTIME_KEYS = {"available", "build", "device_count", "driver", "note"}

# Pinned as text because it is user-facing: this is the sentence that replaces the wrong one, and it
# matches `tests/test_xpu.py`'s NOTE_CUDA for the same build.
NOTE_CUDA_BUILD = (
    "PyTorch here is a CUDA build, so this Intel GPU cannot train "
    "until the XPU build is installed."
)
# What the registry's own `torch_missing` token says today, and what a caller must never see from a
# runtime read that failed: the read failing says nothing about whether torch is installed.
NOTE_TORCH_MISSING = "PyTorch is not installed in this environment."
# The route's own fallback, for a runtime read that failed. It names no build and does not say torch
# is missing, because the read failing is the only fact it has.
NOTE_RUNTIME_UNREADABLE = (
    "The Intel GPU runtime could not be read here, so whether PyTorch can train "
    "on this GPU is unknown."
)

CUDA_RUNTIME = {
    "available": False,
    "build": "cuda",
    "device_count": 0,
    "driver": None,
    "note": NOTE_CUDA_BUILD,
}

NPU_CAPABILITY = {
    "mode": "unavailable",
    "backward_pass": False,
    "device": None,
    "tasks": [],
    "providers": [],
    "reasons": ["absent"],
    "notes": "No NPU was detected on this machine.",
}


def _gpu(**overrides) -> dict:
    base = dict(
        index = 0,
        name = "Intel(R) Arc(TM) Graphics",
        vendor = "intel",
        integrated = True,
        memory_bytes = 512 * 1024**2,
        arch = None,
        unified = True,
    )
    base.update(overrides)
    return base


def _probe(**overrides) -> TargetProbe:
    """An Intel iGPU host whose PyTorch cannot be read: the registry's `torch_missing` row exactly.

    `torch_available` False is what the registry reads, and a CUDA wheel on an Intel host reports no
    device to `torch.xpu`, which is the same fact from the other side.
    """
    base = dict(
        cpu_name = "Intel(R) Core(TM) Ultra 7 155H",
        cpu_cores = 16,
        system_memory_bytes = 64 * GB,
        system_memory_available_bytes = 40 * GB,
        torch_available = False,
        torch_device = None,
        gpus = [_gpu()],
        npu = None,
        bnb_available = False,
        triton_available = False,
        flash_attn_available = False,
    )
    base.update(overrides)
    return TargetProbe(**base)


@pytest.fixture()
def host(monkeypatch):
    """The route, with a switchable probe and the session-only reads stubbed.

    The probe cache is process-wide, so it is emptied before and after every test here: a cached probe
    from another file would serve this one a different machine, and a probe left behind would do the
    same to them.
    """
    from routes import compute
    from utils import compute_target_settings

    state = SimpleNamespace(probe = _probe())

    monkeypatch.setattr(compute, "_probe", lambda: state.probe)
    monkeypatch.setattr(compute, "_npu_capability", lambda: dict(NPU_CAPABILITY))
    monkeypatch.setattr(compute_target_settings, "get_selected_compute_target", lambda: "auto")
    compute._invalidate_probe_cache()
    yield state
    compute._invalidate_probe_cache()


@pytest.fixture()
def client(host):
    from routes import compute

    app = FastAPI()
    app.include_router(compute.router, prefix = "/api/compute")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    return TestClient(app)


def _counting_runtime(monkeypatch) -> list:
    """Replace the route's runtime read with one that records every call, answering as a CUDA host."""
    from routes import compute

    calls: list = []

    def read():
        calls.append(1)
        return dict(CUDA_RUNTIME)

    monkeypatch.setattr(compute, "_xpu_torch_status", read)
    return calls


def _replace_probe_module_status(monkeypatch, behaviour) -> None:
    """Replace `utils.hardware.xpu.xpu_torch_status` itself, leaving the route's reader alone.

    The route imports it inside the call, so the patch is what the route's own guard sees: this is how
    a probe that raises or answers with half a dict reaches the payload.
    """
    from utils.hardware import xpu

    monkeypatch.setattr(xpu, "xpu_torch_status", behaviour)


def test_the_intel_gpu_row_that_cannot_train_carries_the_real_reason(client, monkeypatch):
    calls = _counting_runtime(monkeypatch)

    body = client.get("/api/compute/targets").json()

    row = [target for target in body["targets"] if target["kind"] == "igpu"][0]
    # The premise, from the real registry: this is a row that cannot train and says only so.
    assert row["supported"] is False
    assert row["reasons"] == ["torch_missing"]
    assert set(body["gpu_runtime"]) == GPU_RUNTIME_KEYS
    assert body["gpu_runtime"]["build"] == "cuda"
    # The sentence the page shows instead of the registry's hint, and it is not the wrong one.
    assert body["gpu_runtime"]["note"] == NOTE_CUDA_BUILD
    assert body["gpu_runtime"]["note"] != NOTE_TORCH_MISSING
    assert len(calls) == 1


def test_the_added_key_leaves_every_existing_one_alone(client, monkeypatch):
    _counting_runtime(monkeypatch)

    body = client.get("/api/compute/targets").json()

    # Same keys, same shapes: `npu` and `budget` are not reshaped to make room for the runtime.
    assert body["npu"] == NPU_CAPABILITY
    assert set(body["budget"]) == {
        "integrated",
        "dedicated_bytes",
        "system_memory_bytes",
        "shared_fraction",
        "usable_bytes",
        "notes",
    }
    assert body["budget"]["integrated"] is False
    assert body["budget"]["dedicated_bytes"] == 40 * GB
    assert body["resolved"] == "cpu"
    assert body["default"] == "cpu"
    assert body["requested"] == "auto"
    assert body["targets"]


def test_a_runtime_probe_that_raises_degrades_instead_of_failing_the_get(client, monkeypatch):
    def boom():
        raise RuntimeError("oneAPI level-zero hung")

    _replace_probe_module_status(monkeypatch, boom)

    response = client.get("/api/compute/targets")

    assert response.status_code == 200
    body = response.json()
    # The unknown shape, not an absent key and not a claim about what is installed.
    assert body["gpu_runtime"] == {
        "available": False,
        "build": "unknown",
        "device_count": 0,
        "driver": None,
        "note": NOTE_RUNTIME_UNREADABLE,
    }
    assert body["gpu_runtime"]["note"] != NOTE_TORCH_MISSING
    # The rest of the page is untouched by the failure it absorbed.
    assert body["npu"] == NPU_CAPABILITY
    assert [target["kind"] for target in body["targets"]] == ["cpu", "igpu"]


def test_a_runtime_answer_missing_keys_still_publishes_all_five(client, monkeypatch):
    _replace_probe_module_status(monkeypatch, lambda: {"build": "xpu", "available": True})

    body = client.get("/api/compute/targets").json()

    assert set(body["gpu_runtime"]) == GPU_RUNTIME_KEYS
    assert body["gpu_runtime"]["build"] == "xpu"
    assert body["gpu_runtime"]["available"] is True
    # Unanswered fields are unmeasured, never invented: zero devices and no driver, with the route's
    # own note rather than a build-specific one.
    assert body["gpu_runtime"]["device_count"] == 0
    assert body["gpu_runtime"]["driver"] is None
    assert body["gpu_runtime"]["note"] == NOTE_RUNTIME_UNREADABLE


def test_a_host_with_no_intel_gpu_gets_none_and_pays_for_no_probe(client, host, monkeypatch):
    host.probe = _probe(
        cpu_name = "AMD Ryzen 9 7950X",
        torch_available = True,
        torch_device = "cuda",
        gpus = [
            _gpu(
                name = "NVIDIA GeForce RTX 4090",
                vendor = "nvidia",
                integrated = False,
                memory_bytes = 24 * GB,
                arch = "sm_89",
                unified = False,
            )
        ],
    )
    calls = _counting_runtime(monkeypatch)

    body = client.get("/api/compute/targets").json()

    # None, not the unknown shape: "there is no Intel GPU to explain" is different information from
    # "this Intel GPU's status could not be read", and the UI must be able to tell them apart.
    assert body["gpu_runtime"] is None
    assert [target["kind"] for target in body["targets"]] == ["cpu", "dgpu"]
    assert len(calls) == 0


def test_the_runtime_status_is_read_once_per_probe_cache_window(client, monkeypatch):
    calls = _counting_runtime(monkeypatch)

    assert client.get("/api/compute/targets").status_code == 200
    assert client.get("/api/compute/targets").status_code == 200

    # The Train page polls, and this is a torch read: one per cache window, like the probe beside it.
    assert len(calls) == 1

    # The route's own invalidator is what the POST /target handler calls, so emptying the cache here
    # is the deliberate-change path: the next poll must re-read rather than reuse the old answer.
    from routes import compute

    compute._invalidate_probe_cache()

    assert client.get("/api/compute/targets").status_code == 200
    assert len(calls) == 2
