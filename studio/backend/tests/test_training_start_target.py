# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a training start request decides to train, pinned without a machine or a model.

`start_training` is the one place a user's "train here" can be lost: the request is copied into the
worker's config, the worker's first act is to write that target into the environment torch reads at
import time, and nothing after that point can notice the choice went missing. So the three decisions
are held here: the request wins over the remembered choice, an unusable device is refused before a
job id exists rather than mid-step, and the three config keys that carry the decision are the same
three the environment writer reads.

The registry is stubbed rather than probed, because a real probe would make these tests answer
"which machine am I on". No torch, no network, no server: the target objects are the real dataclass,
the config keys are checked against the real consumer.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# The three keys the start path writes into the worker's config; the environment writer reads
# exactly these, and a rename on either side is otherwise silent.
_TARGET_CONFIG_KEYS = (
    "compute_target",
    "compute_target_vendor",
    "compute_target_integrated",
)


def _target(kind: str, *, name = None, vendor = "other"):
    """A real registry target, so the code under test sees the type it sees in production."""
    from core.training.device_targets import TrainingTarget

    return TrainingTarget(
        id = kind,
        device = "cpu" if kind == "cpu" else f"{kind}:0",
        kind = kind,
        name = name or f"{kind} device",
        vendor = vendor,
        memory_bytes = 8 * 1024**3,
        unified_memory = kind in ("cpu", "igpu"),
        supported = True,
        capabilities = {},
        recommended = {},
        notes = (),
    )


def _stub_registry(monkeypatch, resolver):
    """Replace the registry's resolver with `resolver`, returning the list of what was asked for.

    The route imports the resolver inside the function, so the replacement belongs on the registry
    module and is picked up on the next call.
    """
    from core.training import device_targets

    asked: list = []

    def fake_resolve(requested = None, probe = None):
        asked.append(requested)
        return resolver(requested)

    monkeypatch.setattr(device_targets, "resolve_training_target", fake_resolve)
    return asked


def _from_map(by_request: dict, requested):
    """A stub registry: answer what the map holds, refuse the rest the way the registry does."""
    from core.training.device_targets import TrainingTargetError

    resolved = by_request.get((requested or "auto").strip().lower())
    if resolved is None:
        raise TrainingTargetError(f"'{requested}' is not a training target.")
    return resolved


def _stub_stored_target(monkeypatch, value: str):
    """The Train page's remembered target, without a settings row behind it."""
    from utils import compute_target_settings

    monkeypatch.setattr(compute_target_settings, "get_selected_compute_target", lambda: value)


def _resolved(compute_target):
    """Resolve a target for a request that carries nothing but the field under test."""
    from routes.training import _resolved_compute_target

    return _resolved_compute_target(SimpleNamespace(compute_target = compute_target))


# --- which device a request resolves to ------------------------------------------------------


def test_a_requested_cpu_is_the_device_the_run_trains_on(monkeypatch):
    # The remembered choice says otherwise; the request is what the user just clicked.
    _stub_stored_target(monkeypatch, "dgpu")
    by_request = {"cpu": _target("cpu")}
    asked = _stub_registry(monkeypatch, lambda requested: _from_map(by_request, requested))

    assert _resolved("cpu").kind == "cpu"
    assert asked == ["cpu"], "the request must be resolved, not the stored setting"


def test_an_absent_target_falls_back_to_the_remembered_choice(monkeypatch):
    _stub_stored_target(monkeypatch, "cpu")
    by_request = {"auto": _target("dgpu"), "cpu": _target("cpu")}
    asked = _stub_registry(monkeypatch, lambda requested: _from_map(by_request, requested))

    assert _resolved(None).kind == "cpu"
    # A stored "cpu" is handed to the registry as itself, so it wins over the auto-detected card.
    assert asked == ["cpu"]


def test_a_blank_target_is_treated_as_absent(monkeypatch):
    _stub_stored_target(monkeypatch, "cpu")
    by_request = {"cpu": _target("cpu")}
    asked = _stub_registry(monkeypatch, lambda requested: _from_map(by_request, requested))

    assert _resolved("   ").kind == "cpu"
    assert asked == ["cpu"]


def test_an_unusable_target_is_refused_with_the_registry_reason(monkeypatch):
    from core.training.device_targets import TrainingTargetError

    message = "This machine has no npu device for training. Available: cpu."
    _stub_stored_target(monkeypatch, "auto")

    def refuse(requested = None, probe = None):
        raise TrainingTargetError(message)

    from core.training import device_targets

    monkeypatch.setattr(device_targets, "resolve_training_target", refuse)

    with pytest.raises(HTTPException) as raised:
        _resolved("npu")
    assert raised.value.status_code == 400
    # The registry's own sentence reaches the user: "not supported" alone says nothing actionable.
    assert raised.value.detail == message


# --- what the resolved device changes about the request --------------------------------------


def _request(load_in_4bit = True):
    return SimpleNamespace(load_in_4bit = load_in_4bit)


def test_an_integrated_gpu_drops_the_4bit_load():
    from routes.training import _apply_target_constraints

    request = _request()
    _apply_target_constraints(request, _target("igpu", name = "Intel(R) Arc(TM) Graphics"))

    # bitsandbytes has no iGPU kernels: keeping the flag would dequantize on the CPU every step.
    assert request.load_in_4bit is False


def test_a_discrete_gpu_keeps_the_4bit_load():
    from routes.training import _apply_target_constraints

    request = _request()
    _apply_target_constraints(request, _target("dgpu", vendor = "nvidia"))

    assert request.load_in_4bit is True


def test_no_resolved_target_leaves_the_request_untouched():
    from routes.training import _apply_target_constraints

    request = _request()
    # The auto path resolves to nothing here; there is no device to justify a change, and guessing
    # one is how a 4-bit run gets silently downgraded on a machine that never asked.
    _apply_target_constraints(request, None)

    assert request.load_in_4bit is True


# --- the config keys that carry the decision into the worker ---------------------------------


def test_the_start_path_config_keys_are_the_ones_the_worker_reads():
    from core.training.compute_target_env import apply_compute_target_environment
    from routes.training import _compute_target_config

    config = _compute_target_config(_target("cpu", vendor = "intel"))

    assert tuple(config) == _TARGET_CONFIG_KEYS
    # A CPU run must arrive as a pin: the worker sets UNSLOTH_TRAINING_DEVICE before torch loads.
    applied = apply_compute_target_environment(config, environ = {})
    assert applied["UNSLOTH_TRAINING_DEVICE"] == "cpu"


def test_an_integrated_target_carries_its_vendor_and_the_shared_memory_flag():
    from core.training.compute_target_env import (
        apply_compute_target_environment,
        describe_compute_target,
    )
    from routes.training import _compute_target_config

    config = _compute_target_config(_target("igpu", vendor = "intel"))

    assert config["compute_target"] == "igpu"
    assert config["compute_target_vendor"] == "intel"
    assert config["compute_target_integrated"] is True
    # An iGPU is deliberately not device-pinned (the Studio sets the visibility mask instead); what
    # it needs from this config is the allocator defaults, which only reach it via these keys.
    applied = apply_compute_target_environment(config, environ = {})
    assert "UNSLOTH_TRAINING_DEVICE" not in applied
    assert applied["UNSLOTH_IGPU"] == "1"
    assert "integrated GPU" in describe_compute_target(config)


def test_an_auto_run_writes_auto_and_pins_nothing():
    from core.training.compute_target_env import (
        apply_compute_target_environment,
        describe_compute_target,
    )
    from routes.training import _compute_target_config

    config = _compute_target_config(None)

    assert config["compute_target"] == "auto"
    assert config["compute_target_vendor"] is None
    assert config["compute_target_integrated"] is False
    # Nothing was chosen, so nothing is forced: detection on the host decides.
    assert apply_compute_target_environment(config, environ = {}) == {}
    assert "auto" in describe_compute_target(config)


def test_the_start_route_still_uses_both_helpers():
    from routes import training

    source = inspect.getsource(training.start_training)

    # Extracting the decisions is only worth anything if the start path still consults them.
    assert "_apply_target_constraints(request, compute_target)" in source
    assert "_compute_target_config(compute_target)" in source
