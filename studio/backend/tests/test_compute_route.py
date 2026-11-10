# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The compute-target route: what the Train page reads, and what it can change.

The route is thin on purpose -- the registry owns the rules -- so what is pinned here is the part
only the route can get wrong. A remembered target that this machine can no longer serve must not
strand the page; an unknown or unsupported target must be refused before anything is stored; and a
refusal must leave the stored choice intact, so the user can still see what they asked for.

The registry is stubbed rather than probed: these are route tests, and a real probe would make them
depend on the machine they run on.
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


def _target(kind: str, *, supported = True, reasons = (), memory = None, vendor = "other"):
    """A real registry target, so the route's own `asdict` path is exercised, not stubbed past."""
    from core.training.device_targets import TrainingTarget
    return TrainingTarget(
        id = kind,
        device = "cpu" if kind == "cpu" else f"{kind}:0",
        kind = kind,
        name = f"{kind} device",
        vendor = vendor,
        memory_bytes = memory,
        unified_memory = kind in ("cpu", "igpu"),
        supported = supported,
        reasons = tuple(reasons),
        capabilities = {
            "bf16": False,
            "fp16": False,
            "bnb4bit": False,
            "triton": False,
            "grad_checkpoint": True,
            "compile": False,
            "flash_attn": False,
        },
        recommended = {
            "batch_size": 1,
            "grad_accum": 8,
            "dtype": "float32",
            "gradient_checkpointing": True,
        },
        notes = ("A sentence.",),
    )


@pytest.fixture()
def registry(monkeypatch):
    """The real registry module, with its two decisions replaced by a switchable device list."""
    from core.training import device_targets

    state = SimpleNamespace(
        targets = [_target("cpu", memory = 64 * 1024**3)],
        probe = SimpleNamespace(system_memory_bytes = 64 * 1024**3, gpus = []),
    )

    def resolve(requested = None, probe = None):
        wanted = (requested or "auto").strip().lower()
        if wanted in ("", "auto"):
            return state.targets[0]
        for target in state.targets:
            if wanted in (target.id, target.device):
                if not target.supported:
                    raise device_targets.TrainingTargetError(
                        f"{target.id} cannot train here: {', '.join(target.reasons)}"
                    )
                return target
        raise device_targets.TrainingTargetError(f"'{requested}' is not a training target.")

    def payload(probe = None):
        return {
            "targets": [
                {
                    "id": target.id,
                    "kind": target.kind,
                    "device": target.device,
                    "name": target.name,
                    "supported": target.supported,
                    "reasons": list(target.reasons),
                }
                for target in state.targets
            ],
            "default": state.targets[0].id,
            "resolved": state.targets[0].id,
        }

    # The route imports these names directly, so the replacement belongs on the route module.
    from routes import compute

    monkeypatch.setattr(compute, "resolve_training_target", resolve)
    monkeypatch.setattr(compute, "training_target_payload", payload)
    monkeypatch.setattr(compute, "TargetProbe", SimpleNamespace(detect = lambda: state.probe))
    return state


@pytest.fixture()
def stored(monkeypatch):
    """The settings module's two functions, backed by a dict."""
    from utils import compute_target_settings

    values = {"target": "auto"}
    monkeypatch.setattr(
        compute_target_settings, "get_selected_compute_target", lambda: values["target"]
    )

    def set_target(value):
        normalized = compute_target_settings.normalize_compute_target(value)
        if normalized is None:
            raise ValueError(f"{value!r} is not a compute target.")
        values["target"] = normalized
        return normalized

    monkeypatch.setattr(compute_target_settings, "set_selected_compute_target", set_target)
    return values


@pytest.fixture()
def client(registry, stored, monkeypatch):
    from routes import compute

    monkeypatch.setattr(
        compute,
        "_npu_capability",
        lambda: {
            "mode": "unavailable",
            "backward_pass": False,
            "device": None,
            "tasks": [],
            "providers": [],
            "reasons": ["absent"],
            "notes": "No NPU was detected on this machine.",
        },
    )
    app = FastAPI()
    app.include_router(compute.router, prefix = "/api/compute")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    return TestClient(app)


def test_the_targets_read_reports_the_default_and_the_budget(client):
    response = client.get("/api/compute/targets")

    assert response.status_code == 200
    body = response.json()
    assert body["resolved"] == "cpu"
    assert body["requested"] == "auto"
    # A CPU run's ceiling is system RAM, and the sentence says where it came from.
    assert body["budget"]["integrated"] is False
    assert body["budget"]["usable_bytes"] == 64 * 1024**3
    assert body["npu"]["mode"] == "unavailable"


def test_a_remembered_target_that_is_gone_falls_back_instead_of_failing(client, stored, registry):
    stored["target"] = "dgpu"
    registry.targets = [
        _target("cpu", memory = 64 * 1024**3),
        _target("dgpu", supported = False, reasons = ("torch_missing",)),
    ]

    response = client.get("/api/compute/targets")

    assert response.status_code == 200
    body = response.json()
    # The page still renders, reports what it resolved to, and keeps what was asked for visible.
    assert body["resolved"] == "cpu"
    assert body["requested"] == "dgpu"
    assert "torch_missing" in body["fallback_reason"]


def test_choosing_a_target_stores_it_and_answers_what_it_resolves_to(client, stored):
    response = client.post("/api/compute/target", json = {"target": "cpu"})

    assert response.status_code == 200
    assert stored["target"] == "cpu"
    assert response.json()["resolved"] == "cpu"
    assert response.json()["target"]["kind"] == "cpu"


def test_choosing_an_unsupported_target_is_refused_and_changes_nothing(client, stored, registry):
    registry.targets = [
        _target("cpu", memory = 64 * 1024**3),
        _target("npu", supported = False, reasons = ("no_backward_pass",)),
    ]

    response = client.post("/api/compute/target", json = {"target": "npu"})

    assert response.status_code == 400
    assert "no_backward_pass" in response.json()["detail"]
    # The stored choice survives a refusal: the user must still see what they asked for.
    assert stored["target"] == "auto"


def test_choosing_an_unknown_target_is_refused(client, stored):
    response = client.post("/api/compute/target", json = {"target": "tpu"})

    assert response.status_code == 400
    assert "tpu" in response.json()["detail"]
    assert stored["target"] == "auto"


def test_an_empty_target_is_rejected_by_the_schema(client):
    assert client.post("/api/compute/target", json = {"target": ""}).status_code == 422


def test_the_npu_assist_preview_answers_even_without_a_usable_npu(client):
    response = client.get("/api/compute/npu/assist", params = {"requested": "assist"})

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    # A refusal names the reason, which is the whole point of exposing the mode.
    assert body["refused_reason"] == "absent"
    assert body["capability"]["mode"] == "unavailable"


def test_an_auto_npu_request_that_cannot_be_helped_explains_itself(client):
    body = client.get("/api/compute/npu/assist").json()

    # Auto asked for nothing, so it is not a refusal -- but the notes still say why nothing happened.
    assert body["enabled"] is False
    assert body["refused_reason"] is None
    assert body["notes"]


def test_the_npu_assist_preview_refuses_a_full_request_with_the_reason(client):
    response = client.get("/api/compute/npu/assist", params = {"requested": "full"})

    body = response.json()
    assert body["enabled"] is False
    assert body["refused_reason"] == "no_backward_pass"
