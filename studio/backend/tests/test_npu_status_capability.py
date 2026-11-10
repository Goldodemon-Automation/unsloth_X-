# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The NPU status route carries the training capability, so the panel reads one answer.

``/api/compute/npu/assist`` already answers what training can use the NPU for, and the NPU panel
and the model picker poll ``/api/npu/status`` for the rest of the story. Without the capability in
``/status`` the UI had to make two calls to say one thing, and the two halves of a panel could
disagree. The capability is additive: every key ``/status`` already carried keeps its value.

The capability is stubbed rather than probed. These are route tests, and a real probe would make
them depend on the machine they run on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth.authentication import get_current_subject  # noqa: E402

# The status a backend reports, key for key, so a change to the payload shows up here.
_STATUS = {
    "supported": True,
    "hardware": {"present": True, "supported": True},
    "runtime_installed": True,
    "runtime_running": True,
    "state": "ready",
    "ready": True,
    "error": None,
    "validation": "ok",
    "help_url": None,
    "loaded_model": "lemonade:qwen3-0.6b-FLM",
    "context_length": 8192,
    "loading_model": None,
    "versions": {"lemonade": "11.9.0", "fastflowlm": "1.0.3"},
}

_CAPABILITY = {
    "mode": "assist",
    "backward_pass": False,
    "device": {"present": True, "family": "XDNA2", "name": "AMD Ryzen AI NPU"},
    "tasks": ["tokenize", "eval", "inference"],
    "providers": [{"name": "lemonade", "installed": True}],
    "reasons": [],
    "notes": "The NPU has no backward pass, so it helps with the forward work only.",
}


@pytest.fixture()
def client(monkeypatch):
    """The status route over a faked backend, with the hardware probe left alone."""
    from routes import npu as npu_routes

    class _Backend:
        def status(self) -> dict:
            return dict(_STATUS)

    monkeypatch.setattr(npu_routes, "get_npu_backend", lambda: _Backend())
    app = FastAPI()
    app.include_router(npu_routes.router, prefix = "/api/npu")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    return TestClient(app)


@pytest.fixture()
def capability(monkeypatch):
    """The capability read, replaced by one that says what this test needs it to say."""
    from routes import npu as npu_routes

    def _install(value):
        monkeypatch.setattr(npu_routes, "_training_capability", lambda: dict(value))

    return _install


def test_the_status_carries_the_training_capability(client, capability):
    capability(_CAPABILITY)
    response = client.get("/api/npu/status")

    assert response.status_code == 200
    body = response.json()
    assert body["training_capability"] == _CAPABILITY


def test_every_other_key_keeps_its_value(client, capability):
    """The capability is additive: the payload the panel already reads is untouched."""
    capability(_CAPABILITY)
    response = client.get("/api/npu/status")

    body = response.json()
    assert set(body) == set(_STATUS) | {"training_capability"}
    assert {key: value for key, value in body.items() if key != "training_capability"} == _STATUS


def test_the_managed_account_status_carries_it_too(client, capability, monkeypatch):
    from hub.services.models import account_access
    from routes import npu as npu_routes

    capability(_CAPABILITY)
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    response = client.get("/api/npu/status")

    assert response.status_code == 200
    body = response.json()
    assert body["training_capability"] == _CAPABILITY
    assert set(body) == set(npu_routes._MANAGED_ACCOUNT_STATUS) | {"training_capability"}
    assert {
        key: value for key, value in body.items() if key != "training_capability"
    } == npu_routes._MANAGED_ACCOUNT_STATUS
    # The shared constant stays a constant: one caller's capability must not leak into the next.
    assert "training_capability" not in npu_routes._MANAGED_ACCOUNT_STATUS


def test_a_capability_that_raises_still_serves_the_status(client, monkeypatch):
    from routes import npu as npu_routes

    def _boom():
        raise RuntimeError("the probe fell over")

    monkeypatch.setattr(npu_routes, "_training_capability", _boom)
    response = client.get("/api/npu/status")

    assert response.status_code == 200
    capability = response.json()["training_capability"]
    assert capability["mode"] == "unavailable"
    assert capability["reasons"] == ["not_measured"]


def test_a_probe_that_raises_degrades_to_the_unavailable_stand_in(client, monkeypatch):
    """The read the compute route uses is the one that fails, and it must not fail /status."""
    from utils.hardware import npu as npu_hardware

    def _boom():
        raise RuntimeError("the probe fell over")

    monkeypatch.setattr(npu_hardware, "npu_training_capability", _boom)
    response = client.get("/api/npu/status")

    assert response.status_code == 200
    capability = response.json()["training_capability"]
    assert capability["mode"] == "unavailable"
    assert capability["reasons"] == ["not_measured"]


def test_the_unavailable_stand_in_has_the_shape_of_a_measured_capability():
    """The degraded answer must be readable the same way as a measured one."""
    from routes import npu as npu_routes
    from utils.hardware import npu as npu_hardware

    measured = npu_hardware.npu_training_capability()
    assert set(npu_routes._UNAVAILABLE_TRAINING_CAPABILITY) == set(measured)
    assert npu_routes._UNAVAILABLE_TRAINING_CAPABILITY["mode"] == "unavailable"


def test_the_unavailable_stand_in_is_the_one_the_compute_route_uses(monkeypatch):
    """One answer in one place: the degraded capability must not drift from the compute route's."""
    from routes import compute as compute_routes
    from routes import npu as npu_routes
    from utils.hardware import npu as npu_hardware

    fallback = getattr(compute_routes, "_npu_capability", None)
    if fallback is None:
        pytest.skip("the compute route no longer exposes the read this stand-in mirrors")

    def _boom():
        raise RuntimeError("the probe fell over")

    monkeypatch.setattr(npu_hardware, "npu_training_capability", _boom)
    assert fallback() == npu_routes._UNAVAILABLE_TRAINING_CAPABILITY
