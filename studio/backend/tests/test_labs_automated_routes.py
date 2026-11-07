# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automated training over HTTP: the gate, the plan, the job and the adapters.

The engine itself (``core.labs.automated_training``) is written beside these
routes, so every test here patches a fake module into its slot instead of
importing it: these tests pin the router's contract, not the engine's numbers.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core import labs
from core.labs import benchmarks
from routes import labs as labs_routes


ENGINE_NAME = "core.labs.automated_training"


class FakeAutomatedTrainingError(ValueError):
    """Stands in for the engine's own error type."""


def _store(monkeypatch, state: dict | None = None) -> dict:
    store = {"value": state}

    def get(key, default = None):
        assert key == labs.SETTING_KEY
        return store["value"] if store["value"] is not None else default

    def upsert(values):
        store["value"] = values[labs.SETTING_KEY]
        return values

    monkeypatch.setattr("storage.studio_db.get_app_setting", get)
    monkeypatch.setattr("storage.studio_db.upsert_app_settings", upsert)
    return store


def _client(monkeypatch, state: dict | None = None) -> tuple[TestClient, dict]:
    store = _store(monkeypatch, state)
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    app.dependency_overrides[labs_routes._require_owner] = lambda: None
    app.include_router(labs_routes.router, prefix = "/api/labs")
    return TestClient(app), store


def _fake_engine(
    monkeypatch,
    *,
    runnable: bool = True,
    blocked_reason: str | None = None,
    presets: list | None = None,
    plan_error: Exception | None = None,
):
    """Install a fake engine in the slot the routes import lazily.

    Both the ``sys.modules`` entry and the package attribute are set: a plain
    ``from core.labs import automated_training`` reads the attribute once an
    earlier import has set one.
    """
    module = types.ModuleType(ENGINE_NAME)
    module.AutomatedTrainingError = FakeAutomatedTrainingError
    module.calls = []
    module.ran = threading.Event()
    listed = (
        presets
        if presets is not None
        else [
            {
                "id": "balanced",
                "label": "Balanced",
                "description": "Four datasets, eight runs a round.",
                "spec": {"runs_per_round": 8},
            }
        ]
    )

    def presets_route():
        module.calls.append(("presets",))
        return list(listed)

    def plan(spec):
        module.calls.append(("plan", spec))
        if plan_error is not None:
            raise plan_error
        return {
            "runnable": runnable,
            "blocked_reason": blocked_reason,
            "rounds": [],
            "warnings": [],
            "projection": {"total_train_runs": 0, "datasets_per_round": 0, "rounds": 0},
            "presets": list(listed),
        }

    def run(spec, *, output_dir, progress = None, train = None, benchmark = None):
        module.calls.append(("run", spec, output_dir, train, benchmark, progress))
        module.ran.set()
        return {"best": {"output_dir": str(output_dir), "score": 0.5}, "totals": {"train_runs": 1}}

    module.presets = presets_route
    module.plan = plan
    module.run = run
    monkeypatch.setitem(sys.modules, ENGINE_NAME, module)
    monkeypatch.setattr(labs, "automated_training", module, raising = False)
    return module


def _grant_resources(monkeypatch) -> list:
    """Let ``_start_job`` take its lease, and record what it asked for."""
    asked = []

    def acquire(job_id, kind, *, sources = None, output_dir = None, **rest):
        asked.append({"job_id": job_id, "kind": kind, "sources": sources})
        return {"status": "granted", "estimate": {"need": {}, "notes": []}}

    monkeypatch.setattr("core.advanced.resources.acquire_for_job", acquire)
    return asked


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_every_automated_route_is_closed_while_labs_is_off(monkeypatch):
    _fake_engine(monkeypatch)
    client, _ = _client(monkeypatch)

    for call in (
        client.get("/api/labs/automated/presets"),
        client.post("/api/labs/automated/plan", json = {"spec": {}}),
        client.post("/api/labs/automated/run", json = {"spec": {}}),
    ):
        assert call.status_code == 403, call.request.url
        assert "Labs" in call.json()["detail"]


def test_the_automated_switch_closes_only_the_automated_routes(monkeypatch, tmp_path):
    _fake_engine(monkeypatch)
    monkeypatch.setattr("core.labs.benchmarks.outputs", lambda: tmp_path)
    client, _ = _client(
        monkeypatch,
        {"enabled": True, "features": {"automated-training": False, "benchmarks": True}},
    )

    assert client.get("/api/labs/benchmarks").status_code == 200

    closed = client.get("/api/labs/automated/presets")
    assert closed.status_code == 403
    assert "Automated training" in closed.json()["detail"]
    assert client.post("/api/labs/automated/plan", json = {"spec": {}}).status_code == 403
    assert client.post("/api/labs/automated/run", json = {"spec": {}}).status_code == 403


# ---------------------------------------------------------------------------
# Presets and plan
# ---------------------------------------------------------------------------


def test_presets_route_lists_what_the_engine_offers(monkeypatch):
    engine = _fake_engine(monkeypatch)
    client, _ = _client(monkeypatch, {"enabled": True, "features": {}})

    response = client.get("/api/labs/automated/presets")

    assert response.status_code == 200
    assert [preset["id"] for preset in response.json()["presets"]] == ["balanced"]
    assert ("presets",) in engine.calls


def test_plan_route_returns_the_engine_plan_for_the_spec_it_was_sent(monkeypatch):
    engine = _fake_engine(monkeypatch)
    client, _ = _client(monkeypatch, {"enabled": True, "features": {}})

    spec = {"source": "/models/base", "datasets": [{"path": "/data/a", "category": "code"}]}
    response = client.post("/api/labs/automated/plan", json = {"spec": spec})

    assert response.status_code == 200
    body = response.json()
    assert body["runnable"] is True
    assert body["rounds"] == []
    assert ("plan", spec) in engine.calls


def test_plan_route_maps_engine_errors_to_http_errors(monkeypatch):
    _fake_engine(monkeypatch, plan_error = FakeAutomatedTrainingError("A source is required."))
    client, _ = _client(monkeypatch, {"enabled": True, "features": {}})

    response = client.post("/api/labs/automated/plan", json = {"spec": {}})

    assert response.status_code == 400
    assert response.json()["detail"] == "A source is required."


def test_translate_maps_each_error_class_to_its_status():
    """A refusal is a 403, a missing file a 404, anything else a 400 with its reason."""
    from fastapi import HTTPException

    for error, status in (
        (PermissionError("not yours"), 403),
        (FileNotFoundError("no such checkpoint"), 404),
        (ValueError("bad spec"), 400),
    ):
        translated = labs_routes._translate(error)
        assert isinstance(translated, HTTPException)
        assert translated.status_code == status
        assert translated.detail == str(error)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def test_run_starts_a_job_and_hands_the_engine_its_adapters(monkeypatch, tmp_path):
    engine = _fake_engine(monkeypatch)
    asked = _grant_resources(monkeypatch)
    client, _ = _client(monkeypatch, {"enabled": True, "features": {}})
    output = tmp_path / "automated-run"

    response = client.post(
        "/api/labs/automated/run",
        json = {"spec": {"source": "/models/base"}, "output_dir": str(output)},
    )

    assert response.status_code == 200
    record = response.json()
    assert record["kind"] == "automated-training"
    assert record["job_id"]
    assert record["output_dir"] == str(output)
    assert "progress" in record
    # The lease was taken on the request thread, like every other advanced job.
    assert asked and asked[0]["kind"] == "automated-training"
    assert asked[0]["sources"] == ["/models/base"]

    assert engine.ran.wait(10), "the background job never reached the engine"
    _, _, called_output, train, benchmark, progress = engine.calls[-1]
    assert called_output == str(output)
    assert train is labs_routes._train_adapter
    assert benchmark is labs_routes._benchmark_adapter
    assert callable(progress)


def test_run_refuses_a_blocked_plan_with_its_reason(monkeypatch, tmp_path):
    engine = _fake_engine(
        monkeypatch,
        runnable = False,
        blocked_reason = "The pool holds 1 dataset but each round needs 4.",
    )
    asked = _grant_resources(monkeypatch)
    client, _ = _client(monkeypatch, {"enabled": True, "features": {}})

    response = client.post("/api/labs/automated/run", json = {"spec": {"source": "/models/base"}})

    assert response.status_code == 400
    assert response.json()["detail"] == "The pool holds 1 dataset but each round needs 4."
    assert not any(call[0] == "run" for call in engine.calls)
    assert asked == []


def test_run_is_owner_only_and_presets_is_not(monkeypatch):
    def dependencies(path: str, method: str) -> set[str]:
        for route in labs_routes.router.routes:
            if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
                names, stack = set(), [route.dependant]
                while stack:
                    node = stack.pop()
                    for child in getattr(node, "dependencies", []):
                        names.add(getattr(child.call, "__name__", ""))
                        stack.append(child)
                return names
        raise AssertionError(f"No {method} route for {path}")

    assert "_require_owner" in dependencies("/automated/run", "POST")
    assert "_require_owner" not in dependencies("/automated/presets", "GET")


# ---------------------------------------------------------------------------
# The adapters the engine calls back into
# ---------------------------------------------------------------------------


def test_the_train_adapter_runs_cmt_through_the_cmt_experiment(monkeypatch):
    seen = {}

    def fake_run(spec, *, output_dir, progress = None):
        seen.update({"spec": spec, "output_dir": output_dir, "progress": progress})
        return {"output_dir": output_dir, "manifest_path": "m.json"}

    _fake_engine(monkeypatch)
    monkeypatch.setattr("core.labs.cmt.run", fake_run)
    progress = lambda payload: None
    spec = {"source": "/models/base", "mode": "cpt"}

    result = labs_routes._train_adapter({"method": "cmt", "spec": spec}, "/out/run-01", progress)

    assert result["output_dir"] == "/out/run-01"
    assert seen == {"spec": spec, "output_dir": "/out/run-01", "progress": progress}


def test_the_train_adapter_refuses_cpt_without_training_kwargs(monkeypatch):
    _fake_engine(monkeypatch)

    with pytest.raises(FakeAutomatedTrainingError) as error:
        labs_routes._train_adapter(
            {"method": "cpt", "spec": {"source": "/models/base"}}, "/out", None
        )

    assert "training_kwargs" in str(error.value)
    assert "use CMT or supply them" in str(error.value)


def test_the_train_adapter_hands_cpt_to_the_training_backend(monkeypatch):
    _fake_engine(monkeypatch)
    started = {}

    class Backend:
        def start_training(self, *, job_id, **kwargs):
            started.update({"job_id": job_id, **kwargs})
            return True

    monkeypatch.setattr("core.training.get_training_backend", lambda: Backend())

    result = labs_routes._train_adapter(
        {
            "method": "fft",
            "round": 2,
            "label": "run-03-fft",
            "spec": {
                "source": "/models/base",
                "training_kwargs": {"output_dir": "/out/fft", "max_steps": 20},
            },
        },
        "/out/run-03",
        None,
    )

    assert started["job_id"] == "automated-2-run-03-fft"
    assert started["output_dir"] == "/out/fft"
    assert started["max_steps"] == 20
    assert result["output_dir"] == "/out/fft"
    assert result["started"] is True


def test_the_train_adapter_reports_a_backend_that_refuses_the_run(monkeypatch):
    _fake_engine(monkeypatch)

    class Backend:
        def start_training(self, *, job_id, **kwargs):
            return False

    monkeypatch.setattr("core.training.get_training_backend", lambda: Backend())

    with pytest.raises(FakeAutomatedTrainingError) as error:
        labs_routes._train_adapter(
            {"method": "cpt", "spec": {"training_kwargs": {"output_dir": "/out"}}},
            "/out",
            None,
        )

    assert "refused" in str(error.value)


def test_the_train_adapter_refuses_an_unknown_method(monkeypatch):
    _fake_engine(monkeypatch)

    with pytest.raises(FakeAutomatedTrainingError):
        labs_routes._train_adapter({"method": "lora", "spec": {}}, "/out", None)


def test_the_benchmark_adapter_scores_the_newest_record_for_the_checkpoint(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "checkpoint"
    artifact.mkdir()
    monkeypatch.setattr("core.labs.benchmarks.outputs", lambda: tmp_path)
    record = benchmarks.save_record(
        job_id = "run-1",
        artifact = str(artifact),
        baseline = "org/base",
        report = {
            "baseline_scores": {"mmlu": 0.4, "gsm8k": 0.2},
            "candidate_scores": {"mmlu": 0.5, "gsm8k": 0.7, "humaneval": None},
        },
        suites = ["mmlu", "gsm8k", "humaneval"],
        limit = None,
        batch_size = 1,
        duration_seconds = 1.0,
        output_dir = str(tmp_path / "runs" / "run-1"),
    )

    result = labs_routes._benchmark_adapter({}, {"output_dir": str(artifact)}, None)

    assert result["score"] == pytest.approx(0.6)
    assert result["record_id"] == record["id"]


def test_the_benchmark_adapter_leaves_an_unscored_checkpoint_unscored(monkeypatch, tmp_path):
    """No record means no score, so the engine ranks it last instead of guessing."""
    monkeypatch.setattr("core.labs.benchmarks.outputs", lambda: tmp_path)

    result = labs_routes._benchmark_adapter({}, {"output_dir": str(tmp_path / "unscored")}, None)

    assert result["score"] is None