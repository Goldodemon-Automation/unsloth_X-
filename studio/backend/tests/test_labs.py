# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Labs: one stored switch, one gate, and endpoints that close with it."""

from __future__ import annotations

import json
import os
import time

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core import labs
from core.labs import benchmarks, housekeeping
from routes import labs as labs_routes


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


def _refuse_credential():
    from fastapi import HTTPException

    raise HTTPException(status_code = 401, detail = "Invalid or expired token")


def _client(monkeypatch, state: dict | None = None) -> tuple[TestClient, dict]:
    store = _store(monkeypatch, state)
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    app.dependency_overrides[labs_routes._require_owner] = lambda: None
    app.include_router(labs_routes.router, prefix = "/api/labs")
    return TestClient(app), store


def test_the_labs_router_carries_the_credential_check():
    """Pinned at the router, not per route: a Labs route added later inherits the gate."""
    dependencies = [
        getattr(dependency.dependency, "__name__", "")
        for dependency in labs_routes.router.dependencies
    ]
    assert "get_current_subject" in dependencies


def test_every_labs_read_route_asks_who_is_calling(monkeypatch):
    """A hidden Labs page is not a gate. Reads carry the credential check the writes do."""
    _store(monkeypatch, {"enabled": True, "features": {}})
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = _refuse_credential
    app.include_router(labs_routes.router, prefix = "/api/labs")
    client = TestClient(app)

    for path in (
        "/api/labs",
        "/api/labs/artifacts",
        "/api/labs/artifacts/detail?path=/tmp/none",
        "/api/labs/pipelines",
        "/api/labs/pipelines/presets",
        "/api/labs/benchmarks",
        "/api/labs/lineage",
        "/api/labs/housekeeping",
    ):
        response = client.get(path)
        assert response.status_code == 401, (path, response.status_code)


def test_labs_defaults_off_and_every_feature_on(monkeypatch):
    _store(monkeypatch)
    state = labs.read_state()
    assert state["enabled"] is False
    assert state["features"] == {feature_id: True for feature_id in labs.FEATURE_IDS}
    assert labs.is_enabled() is False
    assert labs.feature_enabled("pipelines") is False


def test_turning_labs_on_does_not_override_an_explicit_off(monkeypatch):
    _store(monkeypatch, {"enabled": False, "features": {"pipelines": False}})
    labs.set_state(enabled = True)
    assert labs.is_enabled() is True
    assert labs.feature_enabled("pipelines") is False
    assert labs.feature_enabled("artifacts") is True


def test_unknown_feature_is_refused_and_never_stored(monkeypatch):
    store = _store(monkeypatch)
    with pytest.raises(ValueError):
        labs.set_state(features = {"not-a-feature": True})
    assert store["value"] is None
    with pytest.raises(ValueError):
        labs.feature_enabled("not-a-feature")


def test_a_disabled_feature_says_which_switch_to_turn_on(monkeypatch):
    _store(monkeypatch, {"enabled": False, "features": {}})
    with pytest.raises(labs.LabsDisabled) as master:
        labs.require("pipelines")
    assert "Settings → Labs" in str(master.value)

    _store(monkeypatch, {"enabled": True, "features": {"pipelines": False}})
    with pytest.raises(labs.LabsDisabled) as feature:
        labs.require("pipelines")
    assert "Pipelines" in str(feature.value)


def test_state_route_is_readable_and_put_saves(monkeypatch):
    client, store = _client(monkeypatch)
    response = client.get("/api/labs")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert {feature["id"] for feature in body["features"]} == set(labs.FEATURE_IDS)
    assert all(feature["enabled"] for feature in body["features"])

    saved = client.put("/api/labs", json = {"enabled": True, "features": {"pipelines": False}})
    assert saved.status_code == 200
    assert saved.json()["enabled"] is True
    assert store["value"]["enabled"] is True
    assert store["value"]["features"]["pipelines"] is False


def test_put_rejects_an_unknown_feature(monkeypatch):
    client, store = _client(monkeypatch)
    response = client.put("/api/labs", json = {"features": {"teleport": True}})
    assert response.status_code == 400
    assert store["value"] is None


def test_every_studio_route_is_closed_while_labs_is_off(monkeypatch):
    client, _ = _client(monkeypatch)
    for path in (
        "/api/labs/artifacts",
        "/api/labs/pipelines",
        "/api/labs/benchmarks",
        "/api/labs/housekeeping",
        "/api/labs/artifacts/manifest?path=/tmp",
    ):
        response = client.get(path)
        assert response.status_code == 403, path
        assert "Labs" in response.json()["detail"]

    planned = client.post(
        "/api/labs/pipelines/plan",
        json = {"spec": {"source": "/tmp/none", "stages": [{"kind": "benchmark"}]}},
    )
    assert planned.status_code == 403
    assert client.post("/api/labs/cmt/plan", json = {"spec": {}}).status_code == 403
    assert client.post("/api/labs/surgery/plan", json = {"spec": {}}).status_code == 403
    assert client.post("/api/labs/arch-swap/plan", json = {"spec": {}}).status_code == 403
    assert (
        client.post("/api/labs/surgery/run", json = {"spec": {"model": "/tmp/none"}}).status_code
        == 403
    )
    assert (
        client.post(
            "/api/labs/arch-swap/run", json = {"spec": {"source": "/tmp/a", "target": "/tmp/b"}}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/labs/benchmarks/run",
            json = {
                "artifact": "/tmp/a",
                "baseline": "/tmp/b",
                "suites": ["mmlu"],
            },
        ).status_code
        == 403
    )


def test_a_feature_that_is_off_closes_only_its_route(monkeypatch, tmp_path):
    client, _ = _client(
        monkeypatch, {"enabled": True, "features": {"pipelines": False, "artifacts": True}}
    )
    monkeypatch.setattr("core.labs.artifacts.root", lambda: tmp_path)
    assert client.get("/api/labs/artifacts").status_code == 200
    assert client.get("/api/labs/pipelines").status_code == 403


def test_surgery_and_arch_swap_close_one_at_a_time(monkeypatch):
    """Two studios, two switches: turning the swap off must not close surgery."""
    client, _ = _client(
        monkeypatch, {"enabled": True, "features": {"surgery": True, "arch-swap": False}}
    )

    open_route = client.post("/api/labs/surgery/plan", json = {"spec": {}})
    assert open_route.status_code == 400
    assert "checkpoint is required" in open_route.json()["detail"]

    closed = client.post("/api/labs/arch-swap/plan", json = {"spec": {}})
    assert closed.status_code == 403
    assert "Arch swap" in closed.json()["detail"]


def test_the_surgery_plans_are_owner_only_like_their_runs(monkeypatch):
    """A plan reads the checkpoint the spec names, so a guest account is refused.

    The run routes were owner-only from the start. A plan opens the same two
    folders before the run starts, so it carries the same gate rather than
    answering a guest account with another account's checkpoint headers.
    """
    _store(monkeypatch, {"enabled": True, "features": {"surgery": True, "arch-swap": True}})
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "guest"

    async def refuse():
        from fastapi import HTTPException

        raise HTTPException(status_code = 403, detail = "Owner only.")

    app.dependency_overrides[labs_routes._require_owner] = refuse
    app.include_router(labs_routes.router, prefix = "/api/labs")
    client = TestClient(app)

    assert client.post("/api/labs/surgery/plan", json = {"spec": {}}).status_code == 403
    assert (
        client.post("/api/labs/arch-swap/plan", json = {"spec": {"source": "/tmp/a"}}).status_code
        == 403
    )


def test_both_new_studios_are_registered_for_the_settings_tab(monkeypatch):
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda key, default = None: None)
    state = labs.describe_state()
    by_id = {feature["id"]: feature for feature in state["features"]}
    for feature_id, word in (("surgery", "Prune"), ("arch-swap", "architecture")):
        assert by_id[feature_id]["studio"] is True
        assert by_id[feature_id]["status"] == "alpha"
        assert word in by_id[feature_id]["description"]


def test_manifest_route_needs_the_manifest_feature(monkeypatch, tmp_path):
    model = tmp_path / "trained"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding = "utf-8")
    (model / "run_manifest.json").write_text('{"stage": "train"}', encoding = "utf-8")
    client, _ = _client(
        monkeypatch,
        {"enabled": True, "features": {"artifacts": True, "manifest-viewer": False}},
    )
    monkeypatch.setattr("core.labs.artifacts.root", lambda: tmp_path)
    response = client.get(f"/api/labs/artifacts/manifest?path={model}")
    assert response.status_code == 403

    client, _ = _client(
        monkeypatch,
        {"enabled": True, "features": {"artifacts": True, "manifest-viewer": True}},
    )
    monkeypatch.setattr("core.labs.artifacts.root", lambda: tmp_path)
    allowed = client.get(f"/api/labs/artifacts/manifest?path={model}")
    assert allowed.status_code == 200
    assert allowed.json()["manifest"]["stage"] == "train"


def test_artifact_detail_attaches_the_raw_manifest_only_when_its_feature_is_on(
    monkeypatch, tmp_path
):
    model = tmp_path / "trained"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,'
        ' "intermediate_size": 128, "vocab_size": 256,'
        ' "num_attention_heads": 2, "num_key_value_heads": 2}',
        encoding = "utf-8",
    )
    (model / "run_manifest.json").write_text('{"stage": "train"}', encoding = "utf-8")
    base = {"enabled": True, "features": {"artifacts": True, "manifest-viewer": False}}
    client, store = _client(monkeypatch, dict(base))
    monkeypatch.setattr("core.labs.artifacts.root", lambda: tmp_path)
    without = client.get(f"/api/labs/artifacts/detail?path={model}").json()
    assert "raw_manifest" not in without
    assert without["manifest"]["stage"] == "train"

    store["value"]["features"]["manifest-viewer"] = True
    with_manifest = client.get(f"/api/labs/artifacts/detail?path={model}").json()
    assert with_manifest["raw_manifest"]["stage"] == "train"


def test_describe_state_lists_the_registry_without_duplicating_it(monkeypatch):
    _store(monkeypatch, {"enabled": True, "features": {}})
    state = labs.describe_state()
    assert [feature["id"] for feature in state["features"]] == list(labs.FEATURE_IDS)
    assert all(feature["enabled"] for feature in state["features"])
    assert {feature["studio"] for feature in state["features"]} == {True, False}


# ---------------------------------------------------------------------------
# Diff, pins and janitor rules over HTTP
# ---------------------------------------------------------------------------


def _artifact(root, name: str, learning_rate: float) -> None:
    model = root / name
    model.mkdir(parents = True)
    (model / "config.json").write_text("{}", encoding = "utf-8")
    (model / "run_manifest.json").write_text(
        json.dumps(
            {"stage": "train", "run_id": name, "config": {"learning_rate": learning_rate}}
        ),
        encoding = "utf-8",
    )


def test_the_diff_route_names_the_field_that_moved(monkeypatch, tmp_path):
    root = tmp_path / "outputs"
    _artifact(root, "run-a", 2e-4)
    _artifact(root, "run-b", 1e-4)
    monkeypatch.setattr("core.labs.artifacts.root", lambda: root)
    monkeypatch.setattr("core.labs.benchmarks.outputs", lambda: tmp_path)
    client, _ = _client(monkeypatch, {"enabled": True, "features": {"artifacts": True}})

    response = client.get(
        "/api/labs/artifacts/diff",
        params = {"left": str(root / "run-a"), "right": str(root / "run-b")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["changes"][0]["field"] == "config.learning_rate"
    assert body["left"]["name"] == "run-a"


def test_pin_and_unpin_change_what_the_list_reports(monkeypatch, tmp_path):
    monkeypatch.setattr("core.labs.benchmarks.outputs", lambda: tmp_path)
    artifact = tmp_path / "merged"
    artifact.mkdir()
    record = benchmarks.save_record(
        job_id = "run-1",
        artifact = str(artifact),
        baseline = "org/base",
        report = {"baseline_scores": {"mmlu": 0.4}, "candidate_scores": {"mmlu": 0.5}},
        suites = ["mmlu"],
        limit = None,
        batch_size = 1,
        duration_seconds = 1.0,
        output_dir = str(tmp_path / "runs" / "run-1"),
    )
    client, _ = _client(monkeypatch, {"enabled": True, "features": {"benchmarks": True}})

    pinned = client.post(f"/api/labs/benchmarks/{record['id']}/pin")
    assert pinned.status_code == 200
    assert pinned.json()["pin"]["record_id"] == record["id"]

    listed = client.get("/api/labs/benchmarks").json()
    assert listed["records"][0]["pin"]["is_self"] is True
    assert listed["records"][0]["vs_pinned"]["status"] == "pinned"
    assert listed["pins"][0]["record_id"] == record["id"]

    assert client.delete(f"/api/labs/benchmarks/{record['id']}/pin").status_code == 200
    after = client.get("/api/labs/benchmarks").json()
    assert after["records"][0]["pin"] is None
    assert after["records"][0]["vs_pinned"]["status"] == "unpinned"
    assert client.delete(f"/api/labs/benchmarks/{record['id']}/pin").status_code == 404


def test_janitor_rules_save_preview_and_apply_over_http(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for name, days in (("run-a", 1), ("run-b", 40)):
        directory = outputs / name
        directory.mkdir()
        (directory / "config.json").write_text("{}", encoding = "utf-8")
        stamp = time.time() - days * 86_400
        os.utime(directory, (stamp, stamp))
    monkeypatch.setattr("core.labs.housekeeping._paths", lambda: {"outputs": outputs})
    monkeypatch.setattr("core.labs.housekeeping._running_outputs", lambda: [])
    monkeypatch.setattr("core.labs.benchmarks.pinned_artifacts", lambda: set())
    client, _ = _client(monkeypatch, {"enabled": True, "features": {"housekeeping": True}})

    assert client.get("/api/labs/housekeeping/rules").json()["rules"]["keep_newest"] == 3
    saved = client.put(
        "/api/labs/housekeeping/rules", json = {"keep_newest": 1, "min_age_days": 7}
    )
    assert saved.status_code == 200
    assert saved.json()["rules"]["keep_newest"] == 1
    assert (outputs / "run-a").is_dir()

    preview = client.post("/api/labs/housekeeping/rules/preview", json = {}).json()
    assert [entry["name"] for entry in preview["candidates"]] == ["run-b"]
    assert [entry["name"] for entry in preview["kept"]] == ["run-a"]
    # A preview touches nothing.
    assert (outputs / "run-b").is_dir()

    applied = client.post("/api/labs/housekeeping/rules/apply", json = {})
    assert applied.status_code == 200
    assert applied.json()["result"]["freed_bytes"] > 0
    assert not (outputs / "run-b").exists()
    assert (outputs / "run-a").is_dir()


def test_switched_off_rules_are_refused_by_the_apply_route(monkeypatch, tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr("core.labs.housekeeping._paths", lambda: {"outputs": outputs})
    monkeypatch.setattr("core.labs.housekeeping._running_outputs", lambda: [])
    client, _ = _client(monkeypatch, {"enabled": True, "features": {"housekeeping": True}})

    response = client.post(
        "/api/labs/housekeeping/rules/apply", json = {"enabled": False}
    )
    assert response.status_code == 400
    assert housekeeping.read_rules()["enabled"] is True
