# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""API tests: auth, ingest validation, the admin payloads and the static console."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from telemetry import server  # noqa: E402
from telemetry.tests.test_store import make_report  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("X8_TELEMETRY_DB", str(tmp_path / "telemetry.sqlite3"))
    monkeypatch.delenv("X8_TELEMETRY_ADMIN_TOKEN", raising = False)
    monkeypatch.delenv("X8_TELEMETRY_INGEST_TOKEN", raising = False)
    with TestClient(server.app) as test_client:
        yield test_client


def test_health_reports_an_empty_store(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "installs": 0}


def test_ingest_then_overview(client):
    response = client.post("/api/v1/report", json = make_report())
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True

    overview = client.get("/api/admin/overview", params = {"days": 30})
    assert overview.status_code == 200
    body = overview.json()
    assert body["totals"]["installs"] == 1
    assert body["totals"]["prompt_tokens"] == 1350
    assert body["models"][0]["model"] == "unsloth/Qwen3-8B"
    assert body["devices"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 4090"
    assert len(body["series"]) == 180


def test_reposting_the_same_report_does_not_double_count(client):
    report = make_report()
    client.post("/api/v1/report", json = report)
    client.post("/api/v1/report", json = report)
    body = client.get("/api/admin/overview", params = {"days": 30}).json()
    assert body["totals"]["prompt_tokens"] == 1350
    assert body["totals"]["requests"] == 7


def test_validation_rejects_a_bad_install_id_and_bad_day(client):
    bad_id = make_report()
    bad_id["install_id"] = "short"
    assert client.post("/api/v1/report", json = bad_id).status_code == 422

    bad_day = make_report()
    bad_day["usage"][0]["day"] = "yesterday"
    assert client.post("/api/v1/report", json = bad_day).status_code == 422


def test_installs_and_csv(client):
    client.post("/api/v1/report", json = make_report())
    installs = client.get("/api/admin/installs").json()["installs"]
    assert installs[0]["short_id"] == "abcdefgh"
    assert installs[0]["top_model"] == "unsloth/Qwen3-8B"

    csv = client.get("/api/admin/export.csv")
    assert csv.status_code == 200
    assert csv.headers["content-type"].startswith("text/csv")
    assert "abcdefgh-0001" in csv.text


def test_loopback_admin_is_refused_through_a_public_host_header(client):
    assert client.get("/api/admin/overview").status_code == 200
    response = client.get(
        "/api/admin/overview",
        headers = {"host": "collector.example.com"},
    )
    assert response.status_code == 401


def test_admin_token_is_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("X8_TELEMETRY_ADMIN_TOKEN", "s3cret")
    assert client.get("/api/admin/overview").status_code == 401
    assert client.get("/api/admin/session").json() == {
        "authenticated": False,
        "tokenRequired": True,
    }

    assert client.post("/api/admin/login", json = {"token": "wrong"}).status_code == 401
    assert client.post("/api/admin/login", json = {"token": "s3cret"}).status_code == 200
    assert client.get("/api/admin/overview").status_code == 200
    client.post("/api/admin/logout")
    assert client.get("/api/admin/overview").status_code == 401


def test_bearer_token_also_authorizes_admin(client, monkeypatch):
    monkeypatch.setenv("X8_TELEMETRY_ADMIN_TOKEN", "s3cret")
    response = client.get(
        "/api/admin/overview",
        headers = {"Authorization": "Bearer s3cret"},
    )
    assert response.status_code == 200


def test_ingest_token_is_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("X8_TELEMETRY_INGEST_TOKEN", "ingest-me")
    assert client.post("/api/v1/report", json = make_report()).status_code == 401
    response = client.post(
        "/api/v1/report",
        json = make_report(),
        headers = {"Authorization": "Bearer ingest-me"},
    )
    assert response.status_code == 200


def test_oversized_reports_are_refused(client):
    response = client.post(
        "/api/v1/report",
        content = b"{}",
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(3 * 1024 * 1024),
        },
    )
    assert response.status_code in (400, 413)


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Operator console" in response.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200
