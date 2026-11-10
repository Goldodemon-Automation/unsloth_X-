# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the opt-in UNSLOTH_TRUSTED_HOSTS allow-list."""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.trusted_hosts import TrustedHostsMiddleware

ENV = "UNSLOTH_TRUSTED_HOSTS"


@pytest.fixture(autouse = True)
def _unset_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising = False)


def _app():
    app = FastAPI()
    app.add_middleware(TrustedHostsMiddleware)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    return app


def _get(base_url, monkeypatch, allow = None):
    if allow is None:
        monkeypatch.delenv(ENV, raising = False)
    else:
        monkeypatch.setenv(ENV, allow)
    return TestClient(_app(), base_url = base_url).get("/ping")


def test_unset_allows_a_foreign_host(monkeypatch):
    response = _get("http://evil.example", monkeypatch)
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_listed_host_passes(monkeypatch):
    response = _get("http://studio.example", monkeypatch, "studio.example")
    assert response.status_code == 200


def test_foreign_host_is_refused_naming_the_variable(monkeypatch):
    response = _get("http://evil.example", monkeypatch, "studio.example")
    assert response.status_code == 400
    assert ENV in response.text
    assert "evil.example" in response.text


def test_loopback_literals_always_pass(monkeypatch):
    for base_url in ("http://127.0.0.1:8888", "http://localhost", "http://[::1]:8888"):
        response = _get(base_url, monkeypatch, "studio.example")
        assert response.status_code == 200, base_url


def test_wildcard_matches_subdomain_not_apex(monkeypatch):
    assert _get("http://foo.example", monkeypatch, "*.example").status_code == 200
    assert _get("http://a.b.example", monkeypatch, "*.example").status_code == 200
    assert _get("http://example", monkeypatch, "*.example").status_code == 400
    assert _get("http://example", monkeypatch, "*.example,example").status_code == 200


def test_port_pattern_is_exact(monkeypatch):
    assert _get("http://studio.example:8443", monkeypatch, "studio.example:8443").status_code == 200
    assert _get("http://studio.example:9999", monkeypatch, "studio.example:8443").status_code == 400
    assert _get("http://studio.example:9999", monkeypatch, "studio.example").status_code == 200


def test_non_http_scope_passes_untouched(monkeypatch):
    monkeypatch.setenv(ENV, "studio.example")
    reached = {}

    async def downstream(scope, receive, send):
        reached["scope"] = scope

    async def receive():
        return {"type": "websocket.receive"}

    async def send(message):
        pass

    scope = {
        "type": "websocket",
        "path": "/ws",
        "headers": [(b"host", b"evil.example")],
    }
    asyncio.run(TrustedHostsMiddleware(downstream)(scope, receive, send))
    assert reached["scope"] is scope
