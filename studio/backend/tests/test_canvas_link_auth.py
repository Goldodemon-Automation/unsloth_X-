# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The canvas/artifact routes and the credential they take: a short-lived ``?link=`` or a bearer
header, never the session bearer in ``?token=`` the library download route already refuses."""

import asyncio
import sys
from pathlib import Path
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import auth.authentication as auth_module  # noqa: E402
import routes.inference as inf_mod  # noqa: E402
from utils import scoped_link  # noqa: E402


@pytest.fixture
def clock(monkeypatch):
    """A movable wall clock: ``verify`` reads ``_now``, so expiry is tested without sleeping."""
    now = [1_000_000.0]
    monkeypatch.setattr(scoped_link, "_now", lambda: now[0])
    return now


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """One chat's sandbox, a real directory the listing and file routes can serve from."""
    root = tmp_path / "chat-1"
    root.mkdir()
    (root / "report.txt").write_bytes(b"artifact bytes")
    real = str(root.resolve())
    monkeypatch.setattr(inf_mod, "_sandbox_dir_for", lambda session_id, create = True: real)
    return root


@pytest.fixture
def client(sandbox):
    app = FastAPI()
    app.include_router(inf_mod.router)
    return TestClient(app)


@pytest.fixture
def signed_in(monkeypatch):
    """What the bearer path resolves to, so no test mints a real session JWT."""

    async def resolve(_request, _token):
        return "unsloth"

    monkeypatch.setattr(auth_module, "subject_for_header_or_query_token", resolve)


@pytest.fixture
def no_credential(monkeypatch):
    """The bearer path with nothing to resolve: then only a link can authenticate."""

    async def refuse(_request, _token):
        raise HTTPException(status_code = 401, detail = "Missing authentication token")

    monkeypatch.setattr(auth_module, "subject_for_header_or_query_token", refuse)


def _request(*headers, **query):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": urlencode(query).encode(),
        "headers": [(name.encode(), value.encode()) for name, value in headers],
    }
    return Request(scope)


def test_the_header_bearer_still_authenticates(client, signed_in):
    response = client.get("/sandbox/chat-1", headers = {"Authorization": "Bearer a-session"})
    assert response.status_code == 200
    assert [entry["name"] for entry in response.json()["files"]] == ["report.txt"]


def test_a_minted_link_opens_the_canvas_routes_without_a_bearer(client, no_credential):
    link = inf_mod._sign_canvas_link("unsloth")
    # No Authorization header: the link is the whole credential, on the listing and the file.
    assert client.get("/sandbox/chat-1", params = {"link": link}).status_code == 200
    response = client.get("/sandbox/chat-1/report.txt", params = {"link": link})
    assert (response.status_code, response.content) == (200, b"artifact bytes")


def test_a_link_resolves_to_the_subject_it_was_minted_for():
    link = inf_mod._sign_canvas_link("unsloth")
    subject = asyncio.run(inf_mod._authenticate_header_or_query(_request(link = link), None))
    assert subject == "unsloth"


def test_the_header_bearer_wins_over_a_link(monkeypatch):
    async def resolve(_request, _token):
        return "signed-in"

    monkeypatch.setattr(auth_module, "subject_for_header_or_query_token", resolve)
    link = inf_mod._sign_canvas_link("someone-else")
    request = _request(("authorization", "Bearer a-session"), link = link)
    assert asyncio.run(inf_mod._authenticate_header_or_query(request, None)) == "signed-in"


def test_an_expired_or_wrong_scope_link_is_refused(client, clock, no_credential):
    link = inf_mod._sign_canvas_link("unsloth")
    assert client.get("/sandbox/chat-1", params = {"link": link}).status_code == 200
    # Signed with this route's own secret, but minted for a scope that is not this route's.
    elsewhere = scoped_link.mint(inf_mod._CANVAS_LINK_SECRET, "library-download", "unsloth")
    assert client.get("/sandbox/chat-1", params = {"link": elsewhere}).status_code == 401
    assert client.get("/sandbox/chat-1", params = {"link": link + "x"}).status_code == 401
    clock[0] += inf_mod._CANVAS_LINK_TTL + 1
    assert client.get("/sandbox/chat-1", params = {"link": link}).status_code == 401


def test_a_session_bearer_in_the_query_string_is_refused_by_name(client, no_credential):
    response = client.get("/sandbox/chat-1", params = {"token": "a-real-looking-session-bearer"})
    assert response.status_code == 400
    assert "link=" in response.json()["detail"]
    # Every caller answers the same way, before any file work.
    for method, path, params in (
        ("get", "/sandbox/chat-1/report.txt", {"token": "a-real-looking-session-bearer"}),
        (
            "post",
            "/sandbox/chat-1/open",
            {"token": "a-real-looking-session-bearer", "file": "report.txt"},
        ),
    ):
        refused = getattr(client, method)(path, params = params)
        assert (refused.status_code, "link=" in refused.json()["detail"]) == (400, True)
    # Even next to a link that would have opened the route, and even with a bearer in the
    # header: the query string must never carry a session bearer.
    link = inf_mod._sign_canvas_link("unsloth")
    refused = client.get(
        "/sandbox/chat-1",
        params = {"link": link, "token": "a-real-looking-session-bearer"},
        headers = {"Authorization": "Bearer a-session"},
    )
    assert refused.status_code == 400
    # A blank one is the absent parameter it always was: the route falls through to the bearer
    # path, which is what refuses it here -- never the 400 that names ``link=``.
    assert client.get("/sandbox/chat-1", params = {"token": ""}).status_code == 401
