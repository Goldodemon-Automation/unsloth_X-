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
from auth import storage as auth_storage  # noqa: E402
from utils import scoped_link  # noqa: E402
from utils.account_context import AccountContext, bind_account, reset_account  # noqa: E402
from utils.paths.storage_roots import account_path  # noqa: E402

ALICE = AccountContext("a" * 32, "alice")


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


def _app():
    app = FastAPI()
    app.include_router(inf_mod.router)
    return app


@pytest.fixture
def client(sandbox):
    return TestClient(_app())


@pytest.fixture
def signed_in(monkeypatch):
    """What the bearer path resolves to, so no test mints a real session JWT."""

    async def resolve(_request):
        return "unsloth"

    monkeypatch.setattr(auth_module, "subject_for_header_or_query_token", resolve)


@pytest.fixture
def no_credential(monkeypatch):
    """The bearer path with nothing to resolve: then only a link can authenticate."""

    async def refuse(_request):
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


def test_a_managed_accounts_link_resolves_inside_that_accounts_sandbox(tmp_path, monkeypatch, no_credential):
    """The link carries its account, and following it binds that account. Without the binding
    ``current_account()`` stayed on its OWNER default and the listing answered with the owner's
    root, so a managed account's link opened another workspace's tree."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    connection = auth_storage.get_connection()
    with connection:
        connection.execute(
            "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret,"
            " account_id, role, is_active) VALUES (?, 'salt', 'hash', 'secret', ?, 'user', 1)",
            (ALICE.username, ALICE.account_id),
        )
    connection.close()

    # The real resolver, so the assertion is about where the path lands, not about a stub.
    monkeypatch.setattr(
        inf_mod,
        "_sandbox_dir_for",
        lambda session_id, create = True: str(account_path("sandbox") / session_id),
    )

    account_token = bind_account(ALICE)
    try:
        link = inf_mod._sign_canvas_link(ALICE.username)
    finally:
        reset_account(account_token)

    owner_minted = inf_mod._sign_canvas_link("unsloth")
    client = TestClient(app = _app())

    response = client.get("/sandbox/chat-1", params = {"link": link})
    assert response.status_code == 200, response.text
    managed_root = tmp_path / "accounts" / ALICE.account_id / "sandbox" / "chat-1"
    assert response.json()["path"] == str(managed_root)
    assert str(managed_root).startswith(str(tmp_path / "accounts"))

    # An owner-minted link keeps the owner root, so the bearer path stays what it was.
    owner_response = client.get("/sandbox/chat-1", params = {"link": owner_minted})
    assert owner_response.json()["path"] == str(tmp_path / "sandbox" / "chat-1")


def test_a_link_naming_an_account_that_no_longer_exists_is_refused(monkeypatch):
    """A deleted account's link resolves to nothing rather than to the owner's tree."""
    import hub.services.models.account_access as account_access

    # ``_signed_link_account`` imports the lookup in its own body, so the module it resolves in is
    # what has to answer. The cache is cleared first or an earlier test's row would stand in.
    account_access._link_accounts.clear()
    monkeypatch.setattr(auth_storage, "get_account_by_id", lambda account_id: None)
    link = scoped_link.mint(
        inf_mod._CANVAS_LINK_SECRET, inf_mod._CANVAS_LINK_SCOPE, f"{'c' * 32}:gone"
    )
    request = _request(link = link)

    async def resolve(_request):
        raise HTTPException(status_code = 401, detail = "Missing authentication token")

    monkeypatch.setattr(auth_module, "subject_for_header_or_query_token", resolve)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(inf_mod._authenticate_header_or_query(request, None))
    assert refused.value.status_code == 401


def test_the_header_bearer_wins_over_a_link(monkeypatch):
    async def resolve(_request):
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
