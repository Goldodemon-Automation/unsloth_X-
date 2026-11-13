# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The scoped link token, and the one route that takes one: a link names one item, in one scope, for
minutes, and a session bearer in ``?token=`` no longer downloads anything."""

import base64
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from routes import library as library_routes  # noqa: E402
from utils import scoped_link  # noqa: E402
from utils.scoped_link import ScopedLinkError  # noqa: E402

_SECRET = b"a" * 32
_SCOPE = "test-scope"

_DATA = b"report bytes"


class _Item:
    """What ``library.open_item`` hands back, as far as ``_send`` reads it."""

    def __init__(self, name = "report.txt", data = _DATA):
        self.name = name
        self.size = len(data)
        self.handle = io.BytesIO(data)

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


@pytest.fixture
def clock(monkeypatch):
    """A movable wall clock: ``verify`` reads ``_now``, so expiry is tested without sleeping."""
    now = [1_000_000.0]
    monkeypatch.setattr(scoped_link, "_now", lambda: now[0])
    return now


def _encoder(value) -> str:
    """The one encoding ``mint`` uses, so a test can forge a well-formed segment."""
    return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()


def _claims(token: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(token.split(".")[0] + "=="))


def _resign(secret: bytes, segment: str) -> str:
    return f"{segment}.{scoped_link._b64encode(scoped_link._signature(secret, segment))}"


def test_a_link_round_trips_its_claims(clock):
    token = scoped_link.mint(_SECRET, _SCOPE, "unsloth", {"item": "upload:abc"}, ttl_s = 60)
    claims = scoped_link.verify(_SECRET, token, _SCOPE)
    assert claims["scope"] == _SCOPE and claims["sub"] == "unsloth"
    assert claims["item"] == "upload:abc"
    assert (claims["iat"], claims["exp"]) == (1_000_000, 1_000_060)
    assert claims["nonce"]
    # Two links minted in the same second for the same thing still differ, so one is not a replay
    # of the other.
    assert token != scoped_link.mint(_SECRET, _SCOPE, "unsloth", {"item": "upload:abc"}, ttl_s = 60)
    # It carries no session bearer: the claims above are the whole of the credential.
    assert set(_claims(token)) == {"scope", "sub", "iat", "exp", "nonce", "item"}


def test_a_link_for_another_scope_is_refused(clock):
    token = scoped_link.mint(_SECRET, "library-stream", "unsloth")
    with pytest.raises(ScopedLinkError):
        scoped_link.verify(_SECRET, token, _SCOPE)
    # The scope is inside the signed segment, so rewording it keeps the old signature, which no
    # longer covers these claims: only this process's secret mints a link for this route.
    tampered = _claims(token)
    tampered["scope"] = _SCOPE
    with pytest.raises(ScopedLinkError):
        scoped_link.verify(_SECRET, f"{_encoder(tampered)}.{token.split('.')[1]}", _SCOPE)


def test_a_link_expires_on_its_exp_second(clock):
    token = scoped_link.mint(_SECRET, _SCOPE, "unsloth", ttl_s = 300)
    clock[0] += 299
    assert scoped_link.verify(_SECRET, token, _SCOPE)["sub"] == "unsloth"
    clock[0] += 1
    with pytest.raises(ScopedLinkError):
        scoped_link.verify(_SECRET, token, _SCOPE)


def test_a_tampered_payload_is_refused(clock):
    token = scoped_link.mint(_SECRET, _SCOPE, "unsloth", {"item": "upload:abc"}, ttl_s = 60)
    original_signature = token.split(".")[1]
    forged = _claims(token)
    forged["item"] = "upload:someone-else"
    # The original signature no longer covers the claims...
    with pytest.raises(ScopedLinkError):
        scoped_link.verify(_SECRET, f"{_encoder(forged)}.{original_signature}", _SCOPE)
    # ...and a signature minted for another item does not cover these claims.
    other = scoped_link.mint(_SECRET, _SCOPE, "unsloth", {"item": "upload:other"}, ttl_s = 60)
    moved = f"{token.split('.')[0]}.{other.split('.')[1]}"
    with pytest.raises(ScopedLinkError):
        scoped_link.verify(_SECRET, moved, _SCOPE)


def test_a_link_signed_with_another_secret_is_refused(clock):
    token = scoped_link.mint(b"b" * 32, _SCOPE, "unsloth")
    with pytest.raises(ScopedLinkError):
        scoped_link.verify(_SECRET, token, _SCOPE)
    # Short signatures are refused rather than compared against a prefix, and a non-ASCII one is
    # an error rather than a TypeError out of compare_digest.
    segment, _, signature = token.partition(".")
    for bad in (f"{segment}.{signature[:-2]}", f"{segment}.", f"{segment}.{'é' * 43}", "x", ""):
        with pytest.raises(ScopedLinkError):
            scoped_link.verify(b"b" * 32, bad, _SCOPE)


@pytest.mark.parametrize(
    "claims",
    [
        [],  # not an object
        {"scope": _SCOPE, "sub": "", "iat": 1, "exp": 10**12, "nonce": "n"},  # empty subject
        {"scope": _SCOPE, "iat": 1, "exp": 10**12, "nonce": "n"},  # no subject
        {"scope": _SCOPE, "sub": "s", "exp": 10**12, "nonce": "n"},  # no issued time
        {"scope": _SCOPE, "sub": "s", "iat": 1, "nonce": "n"},  # no expiry: never accepted
        {"scope": _SCOPE, "sub": "s", "iat": 1, "exp": "soon", "nonce": "n"},  # wrong type
        {"scope": _SCOPE, "sub": "s", "iat": 10**12, "exp": True, "nonce": "n"},  # bool is not seconds
        {"scope": _SCOPE, "sub": "s", "iat": 10**12, "exp": 10**12, "nonce": "n"},  # exp <= iat
    ],
)
def test_claims_that_are_not_a_link_are_refused(claims):
    # Correctly signed, so this pins the claim checks and not the signature.
    with pytest.raises(ScopedLinkError):
        scoped_link.verify(_SECRET, _resign(_SECRET, _encoder(claims)), _SCOPE)


@pytest.mark.parametrize(
    "token",
    [
        "nonsense",
        "one.two.three",
        "." + "A" * 43,
        "!" * 10 + "." + "A" * 43,
        "A" * (scoped_link.TOKEN_MAX_LENGTH + 1),
    ],
)
def test_a_malformed_link_is_refused(clock, token):
    with pytest.raises(ScopedLinkError):
        scoped_link.verify(_SECRET, token, _SCOPE)


def test_mint_refuses_what_it_cannot_sign(clock):
    for kwargs in (
        {"scope": "", "subject": "s"},
        {"scope": _SCOPE, "subject": ""},
        {"scope": _SCOPE, "subject": "s", "ttl_s": 0},
        {"scope": _SCOPE, "subject": "s", "ttl_s": -5},
        {"scope": _SCOPE, "subject": "s", "ttl_s": True},
        {"scope": _SCOPE, "subject": "s", "payload": {"exp": 10**12}},  # reserved claim
        {"scope": _SCOPE, "subject": "s", "payload": {"nested": {1, 2}}},  # not JSON
    ):
        with pytest.raises(ScopedLinkError):
            scoped_link.mint(_SECRET, **kwargs)
    with pytest.raises(ScopedLinkError):
        scoped_link.mint(b"", _SCOPE, "s")


# The route: one path, one item, no bearer.


@pytest.fixture
def item(monkeypatch):
    opened = []

    def open_item(item_id):
        opened.append(item_id)
        if item_id == "upload:missing":
            raise LookupError(item_id)
        return _Item()

    monkeypatch.setattr(library_routes.library, "open_item", open_item)
    return opened


@pytest.fixture
def client(item):
    app = FastAPI()
    app.include_router(library_routes.router, prefix = "/api/library")
    return TestClient(app)


def _download(client, item_id, **params):
    return client.get("/api/library/items/download", params = {"id": item_id, **params})


def test_a_minted_link_downloads_without_a_bearer(client):
    item_id = "upload:" + "a" * 32
    link = library_routes._sign_download_link(item_id)
    # No Authorization header, no dependency override: the link is the whole credential.
    response = _download(client, item_id, link = link)
    assert (response.status_code, response.content) == (200, _DATA)
    assert response.headers["content-disposition"].startswith('attachment; filename="report.txt"')
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.head(
        "/api/library/items/download", params = {"id": item_id, "link": link}
    ).status_code == 200


def test_a_link_opens_only_the_item_it_names(client):
    item_id = "upload:" + "a" * 32
    link = library_routes._sign_download_link(item_id)
    response = _download(client, "upload:" + "b" * 32, link = link)
    assert response.status_code == 401
    assert _DATA not in response.content


def test_a_tampered_expired_or_other_scope_link_downloads_nothing(client, clock):
    item_id = "upload:" + "a" * 32
    link = library_routes._sign_download_link(item_id)
    assert _download(client, item_id, link = link + "x").status_code == 401
    assert _download(client, item_id, link = "nonsense").status_code == 401
    # Signed with this route's own secret, but minted for a scope that is not this route's.
    elsewhere = scoped_link.mint(
        library_routes._DOWNLOAD_LINK_SECRET, "library-stream", item_id, {"item": item_id}
    )
    assert _download(client, item_id, link = elsewhere).status_code == 401
    clock[0] += library_routes._DOWNLOAD_LINK_TTL + 1
    assert _download(client, item_id, link = link).status_code == 401


def test_a_session_bearer_in_the_query_string_is_refused_by_name(client, monkeypatch):
    async def no_credential(_request):
        raise HTTPException(status_code = 401, detail = "Missing authentication token")

    monkeypatch.setattr(library_routes, "subject_for_header_or_query_token", no_credential)
    item_id = "upload:" + "a" * 32
    response = _download(client, item_id, token = "a-real-looking-session-bearer")
    assert response.status_code == 400
    assert "link=" in response.json()["detail"]
    # A blank one is the absent parameter it always was: the route falls through to the bearer
    # path, which is what refuses it here -- never the 400 that names ``link=``.
    assert _download(client, item_id, token = "").status_code == 401


def test_the_header_bearer_still_downloads(client, monkeypatch):
    async def signed_in(_request):
        return "unsloth"

    monkeypatch.setattr(library_routes, "subject_for_header_or_query_token", signed_in)
    response = client.get(
        "/api/library/items/download",
        params = {"id": "upload:" + "a" * 32},
        headers = {"Authorization": "Bearer a-session"},
    )
    assert (response.status_code, response.content) == (200, _DATA)