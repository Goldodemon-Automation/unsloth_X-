# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the credential brute-force guard: the limiter's policy on an injected clock, then the
credential-accepting routes it protects with a faked credential check.

Nothing sleeps here. ``AttemptLimiter`` takes its clock, so window expiry and lockout end are
driven by advancing a fake clock to exact instants -- including the exact boundary, because "locked
for exactly ``lockout_s``" is the property being pinned. The route tests run the real /refresh
dispatch through ``TestClient`` with only the store rotation faked; /change-password's auth
dependency chain needs a full signed-in session, so its handler is driven the way
``test_change_password_policy.py`` already drives it, with the credential helpers patched and a
fake request carrying the client address the limiter keys on.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from auth import rate_limit, storage  # noqa: E402
from auth.rate_limit import AttemptLimiter, client_key, token_attempts  # noqa: E402
from models.auth import ChangePasswordRequest  # noqa: E402

# Load routes/auth.py directly so collection does not execute routes/__init__.py, which pulls in
# the heavy training/models/inference routers (the same trick test_change_password_policy.py uses).
_route_path = _BACKEND_ROOT / "routes" / "auth.py"
_spec = importlib.util.spec_from_file_location("_auth_rate_limit_routes", _route_path)
assert _spec is not None and _spec.loader is not None
auth_routes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(auth_routes)


class _Clock:
    """Injectable clock: advance() moves time without sleeping."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeRequest:
    """Just enough of ``Request`` for ``client_key``: a client address and headers."""

    def __init__(self, client_host: str = "127.0.0.1", headers: dict | None = None) -> None:
        self.client = SimpleNamespace(host = client_host)
        self.headers = headers or {}


@pytest.fixture(autouse = True)
def _isolate_token_attempts():
    """The routes share one process-wide limiter; give every test an empty one."""
    token_attempts.reset()
    yield
    token_attempts.reset()


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def limiter(clock):
    # Small numbers so the policy is driven, not configured around: 3 failures in a 60 s window
    # buy a 30 s ban.
    return AttemptLimiter(max_attempts = 3, window_s = 60.0, lockout_s = 30.0, clock = clock)


# --- limiter policy on the injected clock -------------------------------------------------


def test_failures_below_the_limit_are_allowed(limiter):
    limiter.record_failure("a")
    limiter.record_failure("a")
    assert limiter.allow("a") == (True, 0.0)


def test_the_max_attempts_th_failure_inside_the_window_locks_the_key(limiter):
    limiter.record_failure("a")
    limiter.record_failure("a")
    assert limiter.allow("a") == (True, 0.0)
    limiter.record_failure("a")
    allowed, retry_after = limiter.allow("a")
    assert allowed is False
    assert 0.0 < retry_after <= 30.0


def test_a_locked_key_reports_a_shrinking_retry_after(limiter, clock):
    for _ in range(3):
        limiter.record_failure("a")
    _allowed, first = limiter.allow("a")
    clock.advance(10.0)
    allowed, retry_after = limiter.allow("a")
    assert allowed is False
    assert retry_after == first - 10.0


def test_the_lockout_lasts_exactly_lockout_s(limiter, clock):
    for _ in range(3):
        limiter.record_failure("a")
    clock.advance(29.5)
    assert limiter.allow("a")[0] is False
    clock.advance(0.5)
    # Exactly 30 s since the failure that started the ban: admitted, with nothing to wait for.
    assert limiter.allow("a") == (True, 0.0)


def test_retrying_a_ban_does_not_extend_it(limiter, clock):
    for _ in range(3):
        limiter.record_failure("a")
    for _ in range(20):
        assert limiter.allow("a")[0] is False
        # A request that raced past the check and lost must not stretch the ban either.
        limiter.record_failure("a")
        clock.advance(0.25)
    clock.advance(25.0)
    # 30 s total since the ban began, despite 20 checks and 20 failures during it.
    assert limiter.allow("a") == (True, 0.0)


def test_a_success_clears_the_key(limiter):
    limiter.record_failure("a")
    limiter.record_failure("a")
    limiter.record_success("a")
    assert limiter.allow("a") == (True, 0.0)
    # The count really reset: the full budget is owed again before the next lock.
    limiter.record_failure("a")
    limiter.record_failure("a")
    assert limiter.allow("a") == (True, 0.0)
    limiter.record_failure("a")
    assert limiter.allow("a")[0] is False


def test_failures_outside_the_window_do_not_add_up(limiter, clock):
    limiter.record_failure("a")
    limiter.record_failure("a")
    clock.advance(61.0)
    # Both failures have aged out of the 60 s window, so the count restarts at one.
    limiter.record_failure("a")
    limiter.record_failure("a")
    assert limiter.allow("a") == (True, 0.0)
    limiter.record_failure("a")
    assert limiter.allow("a")[0] is False


def test_the_key_bound_evicts_the_oldest_key(clock):
    limiter = AttemptLimiter(max_attempts = 2, window_s = 60.0, lockout_s = 30.0, clock = clock)
    limiter.record_failure("victim")  # one short of locking
    for i in range(rate_limit._MAX_KEYS):
        limiter.record_failure(f"spray-{i}")
    assert len(limiter._entries) == rate_limit._MAX_KEYS
    # The victim (least recently seen) was evicted: its partial count is gone, so the next failure
    # counts as the first, not the second that would have locked.
    limiter.record_failure("victim")
    assert limiter.allow("victim") == (True, 0.0)
    # Re-entering displaced the least recently seen spray key instead of growing past the bound.
    assert len(limiter._entries) == rate_limit._MAX_KEYS


# --- client key derivation ----------------------------------------------------------------


def test_client_key_prefers_the_trusted_forwarded_address():
    # Through the managed tunnel the socket peer is the local cloudflared process; utils.client_ip
    # resolves the real visitor from CF-Connecting-IP for a loopback peer.
    request = _FakeRequest("127.0.0.1", {"cf-connecting-ip": "203.0.113.9"})
    assert client_key(request) == "203.0.113.9"


def test_client_key_falls_back_to_the_socket_peer():
    # A direct LAN caller cannot spoof its way into another bucket with a header.
    request = _FakeRequest("192.168.1.7", {"cf-connecting-ip": "203.0.113.9"})
    assert client_key(request) == "192.168.1.7"


def test_client_key_without_a_request_uses_the_fallback_constant():
    assert client_key(None) == "_unknown"


# --- the routes, with a faked credential check ---------------------------------------------


_GOOD_REFRESH_TOKEN = "good-refresh-token"


@pytest.fixture
def refresh_client(monkeypatch):
    """The /refresh route with the credential check faked at its seam: the real rotation does
    SQLite work and hash compares, while the limiter only ever sees accept or reject."""

    def fake_rotate_refresh_token(token, new_token, new_expires_at):
        if token == _GOOD_REFRESH_TOKEN:
            return SimpleNamespace(
                username = storage.DEFAULT_ADMIN_USERNAME,
                is_desktop = False,
                jwt_secret = "jwt-secret-0123456789abcdef0123456789abcdef",
                must_change_password = False,
                account_id = None,
            )
        return None

    monkeypatch.setattr(auth_routes.storage, "rotate_refresh_token", fake_rotate_refresh_token)
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    return TestClient(app)


def _refresh(client, token: str = "guess"):
    return client.post("/api/auth/refresh", json = {"refresh_token": token})


def test_rejected_refresh_tokens_keep_their_old_answer(refresh_client):
    for _ in range(3):
        response = _refresh(refresh_client)
        assert response.status_code == 401
        # The pre-existing failure answer, byte for byte.
        assert response.json() == {"detail": "Invalid or expired refresh token"}


def test_the_next_refresh_guess_is_answered_with_429_and_retry_after(refresh_client):
    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS):
        assert _refresh(refresh_client).status_code == 401
    response = _refresh(refresh_client)
    assert response.status_code == 429
    wait = int(response.headers["Retry-After"])
    assert 1 <= wait <= rate_limit._TOKEN_LOCKOUT_S
    assert response.json() == {
        "detail": f"Too many failed attempts. Try again in {wait} seconds."
    }


def test_a_refresh_success_in_between_resets_the_count(refresh_client):
    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS - 1):
        assert _refresh(refresh_client).status_code == 401
    ok = _refresh(refresh_client, _GOOD_REFRESH_TOKEN)
    assert ok.status_code == 200
    assert ok.json()["access_token"]
    # Without the reset the next batch would trip the limit partway through.
    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS):
        assert _refresh(refresh_client).status_code == 401
    assert _refresh(refresh_client).status_code == 429


def test_clients_behind_different_addresses_do_not_share_a_budget(refresh_client, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_TRUST_FORWARDED", "1")

    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS + 1):
        response = refresh_client.post(
            "/api/auth/refresh",
            json = {"refresh_token": "guess"},
            headers = {"X-Forwarded-For": "203.0.113.7"},
        )
    assert response.status_code == 429
    other = refresh_client.post(
        "/api/auth/refresh",
        json = {"refresh_token": "guess"},
        headers = {"X-Forwarded-For": "203.0.113.8"},
    )
    # A different forwarded client keeps its own budget: one banned source cannot lock the rest.
    assert other.status_code == 401


def _patch_password_change(monkeypatch, current_password: str) -> None:
    monkeypatch.setattr(
        auth_routes.storage,
        "get_user_and_secret",
        lambda username: ("salt", "hash", "jwt-secret", False),
    )
    monkeypatch.setattr(
        auth_routes.hashing,
        "verify_password",
        lambda password, salt, pwd_hash: password == current_password,
    )
    monkeypatch.setattr(auth_routes.storage, "update_password", lambda *a, **k: "rotated-secret")
    monkeypatch.setattr(auth_routes, "create_access_token", lambda subject, **kwargs: "at")
    monkeypatch.setattr(auth_routes, "create_refresh_token", lambda subject, **kwargs: "rt")


def _change_password(request, current_password: str):
    payload = ChangePasswordRequest(
        current_password = current_password,
        new_password = "brand-new-password",
    )
    return asyncio.run(
        auth_routes.change_password(payload, request, storage.DEFAULT_ADMIN_USERNAME, False)
    )


def test_change_password_guesses_keep_their_401_then_get_429(monkeypatch):
    _patch_password_change(monkeypatch, current_password = "current-password-1")
    request = _FakeRequest("203.0.113.7")
    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS):
        with pytest.raises(HTTPException) as excinfo:
            _change_password(request, "wrong-password-1")
        assert excinfo.value.status_code == 401
        assert excinfo.value.detail == "Current password is incorrect"
    with pytest.raises(HTTPException) as excinfo:
        _change_password(request, "wrong-password-1")
    assert excinfo.value.status_code == 429
    wait = int(excinfo.value.headers["Retry-After"])
    assert 1 <= wait <= rate_limit._TOKEN_LOCKOUT_S
    assert excinfo.value.detail == f"Too many failed attempts. Try again in {wait} seconds."


def test_a_change_password_success_resets_the_count(monkeypatch):
    _patch_password_change(monkeypatch, current_password = "current-password-1")
    request = _FakeRequest("203.0.113.7")
    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS - 1):
        with pytest.raises(HTTPException):
            _change_password(request, "wrong-password-1")
    token = _change_password(request, "current-password-1")
    assert token.access_token == "at"
    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS):
        with pytest.raises(HTTPException) as excinfo:
            _change_password(request, "wrong-password-1")
        assert excinfo.value.status_code == 401
