# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The keyless tool-policy middleware must not pay for an answer that cannot change.

`KeylessToolPolicyMiddleware` ran `run_in_threadpool(asgi_request_is_keyless, ...)` on EVERY request.
With keyless access off -- the shipped default -- the answer is decided by `scope == KEYLESS_SCOPE_OFF`
before the only statement in that function that can reach SQLite (the account counts), so the hop was
pure cost: measured at 3.59 ms of a 14 ms request when the middleware was removed from the stack in
`tests/perf_harness/measure_middleware_cost.py`.

These tests pin both halves of the new contract: no hop when keyless is off, and the hop kept when it
is on (which is where the off-loop SQLite read still matters).
"""

from __future__ import annotations

import asyncio

import pytest

from utils import keyless_api_access as keyless


class _Recorder:
    """A minimal ASGI app that records whether it was called."""

    def __init__(self) -> None:
        self.called = False

    async def __call__(self, scope, receive, send):
        self.called = True


def _scope(method = "GET", path = "/api/models/local"):
    return {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": "",
        "headers": [],
        "state": {},
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 5001),
        "scheme": "http",
        "query_string": b"",
    }


@pytest.fixture
def no_threadpool(monkeypatch):
    """Fail loudly if the middleware hops to a worker thread."""
    calls: list[object] = []
    import starlette.concurrency as concurrency

    async def forbidden(*args, **kwargs):
        calls.append(args[0] if args else kwargs)
        return False

    monkeypatch.setattr(concurrency, "run_in_threadpool", forbidden)
    return calls


@pytest.fixture
def keyless_off(monkeypatch):
    async def settings():
        return keyless.KEYLESS_SCOPE_OFF, False, 0

    monkeypatch.setattr(keyless, "_settings_async", settings)
    monkeypatch.setattr(keyless, "_settings_write_inflight", None, raising = False)
    monkeypatch.setattr(keyless, "_settings_generation", 0, raising = False)


@pytest.fixture
def keyless_full(monkeypatch):
    async def settings():
        return keyless.KEYLESS_SCOPE_FULL, False, 0

    monkeypatch.setattr(keyless, "_settings_async", settings)
    monkeypatch.setattr(keyless, "_settings_write_inflight", None, raising = False)
    monkeypatch.setattr(keyless, "_settings_generation", 0, raising = False)


def test_keyless_off_answers_without_a_worker_thread(keyless_off, no_threadpool):
    app = _Recorder()
    middleware = keyless.KeylessToolPolicyMiddleware(app)
    asyncio.run(middleware(_scope(), None, None))

    assert no_threadpool == [], (
        "with keyless access off the admission answer is constant, so the request must not hop "
        f"to a thread; calls were {no_threadpool}"
    )
    assert app.called is True
    assert _scope() is not None  # the scope stays usable


def test_keyless_off_still_publishes_a_false_admission(keyless_off, no_threadpool):
    scope = _scope()
    middleware = keyless.KeylessToolPolicyMiddleware(_Recorder())
    asyncio.run(middleware(scope, None, None))
    assert scope["state"][keyless.KEYLESS_ADMISSION_STATE_KEY] is False


def test_keyless_on_keeps_the_off_loop_admission_check(keyless_full, no_threadpool):
    app = _Recorder()
    middleware = keyless.KeylessToolPolicyMiddleware(app)
    asyncio.run(middleware(_scope(), None, None))

    assert len(no_threadpool) == 1, (
        "with keyless access on, admission reaches the account counts, which read SQLite; that "
        "lookup must stay off the event loop"
    )
    assert app.called is True


def test_a_non_http_scope_passes_straight_through(no_threadpool):
    app = _Recorder()
    middleware = keyless.KeylessToolPolicyMiddleware(app)
    asyncio.run(middleware({"type": "lifespan"}, None, None))

    assert app.called is True
    assert no_threadpool == []
