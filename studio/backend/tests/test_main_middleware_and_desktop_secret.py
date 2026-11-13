# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two regressions in `main.py`: the security-header middleware behind a path prefix, and the
unthrottled PBKDF2 behind the desktop secret header.

Both findings come from SECURITY_AUDIT.md (SEC-5 and SEC-4). `main` is imported inside the tests
because importing it boots the whole backend stack; doing it at module scope would charge every
collection of this file, and only these two tests need it.
"""

from __future__ import annotations

import asyncio

import pytest


def _headers_for(scope_path: str, *, root_path: str = "", frame_shell: bool = False):
    """Run SecurityHeadersMiddleware over one fake request and return the response headers."""
    import main

    captured: dict[str, list] = {}

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = main.SecurityHeadersMiddleware(app)
    scope = {
        "type": "http",
        "method": "GET",
        "path": scope_path,
        "root_path": root_path,
        "headers": [],
    }

    async def send(message):
        if message["type"] == "http.response.start":
            captured["headers"] = [
                (name.decode("latin-1").lower(), value.decode("latin-1"))
                for name, value in message["headers"]
            ]

    asyncio.run(middleware(scope, lambda: None, send))
    return dict(captured["headers"])


def test_an_api_read_behind_a_path_prefix_still_gets_no_store():
    """`--root-path /x` leaves `/x/api/...` in scope["path"]; the /api no-store rule must still fire."""
    headers = _headers_for("/x/api/models/local", root_path = "/x")
    assert headers.get("cache-control") == "no-store", (
        "the /api no-store rule matched on the raw path and missed behind --root-path; "
        f"headers were {headers}"
    )


def test_an_api_read_without_a_prefix_is_unchanged():
    headers = _headers_for("/api/models/local")
    assert headers.get("cache-control") == "no-store"


def test_a_framed_shell_keeps_its_x_frame_options_exemption_behind_a_prefix():
    """The artifact preview frame must not be handed X-Frame-Options: DENY under a prefix either."""
    import main

    path = next(iter(main._FRAME_SHELL_PATHS))
    framed = _headers_for(path, root_path = "/x")
    assert "x-frame-options" not in framed, (
        f"{path} is a frame shell and must stay embeddable behind --root-path; got {framed}"
    )
    plain = _headers_for("/api/settings/hub", root_path = "/x")
    assert plain.get("x-frame-options") == "DENY"


def test_the_docs_csp_exemption_follows_the_prefix():
    import main

    docs_path = next(iter(main._DOCS_PATHS))
    prefixed = _headers_for(f"/x{docs_path}", root_path = "/x")
    plain = _headers_for("/api/models/local", root_path = "/x")
    assert prefixed.get("content-security-policy") != plain.get("content-security-policy"), (
        "the docs page gets its own CSP; behind a prefix it was built as an ordinary page"
    )


@pytest.fixture
def desktop_secret(monkeypatch):
    """A well-formed secret that never validates, with the shared credential budget reset."""
    import main
    from auth import rate_limit, storage

    rate_limit.token_attempts.reset()
    monkeypatch.setattr(storage, "validate_desktop_secret", lambda _secret: None)
    yield "desktop-" + "0" * 64
    rate_limit.token_attempts.reset()


def _request(secret: str):
    from starlette.requests import Request

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/desktop/shutdown",
            "headers": [(b"x-desktop-secret", secret.encode())],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 5001),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_the_desktop_secret_path_stops_spending_the_kdf_after_repeated_failures(desktop_secret):
    """Every guess costs 100k PBKDF2 rounds, so the budget has to close before the KDF, not after."""
    import main
    from auth import rate_limit
    from fastapi import HTTPException

    codes: list[int] = []
    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS + 3):
        try:
            asyncio.run(main._desktop_shell_subject(_request(desktop_secret)))
        except HTTPException as exc:
            codes.append(exc.status_code)

    assert codes.count(401) == rate_limit._TOKEN_MAX_ATTEMPTS
    assert codes[-1] == 429, f"the attempt after the budget was exhausted must be throttled, got {codes}"
    assert 429 in codes


def test_a_correct_secret_clears_the_budget(desktop_secret, monkeypatch):
    import main
    from auth import rate_limit, storage
    from fastapi import HTTPException

    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS - 1):
        with pytest.raises(HTTPException):
            asyncio.run(main._desktop_shell_subject(_request(desktop_secret)))

    monkeypatch.setattr(storage, "validate_desktop_secret", lambda _secret: "credential")
    assert asyncio.run(main._desktop_shell_subject(_request(desktop_secret))) == (
        storage.DEFAULT_ADMIN_USERNAME
    )

    # A success disproves guessing pressure, so the key starts clean again.
    monkeypatch.setattr(storage, "validate_desktop_secret", lambda _secret: None)
    for _ in range(rate_limit._TOKEN_MAX_ATTEMPTS - 1):
        with pytest.raises(HTTPException):
            asyncio.run(main._desktop_shell_subject(_request(desktop_secret)))


def test_a_request_without_the_header_is_not_charged_to_the_budget(desktop_secret):
    import main
    from starlette.requests import Request

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/desktop/shutdown",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 5001),
            "scheme": "http",
            "query_string": b"",
        }
    )
    assert asyncio.run(main._desktop_shell_subject(request)) is None
