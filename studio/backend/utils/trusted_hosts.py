# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional Host allow-list, off unless ``UNSLOTH_TRUSTED_HOSTS`` is set.

A page on ``evil.example`` rebound to ``127.0.0.1:8888`` is same-origin with the
backend, so it can read every unauthenticated ``/api`` route. The narrower
transport checks in ``keyless_api_access`` already refuse the serious versions
(bootstrap-password injection, keyless admission), which is why this stays
default-off: LAN, tunnel and Colab modes legitimately carry many Host values,
and turning a host check on by default would break them for a smaller gap than
it closes. Setting the variable opts in.

``UNSLOTH_TRUSTED_HOSTS`` is a comma-separated list of ``host[:port]`` patterns.
A leading ``*`` matches subdomains (``*.example`` matches ``a.example`` and
``b.a.example`` but not ``example``); a bare ``*`` matches every host. A pattern
with a port only matches that port; one without a port matches any port.
Matching is case-insensitive and ignores a trailing root dot. Loopback literals
and ``localhost`` names are always allowed, whatever the list says, so an
operator cannot lock themselves out of their own machine.

Pure ASGI, not ``BaseHTTPMiddleware``, matching ``SecurityHeadersMiddleware``,
so streaming responses are not wrapped in an anyio stream. WebSocket and other
non-http scopes pass through untouched.
"""

from __future__ import annotations

import ipaddress
import os

TRUSTED_HOSTS_ENV = "UNSLOTH_TRUSTED_HOSTS"


def _split_authority(value: str) -> tuple[str, int | None] | None:
    """Split a Host header or pattern into a lowercase host and numeric port."""
    raw = value.strip().lower()
    if not raw:
        return None
    port_text = ""
    if raw.startswith("["):
        end = raw.find("]")
        if end == -1:
            return None
        host = raw[1:end]
        suffix = raw[end + 1 :]
        if suffix:
            if not suffix.startswith(":"):
                return None
            port_text = suffix[1:]
    else:
        if raw.count(":") > 1:  # unbracketed IPv6 is not a legal authority
            return None
        host, separator, port_text = raw.partition(":")
        if not separator:
            port_text = ""
    host = host.rstrip(".")
    if not host:
        return None
    port = None
    if port_text:
        if not port_text.isdigit():
            return None
        port = int(port_text)
        if not 1 <= port <= 65535:
            return None
    return host, port


def _split_pattern(value: str) -> tuple[str, int | None, bool] | None:
    """Parse one allow-list entry into host, port and whether it is a subdomain wildcard."""
    raw = value.strip().lower()
    if not raw:
        return None
    wildcard = False
    if raw == "*":
        return "", None, True
    if raw.startswith("*"):
        wildcard = True
        raw = raw[1:]
        if raw.startswith("."):
            raw = raw[1:]
    authority = _split_authority(raw)
    if authority is None:
        return None
    host, port = authority
    return host, port, wildcard


def _is_loopback_or_localhost(host: str) -> bool:
    """Whether ``host`` is a loopback literal or a ``localhost`` name."""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or (mapped is not None and mapped.is_loopback)


def _matches(
    host: str,
    port: int | None,
    patterns: tuple[tuple[str, int | None, bool], ...],
) -> bool:
    for pattern_host, pattern_port, wildcard in patterns:
        if pattern_port is not None and pattern_port != port:
            continue
        if pattern_host == "":
            return True
        if wildcard:
            if host.endswith("." + pattern_host):
                return True
        elif host == pattern_host:
            return True
    return False


def _refusal(host_header: str | None) -> bytes:
    if host_header is None:
        detail = f"the request has no Host header; add one to {TRUSTED_HOSTS_ENV}"
    else:
        detail = f"{host_header!r} is not trusted; add it to {TRUSTED_HOSTS_ENV}"
    return f"400 Bad Request: Host {detail}.\n".encode("utf-8")


class TrustedHostsMiddleware:
    """Refuse an unmatched Host with a plain 400, but only when the env var is set."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        configured = os.environ.get(TRUSTED_HOSTS_ENV, "")
        if not configured.strip():
            await self.app(scope, receive, send)
            return
        patterns = tuple(
            parsed
            for parsed in (_split_pattern(item) for item in configured.split(","))
            if parsed is not None
        )

        host_header = None
        for name, value in scope.get("headers") or ():
            if name.lower() == b"host":
                host_header = bytes(value).decode("latin-1")
                break
        authority = _split_authority(host_header) if host_header is not None else None
        if authority is not None:
            host, port = authority
            if _is_loopback_or_localhost(host) or _matches(host, port, patterns):
                await self.app(scope, receive, send)
                return

        body = _refusal(host_header)
        await send(
            {
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
