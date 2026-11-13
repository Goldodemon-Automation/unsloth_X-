# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Host allow-list, on by default and overridable with ``UNSLOTH_TRUSTED_HOSTS``.

A page on ``evil.example`` rebound to ``127.0.0.1:8888`` is same-origin with the
backend, so it can read every unauthenticated ``/api`` route and drive the
login/bootstrap ones. The narrower transport checks in ``keyless_api_access``
already refuse the serious versions, but a Host check is cheap and this is the
control that closes the rest, so it ships on rather than opt-in.

The default list is the set of hosts this backend legitimately answers to:
loopback, ``localhost``, the documented Tauri desktop origins, a Cloudflare
quick tunnel's own host and Colab's kernel proxy. Private and link-local
address literals and mDNS ``*.local`` names pass too, so a LAN launch keeps
working without configuration. The two in-process hosts stay listed as well:
``testserver`` (Starlette's TestClient) and ``unsloth-mcp.invalid``, which every
MCP tool call is replayed under through this very app.

``UNSLOTH_TRUSTED_HOSTS`` overrides that list wholesale -- a reverse proxy on a
real name sets it and the defaults stop applying. It is a comma-separated list
of ``host[:port]`` patterns.
A leading ``*`` matches subdomains (``*.example`` matches ``a.example`` and
``b.a.example`` but not ``example``); a bare ``*`` matches every host. A pattern
with a port only matches that port; one without a port matches any port.
Matching is case-insensitive and ignores a trailing root dot. Loopback literals
and ``localhost`` names are always allowed, whatever the list says, so an
operator cannot lock themselves out of their own machine. A ``Host`` that
matches nothing is refused with a 400 that names the host and the variable.

Pure ASGI, not ``BaseHTTPMiddleware``, matching ``SecurityHeadersMiddleware``,
so streaming responses are not wrapped in an anyio stream. WebSocket and other
non-http scopes pass through untouched.
"""

from __future__ import annotations

import ipaddress
import os

TRUSTED_HOSTS_ENV = "UNSLOTH_TRUSTED_HOSTS"

# Shipped in place of the variable being unset. ``localhost`` and the loopback literals are already
# always allowed by ``_is_loopback_or_localhost``; they are repeated here so the list reads as the
# whole policy. ``*.localhost`` also covers ``tauri.localhost`` (the Windows WebView2 origin, beside
# ``tauri://localhost`` on macOS and Linux), but the exact name stays for the reader.
_DEFAULT_PATTERNS = (
    "localhost",
    "*.localhost",
    "tauri.localhost",
    # Starlette's TestClient host, and the host the MCP forwarder replays its in-process requests
    # under (studio_mcp/forward.py): every tool call reaches the routes through this app.
    "testserver",
    "unsloth-mcp.invalid",
    # A quick tunnel is published under a random name in this domain and cloudflared forwards the
    # original Host, so the tunnel URL is the Host the browser sends.
    "*.trycloudflare.com",
    # Colab serves the notebook's proxy port from this domain.
    "*.googleusercontent.com",
    # mDNS names (``macbook.local``) are how a LAN is browsed from macOS and from some routers; the
    # suffix is mDNS-only, so an internet page cannot claim one.
    "*.local",
)


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


# Parsed once: the defaults are a constant and ``_matches`` takes parsed entries.
_DEFAULT_PARSED = tuple(
    parsed for parsed in (_split_pattern(item) for item in _DEFAULT_PATTERNS) if parsed is not None
)


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


def _is_lan_literal(host: str) -> bool:
    """Whether ``host`` is a private or link-local address literal.

    A LAN launch publishes the interface address and the browser then sends exactly that as the
    Host, so refusing it would break the mode. Only literals: a name still has to be listed, which
    is the rebinding case this list exists for.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return (mapped if mapped is not None else address).is_private


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
    """Refuse an unmatched Host with a plain 400, against the default list or the override."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        configured = os.environ.get(TRUSTED_HOSTS_ENV, "")
        defaults_in_force = not configured.strip()
        if defaults_in_force:
            patterns = _DEFAULT_PARSED
        else:
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
            # A LAN literal only rides along with the defaults: an operator who set a list has named
            # the addresses their backend answers to, and the list wins over the guess.
            if (
                _is_loopback_or_localhost(host)
                or _matches(host, port, patterns)
                or (defaults_in_force and _is_lan_literal(host))
            ):
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
