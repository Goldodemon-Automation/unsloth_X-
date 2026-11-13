# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bind-host resolver must not run a DNS lookup per request.

`ResearchPortMiddleware` calls `note_server_address` on every HTTP request, which reaches
`scope_request_host` -> `wildcard_loopback_host` -> `wildcard_ip_versions` ->
`_resolved_ip_addresses`. For any ASGI server host that is not an IP literal that used to be an
uncached `socket.getaddrinfo` on the event loop: measured 2.29 s per request while the resolver
timed out, and nothing else could run meanwhile. These tests pin the cache that fixes it, and pin
that the answers themselves are unchanged.
"""

from __future__ import annotations

import ipaddress
import socket

import pytest

from utils import host_policy


@pytest.fixture(autouse = True)
def _clear_cache():
    host_policy.clear_resolution_cache()
    yield
    host_policy.clear_resolution_cache()


@pytest.fixture
def resolve_counter(monkeypatch):
    """Count real resolver calls, and answer with a fixed address table."""
    calls: list[str] = []

    def fake_getaddrinfo(host, _port, *args, **kwargs):
        calls.append(host)
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.10", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:db8::10", 0, 0, 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    return calls


def test_repeated_requests_resolve_a_hostname_once(resolve_counter):
    for _ in range(50):
        host_policy.wildcard_ip_versions("studio-box.example")
    assert resolve_counter == ["studio-box.example"], (
        "a hostname on the request path must be resolved once per TTL, not once per request; "
        f"got {len(resolve_counter)} resolver calls for 50 requests"
    )


def test_the_cache_expires_so_a_renamed_host_is_re_resolved(resolve_counter, monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(host_policy.time, "monotonic", lambda: clock["now"])

    host_policy.wildcard_ip_versions("studio-box.example")
    host_policy.wildcard_ip_versions("studio-box.example")
    assert len(resolve_counter) == 1

    clock["now"] += host_policy._RESOLUTION_TTL_SECONDS + 0.5
    host_policy.wildcard_ip_versions("studio-box.example")
    assert len(resolve_counter) == 2


def test_a_name_that_does_not_resolve_is_cached_too(monkeypatch):
    calls: list[str] = []

    def failing_getaddrinfo(host, *_args, **_kwargs):
        calls.append(host)
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", failing_getaddrinfo)

    for _ in range(20):
        assert host_policy.wildcard_ip_versions("does-not-exist.invalid") == ()
    assert calls == ["does-not-exist.invalid"], (
        "a name that fails to resolve must not cost a resolver timeout on every request"
    )


def test_ip_literals_never_reach_the_resolver(resolve_counter):
    for _ in range(10):
        host_policy.wildcard_ip_versions("0.0.0.0")
        host_policy.wildcard_ip_versions("::")
        host_policy.wildcard_ip_versions("127.0.0.1")
    assert resolve_counter == []


def test_the_cache_is_bounded(monkeypatch):
    calls: list[str] = []

    def fake_getaddrinfo(host, *_args, **_kwargs):
        calls.append(host)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    for index in range(host_policy._RESOLUTION_CACHE_MAX + 25):
        host_policy.wildcard_ip_versions(f"host-{index}.example")
    assert len(host_policy._resolution_cache) == host_policy._RESOLUTION_CACHE_MAX


def test_a_cached_answer_still_says_what_the_uncached_one_said(monkeypatch):
    """The cache may not change the verdict: wildcard detection, loopback choice, escaping."""
    table = {
        "wildcard-box.example": ("0.0.0.0",),
        "loopback-box.example": ("127.0.0.1",),
        "v6-wildcard.example": ("::",),
    }

    def fake_getaddrinfo(host, *_args, **_kwargs):
        addresses = table.get(host, ())
        if not addresses:
            raise socket.gaierror(socket.EAI_NONAME, "no such host")
        return [
            (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))
            for address in addresses
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    first = {
        name: (
            host_policy.wildcard_ip_versions(name),
            host_policy.wildcard_loopback_host(name),
            host_policy.is_wildcard_host(name),
            host_policy.scope_request_host((name, 1)),
        )
        for name in table
    }
    second = {
        name: (
            host_policy.wildcard_ip_versions(name),
            host_policy.wildcard_loopback_host(name),
            host_policy.is_wildcard_host(name),
            host_policy.scope_request_host((name, 1)),
        )
        for name in table
    }
    assert first == second
    assert first["wildcard-box.example"][1] == "127.0.0.1"
    # A non-wildcard name is passed through unchanged: scope_request_host keeps the accepted
    # address and only substitutes loopback for a wildcard bind.
    assert first["loopback-box.example"][3] == "loopback-box.example"
    assert first["v6-wildcard.example"][0] == (6,)


def test_mapped_addresses_keep_their_ipv4_normalisation(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::ffff:0.0.0.0", 0, 0, 0))
        ],
    )
    assert host_policy.wildcard_ip_versions("mapped.example") == (4,)
    assert host_policy.wildcard_ip_versions("mapped.example") == (4,)


def test_the_request_path_helper_reads_the_cache_not_the_resolver(resolve_counter):
    """`scope_request_host` is the function the middleware actually calls."""
    for _ in range(30):
        host_policy.scope_request_host(("studio-box.example", 5001))
    assert len(resolve_counter) == 1


def test_clearing_the_cache_forces_the_next_answer_to_be_resolved_again(resolve_counter):
    host_policy.wildcard_ip_versions("studio-box.example")
    host_policy.clear_resolution_cache()
    host_policy.wildcard_ip_versions("studio-box.example")
    assert len(resolve_counter) == 2


def test_resolved_addresses_are_normalised_not_raw():
    """A guard against caching the wrong layer: results are ipaddress objects, deduplicated."""
    resolved = host_policy._resolved_ip_addresses("127.0.0.1")
    assert resolved == (ipaddress.ip_address("127.0.0.1"),)
