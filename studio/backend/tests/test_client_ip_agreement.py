# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One client-IP trust model, shared by every path that keys on the caller.

``routes.auth`` used to carry its own resolver that took the LEFTMOST ``X-Forwarded-For`` hop
(client-controlled) while ``utils.client_ip`` -- and with it the shared credential lockout --
took the rightmost. Behind a proxy that appends a hop those two names referred to different
people for the same request, so one bucket could be evaded by taking the other path and an
innocent client could be 429'd by a stranger's failures.

Each test therefore asserts the SAME address from all of the paths that key on a client, rather
than pinning any single one:

  * ``utils.client_ip.client_ip`` -- the resolver,
  * ``routes.auth._client_ip`` -- the login limiter's entry point,
  * ``auth.rate_limit.client_key`` -- the shared credential lockout,
  * ``routes.auth._bucket_key`` / ``_unknown_user_key`` -- the login buckets themselves.
"""

import sys
import types
from pathlib import Path

import pytest
from starlette.datastructures import Headers

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from auth.rate_limit import client_key
from routes import auth as auth_routes
from utils.client_ip import client_ip


class _Req:
    def __init__(self, peer, headers = None):
        self.client = types.SimpleNamespace(host = peer) if peer else None
        self.headers = Headers(headers or {})


@pytest.fixture
def trust_off(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)


@pytest.fixture
def trust_on(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_TRUST_FORWARDED", "1")


def _every_path(request) -> set[str]:
    """Every address that some throttle would key ``request`` on."""
    return {
        client_ip(request),
        auth_routes._client_ip(request),
        client_key(request),
        auth_routes._bucket_key(request, "alice")[0],
        auth_routes._unknown_user_key(request)[0],
    }


def _assert_agreed(request, expected: str) -> None:
    seen = _every_path(request)
    assert seen == {expected}, f"{request.headers} resolved differently per path: {seen}"


# ── the five layouts that matter ────────────────────────────────────────────


def test_a_direct_request_keys_on_the_socket_peer(trust_off):
    # No proxy in the path: a header here is the caller's own invention.
    _assert_agreed(_Req("203.0.113.5", {"x-forwarded-for": "198.51.100.7"}), "203.0.113.5")


def test_a_single_hop_proxy_keys_on_the_hop_its_proxy_appended(trust_on):
    _assert_agreed(_Req("127.0.0.1", {"x-forwarded-for": "198.51.100.7"}), "198.51.100.7")
    _assert_agreed(_Req("10.0.0.4", {"x-forwarded-for": "198.51.100.7"}), "198.51.100.7")


def test_a_two_hop_chain_keys_on_the_visitor_not_the_last_proxy(trust_on):
    # proxy -> LAN ingress -> caller: the rightmost hop (10.0.0.1) is ours, not the visitor.
    _assert_agreed(_Req("127.0.0.1", {"x-forwarded-for": "198.51.100.7, 10.0.0.1"}), "198.51.100.7")
    # Two external hops: the rightmost is still the one the trusted proxy appended.
    _assert_agreed(_Req("127.0.0.1", {"x-forwarded-for": "1.2.3.4, 198.51.100.7"}), "198.51.100.7")


def test_a_spoofed_xff_from_a_non_loopback_peer_keys_on_the_peer(trust_off):
    # A direct LAN caller cannot pick its bucket by inventing a chain.
    _assert_agreed(_Req("203.0.113.9", {"x-forwarded-for": "1.2.3.4, 10.0.0.1"}), "203.0.113.9")
    _assert_agreed(_Req("203.0.113.9", {"cf-connecting-ip": "198.51.100.7"}), "203.0.113.9")


def test_a_tunnel_style_cf_header_keys_on_the_visitor(trust_off):
    # cloudflared terminates on loopback; the peer is the tunnel, CF-Connecting-IP is the visitor.
    _assert_agreed(_Req("127.0.0.1", {"cf-connecting-ip": "198.51.100.7"}), "198.51.100.7")
    _assert_agreed(_Req("::1", {"cf-connecting-ip": "2001:db8::1"}), "2001:db8::1")


# ── the rule itself ────────────────────────────────────────────────────────


def test_forwarded_and_xff_are_parsed_by_the_same_rules(trust_on):
    xff = _Req("127.0.0.1", {"x-forwarded-for": "198.51.100.7, 10.0.0.1"})
    forwarded = _Req(
        "127.0.0.1", {"forwarded": "for=198.51.100.7, for=10.0.0.1;proto=https"}
    )
    assert client_ip(xff) == client_ip(forwarded) == "198.51.100.7"


def test_a_spoofed_leftmost_hop_cannot_move_the_bucket(trust_on):
    # The proxy appends the same real visitor; only the client-controlled leftmost entry differs,
    # so both requests must land in one bucket (this is the evasion the split models allowed).
    first = _Req("127.0.0.1", {"x-forwarded-for": "203.0.113.50, 198.51.100.7, 10.0.0.1"})
    second = _Req("127.0.0.1", {"x-forwarded-for": "192.0.2.99, 198.51.100.7, 10.0.0.1"})
    _assert_agreed(first, "198.51.100.7")
    _assert_agreed(second, "198.51.100.7")


def test_garbage_hops_are_skipped_and_a_bare_chain_falls_back_to_the_peer(trust_on):
    _assert_agreed(_Req("127.0.0.1", {"x-forwarded-for": "not-an-ip"}), "127.0.0.1")
    _assert_agreed(_Req("127.0.0.1", {"forwarded": "for=unknown"}), "127.0.0.1")
    # A valid hop behind a garbage one is still found from the right.
    _assert_agreed(_Req("127.0.0.1", {"x-forwarded-for": "junk, 198.51.100.7"}), "198.51.100.7")


def test_ports_and_brackets_are_stripped_identically(trust_on):
    _assert_agreed(_Req("127.0.0.1", {"x-forwarded-for": "198.51.100.7:50001"}), "198.51.100.7")
    _assert_agreed(
        _Req("127.0.0.1", {"x-forwarded-for": "[2001:db8::1]:50001"}), "2001:db8::1"
    )
    _assert_agreed(
        _Req("127.0.0.1", {"forwarded": 'for="[2001:db8::1]:50001";proto=https'}), "2001:db8::1"
    )


def test_an_addressless_request_is_unknown_everywhere(trust_off):
    _assert_agreed(_Req(None), "_unknown")
    # The utils resolver keeps its None-request contract; the login paths pass it through too.
    assert client_ip(None) == "_unknown"
    assert auth_routes._client_ip(None) == "_unknown"
    assert client_key(None) == "_unknown"


def test_a_loopback_peer_without_a_tunnel_header_is_itself(trust_off):
    # The direct loopback case the login limiter has always keyed on: unchanged.
    _assert_agreed(_Req("127.0.0.1"), "127.0.0.1")
    _assert_agreed(_Req("::1"), "::1")


def test_the_desktop_login_bucket_is_the_same_address_with_its_own_suffix(trust_off):
    request = _Req("127.0.0.1", {"cf-connecting-ip": "198.51.100.7"})
    address, slot = auth_routes._desktop_login_key(request)
    assert slot == auth_routes._DESKTOP_LOGIN_USER
    assert address == client_ip(request) + auth_routes._DESKTOP_LOGIN_USER
