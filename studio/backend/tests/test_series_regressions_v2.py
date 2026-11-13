# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pins for the behaviours this round added, so the next refactor cannot quietly undo them.

``tests/test_series_regressions.py`` pins the previous round. This file pins the round that came
after it: the scoped download/canvas link, the optional Host allow-list, the unified client-IP
resolver, the Hub probe's negative cache, and the probe caches that arrived with them -- and, from
the round that closed the query-token residue, the auth resolver's loss of its token parameter.

Each test answers one question: "if someone deleted the behaviour, would this fail?" The comment
on each test names the behaviour it holds in place and what removing it would break.

Three kinds of pin are used, chosen per behaviour:

* Behavioural pins drive the real code (the auth helpers, the Trusted Hosts ASGI middleware, the
  client-IP resolver, ``cached_probe`` with an injected clock, each cache's own store) and assert
  what it does.
* ``ast`` pins read a file and assert the *shape* of the code, for the places whose behaviour
  cannot be exercised without a live server or a real download: the two credential helpers and
  ``routes/auth.py``'s resolution rule.
* Bounded-cache pins fill a cache past its bound and assert the bound holds, then assert the
  module's own ``invalidate`` drops what it filled.

Nothing here imports torch, starts a server, opens a socket or touches the network.
"""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path
from urllib.parse import urlencode

import pytest
from starlette.requests import Request

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# --- helpers ---------------------------------------------------------------------------------

def _read(relative: str) -> str:
    return (_BACKEND / relative).read_text(encoding = "utf-8")


def _tree(relative: str) -> ast.Module:
    return ast.parse(_read(relative))


def _functions(node: ast.AST) -> dict:
    """Every function/method definition under ``node``, by name."""
    found = {}
    for sub in ast.walk(node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(sub.name, []).append(sub)
    return found


def _called_names(node: ast.AST) -> set:
    """Every name a call inside ``node`` refers to (attribute tails included)."""
    names = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Attribute):
            names.add(func.attr)
        elif isinstance(func, ast.Name):
            names.add(func.id)
    return names


def _query_param_keys(node: ast.AST) -> set:
    """Every string key read off a ``query_params`` mapping inside ``node``."""
    keys = set()
    for sub in ast.walk(node):
        # request.query_params.get("link")
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            owner = sub.func.value
            if (
                sub.func.attr == "get"
                and isinstance(owner, ast.Attribute)
                and owner.attr == "query_params"
                and sub.args
                and isinstance(sub.args[0], ast.Constant)
                and isinstance(sub.args[0].value, str)
            ):
                keys.add(sub.args[0].value)
        # request.query_params["token"]
        if isinstance(sub, ast.Subscript):
            owner = sub.value
            if isinstance(owner, ast.Attribute) and owner.attr == "query_params":
                index = sub.slice
                if isinstance(index, ast.Constant) and isinstance(index.value, str):
                    keys.add(index.value)
    return keys


def _header_names_read(node: ast.AST) -> set:
    """Every header name read as ``something.headers.get("<name>")`` inside ``node``."""
    names = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call) or not isinstance(sub.func, ast.Attribute):
            continue
        if sub.func.attr != "get" or not sub.args:
            continue
        owner = sub.func.value
        if not (isinstance(owner, ast.Attribute) and owner.attr == "headers"):
            continue
        first = sub.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            names.add(first.value.lower())
    return names


def _request(*headers, client = ("10.0.0.5", 51234), **query) -> Request:
    """A ``Request`` built from a scope alone: no server, no socket, no ASGI app."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": urlencode(query).encode(),
        "headers": [(name.encode(), value.encode()) for name, value in headers],
        "client": client,
    }
    return Request(scope)


def _http_scope(host: str, path: str = "/api/system") -> dict:
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [(b"host", host.encode("latin-1"))],
    }


async def _run_asgi(middleware, scope) -> list:
    """Run ``middleware`` over ``scope`` and return the messages it sent."""
    messages: list = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await middleware(scope, receive, send)
    return messages


# --- 1. The canvas credential is a header or a scoped link, never a query bearer ---------------

def test_canvas_auth_helper_reads_only_the_link_query_param():
    """Source-level pin: ``_authenticate_header_or_query`` never reads ``?token=``.

    The helper is the one place the canvas/artifact routes resolve a credential. If it starts
    reading ``query_params["token"]`` (or ``.get("token")``) a session bearer goes back into a URL,
    where it survives the click in history, in a proxy's access log and in the ``Referer`` of
    whatever the page links to. Add that read and this test fails.
    """
    definitions = _functions(_tree("routes/inference.py")).get("_authenticate_header_or_query")
    assert definitions, "routes/inference.py no longer defines _authenticate_header_or_query"

    for definition in definitions:
        keys = _query_param_keys(definition)
        assert keys, (
            "the helper no longer reads any query parameter: the ?link= path it exists for is gone"
        )
        assert "token" not in keys, (
            "the helper reads ?token= from the query string again: a session bearer in a URL is "
            "exactly what the link replaced"
        )
        assert keys == {"link"}, f"the helper resolves credentials from unexpected query keys: {keys}"


def test_canvas_auth_helper_takes_no_credential_besides_the_deprecated_token():
    """Source-level pin: no new credential parameter can enter the helper unnoticed.

    The helper still carries the ``token`` argument FastAPI passes from the routes' own query
    declaration; the pin is that nothing else joins it. A refactor that accepts the caller's token
    (rather than only refusing it) or adds a second credential parameter fails here.
    """
    definitions = _functions(_tree("routes/inference.py")).get("_authenticate_header_or_query")
    assert definitions, "routes/inference.py no longer defines _authenticate_header_or_query"

    for definition in definitions:
        names = [argument.arg for argument in definition.args.args]
        assert names and names[0] == "request", (
            f"the helper's first parameter is no longer the request: {names}"
        )
        extra = set(names[1:]) - {"token"}
        assert not extra, (
            f"the helper accepts a credential parameter beyond the refused token: {sorted(extra)}"
        )


def test_a_valid_link_in_the_token_query_param_authenticates_nothing(monkeypatch):
    """Behaviour pinned: ``?token=`` is refused, and only ``?link=`` is a credential.

    Two halves: a caller that sends a session bearer in ``?token=`` gets a 400 that names
    ``link=``, and a real, still-valid canvas link placed under ``?token=`` grants nothing (the
    helper falls through to the bearer resolver, which refuses). Delete the refusal, or teach the
    helper to read ``?token=``, and this test fails.
    """
    from fastapi import HTTPException

    import auth.authentication as auth_module
    from routes import inference as inf

    async def refuse(_request):
        raise HTTPException(status_code = 401, detail = "Missing authentication token")

    # The helper imports the resolver inside its own body, so patching the module is what it sees.
    monkeypatch.setattr(auth_module, "subject_for_header_or_query_token", refuse)

    with pytest.raises(HTTPException) as bearer_in_url:
        asyncio.run(
            inf._authenticate_header_or_query(
                _request(token = "a-real-looking-session-bearer"), "a-real-looking-session-bearer"
            )
        )
    assert bearer_in_url.value.status_code == 400
    assert "link=" in bearer_in_url.value.detail, (
        "the refusal no longer tells the caller to use a ?link= instead"
    )

    link = inf._sign_canvas_link("unsloth")
    with pytest.raises(HTTPException) as link_in_the_wrong_param:
        asyncio.run(inf._authenticate_header_or_query(_request(token = link), None))
    assert link_in_the_wrong_param.value.status_code == 401, (
        "a link carried in ?token= was accepted: the helper reads the query key it must not read"
    )


# --- 2. The library download route refuses a query bearer, by name ------------------------------

def test_library_download_refuses_a_session_bearer_in_the_query_string():
    """Behaviour pinned: ``/items/download?token=`` is a 400 that names ``link=``.

    The download route is reached by a native save and an ``<a href>``, which is why it needs a
    headerless path at all -- and why the answer to "put your bearer in the URL" has to be a
    refusal that points at the scoped link. Drop the guard and this test fails.
    """
    from fastapi import HTTPException

    from routes import library as library_routes

    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            library_routes.download_item(
                _request(), id = "item-1", token = "a-real-looking-session-bearer", link = None
            )
        )
    assert refused.value.status_code == 400
    assert "link=" in refused.value.detail
    assert "?token=" in refused.value.detail, (
        "the refusal no longer names the parameter it refuses, so a caller cannot tell what to "
        "change"
    )


def test_library_download_never_passes_the_token_param_to_an_authenticator():
    """Source-level pin: the route's ``token`` argument is only ever refused.

    The handler must not hand its ``token`` argument to the bearer resolver (that would make
    ``?token=`` a credential on the one route that refuses it) and must not read it back off the
    query string. Give the token argument any other job and this test fails.
    """
    definitions = _functions(_tree("routes/library.py")).get("download_item")
    assert definitions, "routes/library.py no longer defines the download_item handler"

    for definition in definitions:
        # The token may not be forwarded into any call.
        for sub in ast.walk(definition):
            if not isinstance(sub, ast.Call):
                continue
            forwarded = [
                argument
                for argument in [*sub.args, *(keyword.value for keyword in sub.keywords)]
                if isinstance(argument, ast.Name) and argument.id == "token"
            ]
            assert not forwarded, "the download route forwards its ?token= argument into a call"

        # And the refusal itself: an `if` on the token that raises a 400 naming link=.
        refused = False
        for node in ast.walk(definition):
            if not isinstance(node, ast.If):
                continue
            if not any(
                isinstance(sub, ast.Name) and sub.id == "token" for sub in ast.walk(node.test)
            ):
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Raise):
                    continue
                text = ast.dump(sub)
                if "status_code" in text and "400" in text and "link=" in text:
                    refused = True
        assert refused, (
            "download_item no longer refuses ?token= with a 400 that names link=: the headerless "
            "path would silently fall back to a session bearer in a URL"
        )


# --- 3. The Host allow-list ships on with a mode-covering default and never locks out loopback ---

def test_trusted_hosts_ships_on_with_the_apps_own_hosts(monkeypatch):
    """Behaviour pinned: with ``UNSLOTH_TRUSTED_HOSTS`` unset or empty the check still runs, and a
    foreign Host is refused.

    It used to be inert; a DNS-rebinding page was therefore same-origin with the backend. The
    default list carries every host a shipped mode sends (loopback, localhost, the Tauri origins,
    a quick tunnel, Colab's proxy, a LAN literal), which is what makes default-on safe. Go back to
    skipping the middleware and this fails; drop one of those hosts from the default list and the
    per-mode tests in ``test_trusted_hosts.py`` fail.
    """
    from utils.trusted_hosts import TRUSTED_HOSTS_ENV, TrustedHostsMiddleware

    seen: list = []

    async def app(scope, receive, send):
        # The stub echoes the scope type back so a pass-through is distinguishable from a refusal
        # without the middleware having to emit anything itself.
        seen.append(scope.get("path"))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": scope.get("type", "").encode("ascii")})

    middleware = TrustedHostsMiddleware(app)

    for configured in (None, "", "   "):
        seen.clear()
        if configured is None:
            monkeypatch.delenv(TRUSTED_HOSTS_ENV, raising = False)
        else:
            monkeypatch.setenv(TRUSTED_HOSTS_ENV, configured)
        messages = asyncio.run(_run_asgi(middleware, _http_scope("evil.example")))
        assert messages[0]["status"] == 400, (
            f"a foreign Host was admitted while the allow-list was {configured!r}: the check is off"
        )
        assert seen == [], "a refused request still reached the app"
        assert b"UNSLOTH_TRUSTED_HOSTS" in messages[-1]["body"]

    # A non-http scope is passed through untouched whatever the variable says: the middleware adds
    # no message of its own, so the only body is the stub's echo of the scope type.
    seen.clear()
    monkeypatch.setenv(TRUSTED_HOSTS_ENV, "example.com")
    messages = asyncio.run(_run_asgi(middleware, {"type": "websocket", "path": "/ws"}))
    statuses = [message.get("status") for message in messages if "status" in message]
    assert 400 not in statuses, "the middleware answered a websocket handshake with a refusal"
    assert messages[-1]["body"] == b"websocket", (
        "the websocket scope never reached the app: the Host check ran on a non-http scope"
    )
    assert seen == ["/ws"]


def test_trusted_hosts_admits_loopback_whatever_the_allow_list_says(monkeypatch):
    """Behaviour pinned: loopback literals and localhost pass even when the list excludes them.

    The operator who sets the variable is the one who would otherwise lock themselves (and the
    desktop shell, and cloudflared) out of their own backend. Drop the loopback exemption and this
    test fails.
    """
    from utils.trusted_hosts import TRUSTED_HOSTS_ENV, TrustedHostsMiddleware

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = TrustedHostsMiddleware(app)
    monkeypatch.setenv(TRUSTED_HOSTS_ENV, "example.com")

    for host in ("127.0.0.1:8888", "localhost:8888", "[::1]:8888", "localhost", "127.0.0.1"):
        messages = asyncio.run(_run_asgi(middleware, _http_scope(host)))
        assert messages[0]["status"] == 200, f"loopback Host {host!r} was refused"


def test_trusted_hosts_refuses_an_unlisted_host_once_configured(monkeypatch):
    """Behaviour pinned: a configured list still refuses a Host that is not on it, and says how.

    The loopback exemption above must not become the whole middleware: a DNS-rebinding page on
    ``evil.example`` is refused with a 400 whose body names the variable to add the host to.
    Empty the middleware's body (or accept everything) and this test fails.
    """
    from utils.trusted_hosts import TRUSTED_HOSTS_ENV, TrustedHostsMiddleware

    reached: list = []

    async def app(scope, receive, send):
        reached.append(scope.get("path"))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = TrustedHostsMiddleware(app)
    monkeypatch.setenv(TRUSTED_HOSTS_ENV, "studio.example, *.internal")

    refused = asyncio.run(_run_asgi(middleware, _http_scope("evil.example")))
    assert refused[0]["status"] == 400
    assert TRUSTED_HOSTS_ENV in refused[-1]["body"].decode("utf-8")
    assert reached == [], "the refusal still ran the app"

    for host in ("studio.example:8888", "a.internal", "b.a.internal"):
        allowed = asyncio.run(_run_asgi(middleware, _http_scope(host)))
        assert allowed[0]["status"] == 200, f"an allow-listed Host {host!r} was refused"


# --- 4. One client-IP resolver for every per-client decision ----------------------------------

def test_the_login_limiter_resolves_the_client_through_utils_client_ip(monkeypatch):
    """Behaviour pinned: ``routes/auth.py`` attributes a request exactly as ``utils.client_ip`` does.

    The login buckets used to key on the LEFTMOST ``X-Forwarded-For`` hop while the shared lockout
    took the rightmost, so behind an appending proxy one request was two different clients to the
    two throttles: either bucket could be dodged, and an innocent client could inherit a stranger's
    lockout. Re-add a private rule in routes/auth.py and this test fails.
    """
    from utils.client_ip import client_ip
    from utils.client_ip import _TRUST_FORWARDED_ENV

    from routes import auth as auth_routes

    monkeypatch.setenv(_TRUST_FORWARDED_ENV, "1")

    # A caller-chosen public hop prepended to the proxy's own append: the rightmost untrusted hop
    # is the answer, the leftmost is fiction.
    forwarded = _request(
        ("x-forwarded-for", "1.2.3.4, 203.0.113.9"),
        ("host", "127.0.0.1:8888"),
        client = ("10.0.0.5", 51234),
    )
    assert client_ip(forwarded) == "203.0.113.9"
    assert auth_routes._client_ip(forwarded) == client_ip(forwarded), (
        "the login limiter derived its own client address instead of using utils.client_ip"
    )

    # A loopback peer relaying a tunnel visitor, which is the managed-Cloudflare case.
    tunneled = _request(
        ("cf-connecting-ip", "198.51.100.7"),
        ("host", "127.0.0.1:8888"),
        client = ("127.0.0.1", 51234),
    )
    assert client_ip(tunneled) == "198.51.100.7"
    assert auth_routes._client_ip(tunneled) == "198.51.100.7"

    # And with the opt-in off, the forwarded chain is not trusted at all: one rule, one answer.
    monkeypatch.delenv(_TRUST_FORWARDED_ENV, raising = False)
    assert client_ip(forwarded) == "10.0.0.5"
    assert auth_routes._client_ip(forwarded) == "10.0.0.5"


def test_auth_defines_no_private_forwarded_header_rule():
    """Source-level pin for the same unification: routes/auth.py parses no address of its own.

    No forwarded-header read and no ``ipaddress`` use may live in routes/auth.py; its
    ``_client_ip`` -- where it still exists -- may only delegate to ``utils.client_ip``. Restore a
    local leftmost-hop rule and this test fails.
    """
    tree = _tree("routes/auth.py")

    forwarded_headers = {"x-forwarded-for", "forwarded", "cf-connecting-ip", "x-real-ip"}
    read = _header_names_read(tree)
    assert not (read & forwarded_headers), (
        f"routes/auth.py reads a forwarded header itself: {sorted(read & forwarded_headers)}"
    )

    imported = set()
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in sub.names)
        elif isinstance(sub, ast.ImportFrom) and sub.module:
            imported.add(sub.module.split(".")[0])
    assert "ipaddress" not in imported, (
        "routes/auth.py parses addresses itself again instead of asking utils.client_ip"
    )

    shims = _functions(tree).get("_client_ip", [])
    for shim in shims:
        assert _called_names(shim) == {"client_ip"}, (
            "routes/auth.py's _client_ip does more than delegate to utils.client_ip: "
            f"{sorted(_called_names(shim))}"
        )
        assert not any(isinstance(sub, ast.If) for sub in ast.walk(shim)), (
            "routes/auth.py's _client_ip branches on its own rule again"
        )


# --- 5. The Hub endpoint probe caches failures and only failures -------------------------------

def _counting_probe(verdict, calls):
    def probe():
        calls.append(1)
        return verdict

    return probe


def test_hf_probe_reuses_an_unreachable_verdict_inside_the_window():
    """Behaviour pinned: an unreachable verdict is answered from the cache, not re-probed.

    This is the whole point of the negative cache: a download, a dataset load and a fit estimate
    each probing a down endpoint pay one connection attempt between them. Stop storing (or stop
    reading) a negative and this test fails.
    """
    from utils import hf_probe

    clock = [1_000.0]
    cache = hf_probe.NegativeVerdictCache(ttl = 30.0, timeout_ttl = 5.0, clock = lambda: clock[0])
    calls: list = []
    probe = _counting_probe(hf_probe.ProbeVerdict(reachable = False), calls)

    first = hf_probe.cached_probe(probe, ("https://endpoint", 3.0), cache = cache)
    assert first.reachable is False
    assert len(calls) == 1

    clock[0] += hf_probe.NEGATIVE_TTL_S / 2
    second = hf_probe.cached_probe(probe, ("https://endpoint", 3.0), cache = cache)
    assert second == first
    assert len(calls) == 1, "an unreachable verdict inside the window was re-probed"


def test_hf_probe_never_reuses_a_reachable_verdict():
    """Behaviour pinned: a reachable verdict is never remembered, so every caller re-probes.

    A stale "reachable" is the failure that matters: it sends the next download at an endpoint that
    has stopped answering. Start caching a positive and this test fails.
    """
    from utils import hf_probe

    clock = [1_000.0]
    cache = hf_probe.NegativeVerdictCache(ttl = 30.0, timeout_ttl = 5.0, clock = lambda: clock[0])
    calls: list = []
    probe = _counting_probe(hf_probe.ProbeVerdict(reachable = True), calls)

    for _ in range(3):
        assert hf_probe.cached_probe(probe, ("https://endpoint", 3.0), cache = cache).reachable
        clock[0] += 1.0

    assert len(calls) == 3, "a reachable verdict was served from the cache"
    assert len(cache) == 0, "a reachable verdict was stored in the negative cache"


def test_hf_probe_drops_a_negative_when_its_window_closes():
    """Behaviour pinned: the remembered negative expires, and the next caller probes again.

    An entry held forever would declare an endpoint dead long after it came back. Remove the
    expiry (or keep the entry past it) and this test fails.
    """
    from utils import hf_probe

    clock = [1_000.0]
    cache = hf_probe.NegativeVerdictCache(ttl = 30.0, timeout_ttl = 5.0, clock = lambda: clock[0])
    calls: list = []
    probe = _counting_probe(hf_probe.ProbeVerdict(reachable = False), calls)

    hf_probe.cached_probe(probe, "endpoint", cache = cache)
    assert len(calls) == 1

    clock[0] += hf_probe.NEGATIVE_TTL_S + 1.0
    hf_probe.cached_probe(probe, "endpoint", cache = cache)
    assert len(calls) == 2, "an expired negative verdict was still served"
    assert len(cache) == 1


def test_hf_probe_holds_a_timeout_for_a_shorter_window_than_a_refusal():
    """Behaviour pinned: a timeout is remembered for less time than a definite refusal.

    A timeout is evidence of a slow answer, not of a dead endpoint, so it must be re-probed sooner;
    holding one for the full window writes off a slow mirror long after it recovered. Make the two
    windows equal and this test fails.
    """
    from utils import hf_probe

    assert 0 < hf_probe.TIMEOUT_TTL_S < hf_probe.NEGATIVE_TTL_S, (
        "a timeout is no longer held for a shorter window than a refusal"
    )

    clock = [1_000.0]
    cache = hf_probe.NegativeVerdictCache(ttl = 30.0, timeout_ttl = 5.0, clock = lambda: clock[0])
    calls: list = []
    timeout_probe = _counting_probe(
        hf_probe.ProbeVerdict(reachable = False, timed_out = True), calls
    )
    refusal_probe = _counting_probe(hf_probe.ProbeVerdict(reachable = False), calls)

    hf_probe.cached_probe(timeout_probe, "slow", cache = cache)
    hf_probe.cached_probe(refusal_probe, "down", cache = cache)
    assert len(calls) == 2

    clock[0] += hf_probe.TIMEOUT_TTL_S + 1.0
    hf_probe.cached_probe(timeout_probe, "slow", cache = cache)
    hf_probe.cached_probe(refusal_probe, "down", cache = cache)
    assert len(calls) == 3, "the timed-out verdict was still served past its shorter window"
    # The refusal is still inside its own window: only the timeout was dropped. The re-probe of the
    # timed-out key stored a fresh verdict for it, so the cache holds two live entries, not one.
    assert len(cache) == 2, "the re-probe did not replace the expired timeout entry"


def test_hf_probe_negative_cache_is_bounded_and_invalidatable():
    """Behaviour pinned: the negative cache keeps a finite number of entries and can be emptied.

    The bound is what stops a caller keying on something unbounded from growing the cache for the
    life of the process; ``invalidate`` is what lets a network change be noticed at once. Remove
    either and this test fails.
    """
    from utils import hf_probe

    assert isinstance(hf_probe.MAX_CACHED_NEGATIVES, int)
    assert 0 < hf_probe.MAX_CACHED_NEGATIVES < 1_000_000

    cache = hf_probe.NegativeVerdictCache(max_entries = 3, clock = lambda: 1_000.0)
    for index in range(12):
        cache.put(f"endpoint-{index}", hf_probe.ProbeVerdict(reachable = False))
    assert len(cache) <= 3, "the negative cache grew past its bound"

    cache.invalidate()
    assert len(cache) == 0, "invalidate() left a negative verdict behind"
    assert cache.get("endpoint-11") is None

    cache.put("endpoint", hf_probe.ProbeVerdict(reachable = False))
    cache.invalidate("endpoint")
    assert len(cache) == 0, "invalidate(key) did not drop that key"


# --- 6. Every probe cache is bounded and has an invalidate ------------------------------------

def test_hub_metadata_cache_is_bounded_and_invalidatable():
    """Behaviour pinned: the ``model_info`` cache keeps at most ``MAX_ENTRIES`` answers.

    A process that walks many repos -- or many users -- would otherwise grow this for its lifetime,
    and a licence accepted on the Hub would stay refused behind a stale answer. Remove the bound or
    ``invalidate`` and this test fails.
    """
    from utils import hub_metadata_cache as cache

    assert isinstance(cache.MAX_ENTRIES, int)
    assert 0 < cache.MAX_ENTRIES < 1_000_000

    fetches: list = []

    def fetch(repo, revision):
        fetches.append((repo, revision))
        return {"repo": repo, "revision": revision}

    cache.invalidate()
    try:
        for index in range(cache.MAX_ENTRIES + 32):
            cache.model_info_cached(fetch, f"repo-{index}", "main")
        assert len(fetches) == cache.MAX_ENTRIES + 32, "a first fetch was served from the cache"
        assert cache.cache_size() <= cache.MAX_ENTRIES, "the metadata cache grew past its bound"

        # A recent answer is a hit; invalidate() makes the next call fetch again.
        last = f"repo-{cache.MAX_ENTRIES + 31}"
        cache.model_info_cached(fetch, last, "main")
        assert len(fetches) == cache.MAX_ENTRIES + 32, "a fresh metadata answer was not reused"

        cache.invalidate()
        assert cache.cache_size() == 0
        cache.model_info_cached(fetch, last, "main")
        assert len(fetches) == cache.MAX_ENTRIES + 33, "invalidate() did not drop the answers"
    finally:
        cache.invalidate()


def test_dataset_fingerprint_cache_is_bounded_and_invalidatable(tmp_path):
    """Behaviour pinned: the dataset hash memos are bounded and can be dropped.

    A Start hashes every file of a dataset and the tokenizer; an unbounded memo over every dataset a
    user ever trains on is a memory leak with a re-hash's worth of correctness risk. Remove the
    bound (or ``invalidate_hash_cache``) and this test fails.
    """
    from core.training import dataset_fingerprint as fingerprint

    assert isinstance(fingerprint.CACHE_ENTRIES, int)
    assert 0 < fingerprint.CACHE_ENTRIES < 1_000_000
    for store in (fingerprint._FILE_HASHES, fingerprint._TREE_ENTRIES):
        assert store.max_entries == fingerprint.CACHE_ENTRIES, (
            "a dataset hash memo is no longer bounded by CACHE_ENTRIES"
        )

    root = tmp_path / "dataset"
    root.mkdir()
    for index in range(fingerprint.CACHE_ENTRIES + 8):
        (root / f"part-{index:03d}.jsonl").write_text("{}\n", encoding = "utf-8")

    fingerprint.invalidate_hash_cache()
    try:
        first = fingerprint.fingerprint_source(str(root))
        assert first["status"] == "hashed"
        assert 0 < len(fingerprint._FILE_HASHES) <= fingerprint.CACHE_ENTRIES, (
            "the per-file hash memo grew past its bound"
        )
        assert len(fingerprint._TREE_ENTRIES) == 1

        second = fingerprint.fingerprint_source(str(root))
        assert second["sha256"] == first["sha256"], "the memoized tree hash changed its answer"

        fingerprint.invalidate_hash_cache()
        assert len(fingerprint._FILE_HASHES) == 0
        assert len(fingerprint._TREE_ENTRIES) == 0
    finally:
        fingerprint.invalidate_hash_cache()


def test_gguf_header_cache_is_bounded_and_invalidatable(tmp_path):
    """Behaviour pinned: the GGUF header memo evicts past its bound and can be invalidated.

    The header read exists to avoid re-parsing every listed model; an unbounded memo over every
    model ever listed is the same defect it was built to prevent. Remove the eviction or the
    invalidate and this test fails.
    """
    from utils import gguf_archs

    assert 0 < gguf_archs.HEADER_CACHE.max_entries < 1_000_000

    paths = []
    for index in range(3):
        path = tmp_path / f"model-{index}.gguf"
        path.write_bytes(b"GGUF" + bytes([index]))
        paths.append(str(path))

    parses: list = []

    def parse(path):
        parses.append(path)
        return {"path": path}

    cache = gguf_archs.GGUFHeaderCache(max_entries = 2)
    for path in paths:
        cache.get(path, parse)
    assert len(cache) == 2, "the header cache grew past its bound"

    # The last two are remembered; the first was evicted and is parsed again.
    before = len(parses)
    cache.get(paths[2], parse)
    assert len(parses) == before, "an unchanged header was parsed twice"
    cache.get(paths[0], parse)
    assert len(parses) == before + 1, "an evicted header was served from the cache anyway"

    cache.invalidate(paths[2])
    assert len(cache) == 1, "invalidate(path) did not drop that file's header"
    cache.invalidate()
    assert len(cache) == 0

    gguf_archs.HEADER_CACHE.get(paths[0], parse)
    gguf_archs.invalidate_header_cache()
    assert len(gguf_archs.HEADER_CACHE) == 0, (
        "invalidate_header_cache() no longer drops the shared header memo"
    )


def test_model_config_parse_cache_is_bounded_and_invalidatable():
    """Behaviour pinned: the ``config.json`` parse memo evicts past its bound and can be dropped.

    A sweep over many models -- many users, many revisions -- would otherwise pin every parsed
    config in memory; and a licence accepted on the Hub must not be answered from a stale entry.
    Remove the bound or ``invalidate`` and this test fails.
    """
    from utils.models import model_config

    bound = model_config._CONFIG_CACHE_MAX_ENTRIES
    assert isinstance(bound, int)
    assert 0 < bound < 1_000_000

    cache = model_config._config_parse_cache
    model_config.invalidate()
    try:
        keys = [("auto", f"/models/m-{index}/config.json", index) for index in range(bound + 8)]
        for key in keys:
            cache.put(key, {"key": key})

        assert cache.get(keys[-1]) == {"key": keys[-1]}, "a stored config answer was lost"
        assert cache.get(keys[0]) is None, "the config parse cache grew past its bound"

        model_config.invalidate()
        assert cache.get(keys[-1]) is None, "invalidate() left a parsed config behind"
    finally:
        model_config.invalidate()


def test_npu_provider_cache_holds_one_bounded_entry_and_can_be_invalidated(monkeypatch):
    """Behaviour pinned: the NPU provider rows are a one-slot cache that ``invalidate`` clears.

    Six PATH walks per call is what the TTL removes; a per-caller key table would be the memory
    defect the TTL was meant to avoid. Remove the cache (or let it grow per call) and this test
    fails; remove ``invalidate`` and a runtime installed under the server is never noticed.
    """
    from utils.hardware import npu_providers

    assert 0 < npu_providers._CACHE_TTL_SECONDS < 3600

    calls: list = []

    def probe_providers():
        calls.append(1)
        return [{"name": "stub", "installed": False}], True

    monkeypatch.setattr(npu_providers, "_probe_providers", probe_providers)
    npu_providers.invalidate()
    assert npu_providers._cache is None

    first = npu_providers.npu_runtime_providers()
    second = npu_providers.npu_runtime_providers()
    assert len(calls) == 1, "the provider rows were re-probed inside the TTL"
    assert first == second
    assert first is not second, "callers share one row list, so one could edit the cache"

    for _ in range(20):
        npu_providers.npu_runtime_providers()
    assert len(calls) == 1, "the cache stopped answering inside its TTL"
    assert len(npu_providers._cache) == 3, (
        "the provider cache is no longer a single (fingerprint, expiry, rows) slot"
    )

    npu_providers.invalidate()
    assert npu_providers._cache is None
    npu_providers.npu_runtime_providers()
    assert len(calls) == 2, "invalidate() did not force a fresh probe"
    npu_providers.invalidate()


def test_compute_probe_cache_is_one_slot_and_can_be_invalidated(monkeypatch):
    """Behaviour pinned: the Train page's probe cache is a single slot with a finite TTL.

    ``/targets`` is polled, and one payload runs a torch device probe plus six PATH scans; the cache
    is one slot keyed on nothing but the probe callable, so it cannot grow with polls. Remove the
    once-per-TTL reuse and this test fails; remove ``_invalidate_probe_cache`` and a deliberate
    target change is not reflected until the TTL passes.
    """
    from routes import compute

    assert 0 < compute._PROBE_TTL_S < 3600

    calls: list = []

    def probe():
        calls.append(1)
        return {"kind": "cpu"}

    monkeypatch.setattr(compute, "_probe", probe)
    monkeypatch.setattr(compute, "_npu_capability", lambda: {"mode": "unavailable"})
    monkeypatch.setattr(compute, "_has_intel_gpu", lambda _probe: False)
    monkeypatch.setattr(compute, "_xpu_torch_status", lambda: {"status": "unknown"})

    compute._invalidate_probe_cache()
    keys = set(compute._probe_cache)

    first = compute._cached_probe_state()
    second = compute._cached_probe_state()
    assert len(calls) == 1, "the hardware probe ran twice inside the TTL"
    assert first == second

    for _ in range(20):
        compute._cached_probe_state()
    assert len(calls) == 1
    assert set(compute._probe_cache) == keys, (
        "the probe cache grew a key table instead of holding one slot"
    )

    compute._invalidate_probe_cache()
    compute._cached_probe_state()
    assert len(calls) == 2, "_invalidate_probe_cache() no longer forces a fresh probe"
    compute._invalidate_probe_cache()


# --- 7. The auth resolver cannot be handed a query token again --------------------------------

def test_the_auth_resolver_takes_the_request_and_no_token_parameter():
    """Source-level pin: ``subject_for_header_or_query_token`` cannot receive a token at all.

    The finding this closes was a session bearer resolved from a query string. The parameter is
    what carried it: a route read ``?token=`` for itself and the resolver treated it like a header
    bearer, where the URL outlives the click in history, in a proxy's access log and in the
    ``Referer`` of whatever the page links to. Dropping the parameter makes the header the only
    read, so a new route cannot re-open the finding even if it starts reading ``?token=`` again.
    Re-add a parameter and this test fails.
    """
    definitions = _functions(_tree("auth/authentication.py")).get("subject_for_header_or_query_token")
    assert definitions, "auth/authentication.py no longer defines subject_for_header_or_query_token"

    for definition in definitions:
        names = [argument.arg for argument in definition.args.args]
        assert names == ["request"], (
            f"the bearer resolver accepts something besides the request again: {names} -- "
            "anything passed there can come from a query string"
        )
        assert definition.args.vararg is None and not definition.args.kwonlyargs, (
            "the bearer resolver regained a slot to receive a token without naming it"
        )
        keys = _query_param_keys(definition)
        assert not keys, f"the bearer resolver reads the query string itself again: {sorted(keys)}"


def test_no_route_passes_a_query_token_into_the_auth_resolver():
    """Source-level pin: route code hands the resolver the request, and a token only as ``None``.

    With the parameter gone, a route that reads ``?token=`` and passes it would fail only at
    runtime, for whoever tries it first. This says so at review time instead: across every module
    under ``routes/``, a call to the bearer resolver or its credential helper may carry the request
    and at most a literal ``None`` -- never a name, a constant or a ``query_params`` read, which is
    what a query-derived token would look like. ``main.py`` is outside the scan on purpose: its one
    helper call passes the ``Authorization`` bearer it read for itself
    (``studio/backend/main.py:2267``), the header path this pin exists to keep the only one.
    """
    resolver_names = {"subject_for_header_or_query_token", "credentials_for_token"}
    checked = 0
    for path in sorted((_BACKEND / "routes").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        relative = path.relative_to(_BACKEND).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            if name not in resolver_names:
                continue
            checked += 1
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            assert arguments, f"{relative}:{node.lineno}: {name} is called without the request"
            extras = [
                *node.args[1:],
                *(keyword.value for keyword in node.keywords if keyword.arg != "request"),
            ]
            for extra in extras:
                assert isinstance(extra, ast.Constant) and extra.value is None, (
                    f"{relative}:{node.lineno}: {name} is handed a token instead of the request "
                    f"alone: {ast.dump(extra)[:120]}"
                )
            for argument in arguments:
                assert not _query_param_keys(argument), (
                    f"{relative}:{node.lineno}: {name} is handed a query-string read directly"
                )
    assert checked, "no route calls the bearer resolver any more: this pin is scanning nothing"


if __name__ == "__main__":  # pragma: no cover - manual run helper
    raise SystemExit(pytest.main([__file__, "-q"]))
