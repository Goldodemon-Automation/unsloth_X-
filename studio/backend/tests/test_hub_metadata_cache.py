# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Hub metadata cache reuses one answer per (repo, revision) instead of re-asking."""

from __future__ import annotations

import threading
from pathlib import Path
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest

import utils.hub_metadata_cache as hub_cache


@pytest.fixture(autouse = True)
def _clean_cache():
    """Every test starts from an empty cache and leaves it empty."""
    hub_cache.invalidate()
    yield
    hub_cache.invalidate()


class _Info:
    def __init__(self, gated: bool = False):
        self.gated = gated


def test_two_calls_fetch_once():
    calls = []

    def fetch(repo, revision):
        calls.append((repo, revision))
        return _Info(gated = True)

    first = hub_cache.model_info_cached(fetch, "org/model")
    second = hub_cache.model_info_cached(fetch, "org/model")

    assert len(calls) == 1
    assert first is second
    assert first.gated is True


def test_different_revision_fetches_again():
    calls = []

    def fetch(repo, revision):
        calls.append((repo, revision))
        return _Info()

    hub_cache.model_info_cached(fetch, "org/model", "main")
    hub_cache.model_info_cached(fetch, "org/model", "v2.0")

    assert calls == [("org/model", "main"), ("org/model", "v2.0")]


def test_different_repo_fetches_again():
    calls = []

    def fetch(repo, revision):
        calls.append(repo)
        return _Info()

    hub_cache.model_info_cached(fetch, "org/one")
    hub_cache.model_info_cached(fetch, "org/two")

    assert calls == ["org/one", "org/two"]


def test_raising_fetch_is_not_cached():
    calls = []

    def fetch(repo, revision):
        calls.append(repo)
        raise RuntimeError("hub is down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            hub_cache.model_info_cached(fetch, "org/model")

    # A failure is not an answer: the next caller must ask again.
    assert len(calls) == 2


def test_invalidate_forces_a_refetch():
    calls = []

    def fetch(repo, revision):
        calls.append(repo)
        return _Info()

    hub_cache.model_info_cached(fetch, "org/model")
    hub_cache.invalidate()
    hub_cache.model_info_cached(fetch, "org/model")

    assert len(calls) == 2


def test_invalidate_one_repo_leaves_others():
    calls = []

    def fetch(repo, revision):
        calls.append(repo)
        return _Info()

    hub_cache.model_info_cached(fetch, "org/one")
    hub_cache.model_info_cached(fetch, "org/two")
    hub_cache.invalidate("org/one")
    hub_cache.model_info_cached(fetch, "org/one")
    hub_cache.model_info_cached(fetch, "org/two")

    assert calls == ["org/one", "org/two", "org/one"]


def test_expired_entry_refetches():
    now = [1000.0]
    calls = []

    def fetch(repo, revision):
        calls.append(repo)
        return _Info()

    hub_cache.model_info_cached(fetch, "org/model", ttl_s = 10.0, clock = lambda: now[0])
    now[0] = 1009.0
    hub_cache.model_info_cached(fetch, "org/model", ttl_s = 10.0, clock = lambda: now[0])
    assert len(calls) == 1

    # Past the TTL the answer is stale, so it is fetched again.
    now[0] = 1011.0
    hub_cache.model_info_cached(fetch, "org/model", ttl_s = 10.0, clock = lambda: now[0])
    assert len(calls) == 2


def test_bound_evicts_the_oldest():
    calls = []

    def fetch(repo, revision):
        calls.append(repo)
        return _Info()

    for index in range(hub_cache.MAX_ENTRIES + 1):
        hub_cache.model_info_cached(fetch, f"org/model-{index}")

    assert len(calls) == hub_cache.MAX_ENTRIES + 1
    assert hub_cache.cache_size() == hub_cache.MAX_ENTRIES

    # The oldest entry was evicted, so it is fetched again; the newest is still cached.
    hub_cache.model_info_cached(fetch, "org/model-0")
    assert calls[-1] == "org/model-0"
    assert len(calls) == hub_cache.MAX_ENTRIES + 2

    hub_cache.model_info_cached(fetch, f"org/model-{hub_cache.MAX_ENTRIES}")
    assert len(calls) == hub_cache.MAX_ENTRIES + 2


def test_concurrent_callers_share_one_fetch():
    calls = []
    start = threading.Barrier(8)

    def fetch(repo, revision):
        calls.append(repo)
        # A little work so the other threads pile up on the key while it is in flight.
        threading.Event().wait(0.05)
        return _Info()

    results = []

    def worker():
        start.wait()
        results.append(hub_cache.model_info_cached(fetch, "org/model"))

    threads = [threading.Thread(target = worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1
    assert len(results) == 8
    assert all(result is results[0] for result in results)


def test_scope_keeps_credentials_apart():
    calls = []

    def fetch(repo, revision):
        calls.append(repo)
        return _Info()

    anon = hub_cache.token_scope(None)
    token = hub_cache.token_scope("hf_secret")

    hub_cache.model_info_cached(fetch, "org/model", scope = anon)
    hub_cache.model_info_cached(fetch, "org/model", scope = token)

    assert len(calls) == 2
    assert anon != token
    # The same credential reuses the entry.
    hub_cache.model_info_cached(fetch, "org/model", scope = token)
    assert len(calls) == 2


def test_client_scope_separates_different_hub_clients():
    calls = []

    def fetch(repo, revision):
        calls.append(repo)
        return _Info()

    def client_a(repo, revision):
        return _Info()

    def client_b(repo, revision):
        return _Info()

    first = hub_cache.client_scope(client_a)
    assert first == hub_cache.client_scope(client_a)
    assert first != hub_cache.client_scope(client_b)

    hub_cache.model_info_cached(fetch, "org/model", scope = first)
    hub_cache.model_info_cached(fetch, "org/model", scope = hub_cache.client_scope(client_b))

    assert len(calls) == 2
