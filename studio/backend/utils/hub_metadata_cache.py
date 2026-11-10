# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Short-lived cache for Hub model metadata (``model_info``).

``model_info`` answers one question -- is this repo gated, and does it exist for this
credential -- and the answer for a given ``(repo, revision)`` does not change from one
minute to the next. Asking the Hub on every model load spends a network round-trip
(and a Hub request budget) to learn something the previous load already learned.

The TTL is deliberately short (5 minutes by default). Gated status is not static: the
moment a user accepts a repository's licence on the Hub, ``info.gated`` stops being the
reason a load is refused. A long-lived cache would keep refusing that load with a stale
verdict; a five-minute window keeps the round-trip saving while letting an accepted
licence take effect within minutes. ``invalidate()`` exists for callers that know a
licence was just accepted and want the answer now.

Only successful fetches are cached. A fetch that raises carries no verdict (offline, a
timeout, a 500), so it is never stored and never replayed: the next caller hits the Hub
again and sees the same failure the uncached path would have produced.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
import weakref
from collections import OrderedDict
from itertools import count
from typing import Any, Callable, Optional, Tuple

__all__ = [
    "DEFAULT_TTL_S",
    "MAX_ENTRIES",
    "cache_size",
    "client_scope",
    "invalidate",
    "model_info_cached",
    "token_scope",
]

# Five minutes: long enough that a training session's repeated loads reuse one answer,
# short enough that a freshly accepted licence is picked up within a coffee break.
DEFAULT_TTL_S = 300.0

# Bounded so a process that walks many repos (or many users) cannot grow this forever.
MAX_ENTRIES = 256

# A process-local salt so the credential fingerprint stored in the key is not a plain
# hash of the token: the key never holds anything derived from a secret in the clear.
_SALT = secrets.token_bytes(16)


class _Entry:
    """A cached value and the monotonic time it stops being usable."""

    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float) -> None:
        self.value = value
        self.expires_at = expires_at


# Guards the two module-level dicts below.
_LOCK = threading.Lock()
_ENTRIES: "OrderedDict[Tuple[Any, ...], _Entry]" = OrderedDict()
# One lock per key so two threads asking for two different repos do not queue behind
# each other's network call, while two threads asking for the *same* repo collapse
# onto a single fetch.
_KEY_LOCKS: "dict[Tuple[Any, ...], threading.Lock]" = {}

# Injectable clock. Tests replace it; production uses the monotonic clock so that a
# wall-clock jump (NTP step, DST) cannot make an entry look fresh or stale.
_CLOCK: Callable[[], float] = time.monotonic

# One scope per distinct Hub client callable. Two different callables are not asking
# the same question: a process that swaps its Hub client, or a test that stubs the
# client with another implementation, must not read the previous client's answers.
# The counter never repeats, so a replaced callable cannot inherit an old scope.
_CLIENT_EPOCHS: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()
_CLIENT_EPOCH_LOCK = threading.Lock()
_CLIENT_COUNTER = count(1)


def _key(
    repo: str,
    revision: Optional[str],
    scope: Optional[str],
) -> Tuple[Any, ...]:
    """Build the cache key.

    The key is ``(repo, revision)``. ``scope`` is an optional extra discriminator for
    callers whose fetch also depends on something outside the key -- the credential in
    use, for instance. Without it, an entry fetched with one credential could be
    replayed for a caller holding another one, which would change the answer the fetch
    would have given.
    """
    if scope is None:
        return (repo, revision)
    return (repo, revision, scope)


def token_scope(token: Optional[str]) -> str:
    """Return a stable, non-reversible scope for a credential.

    ``model_info`` can answer differently depending on who asks: a private repo is
    invisible to an anonymous caller and visible to one holding a token. Callers that
    pass a token should pass the matching scope so the two never share an entry.
    """
    if not token:
        return "anon"
    digest = hashlib.sha256(_SALT + token.encode("utf-8", "replace")).hexdigest()
    return f"tok:{digest[:32]}"


def client_scope(client: Any) -> str:
    """Return a stable scope for a Hub client callable.

    Callers that fetch through ``huggingface_hub.model_info`` should pass this scope so
    answers from one client implementation are never replayed for another.
    """
    try:
        with _CLIENT_EPOCH_LOCK:
            epoch = _CLIENT_EPOCHS.get(client)
            if epoch is None:
                epoch = next(_CLIENT_COUNTER)
                _CLIENT_EPOCHS[client] = epoch
            return f"client{epoch}"
    except TypeError:
        # Not weak-referenceable or not hashable: it cannot be tracked, so give it a
        # scope of its own that never matches a tracked client.
        return "client?"


def model_info_cached(
    fetch: Callable[[str, Optional[str]], Any],
    repo: str,
    revision: Optional[str] = None,
    *,
    ttl_s: float = DEFAULT_TTL_S,
    scope: Optional[str] = None,
    clock: Optional[Callable[[], float]] = None,
) -> Any:
    """Return ``fetch(repo, revision)``, reusing a recent answer for the same key.

    ``fetch`` is called as ``fetch(repo, revision)`` and must raise to signal failure.
    A raising fetch is never cached, so the caller sees the same exception on every
    attempt rather than a replayed one.
    """
    key = _key(repo, revision, scope)
    now = (clock or _CLOCK)()

    with _LOCK:
        entry = _ENTRIES.get(key)
        if entry is not None:
            if entry.expires_at > now:
                _ENTRIES.move_to_end(key)
                return entry.value
            del _ENTRIES[key]
        key_lock = _KEY_LOCKS.get(key)
        if key_lock is None:
            key_lock = threading.Lock()
            _KEY_LOCKS[key] = key_lock

    with key_lock:
        # Another thread may have filled the entry while this one waited for the lock.
        with _LOCK:
            entry = _ENTRIES.get(key)
            now = (clock or _CLOCK)()
            if entry is not None and entry.expires_at > now:
                _ENTRIES.move_to_end(key)
                return entry.value

        value = fetch(repo, revision)

        expires_at = (clock or _CLOCK)() + max(0.0, float(ttl_s))
        with _LOCK:
            _ENTRIES[key] = _Entry(value, expires_at)
            _ENTRIES.move_to_end(key)
            _evict()
        return value


def cache_size() -> int:
    """Return the number of cached answers currently held."""
    with _LOCK:
        return len(_ENTRIES)


def invalidate(repo: Optional[str] = None) -> None:
    """Drop cached answers for ``repo``, or every cached answer when it is None."""
    with _LOCK:
        if repo is None:
            _ENTRIES.clear()
            return
        for key in [k for k in _ENTRIES if k[0] == repo]:
            del _ENTRIES[key]


def _evict() -> None:
    """Keep the cache bounded by dropping the oldest entries. Caller holds ``_LOCK``."""
    while len(_ENTRIES) > MAX_ENTRIES:
        oldest, _ = _ENTRIES.popitem(last = False)
        _drop_lock(oldest)


def _drop_lock(key: Tuple[Any, ...]) -> None:
    """Forget a key's lock when nobody holds it. Caller holds ``_LOCK``."""
    lock = _KEY_LOCKS.get(key)
    if lock is None:
        return
    # Only forget a key's lock when nobody holds it: a thread that already took this
    # lock object must keep being the only fetcher for that key. A thread still waiting
    # for it counts as not holding it, so a key whose fetch is in flight keeps its lock:
    # the entry is only evicted once the answer is already stored.
    if lock.acquire(blocking = False):
        try:
            if _KEY_LOCKS.get(key) is lock:
                del _KEY_LOCKS[key]
        finally:
            lock.release()
