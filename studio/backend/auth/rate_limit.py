# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Failure-counting lockout for credential entry: sign-in tokens and passwords.

The Studio is local-first but can be served on a LAN or through a Cloudflare tunnel, so the routes
that check a credential face guesses from the network. This module is the shared guard those routes
consult before checking a credential and report each outcome to afterwards. The shape follows
``utils/preview_rate_limit.py``: a sliding window per key under one lock, in-process and
single-worker (Unsloth runs as one uvicorn process), so no shared store is needed.

``token_attempts`` is one budget for every credential checked here, so a guesser rotating between a
refresh token and a password still spends one budget per client. The key is the client address
(``client_key``), never the submitted secret: the credential is the thing under attack and must not
become a map key, a log line, or part of an error message.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable

from utils.client_ip import client_ip

# Tracked-key bound. An unbounded dict keyed by client address is a memory-exhaustion vector: a
# flood of distinct (or forged) source addresses would mint one entry per source until the process
# dies, which is strictly worse than the guessing it exists to stop.
_MAX_KEYS = 1024


class _Entry:
    """One key's state: failure timestamps inside the current window, and the ban's end time."""

    __slots__ = ("failures", "locked_until")

    def __init__(self) -> None:
        self.failures: list[float] = []
        self.locked_until = 0.0


class AttemptLimiter:
    """Per-key failure counter that locks a key out once ``max_attempts`` failures land inside
    ``window_s``, for ``lockout_s`` from the triggering failure.

    The ban's duration is fixed at the moment it starts: ``allow`` is a pure read, and a failure
    recorded while the ban is up is dropped. A locked-out caller who keeps retrying therefore
    cannot lengthen their own ban, and is admitted again at exactly ``lockout_s`` -- not sooner,
    and not later. A success clears the key outright, because one accepted credential disproves
    guessing pressure from that client. The clock is injected so tests are instant and
    deterministic; nothing here sleeps.
    """

    def __init__(
        self,
        *,
        max_attempts: int,
        window_s: float,
        lockout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_attempts = max_attempts
        self._window_s = window_s
        self._lockout_s = lockout_s
        self._clock = clock
        self._lock = threading.Lock()
        # Insertion-ordered so the bound can evict in O(1). Every lookup moves its key to the end,
        # so the key evicted is the least recently seen: a key under an active ban is being retried,
        # and evicting it would hand the banned caller a fresh budget mid-ban (the same trap
        # ``utils/preview_rate_limit.py`` documents for its bucket table).
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    def allow(self, key: str) -> tuple[bool, float]:
        """Whether ``key`` may attempt a credential now, with seconds to wait when it may not.

        A pure read: the check must not charge the key or touch a ban's end time, or a caller
        retrying every millisecond would sentence themselves to a longer lockout.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return True, 0.0
            self._entries.move_to_end(key)
            retry_after = entry.locked_until - self._clock()
            if retry_after > 0:
                return False, retry_after
            return True, 0.0

    def record_failure(self, key: str) -> None:
        """Count one rejected credential for ``key``; the ``max_attempts``-th inside the window
        starts the ``lockout_s`` ban."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                # Only outcomes allocate. ``allow`` on an unknown key stays free, so a flood of
                # never-resolved checks cannot grow the table either.
                while len(self._entries) >= _MAX_KEYS:
                    self._entries.popitem(last = False)
                entry = self._entries[key] = _Entry()
            else:
                self._entries.move_to_end(key)
            if now < entry.locked_until:
                # Raced past ``allow`` and lost after the ban began. Dropping the failure keeps the
                # ban's end fixed; counting it would let the caller stretch their own ban.
                return
            failures = entry.failures
            while failures and now - failures[0] > self._window_s:
                failures.pop(0)
            failures.append(now)
            if len(failures) >= self._max_attempts:
                entry.locked_until = now + self._lockout_s
                # The ban consumes these failures: after it ends the key starts clean, so the
                # lockout is exactly ``lockout_s`` instead of re-locking on the very next attempt.
                failures.clear()

    def record_success(self, key: str) -> None:
        """Clear ``key``: one accepted credential disproves guessing pressure from it."""
        with self._lock:
            self._entries.pop(key, None)

    def reset(self, key: str | None = None) -> None:
        """Forget one key, or every key when ``key`` is None (test isolation)."""
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries.pop(key, None)


# Policy for the shared sign-in budget. The window and lockout mirror the login limiter already in
# ``routes/auth.py`` (_LOGIN_WINDOW_SECONDS / _LOGIN_LOCKOUT_SECONDS) so every credential surface
# answers with the same one-minute vocabulary. The count is 10 rather than that limiter's 5: the
# hot guesses here are against an opaque 48-byte refresh token, where no number of attempts is
# realistic, and legit 401s on this budget are common and key-sharing (a stale tab, an expired
# token, several devices behind one NAT or one tunnel edge). Ten per minute per client still caps
# scripted guessing and the per-guess store work at noise level.
_TOKEN_MAX_ATTEMPTS = 10
_TOKEN_WINDOW_S = 60.0
_TOKEN_LOCKOUT_S = 60.0

token_attempts = AttemptLimiter(
    max_attempts = _TOKEN_MAX_ATTEMPTS,
    window_s = _TOKEN_WINDOW_S,
    lockout_s = _TOKEN_LOCKOUT_S,
)


def client_key(request) -> str:
    """Throttle key for the caller of ``request``: the forwarded client address the rest of the
    backend trusts (``utils.client_ip``, which resolves the tunnel visitor behind a loopback
    cloudflared peer), falling back to a constant so address-less callers share one bounded bucket
    instead of getting one key each."""
    return client_ip(request) or "_unknown"
