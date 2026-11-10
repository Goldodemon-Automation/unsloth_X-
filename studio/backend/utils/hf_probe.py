# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe optional Hub files without caching missing entries.

``hf_hub_download`` can leave a ref pointing to a snapshot that is not on disk after a 404, causing
``scan_cache_dir`` to omit the repo. Metadata requests answer the same question without changing
the cache.

The other probe question is answered here as well: does the configured Hub endpoint answer at all?
That answer is re-made on the paths that run often -- a download, a dataset load, the fit estimate
-- and while the endpoint is down the answer is the same for minutes, so each of those paths pays
a connection attempt to learn what the previous one already knew. A short negative cache removes
that repetition.

The asymmetry is deliberate: an unreachable verdict is remembered for a short window, a reachable
verdict never is. A stale "reachable" is the failure mode that matters, because it is what sends
the next download at an endpoint that has stopped answering; a stale "unreachable" only costs a
short delay before the endpoint is tried again.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional, Union

from loggers import get_logger

logger = get_logger(__name__)


def hf_file_definitely_absent(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Return True only when the Hub confirms that *filename* is absent.

    Offline, authentication, rate-limit, and resolution failures return False so callers preserve
    their existing download and error handling.
    """
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        from hub.utils.hf_tokens import call_with_anonymous_retry
    except Exception:  # noqa: BLE001 - an unimportable hub is the caller's problem, not ours
        return False

    try:
        url = hf_hub_url(
            repo_id = repo_id,
            filename = filename,
            repo_type = repo_type,
            revision = revision,
        )
        # A refused credential is not an answer about the file: ask once more without it.
        call_with_anonymous_retry(lambda sent: get_hf_file_metadata(url, token = sent), token)
    except LocalEntryNotFoundError:
        return False
    except EntryNotFoundError:
        return True
    except Exception as exc:  # noqa: BLE001 - no answer is not an answer of "absent"
        logger.debug("Could not probe %s for %s: %s", repo_id, filename, exc)
        return False
    return False


# ── Endpoint reachability ────────────────────────────────────────────────────

# How long an unreachable verdict is reused. Long enough that the paths which ask the
# same question inside one job (a download, a dataset load, the fit estimate) share one
# answer instead of one connection attempt each, and short enough that an endpoint that
# comes back is picked up by the next job rather than the next hour.
NEGATIVE_TTL_S = 30.0

# A timeout is not a refusal. The endpoint may be down or merely slow, and a slow mirror
# written off for the full window would be declared dead long after it came back, so a
# timeout is held for a fraction of the window and re-probed sooner. It is never held
# forever: the entry expires, and ``invalidate()`` drops it.
TIMEOUT_TTL_S = 5.0

# The cache holds one entry per key and only ever negatives, so it stays tiny in practice.
# The bound is the guard against a caller that keys on something unbounded, which would
# otherwise grow the cache for the life of the process.
MAX_CACHED_NEGATIVES = 64

# Probes are serialised in small groups rather than not at all: two threads asking about the
# same endpoint should make one connection attempt between them, not one each. A fixed pool
# keeps the number of locks bounded however many keys there are.
_PROBE_LOCK_SLOTS = 16


@dataclass(frozen = True)
class ProbeVerdict:
    """What one probe attempt concluded.

    ``reachable`` is the answer callers act on: the endpoint answered, or the wire gave a
    reason that nothing is serving there. ``timed_out`` marks the ambiguous case -- the
    attempt ran out of time -- which is remembered for less time than a definite failure,
    because it is evidence of nothing more than a slow answer.
    """

    reachable: bool
    timed_out: bool = False


class NegativeVerdictCache:
    """A bounded, thread-safe store of negative probe verdicts.

    Only negatives are kept. A reachable verdict is the one that must not go stale, so
    every caller re-probes rather than trusting a remembered "it is fine". The clock is
    injected, so the window can be exercised without waiting for it, and the entries are
    ordered so the oldest negative is the one dropped when the bound is reached.
    """

    def __init__(
        self,
        *,
        ttl: float = NEGATIVE_TTL_S,
        timeout_ttl: float = TIMEOUT_TTL_S,
        max_entries: int = MAX_CACHED_NEGATIVES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = float(ttl)
        self._timeout_ttl = float(timeout_ttl)
        self._max_entries = max(1, int(max_entries))
        self._clock = clock
        self._entries: "OrderedDict[Hashable, tuple[float, ProbeVerdict]]" = OrderedDict()
        self._lock = threading.Lock()
        self._probe_locks = [threading.RLock() for _ in range(_PROBE_LOCK_SLOTS)]

    def get(self, key: Hashable) -> Optional[ProbeVerdict]:
        """The remembered negative verdict for *key* while it is fresh, else None."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, verdict = entry
            if self._clock() >= expires_at:
                # Expired: the endpoint may have come back, so this is dropped rather than
                # kept as a stale answer.
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return verdict

    def put(self, key: Hashable, verdict: ProbeVerdict) -> None:
        """Remember a negative verdict for *key*. A reachable verdict is not remembered."""
        if verdict.reachable:
            return
        ttl = self._timeout_ttl if verdict.timed_out else self._ttl
        with self._lock:
            self._entries[key] = (self._clock() + ttl, verdict)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last = False)

    def invalidate(self, key: Optional[Hashable] = None) -> None:
        """Forget one key, or every key when *key* is None.

        Called when the network changes under the process -- the operator reconnects, the
        mirror is swapped -- so the next probe answers for the world as it is now.
        """
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries.pop(key, None)

    def probe_lock(self, key: Hashable) -> Any:
        """The reentrant lock that serialises probes for *key*.

        Reentrant, so a probe that asks this cache about its own key cannot deadlock.
        """
        return self._probe_locks[hash(key) % _PROBE_LOCK_SLOTS]

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_default_cache = NegativeVerdictCache()


def _as_verdict(value: Union[ProbeVerdict, bool]) -> ProbeVerdict:
    """Accept a bare bool from a simple probe."""
    if isinstance(value, ProbeVerdict):
        return value
    return ProbeVerdict(reachable = bool(value))


def cached_probe(
    probe: Callable[[], Union[ProbeVerdict, bool]],
    key: Hashable,
    *,
    cache: Optional[NegativeVerdictCache] = None,
) -> ProbeVerdict:
    """Run *probe* unless a fresh negative verdict for *key* is already remembered.

    *probe* takes no arguments and returns a :class:`ProbeVerdict` (a bare bool is read as
    "reachable" or not). A negative verdict is remembered for the window and reused by the
    callers that follow; a reachable one is not remembered, so the next caller probes again.

    The key carries the question, not the answer: the endpoint URL and the probe's own
    parameters. A mirror that is swapped under Studio is therefore a different question,
    never answered from the previous mirror's verdict.
    """
    store = _default_cache if cache is None else cache
    remembered = store.get(key)
    if remembered is not None:
        return remembered
    with store.probe_lock(key):
        # Another thread may have probed this key while this one waited for the lock.
        remembered = store.get(key)
        if remembered is not None:
            return remembered
        try:
            verdict = _as_verdict(probe())
        except Exception as exc:  # noqa: BLE001 - a broken probe is not a verdict
            logger.debug("Probe %s failed: %s", key, exc)
            return ProbeVerdict(reachable = True)
        store.put(key, verdict)
        return verdict


def endpoint_probe_key(
    endpoint: Optional[str] = None,
    *,
    timeout: float = 3.0,
) -> tuple[str, float]:
    """The cache key for one endpoint probe: the URL plus the probe's own parameters.

    The URL is part of the key, so a mirror that is swapped (or an ``HF_ENDPOINT`` that
    changes) is a different question and is never answered from the previous URL's verdict.
    The parameters are part of it for the same reason: a probe with a different timeout is
    a different probe.
    """
    return (_endpoint_url(endpoint), float(timeout))


def probe_endpoint(
    endpoint: Optional[str] = None,
    *,
    timeout: float = 3.0,
    probe: Optional[Callable[[str, float], Union[ProbeVerdict, bool]]] = None,
    cache: Optional[NegativeVerdictCache] = None,
) -> ProbeVerdict:
    """Whether *endpoint* answers, reusing a recent negative verdict.

    *endpoint* defaults to the configured Hub endpoint. *probe* is called as
    ``probe(url, timeout)``; the default makes one bounded HEAD request.
    """
    key = endpoint_probe_key(endpoint, timeout = timeout)
    runner = probe if probe is not None else _default_endpoint_probe
    return cached_probe(lambda: runner(key[0], key[1]), key, cache = cache)


def invalidate_probe_cache(key: Optional[Hashable] = None) -> None:
    """Forget the remembered negative verdicts: one key, or all of them."""
    _default_cache.invalidate(key)


def _endpoint_url(endpoint: Optional[str]) -> str:
    """The URL to probe: the caller's, else the configured Hub endpoint.

    The configured value goes through ``utils.hf_endpoint``, so the probe asks about the
    endpoint the backend itself uses: a value that module rejects is replaced by the
    official Hub there, and probing the rejected value instead would answer a question
    nothing else in the process asks.
    """
    if endpoint:
        return str(endpoint).strip().rstrip("/")
    try:
        from utils.hf_endpoint import get_hf_endpoint

        return get_hf_endpoint().rstrip("/")
    except Exception:  # noqa: BLE001 - fall through to the raw value
        pass
    try:
        from utils.utils import hf_endpoint_url

        return hf_endpoint_url().rstrip("/")
    except Exception:  # noqa: BLE001 - the default Hub is a safe answer to fall back to
        return "https://huggingface.co"


def _urllib_opener(url: str):
    """An opener that follows the same proxy the Hub client would use.

    None when egress goes through a proxy urllib cannot speak (socks5): urllib's failure
    then proves nothing about the endpoint, and the probe must not read it as "down".
    """
    import urllib.request

    try:
        from utils.utils import hf_proxy_for_endpoint, hf_proxy_usable_by_urllib
    except Exception:  # noqa: BLE001 - no proxy helpers means no proxy
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        proxy = hf_proxy_for_endpoint(url)
    except Exception:  # noqa: BLE001 - an unreadable proxy setting is no proxy
        proxy = None
    if proxy and not hf_proxy_usable_by_urllib(proxy):
        return None
    if not proxy:
        # An empty ProxyHandler keeps urllib from picking up a proxy the Hub client
        # would not use, which would probe the wrong route entirely.
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    return urllib.request.build_opener(handler)


def _default_endpoint_probe(url: str, timeout: float) -> ProbeVerdict:
    """One bounded HEAD request against *url*.

    The question is "does the endpoint answer", not "is the Hub healthy": any HTTP
    response, including an error, means something is serving there. A refused connection
    or a host that does not resolve means nothing is. A timeout is reported as such,
    because it is ambiguous and the cache treats it differently from a refusal.

    The attempt runs on a daemon thread joined with a deadline, so a resolver or a server
    that never answers cannot hold the caller past the timeout.
    """
    import ssl
    import urllib.error
    import urllib.request

    opener = _urllib_opener(url)
    if opener is None:
        # urllib cannot speak this proxy, so its failure is not evidence about the
        # endpoint. Failing open is the safe direction: the caller then tries for real.
        return ProbeVerdict(reachable = True)

    outcome = {"reachable": True, "timed_out": False}

    def _attempt() -> None:
        try:
            request = urllib.request.Request(url, method = "HEAD")
            with opener.open(request, timeout = timeout):
                outcome["reachable"] = True
        except urllib.error.HTTPError:
            # The endpoint answered. A 404 or a 503 still proves it is there.
            outcome["reachable"] = True
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                outcome["reachable"] = False
                outcome["timed_out"] = True
            elif isinstance(reason, ConnectionRefusedError):
                # Nothing is serving on that port: the endpoint is down.
                outcome["reachable"] = False
            elif isinstance(reason, (ssl.SSLError, ConnectionError)):
                # The wire answered (a reset, a broken TLS handshake): egress exists, and
                # the real request is what surfaces the problem.
                outcome["reachable"] = True
            elif isinstance(reason, OSError):
                # gaierror and friends: the host does not resolve, or there is no route.
                outcome["reachable"] = False
            else:
                # A client-side reason ("no host given") says nothing about the endpoint.
                outcome["reachable"] = True
        except TimeoutError:
            outcome["reachable"] = False
            outcome["timed_out"] = True
        except ConnectionError:
            outcome["reachable"] = True
        except OSError:
            outcome["reachable"] = False
        except Exception:  # noqa: BLE001 - a broken endpoint or proxy is not a verdict
            outcome["reachable"] = True

    thread = threading.Thread(target = _attempt, daemon = True)
    thread.start()
    thread.join(timeout + 1)
    if thread.is_alive():
        # Hung past the deadline: it did not answer in time.
        return ProbeVerdict(reachable = False, timed_out = True)
    return ProbeVerdict(reachable = outcome["reachable"], timed_out = outcome["timed_out"])
