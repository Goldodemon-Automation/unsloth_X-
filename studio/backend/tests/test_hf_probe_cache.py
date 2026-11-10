# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The negative reachability cache.

While the Hub endpoint is down the answer is the same for minutes, so the paths that ask
again (a download, a dataset load, the fit estimate) should not each pay a connection
attempt to hear it a second time. The cache is deliberately asymmetric: an unreachable
verdict is reused for a short window, a reachable verdict never is.
"""

from __future__ import annotations

import threading
import time

import pytest

from utils import hf_probe
from utils.hf_probe import ProbeVerdict


class _Clock:
    """A clock the test moves by hand, so the window can be crossed without waiting."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Probe:
    """A stand-in for the network: counts attempts and hands back a canned verdict."""

    def __init__(self, verdict) -> None:
        self._verdict = verdict
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> ProbeVerdict:
        self.calls.append((url, timeout))
        if callable(self._verdict):
            return self._verdict(url, timeout)
        return self._verdict

    @property
    def urls(self) -> list[str]:
        return [url for url, _ in self.calls]


@pytest.fixture(autouse = True)
def _clean_module_cache():
    """The module-level cache outlives a test, so it starts and ends empty."""
    hf_probe.invalidate_probe_cache()
    yield
    hf_probe.invalidate_probe_cache()


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def cache(clock: _Clock) -> hf_probe.NegativeVerdictCache:
    return hf_probe.NegativeVerdictCache(clock = clock)


# --- the negative verdict is reused -------------------------------------------


def test_two_probes_inside_the_window_make_one_connection_attempt(cache):
    """The second probe inside the window reuses the first answer instead of dialing again."""
    probe = _Probe(ProbeVerdict(reachable = False))

    first = hf_probe.probe_endpoint("https://hub.example", timeout = 3.0, probe = probe, cache = cache)
    second = hf_probe.probe_endpoint("https://hub.example", timeout = 3.0, probe = probe, cache = cache)

    assert first.reachable is False
    assert second.reachable is False
    assert len(probe.calls) == 1, "an unreachable endpoint must not be dialed twice in the window"


def test_the_window_is_the_documented_one(cache, clock):
    """The negative verdict lives for 30s, and not a moment longer."""
    probe = _Probe(ProbeVerdict(reachable = False))

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    clock.advance(hf_probe.NEGATIVE_TTL_S - 0.1)
    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    assert len(probe.calls) == 1, "the verdict must still be inside the window"

    clock.advance(0.2)
    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    assert len(probe.calls) == 2, "the verdict must not outlive the window"


# --- the asymmetry -------------------------------------------------------------


def test_a_reachable_endpoint_is_probed_every_time(cache):
    """A stale "healthy" is the failure mode that matters, so it is never remembered."""
    probe = _Probe(ProbeVerdict(reachable = True))

    for _ in range(3):
        verdict = hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
        assert verdict.reachable is True

    assert len(probe.calls) == 3, "a reachable verdict must never be cached"


def test_a_reachable_verdict_leaves_the_cache_empty(cache):
    probe = _Probe(ProbeVerdict(reachable = True))

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)

    assert len(cache) == 0


# --- the key -------------------------------------------------------------------


def test_a_changed_url_is_never_answered_from_the_previous_one(cache):
    """A different mirror is a different question."""
    probe = _Probe(ProbeVerdict(reachable = False))

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    hf_probe.probe_endpoint("https://mirror.example", probe = probe, cache = cache)

    assert probe.urls == ["https://hub.example", "https://mirror.example"]


def test_a_cached_negative_does_not_leak_to_another_url(cache):
    """The unreachable answer for one endpoint says nothing about another."""
    probe = _Probe(
        lambda url, _timeout: ProbeVerdict(reachable = url != "https://hub.example")
    )

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    verdict = hf_probe.probe_endpoint("https://mirror.example", probe = probe, cache = cache)

    assert verdict.reachable is True, "the mirror answered, so the hub's negative must not apply"


def test_the_configured_endpoint_is_part_of_the_key(monkeypatch, cache):
    """A changed HF_ENDPOINT is a different endpoint, so it is probed again."""
    monkeypatch.setenv("HF_ENDPOINT", "https://hub.example")
    probe = _Probe(ProbeVerdict(reachable = False))

    hf_probe.probe_endpoint(probe = probe, cache = cache)
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    hf_probe.probe_endpoint(probe = probe, cache = cache)

    assert probe.urls == ["https://hub.example", "https://mirror.example"]


def test_the_probe_parameters_are_part_of_the_key(cache):
    """A different timeout is a different probe, so the verdict is not shared."""
    probe = _Probe(ProbeVerdict(reachable = False))

    hf_probe.probe_endpoint("https://hub.example", timeout = 3.0, probe = probe, cache = cache)
    hf_probe.probe_endpoint("https://hub.example", timeout = 10.0, probe = probe, cache = cache)

    assert len(probe.calls) == 2


def test_the_key_carries_the_url_and_the_parameters():
    key = hf_probe.endpoint_probe_key("https://hub.example/", timeout = 3.0)

    assert key == ("https://hub.example", 3.0)


# --- invalidation --------------------------------------------------------------


def test_invalidate_forces_the_next_probe(cache):
    probe = _Probe(ProbeVerdict(reachable = False))

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    assert len(probe.calls) == 1

    cache.invalidate()

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    assert len(probe.calls) == 2, "invalidate() must force a fresh answer"


def test_invalidate_one_key_leaves_the_others(cache):
    probe = _Probe(ProbeVerdict(reachable = False))
    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    hf_probe.probe_endpoint("https://mirror.example", probe = probe, cache = cache)

    cache.invalidate(hf_probe.endpoint_probe_key("https://hub.example"))

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    hf_probe.probe_endpoint("https://mirror.example", probe = probe, cache = cache)
    assert probe.urls == [
        "https://hub.example",
        "https://mirror.example",
        "https://hub.example",
    ]


def test_the_module_cache_forgets_on_invalidate():
    """The process-wide cache drops its verdicts on request too."""
    probe = _Probe(ProbeVerdict(reachable = False))

    hf_probe.probe_endpoint("https://hub.example", probe = probe)
    hf_probe.probe_endpoint("https://hub.example", probe = probe)
    assert len(probe.calls) == 1

    hf_probe.invalidate_probe_cache()

    hf_probe.probe_endpoint("https://hub.example", probe = probe)
    assert len(probe.calls) == 2


# --- timeouts ------------------------------------------------------------------


def test_a_timeout_is_not_remembered_as_long_as_a_refusal(cache, clock):
    """A timeout is ambiguous, so it is held for less time than a definite failure."""
    probe = _Probe(ProbeVerdict(reachable = False, timed_out = True))

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    assert len(probe.calls) == 1

    clock.advance(hf_probe.TIMEOUT_TTL_S + 0.1)

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    assert len(probe.calls) == 2, "a timeout must be re-probed once its shorter window ends"


def test_a_timeout_never_becomes_unreachable_forever(cache, clock):
    """The endpoint may come back; the timeout must not outlive the window by much."""
    probe = _Probe(ProbeVerdict(reachable = False, timed_out = True))

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)
    clock.advance(hf_probe.NEGATIVE_TTL_S * 10)

    hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)

    assert len(probe.calls) == 2, "a timeout must never be remembered forever"


def test_a_timeout_verdict_is_marked_as_such(cache):
    probe = _Probe(ProbeVerdict(reachable = False, timed_out = True))

    verdict = hf_probe.probe_endpoint("https://hub.example", probe = probe, cache = cache)

    assert verdict.reachable is False
    assert verdict.timed_out is True


# --- bounds, failures and concurrency ------------------------------------------


def test_the_cache_is_bounded(clock):
    """A caller that keys on something unbounded cannot grow the cache for ever."""
    cache = hf_probe.NegativeVerdictCache(max_entries = 4, clock = clock)

    for index in range(20):
        cache.put((f"https://hub{index}.example", 3.0), ProbeVerdict(reachable = False))

    assert len(cache) == 4


def test_the_oldest_negative_is_the_one_dropped(clock):
    cache = hf_probe.NegativeVerdictCache(max_entries = 2, clock = clock)
    cache.put(("a", 3.0), ProbeVerdict(reachable = False))
    cache.put(("b", 3.0), ProbeVerdict(reachable = False))
    cache.put(("c", 3.0), ProbeVerdict(reachable = False))

    assert cache.get(("a", 3.0)) is None
    assert cache.get(("c", 3.0)) is not None


def test_a_broken_probe_is_not_a_verdict(cache):
    """A probe that raises is not an answer, so nothing is remembered and the caller fails open."""

    def _boom(_url: str, _timeout: float) -> ProbeVerdict:
        raise RuntimeError("no network stack here")

    verdict = hf_probe.probe_endpoint("https://hub.example", probe = _boom, cache = cache)

    assert verdict.reachable is True, "a broken probe must fail open"
    assert len(cache) == 0, "a broken probe must not be cached"


def test_concurrent_callers_make_one_connection_attempt(cache):
    """Two threads asking at once should not both dial."""
    calls: list[str] = []
    start = threading.Event()

    def _slow(url: str, _timeout: float) -> ProbeVerdict:
        calls.append(url)
        time.sleep(0.05)
        return ProbeVerdict(reachable = False)

    def _ask() -> None:
        start.wait(2.0)
        hf_probe.probe_endpoint("https://hub.example", probe = _slow, cache = cache)

    threads = [threading.Thread(target = _ask) for _ in range(8)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(10.0)

    assert len(calls) == 1, "one endpoint down must not mean one attempt per thread"


def test_a_bare_bool_from_a_probe_is_understood(cache):
    """A simple probe may answer with a plain bool."""

    def _plain(_url: str, _timeout: float) -> bool:
        return False

    verdict = hf_probe.probe_endpoint("https://hub.example", probe = _plain, cache = cache)

    assert verdict.reachable is False
    assert verdict.timed_out is False
