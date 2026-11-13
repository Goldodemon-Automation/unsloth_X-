# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression test and benchmark for the polled llama.cpp architecture status.

GET /api/llama/architectures is called on every Settings open and roughly once a
second while an install runs, so what matters is not how long one call takes on
one machine but how much expensive work each call does: GitHub release lookups,
marker reads off disk, installed-tree scans for llama-server, and the cmake /
compiler probe. Every test here counts those instead of timing anything, which is
why they stay green on a slow or loaded Windows box.

The numbers this pins (per GET, six architectures in the manifest); "tree scan"
is the llama-server rglob an install marker without a binary_relpath needs:

    first call      4 release lookups, 6 marker reads, 0-2 tree scans, 1-2 probes
    repeated calls  0 release lookups, 0 marker reads, 0 tree scans, 1 probe

The 1 probe per call is the fingerprint asking whether this host can still build
from source (dory's availability depends on it); the marker identity is one stat
per architecture. Neither is the expensive work, and both are the price of never
serving an answer the disk contradicts.

Hermetic: the release lookup is stubbed at utils.llama_cpp_architectures'
_fetch_release, so nothing here touches the network, and the install root is a
temporary directory.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.llama_cpp_architectures as arch  # noqa: E402
import utils.llama_cpp_update as update  # noqa: E402

# The only installed asset any test needs: a CPU bundle ik_llama.cpp publishes.
_IK_ASSET = "ik_llama-main-b5658-bin-win-cpu-x86_64_v3-avx2.zip"


class _Work:
    """Counts of the expensive work one status call may do.

    Instrumented at the sinks the payload is built from, not at the wrapper, so a
    change that stops calling the wrapper but still scans the disk is caught.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.release_lookups = 0
        self.marker_reads = 0
        self.tree_scans = 0
        self.tool_probes = 0
        # Repo -> number of times a fetch actually started for it.
        self.fetched_repos: list[str] = []

    def bump(self, name: str, repo: str = "") -> None:
        with self.lock:
            setattr(self, name, getattr(self, name) + 1)
            if repo:
                self.fetched_repos.append(repo)

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                "release_lookups": self.release_lookups,
                "marker_reads": self.marker_reads,
                "tree_scans": self.tree_scans,
                "tool_probes": self.tool_probes,
            }

    def reset(self) -> None:
        with self.lock:
            self.release_lookups = 0
            self.marker_reads = 0
            self.tree_scans = 0
            self.tool_probes = 0
            self.fetched_repos = []


def _release(tag: str = "main-b5658") -> dict:
    return {
        "tag": tag,
        "html_url": f"https://example.test/{tag}",
        "assets": [
            {
                "name": _IK_ASSET,
                "url": "https://example.test/ik.zip",
                "size": 12345,
                "digest": None,
            }
        ],
    }


@pytest.fixture
def work(monkeypatch):
    """Count the expensive sinks and keep the release lookup offline."""
    counters = _Work()

    def fetch(repo: str):
        counters.bump("release_lookups", repo)
        return _release()

    def raise_offline(repo: str):
        counters.bump("release_lookups", repo)
        raise OSError(f"no route to {repo}")

    real_read_marker = arch._read_marker
    real_safe_binary = arch._safe_marker_binary
    real_source_tools = arch._source_build_tools

    def read_marker(architecture: str):
        counters.bump("marker_reads")
        return real_read_marker(architecture)

    def safe_binary(architecture: str, marker: dict):
        counters.bump("tree_scans")
        return real_safe_binary(architecture, marker)

    def source_tools():
        counters.bump("tool_probes")
        return real_source_tools()

    monkeypatch.setattr(arch, "_fetch_release", fetch)
    monkeypatch.setattr(arch, "_read_marker", read_marker)
    monkeypatch.setattr(arch, "_safe_marker_binary", safe_binary)
    monkeypatch.setattr(arch, "_source_build_tools", source_tools)
    monkeypatch.setattr(arch, "_host", lambda: ("windows", "x64"))
    counters.offline = raise_offline
    counters.fetch = fetch
    return counters


@pytest.fixture(autouse = True)
def _isolated(monkeypatch, tmp_path):
    arch.reset_caches()
    arch._reset_for_tests()
    update.reset_architecture_status_cache()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    for name in (
        "LLAMA_SERVER_PATH",
        "UNSLOTH_LLAMA_CPP_PATH",
        "UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(arch, "_active_id", lambda: "normal")
    yield
    arch.reset_caches()
    arch._reset_for_tests()
    update.reset_architecture_status_cache()


_BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"


def _install(architecture: str, tag: str = "main-b5658") -> Path:
    """Write the install marker and binary _install_dir expects for a fork."""
    install_dir = arch._install_dir(architecture)
    binary = install_dir / "build" / "bin" / _BINARY_NAME
    binary.parent.mkdir(parents = True)
    binary.write_text("stub", encoding = "utf-8")
    (install_dir / arch.ARCHITECTURE_MARKER_NAME).write_text(
        '{"architecture":"%s","binary_relpath":"build/bin/%s",'
        '"release_tag":"%s"}' % (architecture, _BINARY_NAME, tag),
        encoding = "utf-8",
    )
    return binary


def _install_without_binary_relpath(architecture: str, tag: str = "v0.4.9") -> Path:
    """A marker with no binary_relpath, which forces the llama-server tree scan.

    Every real install written by an older Unsloth looks like this, so the scan
    -- not the marker lookup -- is the branch a warm poll must not repeat.
    """
    install_dir = arch._install_dir(architecture)
    binary = install_dir / "build" / "bin" / _BINARY_NAME
    binary.parent.mkdir(parents = True)
    binary.write_text("stub", encoding = "utf-8")
    for index in range(50):
        (binary.parent / f"ggml-{index}.dll").write_text("x", encoding = "utf-8")
    (install_dir / arch.ARCHITECTURE_MARKER_NAME).write_text(
        '{"architecture":"%s","release_tag":"%s"}' % (architecture, tag),
        encoding = "utf-8",
    )
    return binary


def _by_id(status: dict) -> dict[str, dict]:
    return {profile["id"]: profile for profile in status["profiles"]}


def test_repeated_calls_do_not_resolve_or_rescan(work):
    first = update.get_architecture_status()
    cold = work.snapshot()

    # Baseline for the same payload without the cache, for the report and so a
    # future change that quietly makes one call free by making it empty fails.
    assert cold["release_lookups"] == 4, "one per prebuilt repo on a cold cache"
    assert cold["marker_reads"] == 6, "one per manifest row"
    assert cold["tool_probes"] >= 1

    work.reset()
    for _ in range(5):
        repeated = update.get_architecture_status()

    warm = work.snapshot()
    assert warm == {
        "release_lookups": 0,
        "marker_reads": 0,
        "tree_scans": 0,
        # One per call: the fingerprint re-checks whether this host can build
        # dory from source, because that changes the row.
        "tool_probes": 5,
    }
    assert repeated["profiles"] == first["profiles"]
    assert repeated["active"] == "normal"

    # And the release memo behind it was not bypassed either.
    assert arch._RELEASE_CACHE, "releases stay resolved, not re-fetched"


def test_repeated_calls_do_not_scan_an_installed_tree(work):
    binary = _install("ik")
    scanned = _install_without_binary_relpath("bee")
    first = update.get_architecture_status()
    cold = work.snapshot()

    # ik resolves straight from its marker, bee needs the llama-server tree scan.
    assert cold["tree_scans"] == 2
    assert _by_id(first)["ik"]["binary_path"] == str(binary)
    assert _by_id(first)["bee"]["binary_path"] == str(scanned)
    assert _by_id(first)["bee"]["installed"] is True

    work.reset()
    for _ in range(5):
        repeated = update.get_architecture_status()

    warm = work.snapshot()
    assert warm["tree_scans"] == 0, "the scan is what the cache is for"
    assert warm["marker_reads"] == 0
    assert warm["release_lookups"] == 0
    assert _by_id(repeated)["ik"] == _by_id(first)["ik"]
    assert _by_id(repeated)["bee"] == _by_id(first)["bee"]


def test_the_uncached_call_is_the_cost_this_avoids(work):
    """The "before" counts, pinned: bypassing the wrapper pays all of it."""
    _install("ik")
    _install_without_binary_relpath("bee")

    for _ in range(3):
        arch.get_status()

    per_call = {name: count / 3 for name, count in work.snapshot().items()}
    assert per_call == {
        "release_lookups": 4 / 3,  # memoised after the first call, not free
        "marker_reads": 6.0,
        "tree_scans": 2.0,
        "tool_probes": 1.0,
    }


def test_force_refresh_does_resolve_and_rescan(work):
    update.get_architecture_status()
    work.reset()

    forced = update.get_architecture_status(force_refresh = True)

    counts = work.snapshot()
    assert counts["release_lookups"] == 4, "force_refresh must bypass the cache"
    assert counts["marker_reads"] == 6, "and the installed-tree scan with it"
    assert sorted(work.fetched_repos) == sorted(update._architecture_repos())
    assert _by_id(forced)["ik"]["latest_tag"] == "main-b5658"

    # A forced call also refreshes what the plain path serves next.
    work.reset()
    update.get_architecture_status()
    assert work.snapshot()["release_lookups"] == 0


def test_a_changed_marker_invalidates_a_cached_answer(work):
    status = update.get_architecture_status()
    assert _by_id(status)["ik"]["installed"] is False
    work.reset()

    _install("ik", tag = "main-b5700")
    after = update.get_architecture_status()

    assert work.snapshot()["marker_reads"] == 6, "the marker changed, so rescan"
    assert _by_id(after)["ik"]["installed"] is True
    assert _by_id(after)["ik"]["installed_tag"] == "main-b5700"

    # Removing the binary changes the row without touching the marker, so the
    # cheap per-read verification has to notice it.
    Path(_by_id(after)["ik"]["binary_path"]).unlink()
    work.reset()
    gone = update.get_architecture_status()
    assert _by_id(gone)["ik"]["installed"] is False
    assert work.snapshot()["tree_scans"] == 1


def test_the_job_is_read_fresh_while_an_install_runs(work):
    """Progress must move on every poll even when the payload is cached."""
    arch._set_job(state = "running", architecture = "ik", progress = 0.1, message = "Downloading")
    update.get_architecture_status()
    work.reset()

    arch._set_job(progress = 0.4, message = "Downloading 40%")
    second = update.get_architecture_status()

    assert second["job"]["progress"] == 0.4
    assert second["job"]["message"] == "Downloading 40%"
    # A running job is what makes get_status skip its release lookups; the poll
    # itself must still not rescan anything it does not have to.
    assert work.snapshot()["marker_reads"] == 0
    assert work.snapshot()["release_lookups"] == 0

    # Finishing the job changes the key, so the last poll re-reads the tree.
    arch._set_job(state = "success", progress = 1.0)
    work.reset()
    finished = update.get_architecture_status()
    assert finished["job"]["state"] == "success"
    assert work.snapshot()["marker_reads"] == 6


def test_the_status_call_never_needs_the_install_lock(work):
    """An install holds _OPERATION_LOCK for its whole run; polling must not wait.

    No timing here: the assertion is that the call returns at all while the
    operation lock is held, which is the deadlock a status read must never have
    with the install it is reporting on.
    """
    assert arch._OPERATION_LOCK.acquire(blocking = False)
    try:
        arch._set_job(state = "running", architecture = "ik", progress = 0.3)
        polled = update.get_architecture_status()
        forced = update.get_architecture_status(force_refresh = True)
    finally:
        arch._OPERATION_LOCK.release()

    assert polled["job"]["progress"] == 0.3
    assert forced["job"]["state"] == "running"
    assert len(polled["profiles"]) == len(arch.ARCHITECTURES)


def test_a_failed_lookup_still_reports_installed_facts(work, monkeypatch):
    binary = _install("ik", tag = "main-b5658")
    monkeypatch.setattr(arch, "_fetch_release", work.offline)

    status = update.get_architecture_status()
    rows = _by_id(status)

    # Degraded, not broken: the release lookup failed and the response says so.
    assert status["release_lookup_failed"] is True
    assert rows["ik"]["reason"] == "release_lookup_failed"
    assert rows["ik"]["latest_tag"] is None

    # Installed facts survive intact, which is the whole point of not failing the
    # request when GitHub is unreachable.
    assert rows["ik"]["installed"] is True
    assert rows["ik"]["installed_tag"] == "main-b5658"
    assert rows["ik"]["binary_path"] == str(binary)
    assert rows["ik"]["selectable"] is True
    assert len(status["profiles"]) == len(arch.ARCHITECTURES)

    # Dory keeps its pinned-source fallback instead of looking like a dead fork.
    assert rows["dory"]["source_build"] is True
    assert rows["dory"]["reason"] in {"source_build", "build_tools_missing"}


def test_a_failed_lookup_is_not_retried_on_every_poll(work, monkeypatch):
    monkeypatch.setattr(arch, "_fetch_release", work.offline)
    update.get_architecture_status()
    attempts = work.snapshot()["release_lookups"]
    assert attempts == 4

    work.reset()
    for _ in range(5):
        status = update.get_architecture_status()

    # The failure memo _latest_release keeps (45 s) stops the retries, and the
    # answer still reports the failure rather than forgetting it.
    assert work.snapshot()["release_lookups"] == 0
    assert status["release_lookup_failed"] is True

    # A forced check does retry, which is what the panel's "check now" promises.
    work.reset()
    update.get_architecture_status(force_refresh = True)
    assert work.snapshot()["release_lookups"] == 4


def test_a_wedged_github_is_abandoned_not_waited_on(work, monkeypatch):
    """The lookup is bounded by a wall clock, so the status call still answers."""
    finished = threading.Event()

    def wedged(repo: str):
        work.bump("release_lookups", repo)
        # Outlives the deadline; the worker is abandoned, not joined.
        finished.wait(0.6)
        return _release()

    monkeypatch.setattr(arch, "_fetch_release", wedged)
    monkeypatch.setattr(update, "ARCHITECTURE_RELEASE_DEADLINE_SECONDS", 0.05)

    status = update.get_architecture_status()

    # Relative check, not a timing bound: the call returned before the fetch did.
    assert finished.is_set() is False
    assert work.snapshot()["release_lookups"] == 4, "all four were attempted"
    assert status["release_lookup_failed"] is True
    # And the installed facts are still there to render.
    assert len(status["profiles"]) == len(arch.ARCHITECTURES)
    finished.set()


def test_the_status_cache_is_bounded(work, monkeypatch):
    """Many distinct keys must not grow the cache without limit."""
    for index in range(update._ARCHITECTURE_STATUS_CACHE_MAX_ENTRIES * 3):
        monkeypatch.setenv("LLAMA_SERVER_PATH", f"/tmp/llama-{index}/llama-server")
        update.get_architecture_status()
        assert len(update._architecture_status_cache) <= update._ARCHITECTURE_STATUS_CACHE_MAX_ENTRIES


def test_an_expired_entry_is_not_served(work, monkeypatch):
    # The entry carries the TTL it was written with, so expire it from the start.
    monkeypatch.setattr(update, "ARCHITECTURE_STATUS_CACHE_TTL_SECONDS", 0.0)
    update.get_architecture_status()
    work.reset()

    update.get_architecture_status()

    assert work.snapshot()["marker_reads"] == 6, "an expired entry must not answer"


def test_an_expired_entry_recomputes_but_does_not_refetch(work, monkeypatch):
    """Past the TTL the scan repeats, while the release memo still answers.

    That split is the point of two caches: a poll every few seconds re-reads the
    local tree at most once per TTL, and never re-resolves a release that
    utils.llama_cpp_architectures already resolved.
    """
    # The entry carries the TTL it was written with, so expire it from the start.
    monkeypatch.setattr(update, "ARCHITECTURE_STATUS_CACHE_TTL_SECONDS", 0.0)
    update.get_architecture_status()
    work.reset()

    for _ in range(3):
        update.get_architecture_status()

    assert work.snapshot()["marker_reads"] == 18, "one rescan per call, as uncached"
    assert work.snapshot()["release_lookups"] == 0, "the release memo still answers"
