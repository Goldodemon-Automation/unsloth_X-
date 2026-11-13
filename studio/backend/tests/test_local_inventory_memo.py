# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`/api/models/local` must not re-walk every scan root on every poll.

The route is polled by the model picker. `shared_scan` only shields *concurrent* callers, so a
sequential poll re-ran the whole scan -- measured 260 ms of disk work per request on a warm
process. These tests pin the short TTL memo that fixes it, and pin that it cannot outlive the
facts it was measured from: a cache-epoch bump or a scan-folder edit changes the key.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

if "structlog" not in sys.modules:  # keep this runnable without optional logging deps

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
from hub.utils import inventory_scan as hf_cache_scan


@pytest.fixture(autouse = True)
def _fresh_memo():
    models_route._compat_local_inventory_memo_clear()
    yield
    models_route._compat_local_inventory_memo_clear()


@pytest.fixture
def scanned(monkeypatch, tmp_path):
    """Count full scans; each one returns one stable, empty inventory."""
    calls: list[int] = []

    def fake_collect(models_root, custom_folders = None, sources = None):
        calls.append(1)
        return []

    sources = models_route._CompatLocalInventorySources(
        hf_cache_dir = tmp_path / "hf",
        legacy_hf = tmp_path / "legacy",
        hf_default = tmp_path / "default",
        lm_dirs = (),
        known_hf_caches = (),
        hermes_dirs = (),
        ollama_dirs = (),
        omlx_dirs = (),
    )
    monkeypatch.setattr(models_route, "collect_local_models", fake_collect)
    monkeypatch.setattr(models_route, "_compat_local_inventory_sources", lambda: sources)
    monkeypatch.setattr(models_route, "list_scan_folders", None, raising = False)
    monkeypatch.setitem(sys.modules, "storage.studio_db", types.SimpleNamespace(list_scan_folders = lambda: []))
    return calls, sources, tmp_path


def _scan(tmp_path, sources):
    return asyncio.run(
        models_route._shared_compat_local_inventory_scan(tmp_path / "models", sources)
    )


def test_a_second_poll_of_an_unchanged_tree_does_not_rescan(scanned):
    calls, sources, tmp_path = scanned
    _scan(tmp_path, sources)
    _scan(tmp_path, sources)
    asyncio.run(models_route._shared_compat_local_inventory_scan(tmp_path / "models", sources))
    assert len(calls) == 1, (
        "an unchanged inventory must be reused inside the TTL, not re-walked per poll; "
        f"got {len(calls)} scans for 3 requests"
    )


def test_the_memo_expires_so_a_new_folder_contents_is_seen(monkeypatch, scanned):
    calls, sources, tmp_path = scanned
    clock = {"now": 5000.0}
    monkeypatch.setattr(models_route.time, "monotonic", lambda: clock["now"])

    _scan(tmp_path, sources)
    assert len(calls) == 1

    clock["now"] += models_route._COMPAT_LOCAL_INVENTORY_MEMO_TTL_SECONDS + 0.01
    _scan(tmp_path, sources)
    assert len(calls) == 2


def test_a_cache_invalidation_epoch_bump_forces_a_rescan(monkeypatch, scanned):
    calls, sources, tmp_path = scanned
    _scan(tmp_path, sources)
    assert len(calls) == 1

    monkeypatch.setattr(hf_cache_scan, "hf_cache_scans_epoch", lambda: 99)
    _scan(tmp_path, sources)
    assert len(calls) == 2, "a bumped epoch means the tree changed; the memo must not answer"


def test_editing_the_scan_folder_set_forces_a_rescan(monkeypatch, scanned):
    calls, sources, tmp_path = scanned
    _scan(tmp_path, sources)
    assert len(calls) == 1

    monkeypatch.setitem(
        sys.modules,
        "storage.studio_db",
        types.SimpleNamespace(list_scan_folders = lambda: [{"path": str(tmp_path / "extra"), "recursive": False}]),
    )
    _scan(tmp_path, sources)
    assert len(calls) == 2


def test_callers_cannot_mutate_the_memoised_list(scanned):
    calls, sources, tmp_path = scanned
    first = _scan(tmp_path, sources)
    first.append("tampered")
    second = _scan(tmp_path, sources)
    assert second == [], "a caller mutating its own rows must not change the cached inventory"
    assert len(calls) == 1


def test_the_memo_is_bounded(scanned):
    calls, sources, tmp_path = scanned
    for index in range(models_route._COMPAT_LOCAL_INVENTORY_MEMO_MAX + 5):
        asyncio.run(
            models_route._shared_compat_local_inventory_scan(tmp_path / f"models-{index}", sources)
        )
    assert len(models_route._compat_local_inventory_memo) <= (
        models_route._COMPAT_LOCAL_INVENTORY_MEMO_MAX
    )


def test_expired_entries_are_dropped_when_a_new_one_is_stored(monkeypatch, scanned):
    calls, sources, tmp_path = scanned
    clock = {"now": 100.0}
    monkeypatch.setattr(models_route.time, "monotonic", lambda: clock["now"])
    _scan(tmp_path, sources)
    clock["now"] += models_route._COMPAT_LOCAL_INVENTORY_MEMO_TTL_SECONDS + 1
    _scan(tmp_path / "other", sources)
    assert len(models_route._compat_local_inventory_memo) == 1
