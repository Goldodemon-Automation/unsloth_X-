# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The sizing walk runs once per change, not once per read.

cache_inventory() is what the disk-usage card and the settings page poll, so a
walk per read would make every poll as slow as the biggest cache on the
machine. These tests count both walks: the recursive sizing pass, and the
bounded signature check that guards reuse — which is the proof a repeat read
pays for whenever it falls outside the signature-check window
(_SIGNATURE_CHECK_TTL_SECONDS). They pin: two reads over an unchanged tree
walk once and prove the tree once, the proof is repeated past the window but
not inside it, a changed tree re-measures once its mismatch is proven,
refresh=True walks unconditionally, invalidate() and a purge force a walk, and
a served entry is a copy its caller cannot poison the memo with.

The definition table (CACHE_DEFINITIONS) is the module-level constant that
names every cache root, and it is monkeypatched because the real resolvers
read machine state — environment variables, HF settings — that a test cannot
inject per case. The single test definition carries its own always-passing
gate (custom_block) so the standard gate's protected-path policy for the test
host cannot decide whether the walk happens at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils import cache_inventory


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")
    return path


@pytest.fixture
def one_cache(tmp_path, monkeypatch):
    """One cache rooted in tmp_path, a fresh memo, and a counter for its walks."""
    root = tmp_path / "model-cache"
    root.mkdir()
    definition = cache_inventory.CacheDefinition(
        "test_cache",
        cache_inventory.GROUP_MODELS,
        False,
        lambda: [root],
        # Its own gate, always passing. The standard one consults this
        # machine's real protected paths, which would decide from host state
        # whether the test sees a walk at all.
        custom_block = lambda: None,
    )
    monkeypatch.setattr(cache_inventory, "CACHE_DEFINITIONS", (definition,))
    # Purges resolve their key through definition_for's own index, which is
    # built from the real table at import; keep the two in step.
    monkeypatch.setattr(cache_inventory, "_DEFINITIONS_BY_KEY", {definition.key: definition})
    # Sizes are memoized per process in production; a test must never read one
    # another test measured.
    monkeypatch.setattr(cache_inventory, "_size_cache", {})
    monkeypatch.setattr(cache_inventory, "_size_epochs", {})
    walks: list = []
    real_measure = cache_inventory._measure_root

    def counting(target, **kwargs):
        walks.append(Path(target))
        return real_measure(target, **kwargs)

    monkeypatch.setattr(cache_inventory, "_measure_root", counting)
    return root, walks


@pytest.fixture
def sig_walks(monkeypatch):
    """Count the bounded signature walks at BOTH call sites.

    cache_inventory bound directory_signature by import, and is_unchanged calls
    it through stat_signature's own module globals; a counter missing either
    binding would count only half the walks.
    """
    from utils import stat_signature

    walks: list = []
    real_signature = stat_signature.directory_signature

    def counting(root, **kwargs):
        walks.append(Path(root))
        return real_signature(root, **kwargs)

    monkeypatch.setattr(cache_inventory, "directory_signature", counting)
    monkeypatch.setattr(stat_signature, "directory_signature", counting)
    return walks


def test_two_reads_of_an_unchanged_tree_walk_it_once(one_cache):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)

    first = cache_inventory.cache_inventory()
    second = cache_inventory.cache_inventory()

    assert first["total_bytes"] == 10
    assert second["total_bytes"] == 10
    assert len(walks) == 1, "the second read re-walked a tree nothing had changed"


def test_a_file_added_between_reads_walks_again(one_cache, monkeypatch):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)
    cache_inventory.cache_inventory()
    assert len(walks) == 1

    _write(root / "added.bin", "y" * 5)
    # The change lands inside the signature-check window, where the memo is
    # served on purpose; closing the window is exactly what the clock does to
    # it, and the first read past it must prove the trees again.
    monkeypatch.setattr(cache_inventory, "_SIGNATURE_CHECK_TTL_SECONDS", 0.0)
    inventory = cache_inventory.cache_inventory()

    # Fresh figures, not the memoised ones: the signature check saw the change.
    assert len(walks) == 2
    assert inventory["total_bytes"] == 15

    # And the memo is warm again for the unchanged tree.
    cache_inventory.cache_inventory()
    assert len(walks) == 2


def test_invalidate_forces_a_walk(one_cache):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)
    cache_inventory.cache_inventory()
    cache_inventory.cache_inventory()
    assert len(walks) == 1

    # A caller that just deleted something cannot prove it through a
    # signature alone; dropping the memo is what forces the walk.
    cache_inventory.invalidate()
    cache_inventory.cache_inventory()
    assert len(walks) == 2


# --- the signature-check window ------------------------------------------


def test_two_reads_inside_the_signature_window_prove_the_tree_once(one_cache, sig_walks):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)

    first = cache_inventory.cache_inventory()
    second = cache_inventory.cache_inventory()

    assert first["total_bytes"] == 10
    assert second["total_bytes"] == 10
    # The cold read TAKES the signatures; the second read is inside the window
    # that proof opened and must not walk the tree again.
    assert len(sig_walks) == 1, "a repeat read inside the window re-proved the tree"
    assert len(walks) == 1


def test_past_the_signature_window_the_tree_is_proven_again(one_cache, sig_walks, monkeypatch):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)
    cache_inventory.cache_inventory()
    cache_inventory.cache_inventory()
    assert len(sig_walks) == 1

    # The window runs on real monotonic time; closing it is what expiry does.
    monkeypatch.setattr(cache_inventory, "_SIGNATURE_CHECK_TTL_SECONDS", 0.0)
    third = cache_inventory.cache_inventory()

    assert third["total_bytes"] == 10
    assert len(sig_walks) == 2, "a read past the window did not prove the tree"
    assert len(walks) == 1, "a passing proof must not re-measure"


def test_a_forced_read_walks_even_inside_the_signature_window(one_cache, sig_walks):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)
    cache_inventory.cache_inventory()
    cache_inventory.cache_inventory()
    assert len(sig_walks) == 1

    # The Recheck button: refresh=True walks no matter how fresh the proof is.
    refreshed = cache_inventory.cache_inventory(refresh = True)

    assert refreshed["total_bytes"] == 10
    assert len(walks) == 2, "a forced read took the memo instead of walking"
    assert len(sig_walks) == 2


def test_a_changed_tree_is_rememeasured_in_the_windows_successor(
    one_cache, sig_walks, monkeypatch
):
    root, walks = one_cache
    real_window = cache_inventory._SIGNATURE_CHECK_TTL_SECONDS
    _write(root / "model.safetensors", "w" * 10)
    cache_inventory.cache_inventory()

    # The change lands inside the window, where the memo is served on purpose.
    _write(root / "added.bin", "y" * 5)
    monkeypatch.setattr(cache_inventory, "_SIGNATURE_CHECK_TTL_SECONDS", 0.0)
    inventory = cache_inventory.cache_inventory()

    # The successor window's first read proves the trees, sees the mismatch,
    # and re-measures — and its figures are the changed tree's. How many
    # proofs that read pays (the check outside the flight and the one inside)
    # is not the point; that one of them MISMATCHES and forces the walk is.
    assert len(sig_walks) >= 2, "the changed tree was never proven"
    assert len(walks) == 2
    assert inventory["total_bytes"] == 15

    # And the successor window is re-armed by that walk for the unchanged tree.
    monkeypatch.setattr(cache_inventory, "_SIGNATURE_CHECK_TTL_SECONDS", real_window)
    proven = len(sig_walks)
    again = cache_inventory.cache_inventory()
    assert again["total_bytes"] == 15
    assert len(walks) == 2
    assert len(sig_walks) == proven, "the successor window did not shield a proven tree"


def test_a_purge_still_invalidates_inside_the_signature_window(one_cache, sig_walks):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)
    cache_inventory.cache_inventory()
    cache_inventory.cache_inventory()
    assert len(walks) == 1

    # purge_caches drops the memo through invalidate_cache_size, and the
    # window must not shield a tree the purge just emptied.
    purged = cache_inventory.purge_caches(["test_cache"])

    assert not any(root.iterdir())
    assert purged["inventory"]["total_bytes"] == 0
    assert len(walks) == 2, "the window shielded a tree the purge had emptied"
    assert cache_inventory.cache_inventory()["total_bytes"] == 0


def test_a_read_returns_a_copy_not_the_memo(one_cache):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)
    cache_inventory.cache_inventory()

    entry = cache_inventory.cache_inventory()["caches"][0]
    assert entry is not cache_inventory._size_cache["test_cache"][2]
    # A response model that edits its rows would poison the memo otherwise.
    entry["size_bytes"] = 999
    entry["entry_count"] = 999
    entry["paths"].append("poisoned")
    entry["blocked_reason"] = "poisoned"

    again = cache_inventory.cache_inventory()["caches"][0]
    assert again["size_bytes"] == 10
    assert again["entry_count"] == 1
    assert again["paths"] == [str(root)]
    assert again["blocked_reason"] is None
    assert len(walks) == 1, "poisoned fields were served or forced a re-measure"


def test_the_response_shape_is_the_same_with_and_without_the_window(one_cache):
    root, _walks = one_cache
    _write(root / "model.safetensors", "w" * 10)

    cold = cache_inventory.cache_inventory()
    warm = cache_inventory.cache_inventory()

    assert set(cold) == {
        "caches", "total_bytes", "reclaimable_bytes", "free_bytes", "total_disk_bytes"
    }
    assert set(warm) == set(cold)
    assert set(cold["caches"][0]) == {
        "key", "group", "opt_in", "paths", "size_bytes", "entry_count",
        "present", "purgeable", "blocked_reason",
    }
    # Every field survives the windowed serve: same rows, only the copy differs.
    assert warm["caches"] == cold["caches"]
    assert warm["total_bytes"] == cold["total_bytes"]
    assert warm["reclaimable_bytes"] == cold["reclaimable_bytes"]
