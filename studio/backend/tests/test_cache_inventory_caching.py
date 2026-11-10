# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The sizing walk runs once per change, not once per read.

cache_inventory() is what the disk-usage card and the settings page poll, so a
walk per read would make every poll as slow as the biggest cache on the
machine. These tests count the walk itself (the recursive sizing pass, not the
bounded signature check that guards reuse) and pin the three behaviours: two
reads over an unchanged tree walk once, a change between reads walks again,
and invalidate() forces a walk for a caller that just deleted something.

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


def test_two_reads_of_an_unchanged_tree_walk_it_once(one_cache):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)

    first = cache_inventory.cache_inventory()
    second = cache_inventory.cache_inventory()

    assert first["total_bytes"] == 10
    assert second["total_bytes"] == 10
    assert len(walks) == 1, "the second read re-walked a tree nothing had changed"


def test_a_file_added_between_reads_walks_again(one_cache):
    root, walks = one_cache
    _write(root / "model.safetensors", "w" * 10)
    cache_inventory.cache_inventory()
    assert len(walks) == 1

    _write(root / "added.bin", "y" * 5)
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
