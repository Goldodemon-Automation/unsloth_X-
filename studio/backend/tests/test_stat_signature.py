# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tree's fingerprint changes when the tree changes, and only then.

These tests pin the three promises caches rely on: the same tree signs the
same twice in a row (or every read would miss), a real change re-signs (or
every read would serve the stale memo), and the walk never raises — a missing
or unreadable root must invalidate, not crash the request that only wanted a
size. mtimes are pinned with os.utime so a rewrite cannot collide with the
first write inside one clock tick.
"""

from __future__ import annotations

import os
from pathlib import Path

from utils.stat_signature import (
    ABSENT_SIGNATURE,
    directory_signature,
    is_degraded,
    is_unchanged,
)

# Fixed timestamps, so "the mtime changed" is the test's doing, not the clock's.
_FIRST_STAMP_NS = 1_600_000_000_000_000_000
_LATER_STAMP_NS = _FIRST_STAMP_NS + 1_000_000_000


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")
    return path


def _touch(path: Path, stamp_ns: int) -> Path:
    os.utime(path, ns = (stamp_ns, stamp_ns))
    return path


def test_two_calls_over_an_unchanged_tree_agree(tmp_path):
    _write(tmp_path / "a.bin", "a" * 10)
    _write(tmp_path / "sub" / "b.bin", "bb")
    first = directory_signature(tmp_path)
    second = directory_signature(tmp_path)
    assert first == second
    assert first != ABSENT_SIGNATURE
    assert is_unchanged(tmp_path, first)
    assert not is_degraded(first)


def test_adding_a_file_changes_the_signature(tmp_path):
    _write(tmp_path / "a.bin", "a" * 10)
    before = directory_signature(tmp_path)
    _write(tmp_path / "b.bin", "b")
    assert not is_unchanged(tmp_path, before)


def test_removing_a_file_changes_the_signature(tmp_path):
    _write(tmp_path / "a.bin", "a" * 10)
    victim = _write(tmp_path / "b.bin", "b")
    before = directory_signature(tmp_path)
    victim.unlink()
    assert not is_unchanged(tmp_path, before)


def test_renaming_a_file_changes_the_signature(tmp_path):
    """The relative path is part of the digest: a move re-signs even though
    size and mtime travelled with the file."""
    _touch(_write(tmp_path / "old.bin", "x" * 5), _FIRST_STAMP_NS)
    before = directory_signature(tmp_path)
    (tmp_path / "old.bin").rename(tmp_path / "new.bin")
    assert not is_unchanged(tmp_path, before)


def test_rewriting_a_file_changes_the_signature(tmp_path):
    """Same size, so only the mtime can catch it — which is what mtime_ns is for."""
    target = _touch(_write(tmp_path / "weights.bin", "aaa"), _FIRST_STAMP_NS)
    before = directory_signature(tmp_path)
    _write(tmp_path / "weights.bin", "bbb")
    _touch(target, _LATER_STAMP_NS)
    assert not is_unchanged(tmp_path, before)


def test_a_tree_past_the_entry_bound_still_answers_and_reports_degraded(tmp_path):
    root = tmp_path / "big"
    for index in range(10):
        _write(root / f"f{index:02d}.bin")

    complete = directory_signature(root, max_entries = 10)
    assert not is_degraded(complete), "exactly at the bound is still the whole tree"

    _write(root / "zzz-last.bin")
    truncated = directory_signature(root, max_entries = 10)
    assert is_degraded(truncated)
    assert truncated != ABSENT_SIGNATURE
    assert truncated != complete

    # Still an answer: repeatable across calls, and sensitive to changes in the
    # prefix it does cover (the first names in sort order are f00..f04).
    assert directory_signature(root, max_entries = 5) == directory_signature(
        root, max_entries = 5
    )
    small_bound = directory_signature(root, max_entries = 5)
    assert is_degraded(small_bound)
    _write(root / "f00.bin", "yy")
    assert directory_signature(root, max_entries = 5) != small_bound


def test_a_missing_root_is_answered_not_raised(tmp_path):
    missing = tmp_path / "not-here"
    signed = directory_signature(missing)
    assert signed == ABSENT_SIGNATURE
    assert is_unchanged(missing, signed), "absent stays absent"
    # A root that disappears reads as changed against anything it signed while
    # it existed, which is what forces the re-measure.
    _write(tmp_path / "here.bin", "h")
    assert not is_unchanged(missing, directory_signature(tmp_path))
    # A file where a directory is expected is absent too.
    plain_file = _write(tmp_path / "plain.bin")
    assert directory_signature(plain_file) == ABSENT_SIGNATURE


def test_an_unreadable_root_is_answered_not_raised(tmp_path, monkeypatch):
    root = tmp_path / "locked"
    _write(root / "a.bin")
    real_scandir = os.scandir

    def guarded(target):
        if Path(target) == root:
            raise PermissionError(13, "access denied", str(target))
        return real_scandir(target)

    monkeypatch.setattr(os, "scandir", guarded)
    assert directory_signature(root) == ABSENT_SIGNATURE


def test_two_different_trees_do_not_collide(tmp_path):
    one = tmp_path / "one"
    two = tmp_path / "two"
    _write(one / "model.bin", "x" * 10)
    # Same name, different size.
    _write(two / "model.bin", "x" * 11)
    assert directory_signature(one) != directory_signature(two)

    # Same name and size, different mtime.
    three = tmp_path / "three"
    _touch(_write(three / "model.bin", "x" * 10), _FIRST_STAMP_NS)
    _touch(_write(two / "model.bin", "x" * 10), _LATER_STAMP_NS)
    assert directory_signature(three) != directory_signature(two)

    # Same size and mtime, different name.
    four = tmp_path / "four"
    _touch(_write(four / "other.bin", "x" * 10), _LATER_STAMP_NS)
    assert directory_signature(four) != directory_signature(two)


def test_identical_metadata_signs_identically_even_with_different_content(tmp_path):
    """By design: the digest reads metadata only, so a mirror of a tree with
    sizes and mtimes aligned is the same tree to it. That is what keeps the
    check cheap — a content hash would re-read gigabytes per request — and
    nothing here writes model weights twice at the same size and mtime."""
    one = tmp_path / "one"
    two = tmp_path / "two"
    _touch(_write(one / "weights.bin", "aaaa"), _FIRST_STAMP_NS)
    _touch(_write(two / "weights.bin", "bbbb"), _FIRST_STAMP_NS)
    assert directory_signature(one) == directory_signature(two)
