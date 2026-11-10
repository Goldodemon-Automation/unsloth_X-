# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A fingerprint is recomputed when the bytes move, and only then.

The promise the Start path depends on: a second Start over an untouched dataset
must not re-read it (that is seconds of I/O for an answer that did not change),
a rewritten file must be re-read, and a file that moved WHILE it was read must
never contribute a number. Counting ``_sha256_file`` calls is the whole point --
"the hash was reused" is only observable as "the bytes were not read again".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.training import dataset_fingerprint
from utils.stat_signature import (
    ABSENT_SIGNATURE,
    SignatureCache,
    directory_signature,
    file_signature,
)

# Fixed timestamps: a rewrite must be visible through the signature without
# depending on the clock ticking between two writes in one test.
_FIRST_STAMP_NS = 1_600_000_000_000_000_000
_LATER_STAMP_NS = _FIRST_STAMP_NS + 1_000_000_000


@pytest.fixture(autouse = True)
def _empty_caches():
    """Every test starts with nothing memoized, so counts are its own doing."""
    dataset_fingerprint.invalidate_hash_cache()
    yield
    dataset_fingerprint.invalidate_hash_cache()


@pytest.fixture
def counter(monkeypatch):
    """Counts real byte reads, while still doing them."""
    reads: list[Path] = []
    original = dataset_fingerprint._sha256_file

    def counting(path: Path) -> str:
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(dataset_fingerprint, "_sha256_file", counting)
    return reads


def _write(path: Path, text: str, stamp_ns: int | None = None) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")
    if stamp_ns is not None:
        os.utime(path, ns = (stamp_ns, stamp_ns))
    return path


def _dataset(root: Path) -> Path:
    """A three-file dataset, so "hashed once" means three reads once."""
    _write(root / "a.jsonl", '{"text": "a"}\n', _FIRST_STAMP_NS)
    _write(root / "b.jsonl", '{"text": "bb"}\n', _FIRST_STAMP_NS)
    _write(root / "sub" / "c.jsonl", '{"text": "ccc"}\n', _FIRST_STAMP_NS)
    return root


def test_an_unchanged_tree_is_hashed_once_for_two_fingerprints(tmp_path, counter):
    root = _dataset(tmp_path / "dataset")

    first = dataset_fingerprint.fingerprint_source(str(root))
    reads_after_first = len(counter)
    second = dataset_fingerprint.fingerprint_source(str(root))

    assert first["status"] == "hashed"
    assert reads_after_first == 3, "one read per file on the first fingerprint"
    assert len(counter) == reads_after_first, "the second fingerprint re-read nothing"
    assert first["sha256"] == second["sha256"]
    assert (first["files"], first["bytes"]) == (second["files"], second["bytes"])


def test_a_single_file_dataset_is_hashed_once_for_two_fingerprints(tmp_path, counter):
    """The file case keys on its own (size, mtime_ns), no tree walk needed."""
    dataset = _write(tmp_path / "train.jsonl", '{"text": "one"}\n', _FIRST_STAMP_NS)

    first = dataset_fingerprint.fingerprint_source(str(dataset))
    second = dataset_fingerprint.fingerprint_source(str(dataset))

    assert first["sha256"] == second["sha256"]
    assert len(counter) == 1


def test_a_rewritten_file_is_rehashed_and_only_it(tmp_path, counter):
    root = _dataset(tmp_path / "dataset")
    before = dataset_fingerprint.fingerprint_source(str(root))
    assert len(counter) == 3

    # Same path, new bytes, new size, new mtime: a real rewrite.
    _write(root / "b.jsonl", '{"text": "rewritten"}\n', _LATER_STAMP_NS)
    after = dataset_fingerprint.fingerprint_source(str(root))

    assert after["sha256"] != before["sha256"]
    assert [path.name for path in counter[3:]] == ["b.jsonl"], (
        "only the rewritten file is read again; the tree entry expired with the "
        "directory signature, and the per-file entries for a and c survived it"
    )


def test_a_file_that_changes_mid_hash_never_becomes_a_number(tmp_path, monkeypatch):
    """The signature is taken before AND after the read.

    A file whose stat moved across the read was not read as one version, so the
    digest in hand belongs to neither: it is not returned and not memoized.
    """
    dataset = _write(tmp_path / "train.jsonl", '{"text": "one"}\n', _FIRST_STAMP_NS)
    original = dataset_fingerprint._sha256_file
    attempts = []

    def mutating(path: Path) -> str:
        digest = original(path)
        attempts.append(digest)
        # Growth moves the size, so the before/after signatures cannot agree.
        path.write_text(path.read_text(encoding = "utf-8") + "x", encoding = "utf-8")
        os.utime(path, ns = (_LATER_STAMP_NS + len(attempts),) * 2)
        return digest

    monkeypatch.setattr(dataset_fingerprint, "_sha256_file", mutating)
    report = dataset_fingerprint.fingerprint_source(str(dataset))

    assert report["status"] == "unavailable"
    assert report["sha256"] is None
    assert "changed while it was being hashed" in report["reason"]
    assert len(attempts) == 2, "one retry, then an honest refusal"
    assert len(dataset_fingerprint._FILE_HASHES) == 0
    assert len(dataset_fingerprint._TREE_ENTRIES) == 0


def test_a_torn_read_leaves_no_memo_to_serve_later(tmp_path, monkeypatch):
    """After the writer stops, the next Start re-reads and caches the real bytes."""
    dataset = _write(tmp_path / "train.jsonl", '{"text": "one"}\n', _FIRST_STAMP_NS)
    original = dataset_fingerprint._sha256_file
    state = {"moved": False}

    def once_mutating(path: Path) -> str:
        digest = original(path)
        if not state["moved"]:
            state["moved"] = True
            path.write_text('{"text": "two-different"}\n', encoding = "utf-8")
            os.utime(path, ns = (_LATER_STAMP_NS,) * 2)
        return digest

    monkeypatch.setattr(dataset_fingerprint, "_sha256_file", once_mutating)
    report = dataset_fingerprint.fingerprint_source(str(dataset))

    # The retry read the settled file, so this call has a real answer...
    assert report["status"] == "hashed"
    assert report["sha256"] is not None
    # ...and it is filed under the signature the settled file has, not the old
    # one, so nothing is served for the version that was torn.
    settled = file_signature(dataset)
    assert dataset_fingerprint._FILE_HASHES.get(dataset, settled) is not None

    monkeypatch.setattr(dataset_fingerprint, "_sha256_file", original)
    reread = dataset_fingerprint.fingerprint_source(str(dataset))
    assert reread["sha256"] == report["sha256"], "the settled bytes are memoized"


def test_unavailable_is_never_cached_as_a_hash(tmp_path, counter):
    """A Hub/streaming or missing dataset stays unknown, with its reason."""
    streaming = dataset_fingerprint.fingerprint_source("org/dataset", streaming = True)
    missing = dataset_fingerprint.fingerprint_source(str(tmp_path / "not-here.jsonl"))
    empty = dataset_fingerprint.fingerprint_source(str(_empty_dataset(tmp_path)))

    for report in (streaming, missing, empty):
        assert report["status"] == "unavailable"
        assert report["sha256"] is None
        assert report["reason"]
    assert counter == [], "nothing on disk was read for any of them"
    assert len(dataset_fingerprint._FILE_HASHES) == 0
    assert len(dataset_fingerprint._TREE_ENTRIES) == 0

    # Repeating them repeats the reason rather than inventing a value.
    again = dataset_fingerprint.fingerprint_source("org/dataset", streaming = True)
    assert again["sha256"] is None
    assert again["reason"] == streaming["reason"]


def _empty_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "empty"
    root.mkdir()
    return root


def test_a_degraded_tree_signature_is_not_used_as_a_cache_key(tmp_path, monkeypatch, counter):
    """Past the walk bound the signature covers only a prefix.

    A change outside that prefix would leave the key equal, so a whole-tree hit
    off a degraded signature could serve a stale hash. The tree memo must be
    skipped there -- the per-file memo still keeps unchanged files unread.
    """
    root = _dataset(tmp_path / "dataset")
    monkeypatch.setattr(
        dataset_fingerprint,
        "directory_signature",
        lambda path, **kwargs: directory_signature(path, max_entries = 2),
    )
    before = dataset_fingerprint.fingerprint_source(str(root))
    assert len(counter) == 3
    after = dataset_fingerprint.fingerprint_source(str(root))

    assert after["sha256"] == before["sha256"]
    assert len(dataset_fingerprint._TREE_ENTRIES) == 0, "a truncated walk is not a key"
    assert len(counter) == 3, "but no byte was read twice"


def test_the_cache_never_files_a_hash_under_an_absent_signature(tmp_path, counter):
    """A path that cannot be stat'd says nothing about content, so it says nothing."""
    dataset = _write(tmp_path / "train.jsonl", "x", _FIRST_STAMP_NS)
    assert file_signature(tmp_path / "not-here.jsonl") == ABSENT_SIGNATURE
    assert dataset_fingerprint._FILE_HASHES.put(dataset, ABSENT_SIGNATURE, "deadbeef") is False
    assert dataset_fingerprint._FILE_HASHES.get(dataset, ABSENT_SIGNATURE) is None
    assert len(dataset_fingerprint._FILE_HASHES) == 0


def test_the_bound_evicts_the_least_recently_used():
    cache = SignatureCache(max_entries = 2)
    cache.put("a", "1", "hash-a")
    cache.put("b", "1", "hash-b")
    assert len(cache) == 2

    # a is a hit, so b is now the oldest and is the one to go.
    assert cache.get("a", "1") == "hash-a"
    cache.put("c", "1", "hash-c")

    assert len(cache) == 2
    assert cache.get("a", "1") == "hash-a"
    assert cache.get("b", "1") is None, "the bound evicted the least recently used"
    assert cache.get("c", "1") == "hash-c"


def test_invalidate_drops_one_path_or_all_of_them():
    cache = SignatureCache(max_entries = 4)
    cache.put("a", "1", "hash-a1")
    cache.put("a", "2", "hash-a2")
    cache.put("b", "1", "hash-b")

    cache.invalidate("a")
    assert cache.get("a", "1") is None
    assert cache.get("a", "2") is None
    assert cache.get("b", "1") == "hash-b"

    cache.invalidate()
    assert len(cache) == 0


def test_the_shipped_caches_are_bounded_and_forget_on_demand(tmp_path):
    assert dataset_fingerprint.CACHE_ENTRIES == 64
    assert dataset_fingerprint._FILE_HASHES.max_entries == 64
    assert dataset_fingerprint._TREE_ENTRIES.max_entries == 64

    root = _dataset(tmp_path / "dataset")
    assert dataset_fingerprint.fingerprint_source(str(root))["status"] == "hashed"
    assert len(dataset_fingerprint._TREE_ENTRIES) == 1

    dataset_fingerprint.invalidate_hash_cache(str(root))
    assert len(dataset_fingerprint._TREE_ENTRIES) == 0
    assert len(dataset_fingerprint._FILE_HASHES) == 0
