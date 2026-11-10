# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A bounded, content-free fingerprint of a directory tree, for cache invalidation.

Walks are seconds on a multi-gigabyte model cache, so anything that answers
"what is on disk" per request must be able to ask "did anything change" without
re-walking everything. This module answers that second question with a digest
over each entry's (relative path, size, mtime_ns, is_dir) — metadata only.

Why mtime_ns and not a content hash: re-hashing a multi-GB cache to invalidate
a cache is the very cost being avoided. Metadata is one stat per entry, and
stat results are what the sizing walks already read. A writer that preserves
size and mtime while changing content will fool this; no stat-keyed invalidation
avoids that, and model weights are written once.

Why the bound: an unbounded walk is also the thing being avoided, so the walk
stops after ``max_entries`` entries. Past the bound the signature covers only
the entries visited first — in a deterministic order (name-sorted children,
depth first, so two calls over the same tree pick the same prefix) — and is
marked DEGRADED so callers can tell a full fingerprint from a truncated one.
Changes outside the prefix are invisible to a degraded signature; the bound
exists because a bounded answer that re-walks too often beats an unbounded one
that hangs the request. Signatures are only comparable to others computed with
the same ``max_entries``; :func:`is_unchanged` always uses the default, so a
signature taken at another bound reads as "changed" and costs a re-walk rather
than a stale answer.

Never raises: a missing, non-directory, or unreadable root yields
:data:`ABSENT_SIGNATURE`, which is what a caller stored for it while it was
gone, so absence is stable and a root that disappears reads as changed against
any signature it had while it existed. An EMPTY directory is not absent: it
signs as a real (empty) tree.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import List, Optional, Tuple

# What a missing or unreadable root signs as. Plain text, not hex, so it can
# never be confused for a digest of a real tree.
ABSENT_SIGNATURE = "absent"

# Prefix that marks a signature whose walk hit the entry bound (or an
# unreadable subtree). Readable in a debug log; use is_degraded() to test.
DEGRADED_PREFIX = "degraded:"

# Namespaces these digests so one cannot collide with a bare sha256 of the same
# bytes built for another purpose, and carries the degraded flag so a complete
# and a truncated walk of the same entries never agree.
_DIGEST_NAMESPACE = b"unsloth-stat-signature:v1"

Entry = Tuple[str, int, int, bool]

# Relative paths use "/" in the digest so the signature is stable across
# platforms and os.sep spellings; a tree is only ever compared to itself.
_JOIN = "/"


def _scan_sorted(path: Path) -> Optional[List["os.DirEntry"]]:
    """The directory's direct entries in name order, or None when unreadable.

    Name order makes the traversal deterministic, which is what lets a
    truncated walk hash the SAME prefix on every call. The entries are used
    for NAMES only: their cached metadata is not trustworthy, see _collect.
    """
    try:
        with os.scandir(path) as scan:
            return sorted(scan, key = lambda entry: entry.name)
    except OSError:
        return None


def _collect(root: Path, limit: int) -> Optional[Tuple[List[Entry], bool]]:
    """Walk *root* depth-first, returning at most ``limit + 1`` entries.

    The extra entry exists only to tell "exactly at the bound" (a complete
    tree) from "past the bound" (a degraded one): a tree with ``limit``
    entries answers fully, a tree with one more is marked degraded. Returns
    None when *root* itself cannot be read, which means absent.
    """
    entries: List[Entry] = []
    # Unreadable subtrees: the digest would not cover everything, which is
    # exactly what degraded reports.
    incomplete = False
    # Junctions and bind mounts can point back at an ancestor, so a directory
    # is descended into only once per real identity. The entry bound would end
    # such a walk too, but only after filling the prefix with duplicates.
    seen_dirs: set = set()
    # (path, relative path); children are pushed in reverse name order so the
    # stack pops them in name order.
    stack: List[Tuple[Path, str]] = [(root, "")]
    while stack and len(entries) <= limit:
        current, current_rel = stack.pop()
        children = _scan_sorted(current)
        if children is None:
            if current == root:
                return None
            incomplete = True
            continue
        subdirs: List[Tuple[Path, str]] = []
        for entry in children:
            if len(entries) > limit:
                break
            try:
                # A fresh lstat, never entry.stat(): the DirEntry metadata
                # comes from the enumeration snapshot, and on Windows a
                # directory's timestamp in that snapshot lags its real mtime
                # after a child write. Two walks back to back then disagreed
                # about an unchanged tree, which made every read miss. lstat
                # reads the settled value.
                info = os.lstat(entry.path)
            except OSError:
                # Vanished mid-walk; a stable tree has no such entries.
                continue
            is_dir = stat.S_ISDIR(info.st_mode)
            rel = entry.name if not current_rel else current_rel + _JOIN + entry.name
            entries.append((rel, int(info.st_size), int(info.st_mtime_ns), is_dir))
            if is_dir and not stat.S_ISLNK(info.st_mode):
                identity = (info.st_dev, info.st_ino)
                # st_ino is 0 on filesystems without inode numbers, where the
                # entry bound below is the only cycle guard. Fine: it bounds.
                if info.st_ino and identity in seen_dirs:
                    continue
                seen_dirs.add(identity)
                subdirs.append((Path(entry.path), rel))
        for item in reversed(subdirs):
            stack.append(item)
    # A non-empty stack at the exit means the bound cut the walk short, even if
    # the overflow landed exactly on a directory boundary.
    truncated = bool(stack) or len(entries) > limit
    return entries, incomplete or truncated


def directory_signature(root: Path, *, max_entries: int = 2048) -> str:
    """A stable hex digest of the tree under *root*, from metadata only.

    See the module docstring for why the digest is mtime-based and bounded.
    Returns :data:`ABSENT_SIGNATURE` for a missing or unreadable *root*, and
    prefixes :data:`DEGRADED_PREFIX` when the walk hit *max_entries* or found
    an unreadable subtree. Never raises.
    """
    try:
        limit = max(1, int(max_entries))
        collected = _collect(Path(root), limit)
    except Exception:  # noqa: BLE001 - a fingerprint must never break its caller
        return ABSENT_SIGNATURE
    if collected is None:
        return ABSENT_SIGNATURE
    entries, degraded = collected
    digest = hashlib.sha256(_DIGEST_NAMESPACE)
    digest.update(b"1" if degraded else b"0")
    # Sorted by relative path so the digest does not depend on traversal
    # order, and truncated AFTER sorting so it is "the first max_entries".
    for rel, size, mtime_ns, is_dir in sorted(entries)[:limit]:
        digest.update(rel.encode("utf-8", "surrogateescape"))
        digest.update(b"\0")
        digest.update(f"{size} {mtime_ns} {1 if is_dir else 0}".encode("ascii"))
        digest.update(b"\n")
    hexed = digest.hexdigest()
    return DEGRADED_PREFIX + hexed if degraded else hexed


def is_degraded(signature: str) -> bool:
    """True when *signature* was truncated by the entry bound or an unreadable subtree."""
    return signature.startswith(DEGRADED_PREFIX)


def is_unchanged(root: Path, previous: str) -> bool:
    """True when *root* still fingerprints to *previous*.

    One bounded stat walk, no file reads. A *previous* taken at a different
    ``max_entries`` compares unequal and costs a re-walk, never a stale hit.
    """
    return previous == directory_signature(root)
