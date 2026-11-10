# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from utils.stat_signature import (
    ABSENT_SIGNATURE,
    SignatureCache,
    directory_signature,
    file_signature,
    is_degraded,
)

SCHEMA = "dataset-files-v1"
_CHUNK = 4 * 1024 * 1024

# How many memoized answers each cache keeps. A Start hashes the dataset and the
# tokenizer, and one tree entry covers every file in that tree, so a small bound
# already spans the restarts a user does back to back. Past the bound the least
# recently used entry goes, and the cost of a miss is the old cost of every call:
# a re-hash.
CACHE_ENTRIES = 64

# Per-file content hashes, keyed on (file, size:mtime_ns). Surviving a sibling's
# rewrite is the point: an unchanged file must not be re-read because something
# else in the tree moved.
_FILE_HASHES = SignatureCache(max_entries = CACHE_ENTRIES)

# Whole-tree hash entries, keyed on (root, directory_signature). This is what
# makes a second Start over an untouched dataset cost one stat walk. It memoizes
# answers; it does not coalesce two Starts that arrive at once, which just do the
# same work twice (and store the same value twice).
_TREE_ENTRIES = SignatureCache(max_entries = CACHE_ENTRIES)


def invalidate_hash_cache(path: Optional[str] = None) -> None:
    """Drop memoized hashes for *path*, or for every path when it is None.

    Only needed when something outside this module's view can rewrite bytes
    without moving a size or an mtime; the signatures catch everything else.
    """
    _FILE_HASHES.invalidate(path)
    _TREE_ENTRIES.invalidate(path)


class DatasetChangedWhileHashing(OSError):
    """A file kept changing while it was read, so no version of it was hashed.

    An ``OSError`` so the existing "could not be read" path answers it as
    ``unavailable``: a dataset being rewritten under the run has no fingerprint
    yet, and saying so beats recording a digest of half of one version and half
    of another.
    """


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _size_of(signature: str) -> int:
    """The size half of a :func:`file_signature`, which is ``"<size>:<mtime_ns>"``."""
    try:
        return int(signature.split(":", 1)[0])
    except (AttributeError, ValueError):
        return 0


def _file_digest(path: Path) -> tuple[str, int]:
    """(sha256, size) for one file, memoized on its metadata, never torn.

    The signature is taken before the read AND after it. If it moved, the digest
    in hand covers a mix of two versions and belongs to neither, so it is
    neither returned as the answer nor memoized; one retry gets a clean read of
    the new version, and a file that keeps moving raises rather than producing a
    number. Only a digest bracketed by one unchanged signature is returned and
    stored, which is what makes a later hit truthful: the stored digest was
    observed under the very signature it is filed under.

    The size is returned from that same accepting signature, not from a separate
    stat: a file that grew while it was read has a size that does not describe
    the bytes that were hashed, and recording the two side by side would make an
    entry that contradicts itself.
    """
    for _ in range(2):
        before = file_signature(path)
        memoized = _FILE_HASHES.get(path, before)
        if memoized is not None:
            digest, size = memoized
            return str(digest), int(size)
        digest = _sha256_file(path)
        if file_signature(path) == before:
            size = _size_of(before)
            _FILE_HASHES.put(path, before, (digest, size))
            return digest, size
        _FILE_HASHES.invalidate(path)
    raise DatasetChangedWhileHashing(
        f"{Path(path).name} changed while it was being hashed, so it has no "
        "stable fingerprint right now. Start the run again once nothing is "
        "writing to the dataset."
    )


def _payload_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys = True, separators = (",", ":"), default = str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unavailable(source: Optional[str], reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "schema": SCHEMA,
        "source": source,
        "sha256": None,
        "reason": reason,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _files_under(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        child
        for child in root.rglob("*")
        if child.is_file()
        and not child.name.startswith(".")
        and not child.name.endswith(".lock")
    )


def _dataset_entries(root: Path) -> tuple[list[dict[str, Any]], int]:
    """(*root*'s per-file hash entries, total bytes), reusing the tree memo.

    The memo is consulted only under a COMPLETE ``directory_signature`` for
    *root*. A degraded signature (a walk cut at the entry bound) covers only the
    first names it visited, so a change outside that prefix would leave the
    signature equal and serve a stale whole-tree hash; a partial key is not a
    key. Such trees still get the per-file memo, whose keys are exact stats.

    The signature is taken again after the walk and the result is stored only if
    it did not move, the same rule single files get: an entry set assembled while
    the tree was changing describes no tree that ever existed on disk, and must
    not be the thing a later call trusts.
    """
    is_directory = root.is_dir()
    signature = directory_signature(root) if is_directory else ABSENT_SIGNATURE
    reusable = is_directory and signature != ABSENT_SIGNATURE and not is_degraded(signature)
    if reusable:
        memoized = _TREE_ENTRIES.get(root, signature)
        if memoized is not None:
            rows, total_bytes = memoized
            return (
                [
                    {"path": rel, "size_bytes": size, "sha256": sha}
                    for rel, size, sha in rows
                ],
                int(total_bytes),
            )
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for file in _files_under(root):
        sha256, size = _file_digest(file)
        total_bytes += size
        entries.append(
            {
                "path": (
                    file.relative_to(root).as_posix() if is_directory else file.name
                ),
                "size_bytes": size,
                "sha256": sha256,
            }
        )
    if reusable and entries and directory_signature(root) == signature:
        _TREE_ENTRIES.put(
            root,
            signature,
            (
                tuple(
                    (entry["path"], entry["size_bytes"], entry["sha256"])
                    for entry in entries
                ),
                total_bytes,
            ),
        )
    return entries, total_bytes


def fingerprint_source(
    source: Optional[str],
    *,
    subset: Optional[str] = None,
    train_split: Optional[str] = None,
    slice_start: Optional[int] = None,
    slice_end: Optional[int] = None,
    streaming: bool = False,
) -> dict[str, Any]:
    if streaming:
        return _unavailable(
            source,
            "Streaming datasets are read from the Hub while training, so no local files exist to hash.",
        )
    text = str(source or "").strip()
    if not text:
        return _unavailable(None, "This run did not record a dataset path.")
    root = Path(text).expanduser()
    if not root.exists():
        return _unavailable(
            text,
            "The dataset path is not on disk, so its content cannot be hashed. "
            "A Hub dataset needs a local snapshot before it can be fingerprinted.",
        )
    try:
        entries, total_bytes = _dataset_entries(root)
    except OSError as error:
        return _unavailable(text, f"The dataset files could not be read: {error}")
    if not entries:
        return _unavailable(text, "The dataset path contains no files to hash.")

    selection = {
        "subset": subset,
        "train_split": train_split,
        "slice_start": slice_start,
        "slice_end": slice_end,
    }
    return {
        "status": "hashed",
        "schema": SCHEMA,
        "source": text,
        "sha256": _payload_sha(
            {"files": entries, "selection": selection, "schema": SCHEMA}
        ),
        "files": len(entries),
        "bytes": total_bytes,
        "method": "file-bytes",
        "selection": selection,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def dataset_source_from_config(config: dict[str, Any]) -> Optional[str]:
    if config.get("dataset_source") == "s3" or config.get("s3_dataset"):
        return None
    snapshot = config.get("dataset_snapshot_path")
    if isinstance(snapshot, str) and snapshot.strip():
        return snapshot
    local = config.get("dataset_local_path")
    if isinstance(local, str) and local.strip():
        return local
    local_datasets = config.get("local_datasets")
    if isinstance(local_datasets, list):
        first = next(
            (item for item in local_datasets if isinstance(item, str) and item.strip()),
            None,
        )
        if first:
            return first
    uploaded = config.get("uploaded_file")
    if isinstance(uploaded, str) and uploaded.strip():
        return uploaded
    hf_dataset = config.get("hf_dataset")
    if isinstance(hf_dataset, str) and hf_dataset.strip():
        return hf_dataset
    return None


def fingerprint_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("dataset_source") == "s3" or config.get("s3_dataset"):
        return _unavailable(
            None,
            "S3 datasets are streamed by the trainer and have no local file set recorded at start.",
        )
    source = dataset_source_from_config(config)
    slice_start = config.get("dataset_slice_start")
    slice_end = config.get("dataset_slice_end")
    return fingerprint_source(
        source,
        subset=config.get("subset"),
        train_split=config.get("train_split"),
        slice_start=slice_start if isinstance(slice_start, int) else None,
        slice_end=slice_end if isinstance(slice_end, int) else None,
        streaming=bool(config.get("dataset_streaming")),
    )


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "chat_template.json",
)


def fingerprint_tokenizer(source: Optional[str]) -> dict[str, Any]:
    text = str(source or "").strip()
    if not text:
        return _unavailable(
            None, "This run did not record a local model path to read a tokenizer from."
        )
    root = Path(text).expanduser()
    if not root.is_dir():
        return _unavailable(
            text, "The model has no local snapshot here, so its tokenizer cannot be hashed."
        )
    files = [root / name for name in TOKENIZER_FILES if (root / name).is_file()]
    if not files:
        return _unavailable(
            text, "The model snapshot contains no tokenizer files to hash."
        )
    entries = []
    try:
        for file in files:
            sha256, size = _file_digest(file)
            entries.append(
                {
                    "path": file.name,
                    "size_bytes": size,
                    "sha256": sha256,
                }
            )
    except OSError as error:
        return _unavailable(text, f"The tokenizer files could not be read: {error}")
    return {
        "status": "hashed",
        "schema": "tokenizer-files-v1",
        "source": text,
        "sha256": _payload_sha({"files": entries, "schema": "tokenizer-files-v1"}),
        "files": len(entries),
        "file_hashes": {entry["path"]: entry["sha256"] for entry in entries},
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def fingerprint_checkpoint_tokenizer(checkpoint: Path) -> dict[str, Any]:
    return fingerprint_tokenizer(str(checkpoint))


def read_fingerprint(output_dir: Optional[str]) -> Optional[dict[str, Any]]:
    if not output_dir:
        return None
    path = Path(output_dir) / "run_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    inputs = manifest.get("inputs") if isinstance(manifest, dict) else None
    if not isinstance(inputs, dict):
        return None
    dataset = inputs.get("dataset")
    if not isinstance(dataset, dict):
        return None
    fingerprint = dataset.get("fingerprint")
    return fingerprint if isinstance(fingerprint, dict) else None
