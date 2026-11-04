# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Run diff: what actually changed between two checkpoints this Studio wrote.

Every stage already writes a ``run_manifest.json`` beside its output, and the
favourite way to lose an afternoon is to compare two of those by opening them
side by side. This module answers the same question in one pass: the manifest
fields that differ, grouped by the section they came from, plus the checkpoint
facts that are not in the manifest at all (parameters, weight size, family).

The comparison reads JSON and file names only. It never loads a weight, so it
works on a machine that cannot hold both checkpoints, and it never writes to
either folder -- a diff that edited the thing it describes would be worthless as
evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from core.labs import artifacts

_MAX_DEPTH = 5
_MAX_CHANGES = 400

# The facts each side carries outside its manifest, in the order the page shows
# them. ``label`` is the column the UI prints; no translation key is invented
# here because this module has no access to the frontend's strings.
_FACTS: tuple[tuple[str, str], ...] = (
    ("params", "Parameters"),
    ("weight_bytes", "Weights"),
    ("model_type", "Family"),
    ("adapter", "LoRA adapter"),
    ("quantized", "Quantized"),
    ("moe", "Mixture of Experts"),
    ("modified", "Written"),
)


# ---------------------------------------------------------------------------
# Reading a side
# ---------------------------------------------------------------------------


def _key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _flatten(value: Any, prefix: str = "", depth: int = 0) -> dict[str, Any]:
    """Dotted-path view of a manifest, so a nested change has a name.

    Lists and scalars stay whole: a manifest's lists are short (suites, inputs)
    and splitting them into ``inputs.0`` would print an index instead of the
    field a reader recognises.
    """
    if isinstance(value, dict) and value and depth < _MAX_DEPTH:
        flat: dict[str, Any] = {}
        for name, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{name}" if prefix else str(name), depth + 1))
        return flat
    return {prefix: value}


def _serialized(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys = True, default = str)
    except (TypeError, ValueError):
        return repr(value)


def _side(path: str) -> dict[str, Any]:
    record = artifacts.find(path)
    directory = Path(record["path"])
    manifest_path = directory / artifacts.MANIFEST_NAME
    manifest: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    if manifest_path.is_file():
        try:
            value = json.loads(manifest_path.read_text(encoding = "utf-8"))
        except (OSError, ValueError) as failure:
            error = f"{manifest_path.name} could not be read: {failure}"
        else:
            manifest = value if isinstance(value, dict) else None
            if manifest is None:
                error = f"{manifest_path.name} is not a JSON object."
    return {
        "path": record["path"],
        "name": record["name"],
        "modified": record["modified"],
        "record": record,
        "manifest": manifest,
        "manifest_error": error,
    }


def _benchmark(path: str) -> Optional[dict[str, Any]]:
    """The newest benchmark record for one checkpoint, if any was run."""
    from core.labs import benchmarks

    try:
        records = benchmarks.list_records(limit = 200)
    except Exception:  # noqa: BLE001 - a diff without scores is still a diff
        return None
    wanted = _key(path)
    for record in records:
        if _key(str(record.get("artifact") or "")) != wanted:
            continue
        return {
            "id": record.get("id"),
            "created_at": record.get("created_at"),
            "suites": record.get("suites") or [],
            "candidate_scores": record.get("candidate_scores") or {},
            "baseline_scores": record.get("baseline_scores") or {},
            "verdict": record.get("verdict"),
            "vs_pinned": benchmarks.verdict(record),
        }
    return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _facts(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, label in _FACTS:
        before = left["record"].get(field)
        after = right["record"].get(field)
        rows.append(
            {
                "field": field,
                "label": label,
                "left": before,
                "right": after,
                "changed": _serialized(before) != _serialized(after),
            }
        )
    return rows


def _manifest_changes(
    left: Optional[dict[str, Any]], right: Optional[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, bool]:
    # One side having no manifest is missing evidence, not a folder that lost
    # every field. Listing each field as "removed" would be a fiction the page
    # then reports as changes; the note explains the gap instead.
    if not isinstance(left, dict) or not isinstance(right, dict):
        return [], 0, False
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    changes: list[dict[str, Any]] = []
    same = 0
    for field in sorted(set(left_flat) | set(right_flat)):
        before = left_flat.get(field)
        after = right_flat.get(field)
        if _serialized(before) == _serialized(after):
            same += 1
            continue
        if len(changes) >= _MAX_CHANGES:
            continue
        changes.append(
            {
                "field": field,
                "group": field.split(".", 1)[0],
                "left": before,
                "right": after,
            }
        )
    truncated = len(set(left_flat) | set(right_flat)) - same > _MAX_CHANGES
    return changes, same, truncated


def _benchmark_comparison(
    left: Optional[dict[str, Any]], right: Optional[dict[str, Any]]
) -> dict[str, Any]:
    if left is None and right is None:
        return {"left": None, "right": None, "delta": None, "aggregate_delta": None}
    left_scores = (left or {}).get("candidate_scores") or {}
    right_scores = (right or {}).get("candidate_scores") or {}
    delta: list[dict[str, Any]] = []
    for suite in sorted(set(left_scores) | set(right_scores)):
        before = left_scores.get(suite)
        after = right_scores.get(suite)
        change = (
            float(after) - float(before)
            if isinstance(before, (int, float)) and isinstance(after, (int, float))
            else None
        )
        delta.append(
            {
                "suite": suite,
                "left": before,
                "right": after,
                "delta": round(change, 6) if change is not None else None,
            }
        )
    scored = [item["delta"] for item in delta if item["delta"] is not None]
    return {
        "left": left,
        "right": right,
        "delta": delta or None,
        "aggregate_delta": round(sum(scored) / len(scored), 6) if scored else None,
    }


def compare(left_path: str, right_path: str) -> dict[str, Any]:
    """Field-level differences between two artifacts this Studio wrote."""
    left = _side(left_path)
    right = _side(right_path)
    changes, same, truncated = _manifest_changes(left["manifest"], right["manifest"])
    notes: list[str] = []
    if left["manifest"] is None or right["manifest"] is None:
        missing = [
            side["name"]
            for side in (left, right)
            if side["manifest"] is None
        ]
        notes.append(
            "No run manifest for "
            + ", ".join(missing)
            + ", so only the checkpoint facts were compared."
        )
    if _key(left["path"]) == _key(right["path"]):
        notes.append("Both sides name the same folder; every difference below is zero.")
    left_run = (left["manifest"] or {}).get("run_id")
    right_run = (right["manifest"] or {}).get("run_id")
    if left_run and left_run == right_run:
        notes.append(
            "Both manifests carry the same run id, so one of the folders is a copy."
        )
    for side in (left, right):
        if side["manifest_error"]:
            notes.append(side["manifest_error"])

    return {
        "left": {
            "path": left["path"],
            "name": left["name"],
            "modified": left["modified"],
            "has_manifest": left["manifest"] is not None,
        },
        "right": {
            "path": right["path"],
            "name": right["name"],
            "modified": right["modified"],
            "has_manifest": right["manifest"] is not None,
        },
        "facts": _facts(left, right),
        "changes": changes,
        "changed_count": len(changes),
        "same_count": same,
        "truncated": truncated,
        "benchmark": _benchmark_comparison(_benchmark(left["path"]), _benchmark(right["path"])),
        "notes": notes,
    }
