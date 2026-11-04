# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Benchmarks: keep the eval report a finished run actually produced.

The eval engine already scores a baseline and a candidate and writes
``auto_eval_report.json``. What it did not have was a place to remember the
result once the job record was gone, so every comparison was re-run. This module
persists one small record per run under ``outputs/labs-benchmarks``. The
artifact's own folder is never modified: a benchmark is evidence *about* a
checkpoint, and rewriting the checkpoint's manifest would make a read-only
result look like part of the model.

Records are keyed by a random id, never by a user-supplied filename, and every
path is checked against the Studio output folder before it is read.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Optional

_MAX_RECORDS = 500
_ID = re.compile(r"^[a-f0-9]{16,32}$")
_DIRNAME = "labs-benchmarks"
_PINS_NAME = "pins.json"

# A delta counts as movement only above this floor. Two runs of the same
# checkpoint cannot be told apart below it, and a warning that fires on noise
# teaches a reader to ignore the warning.
_PIN_RELATIVE_TOLERANCE = 0.01
_PIN_ABSOLUTE_TOLERANCE = 0.005


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def outputs() -> Path:
    from utils.paths.storage_roots import outputs_root

    return outputs_root()


def root() -> Path:
    return outputs() / _DIRNAME


def runs_root() -> Path:
    return root() / "runs"


def records_root() -> Path:
    return root() / "records"


def pins_path() -> Path:
    return root() / _PINS_NAME


def create_run_dir() -> Path:
    """A fresh directory for one eval job's reports, outside every artifact."""
    runs = runs_root()
    runs.mkdir(parents=True, exist_ok=True)
    target = runs / uuid.uuid4().hex
    target.mkdir()
    return target


def _within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except (OSError, ValueError):
        return False
    return True


def _read(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _scores(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    scores: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, (int, float)):
            scores[str(key)] = float(raw)
    return scores


def _type_name(path: str) -> str:
    from core.labs import artifacts

    try:
        return str(artifacts.find(path)["name"])
    except Exception:  # noqa: BLE001 - a Hub id / bare path has no artifact record
        return Path(path.rstrip("/\\")).name or path


def save_record(
    *,
    job_id: str,
    artifact: str,
    baseline: str,
    report: dict[str, Any],
    suites: Iterable[str],
    limit: Optional[int],
    batch_size: int,
    duration_seconds: Optional[float],
    output_dir: str,
) -> dict[str, Any]:
    """Persist one completed report and return the record the page lists."""
    candidate_scores = _scores(report.get("candidate_scores"))
    baseline_scores = _scores(report.get("baseline_scores"))
    if not candidate_scores and not baseline_scores:
        raise ValueError("The eval report has no scores to record.")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        comparison = {}
    record = {
        "id": uuid.uuid4().hex,
        "job_id": job_id,
        "artifact": str(artifact),
        "artifact_name": _type_name(artifact),
        "baseline": str(baseline),
        "baseline_name": _type_name(baseline),
        "suites": [str(suite) for suite in suites],
        "limit": limit,
        "batch_size": int(batch_size),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": duration_seconds,
        "candidate_scores": candidate_scores,
        "baseline_scores": baseline_scores,
        "comparison": {
            "entries": comparison.get("entries") or [],
            "aggregate_delta": comparison.get("aggregate_delta"),
            "verdict": comparison.get("verdict") or report.get("verdict"),
            "regressions": comparison.get("regressions") or [],
            "improved": comparison.get("improved") or [],
            "flat": comparison.get("flat") or [],
        },
        "verdict": report.get("verdict") or comparison.get("verdict") or "unavailable",
        "errors": list(report.get("errors") or []),
        "warnings": list(report.get("warnings") or []),
        "report_dir": str(output_dir),
    }
    directory = records_root()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{record['id']}.json"
    # Same rule as pipelines.save_pipeline: a crash or a full disk must not leave a
    # half-written record, because list_records skips invalid JSON and the eval
    # result would be lost silently after the hours it took to produce.
    from core.advanced.policy import atomic_write_text

    atomic_write_text(target, json.dumps(record, indent = 2, sort_keys = True))
    _prune(directory)
    return record


def _prune(directory: Path) -> None:
    try:
        files = sorted(
            (path for path in directory.glob("*.json")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in files[_MAX_RECORDS:]:
        try:
            stale.unlink()
        except OSError:
            continue


def list_records(
    *, limit: int = 50, artifact: Optional[str] = None
) -> list[dict[str, Any]]:
    """Newest first, optionally only the records for one checkpoint."""
    directory = records_root()
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    try:
        files = sorted(
            directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    wanted = str(artifact) if artifact else None
    for path in files:
        if len(records) >= max(1, min(int(limit), 200)):
            break
        record = _read(path)
        if record is None or not isinstance(record.get("id"), str):
            continue
        if wanted is not None and str(record.get("artifact")) != wanted:
            continue
        record.setdefault("created_at", "")
        records.append(record)
    return records


def delete_record(record_id: str) -> bool:
    if not _ID.match(str(record_id)):
        raise ValueError(f"Invalid benchmark record id: {record_id!r}")
    path = records_root() / f"{record_id}.json"
    if not path.is_file():
        return False
    path.unlink()
    # A pin that outlives its record would protect a checkpoint nobody can score
    # against and report "missing" on every page load. Drop it with the evidence.
    try:
        pins = _read_pins()
        stale = [key for key, entry in pins.items() if entry.get("record_id") == record_id]
        if stale:
            for key in stale:
                pins.pop(key, None)
            _write_pins(pins)
    except OSError:
        pass
    return True


def find_record(record_id: str) -> Optional[dict[str, Any]]:
    if not _ID.match(str(record_id)):
        raise ValueError(f"Invalid benchmark record id: {record_id!r}")
    return _read(records_root() / f"{record_id}.json")


# ---------------------------------------------------------------------------
# Pinned baselines
# ---------------------------------------------------------------------------
#
# A record is a snapshot of one comparison. A pin says "this snapshot is the
# reference": the score later runs are measured against, and a folder the
# janitor must not remove. Pins live in their own file rather than being edited
# into a record, so pinning never rewrites the evidence a benchmark produced.


def _pin_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _read_pins() -> dict[str, dict[str, Any]]:
    payload = _read(pins_path()) or {}
    raw = payload.get("pins")
    if not isinstance(raw, dict):
        return {}
    pins: dict[str, dict[str, Any]] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if not isinstance(entry.get("record_id"), str):
            continue
        pins[str(key)] = entry
    return pins


def _write_pins(pins: dict[str, dict[str, Any]]) -> None:
    from core.advanced.policy import atomic_write_text

    target = pins_path()
    target.parent.mkdir(parents = True, exist_ok = True)
    payload = json.dumps({"pins": pins}, indent = 2, sort_keys = True)
    atomic_write_text(target, payload)


def pin_record(record_id: str) -> dict[str, Any]:
    """Make one record the baseline for its checkpoint, replacing any earlier pin."""
    record = find_record(record_id)
    if record is None:
        raise FileNotFoundError(f"No benchmark record {record_id!r}.")
    artifact = str(record.get("artifact") or "")
    if not artifact:
        raise ValueError("That record does not name a checkpoint, so it cannot be pinned.")
    entry = {
        "record_id": record["id"],
        "artifact": artifact,
        "artifact_name": record.get("artifact_name") or artifact,
        "suites": list(record.get("suites") or []),
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    pins = _read_pins()
    pins[_pin_key(artifact)] = entry
    _write_pins(pins)
    return entry


def unpin(record_id: str) -> bool:
    """Drop the pin this record holds. False when the record was not pinned."""
    if not _ID.match(str(record_id)):
        raise ValueError(f"Invalid benchmark record id: {record_id!r}")
    pins = _read_pins()
    stale = [key for key, entry in pins.items() if entry.get("record_id") == record_id]
    if not stale:
        return False
    for key in stale:
        pins.pop(key, None)
    _write_pins(pins)
    return True


def list_pins() -> list[dict[str, Any]]:
    """Every pin, with the record it points at when that record still exists."""
    pins = _read_pins()
    entries: list[dict[str, Any]] = []
    for key in sorted(pins):
        entry = pins[key]
        record = None
        try:
            record = find_record(str(entry["record_id"]))
        except (OSError, ValueError):
            record = None
        entries.append(
            {
                **entry,
                "artifact": entry.get("artifact") or key,
                "available": record is not None,
                "candidate_scores": (record or {}).get("candidate_scores") or {},
                "created_at": (record or {}).get("created_at"),
            }
        )
    return entries


def pinned_artifacts() -> set[str]:
    """Normalized checkpoint paths the janitor must leave alone."""
    return {str(key) for key in _read_pins()}


def _tolerance(pinned: float) -> float:
    return max(_PIN_ABSOLUTE_TOLERANCE, _PIN_RELATIVE_TOLERANCE * abs(pinned))


def _compare_scores(
    candidate: dict[str, float], pinned: dict[str, float]
) -> Optional[dict[str, Any]]:
    suites: list[dict[str, Any]] = []
    regressions: list[str] = []
    improved: list[str] = []
    flat: list[str] = []
    for suite in sorted(set(candidate) & set(pinned)):
        before = float(pinned[suite])
        after = float(candidate[suite])
        delta = after - before
        tolerance = _tolerance(before)
        if delta < -tolerance:
            label = "regressed"
            regressions.append(suite)
        elif delta > tolerance:
            label = "improved"
            improved.append(suite)
        else:
            label = "flat"
            flat.append(suite)
        suites.append(
            {
                "suite": suite,
                "pinned": before,
                "candidate": after,
                "delta": round(delta, 6),
                "tolerance": round(tolerance, 6),
                "verdict": label,
            }
        )
    if not suites:
        return None
    if regressions and improved:
        status = "mixed"
    elif regressions:
        status = "regressed"
    elif improved:
        status = "improved"
    else:
        status = "flat"
    return {
        "status": status,
        "aggregate_delta": round(sum(item["delta"] for item in suites) / len(suites), 6),
        "tolerance": round(max(item["tolerance"] for item in suites), 6),
        "suites": suites,
        "regressions": regressions,
        "improved": improved,
        "flat": flat,
    }


def verdict(record: dict[str, Any], pins: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """How this record sits against the pinned baseline for its checkpoint.

    ``status`` is one of ``pinned`` (this record is the baseline), ``regressed``,
    ``improved``, ``mixed``, ``flat``, ``unscored`` (no suite in common),
    ``missing`` (the pinned record was deleted) or ``unpinned`` (nothing to
    compare against). The pinned record is never compared with itself: a page
    that calls its own baseline a regression is noise, not information.
    """
    store = _read_pins() if pins is None else pins
    entry = store.get(_pin_key(str(record.get("artifact") or "")))
    if entry is None:
        return {"status": "unpinned"}
    pinned_id = str(entry.get("record_id") or "")
    if pinned_id == str(record.get("id") or ""):
        return {
            "status": "pinned",
            "pinned_record_id": pinned_id,
            "pinned_at": entry.get("pinned_at"),
            "suites": [],
            "aggregate_delta": None,
            "tolerance": None,
            "regressions": [],
            "improved": [],
            "flat": [],
        }
    try:
        reference = find_record(pinned_id)
    except (OSError, ValueError):
        reference = None
    if reference is None:
        return {"status": "missing", "pinned_record_id": pinned_id}
    comparison = _compare_scores(
        _scores(record.get("candidate_scores")),
        _scores(reference.get("candidate_scores")),
    )
    if comparison is None:
        return {"status": "unscored", "pinned_record_id": pinned_id}
    return {
        **comparison,
        "pinned_record_id": pinned_id,
        "pinned_at": entry.get("pinned_at"),
    }


def annotate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the pin and the against-pin verdict to each record for the page."""
    pins = _read_pins()
    for record in records:
        entry = pins.get(_pin_key(str(record.get("artifact") or "")))
        record["pin"] = (
            {
                "record_id": entry.get("record_id"),
                "pinned_at": entry.get("pinned_at"),
                "is_self": str(entry.get("record_id")) == str(record.get("id")),
            }
            if entry is not None
            else None
        )
        record["vs_pinned"] = verdict(record, pins)
    return records
