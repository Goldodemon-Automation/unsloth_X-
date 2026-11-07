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
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

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


# ---------------------------------------------------------------------------
# Finding a checkpoint's records
# ---------------------------------------------------------------------------
#
# A benchmark can name its artifact exactly (the merged model folder) or a run
# folder whose steps are the real checkpoints. Those two cases need one query
# that answers "what is the latest evidence for this directory?", so the path is
# reduced to a comparable key once and then compared as a prefix. Nothing here
# touches the filesystem beyond the record directory that list_records reads,
# and nothing here decides whether a record is scored: a location query and a
# usefulness question are different questions.


def _path_key(value: str) -> str:
    """One comparable form for a checkpoint path.

    Backslashes become ``/`` so a record written on Windows still matches a
    directory typed with the other separator, trailing separators are removed so
    ``/out/run/`` and ``/out/run`` name the same folder, and ``os.path.normcase``
    applies the platform's own rule (lowercase on Windows, identity elsewhere).
    ``normcase`` turns the slashes back into backslashes on Windows, so the result
    ends with one more ``/`` pass; the prefix test in ``records_under`` can then
    always use a forward slash. No filesystem access happens here.
    """
    text = str(value or "").replace("\\", "/")
    stem = text.rstrip("/")
    if not stem:
        stem = "/" if text else ""
    return os.path.normcase(stem).replace("\\", "/")


def records_under(directory: str, *, limit: int = 200) -> list[dict[str, Any]]:
    """Newest-first records whose checkpoint is ``directory`` or sits under it.

    A record matches when its ``artifact`` key equals the directory key or starts
    with that key plus a ``/``. The separator is what keeps a sibling of the same
    name out: a query for ``/out/run`` must not return ``/out/run-2``. Ordering is
    the same newest-first order ``list_records`` uses (record file mtime,
    descending), and the same cap applies; the input record is never written
    back, only a copy read for this call is returned.

    Unscored records are *not* filtered here. This answers where a checkpoint's
    evidence lives, not whether the evidence has a usable score; a caller that
    cares about scores ranks or checks them itself.
    """
    wanted = _path_key(directory)
    if not wanted:
        return []
    prefix = wanted if wanted.endswith("/") else wanted + "/"
    records_dir = records_root()
    if not records_dir.is_dir():
        return []
    try:
        files = sorted(
            records_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for path in files:
        if len(records) >= max(1, min(int(limit), 200)):
            break
        record = _read(path)
        if record is None or not isinstance(record.get("id"), str):
            continue
        key = _path_key(str(record.get("artifact") or ""))
        if not key or (key != wanted and not key.startswith(prefix)):
            continue
        record.setdefault("created_at", "")
        records.append(record)
    return records


def newest_for(directory: str, *, limit: int = 200) -> Optional[dict[str, Any]]:
    """The record that best answers for ``directory``, or ``None``.

    The rule is exact path first, then newest under the directory. A record whose
    ``artifact`` *is* the directory wins over any record under a step folder
    beneath it, even a newer one, because it is the benchmark of that checkpoint
    itself. Only when no exact record exists does the newest record under the
    directory (``records_under``) stand in: the ``checkpoint-500`` case, where the
    eval scored a step and never the run folder.

    Unscored records are not filtered here either; the caller decides whether a
    found record is usable evidence.
    """
    if not _path_key(directory):
        return None
    exact = list_records(limit = limit, artifact = str(directory))
    if exact:
        return exact[0]
    under = records_under(str(directory), limit = limit)
    return under[0] if under else None


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


def verdict(
    record: dict[str, Any],
    pins: Optional[dict[str, Any]] = None,
    references: Optional[dict[str, Optional[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    """How this record sits against the pinned baseline for its checkpoint.

    ``status`` is one of ``pinned`` (this record is the baseline), ``regressed``,
    ``improved``, ``mixed``, ``flat``, ``unscored`` (no suite in common),
    ``missing`` (the pinned record was deleted) or ``unpinned`` (nothing to
    compare against). The pinned record is never compared with itself: a page
    that calls its own baseline a regression is noise, not information.

    ``references`` is an optional per-call cache of pinned records keyed by id.
    A page loads many records for the same checkpoint, and every one of them was
    reading the same pinned record file again.
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
    if references is not None and pinned_id in references:
        reference = references[pinned_id]
    else:
        try:
            reference = find_record(pinned_id)
        except (OSError, ValueError):
            reference = None
        if references is not None:
            references[pinned_id] = reference
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
    references: dict[str, Optional[dict[str, Any]]] = {}
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
        record["vs_pinned"] = verdict(record, pins, references)
    return records


# ---------------------------------------------------------------------------
# Picking the best of several runs
# ---------------------------------------------------------------------------
#
# An automated round trains more than one candidate and needs to say which one
# came out ahead. That decision is pure arithmetic over records that already
# exist, so it lives here rather than in the loop: no filesystem, no pins, and
# nothing to mock. A record's candidate scores are the mean the rank uses,
# because a checkpoint that wins one suite and loses another still needs a
# single number, and ``_scores`` already dropped the non-numeric entries.


def _record_score(record: Mapping[str, Any]) -> Optional[float]:
    """The mean of this record's finite candidate scores, or ``None``.

    A record carries one score per eval suite. The mean of those candidate
    scores is the one number the ranking compares. Non-numeric entries are
    ignored by ``_scores``, and non-finite ones (``NaN``, ``inf``) are dropped
    here, so a suite that failed to report a usable number cannot poison the
    mean. A record with no finite score at all is *unscored*: it is not zero,
    and it must not rank as if it were the worst result.
    """
    if not isinstance(record, Mapping):
        return None
    values = [
        float(value)
        for value in _scores(record.get("candidate_scores")).values()
        if math.isfinite(value)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def rank_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order records best-first: the higher mean candidate score wins.

    Sorting is by mean candidate score, descending. A record with no finite
    score goes last, not first, because "unknown" is not "worst". Ties keep
    the order the caller supplied: ``sorted`` is stable, and every score is
    finite, so there is nothing to compare beyond the score. The input records
    are never changed and a new list is returned.
    """

    def key(record: Mapping[str, Any]) -> tuple[int, float]:
        score = _record_score(record)
        if score is None:
            return (1, 0.0)
        return (0, -score)

    return sorted(records, key = key)


def best_record(records: Iterable[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    """The top-ranked record, or ``None`` when nothing was supplied.

    This is the pick an automated round keeps: the checkpoint that scored best
    on the mean of its candidate suites. It is ``rank_records`` in one step,
    and it shares that rule -- higher mean wins, unscored last, ties keep the
    caller's order.
    """
    ranked = rank_records(records)
    return ranked[0] if ranked else None
