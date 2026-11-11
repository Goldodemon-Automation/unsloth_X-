# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""SQLite storage for the telemetry collector.

One row per install per day per source for token counts, one row per install per
day per model for the model leaderboard, and all-time rows for training runs and
merges (which are events, not counters). A report is idempotent: the day rows it
covers are replaced, so a retry after a timeout cannot double-count.

Nothing here stores prompts, replies, file paths, hostnames, tokens or email
addresses. The client sends model and dataset identifiers, counters and hardware
classes -- that is the whole record.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA_VERSION = 1

# Fixed windows the console reports. 180 is the widest history the client keeps.
WINDOWS: tuple[int, ...] = (2, 4, 9, 12, 30, 180)
MAX_WINDOW_DAYS = 400
MAX_LIST_ROWS = 500

# The fields an install's feature block may carry. Named here so a new client
# field cannot quietly become an arbitrary JSON dump on the dashboard.
FEATURE_KEYS: tuple[str, ...] = (
    "chat_threads",
    "chat_messages",
    "chat_attachments",
    "trainings",
    "completions",
    "merges",
    "merges_successful",
    "api_requests",
    "labs_runs",
    "video_generations",
    "image_generations",
    "rag_documents",
    "models_loaded",
)

FEATURE_LABELS = {
    "chat_threads": "Chat threads",
    "chat_messages": "Chat messages",
    "chat_attachments": "Attachments",
    "trainings": "Training runs",
    "completions": "Completed runs",
    "merges": "Merges",
    "merges_successful": "Merges succeeded",
    "api_requests": "OpenAI API requests",
    "labs_runs": "Labs runs",
    "video_generations": "Video generations",
    "image_generations": "Image generations",
    "rag_documents": "RAG documents",
    "models_loaded": "Models loaded",
}

REPORT_SCHEMA = 1


def db_path() -> Path:
    """The SQLite file, overridable for deployments and tests."""
    configured = os.environ.get("X8_TELEMETRY_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent / "telemetry.sqlite3"


def connect(path: Optional[Path | str] = None) -> sqlite3.Connection:
    target = Path(path) if path is not None else db_path()
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(target, timeout = 15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


@contextmanager
def transaction(path: Optional[Path | str] = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        init_db(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS installs (
            install_id TEXT PRIMARY KEY,
            first_seen INTEGER NOT NULL,
            last_seen INTEGER NOT NULL,
            version TEXT,
            os TEXT,
            os_version TEXT,
            arch TEXT,
            cpu TEXT,
            cpu_cores INTEGER,
            cpu_threads INTEGER,
            ram_gb REAL,
            gpu TEXT,
            gpu_vendor TEXT,
            vram_gb REAL,
            gpu_count INTEGER,
            backend TEXT,
            apple_silicon INTEGER NOT NULL DEFAULT 0,
            locale TEXT,
            python TEXT,
            torch TEXT,
            transformers TEXT,
            accounts INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS usage_daily (
            install_id TEXT NOT NULL,
            day TEXT NOT NULL,
            source TEXT NOT NULL,
            requests INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (install_id, day, source)
        );

        CREATE INDEX IF NOT EXISTS idx_usage_day ON usage_daily(day);

        CREATE TABLE IF NOT EXISTS model_daily (
            install_id TEXT NOT NULL,
            day TEXT NOT NULL,
            model TEXT NOT NULL,
            kind TEXT NOT NULL,
            uses INTEGER NOT NULL DEFAULT 0,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (install_id, day, model, kind)
        );

        CREATE INDEX IF NOT EXISTS idx_model_day ON model_daily(day);

        CREATE TABLE IF NOT EXISTS trainings (
            install_id TEXT NOT NULL,
            run_key TEXT NOT NULL,
            day TEXT,
            base_model TEXT,
            dataset TEXT,
            status TEXT,
            method TEXT,
            steps INTEGER NOT NULL DEFAULT 0,
            epochs REAL,
            PRIMARY KEY (install_id, run_key)
        );

        CREATE INDEX IF NOT EXISTS idx_trainings_day ON trainings(day);

        CREATE TABLE IF NOT EXISTS merges (
            install_id TEXT NOT NULL,
            merge_key TEXT NOT NULL,
            day TEXT,
            models_json TEXT NOT NULL DEFAULT '[]',
            method TEXT,
            PRIMARY KEY (install_id, merge_key)
        );

        CREATE INDEX IF NOT EXISTS idx_merges_day ON merges(day);

        CREATE TABLE IF NOT EXISTS features (
            install_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL DEFAULT '{}',
            updated_at INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(SCHEMA_VERSION),),
    )


def _clamp_int(value: Any, low: int = 0, high: int = 1 << 50) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def _clamp_float(value: Any, low: float = 0.0, high: float = 1e9) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if number != number:  # NaN
        return low
    return max(low, min(high, number))


def _text(value: Any, limit: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned[:limit]


def _day(value: Any) -> Optional[str]:
    """A real ISO date, or None. Future days beyond tomorrow are clock skew."""
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None
    if parsed > date.today() + timedelta(days = 2):
        return None
    return parsed.isoformat()


def _day_start_epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo = timezone.utc).timestamp())


def _feature_payload(raw: Any) -> dict[str, int]:
    """Only known counters, each a non-negative int, survive."""
    if not isinstance(raw, dict):
        return {}
    return {
        key: _clamp_int(raw.get(key), 0, 1 << 40)
        for key in FEATURE_KEYS
        if key in raw
    }


def ingest(
    conn: sqlite3.Connection,
    report: dict[str, Any],
    *,
    now: Optional[int] = None,
) -> dict[str, int]:
    """Apply one validated report. Replaces the day rows it covers, upserts the rest."""
    stamp = int(now if now is not None else time.time())
    install_id = _text(report.get("install_id"), 64) or ""
    if not install_id:
        raise ValueError("install_id is required")

    device = report.get("device") if isinstance(report.get("device"), dict) else {}
    machine = report.get("machine") if isinstance(report.get("machine"), dict) else {}

    conn.execute(
        """
        INSERT INTO installs (
            install_id, first_seen, last_seen, version, os, os_version, arch,
            cpu, cpu_cores, cpu_threads, ram_gb, gpu, gpu_vendor, vram_gb,
            gpu_count, backend, apple_silicon, locale, python, torch,
            transformers, accounts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(install_id) DO UPDATE SET
            last_seen = excluded.last_seen,
            version = excluded.version,
            os = excluded.os,
            os_version = excluded.os_version,
            arch = excluded.arch,
            cpu = excluded.cpu,
            cpu_cores = excluded.cpu_cores,
            cpu_threads = excluded.cpu_threads,
            ram_gb = excluded.ram_gb,
            gpu = excluded.gpu,
            gpu_vendor = excluded.gpu_vendor,
            vram_gb = excluded.vram_gb,
            gpu_count = excluded.gpu_count,
            backend = excluded.backend,
            apple_silicon = excluded.apple_silicon,
            locale = excluded.locale,
            python = excluded.python,
            torch = excluded.torch,
            transformers = excluded.transformers,
            accounts = excluded.accounts
        """,
        (
            install_id,
            stamp,
            stamp,
            _text(report.get("version"), 64),
            _text(device.get("os"), 48),
            _text(device.get("os_version"), 64),
            _text(device.get("arch"), 32),
            _text(device.get("cpu"), 160),
            _clamp_int(device.get("cpu_cores"), 0, 4096) or None,
            _clamp_int(device.get("cpu_threads"), 0, 4096) or None,
            round(_clamp_float(device.get("ram_gb"), 0, 4096), 2),
            _text(device.get("gpu"), 160),
            _text(device.get("gpu_vendor"), 48),
            round(_clamp_float(device.get("vram_gb"), 0, 4096), 2),
            _clamp_int(device.get("gpu_count"), 0, 64),
            _text(device.get("backend"), 24),
            1 if device.get("apple_silicon") else 0,
            _text(machine.get("locale"), 32),
            _text(machine.get("python"), 24),
            _text(machine.get("torch"), 24),
            _text(machine.get("transformers"), 24),
            _clamp_int(machine.get("accounts"), 0, 1000),
        ),
    )

    window_days = _clamp_int(report.get("window_days"), 1, MAX_WINDOW_DAYS)
    window_start = (date.today() - timedelta(days = window_days - 1)).isoformat()

    # The covered window is replaced wholesale, so a day that went from 3 requests
    # to 0 is recorded as 0 instead of keeping the old number forever.
    conn.execute(
        "DELETE FROM usage_daily WHERE install_id = ? AND day >= ?",
        (install_id, window_start),
    )
    conn.execute(
        "DELETE FROM model_daily WHERE install_id = ? AND day >= ?",
        (install_id, window_start),
    )

    usage_rows = 0
    for entry in report.get("usage") or []:
        if not isinstance(entry, dict):
            continue
        day = _day(entry.get("day"))
        source = _text(entry.get("source"), 16)
        if day is None or source not in ("chat", "api"):
            continue
        conn.execute(
            """
            INSERT INTO usage_daily (install_id, day, source, requests, prompt_tokens, completion_tokens)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(install_id, day, source) DO UPDATE SET
                requests = excluded.requests,
                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens
            """,
            (
                install_id,
                day,
                source,
                _clamp_int(entry.get("requests")),
                _clamp_int(entry.get("prompt_tokens")),
                _clamp_int(entry.get("completion_tokens")),
            ),
        )
        usage_rows += 1

    model_rows = 0
    for entry in report.get("models") or []:
        if not isinstance(entry, dict):
            continue
        day = _day(entry.get("day"))
        model = _text(entry.get("model"), 200)
        if day is None or model is None:
            continue
        kind = _text(entry.get("kind"), 16) or "chat"
        if kind not in ("chat", "api"):
            kind = "chat"
        conn.execute(
            """
            INSERT INTO model_daily (install_id, day, model, kind, uses, prompt_tokens, completion_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(install_id, day, model, kind) DO UPDATE SET
                uses = excluded.uses,
                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens
            """,
            (
                install_id,
                day,
                model,
                kind,
                _clamp_int(entry.get("uses")),
                _clamp_int(entry.get("prompt_tokens")),
                _clamp_int(entry.get("completion_tokens")),
            ),
        )
        model_rows += 1

    training_rows = 0
    for entry in report.get("trainings") or []:
        if not isinstance(entry, dict):
            continue
        run_key = _text(entry.get("run_key"), 64)
        if not run_key:
            continue
        conn.execute(
            """
            INSERT INTO trainings (install_id, run_key, day, base_model, dataset, status, method, steps, epochs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(install_id, run_key) DO UPDATE SET
                day = excluded.day,
                base_model = excluded.base_model,
                dataset = excluded.dataset,
                status = excluded.status,
                method = excluded.method,
                steps = excluded.steps,
                epochs = excluded.epochs
            """,
            (
                install_id,
                run_key,
                _day(entry.get("day")),
                _text(entry.get("base_model"), 200),
                _text(entry.get("dataset"), 200),
                _text(entry.get("status"), 32),
                _text(entry.get("method"), 32),
                _clamp_int(entry.get("steps")),
                round(_clamp_float(entry.get("epochs"), 0, 100000), 3),
            ),
        )
        training_rows += 1

    merge_rows = 0
    for entry in report.get("merges") or []:
        if not isinstance(entry, dict):
            continue
        merge_key = _text(entry.get("merge_key"), 64)
        if not merge_key:
            continue
        parents = entry.get("models")
        models = [m for m in (_text(item, 200) for item in (parents or [])) if m][:8]
        conn.execute(
            """
            INSERT INTO merges (install_id, merge_key, day, models_json, method)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(install_id, merge_key) DO UPDATE SET
                day = excluded.day,
                models_json = excluded.models_json,
                method = excluded.method
            """,
            (
                install_id,
                merge_key,
                _day(entry.get("day")),
                json.dumps(models, separators = (",", ":")),
                _text(entry.get("method"), 32),
            ),
        )
        merge_rows += 1

    conn.execute(
        """
        INSERT INTO features (install_id, payload_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(install_id) DO UPDATE SET
            payload_json = excluded.payload_json,
            updated_at = excluded.updated_at
        """,
        (
            install_id,
            json.dumps(_feature_payload(report.get("features")), separators = (",", ":")),
            stamp,
        ),
    )

    return {
        "usage_days": usage_rows,
        "model_days": model_rows,
        "trainings": training_rows,
        "merges": merge_rows,
    }


def _bucket_gb(value: Optional[float], edges: tuple[int, ...]) -> str:
    if not value or value <= 0:
        return "Unknown"
    for edge in edges:
        if value <= edge:
            return f"≤ {edge} GB"
    return f"> {edges[-1]} GB"


def _bump(counter: dict[str, dict[str, int]], key: Optional[str]) -> None:
    label = (key or "Unknown").strip() or "Unknown"
    entry = counter.setdefault(label, {"installs": 0, "value": 0})
    entry["installs"] += 1


def _ranked(
    counter: dict[str, dict[str, int]],
    *,
    label: str = "name",
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows = sorted(counter.items(), key = lambda item: (-item[1]["installs"], item[0].lower()))
    return [
        {label: name, "installs": data["installs"], "value": data["value"]}
        for name, data in rows[:limit]
    ]


def _dense_series(rows: list[sqlite3.Row], days: int, today: date) -> list[dict[str, Any]]:
    by_day = {row["day"]: row for row in rows}
    start = today - timedelta(days = days - 1)
    series: list[dict[str, Any]] = []
    for offset in range(days):
        day = (start + timedelta(days = offset)).isoformat()
        row = by_day.get(day)
        series.append(
            {
                "day": day,
                "prompt_tokens": int(row["prompt_tokens"]) if row else 0,
                "completion_tokens": int(row["completion_tokens"]) if row else 0,
                "requests": int(row["requests"]) if row else 0,
                "installs": int(row["installs"]) if row else 0,
            }
        )
    return series


def overview(conn: sqlite3.Connection, days: int = 30) -> dict[str, Any]:
    """Everything the console renders for one window, in one response."""
    days = max(1, min(MAX_WINDOW_DAYS, int(days)))
    today = date.today()
    since = (today - timedelta(days = days - 1)).isoformat()
    since_epoch = _day_start_epoch(today - timedelta(days = days - 1))

    totals_row = conn.execute(
        """
        SELECT COALESCE(SUM(requests), 0) AS requests,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
               COUNT(DISTINCT install_id) AS active_installs
        FROM usage_daily WHERE day >= ?
        """,
        (since,),
    ).fetchone()

    install_totals = conn.execute(
        """
        SELECT COUNT(*) AS installs,
               COALESCE(SUM(CASE WHEN last_seen >= ? THEN 1 ELSE 0 END), 0) AS active,
               COALESCE(SUM(CASE WHEN first_seen >= ? THEN 1 ELSE 0 END), 0) AS new
        FROM installs
        """,
        (since_epoch, since_epoch),
    ).fetchone()

    windows: list[dict[str, Any]] = []
    for window in WINDOWS:
        start = (today - timedelta(days = window - 1)).isoformat()
        row = conn.execute(
            """
            SELECT COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                   COALESCE(SUM(requests), 0) AS requests,
                   COUNT(DISTINCT install_id) AS installs
            FROM usage_daily WHERE day >= ?
            """,
            (start,),
        ).fetchone()
        windows.append(
            {
                "days": window,
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "requests": int(row["requests"]),
                "active_installs": int(row["installs"]),
            }
        )

    series_rows = conn.execute(
        """
        SELECT day,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
               COALESCE(SUM(requests), 0) AS requests,
               COUNT(DISTINCT install_id) AS installs
        FROM usage_daily
        WHERE day >= ?
        GROUP BY day
        ORDER BY day
        """,
        ((today - timedelta(days = 179)).isoformat(),),
    ).fetchall()

    model_rows = conn.execute(
        """
        SELECT model,
               COALESCE(SUM(uses), 0) AS uses,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
               COUNT(DISTINCT install_id) AS installs,
               GROUP_CONCAT(DISTINCT kind) AS kinds
        FROM model_daily
        WHERE day >= ?
        GROUP BY model
        ORDER BY uses DESC, model ASC
        LIMIT 15
        """,
        (since,),
    ).fetchall()

    merge_rows = conn.execute(
        "SELECT install_id, day, models_json, method FROM merges WHERE day >= ? ORDER BY day DESC",
        (since,),
    ).fetchall()
    parent_counter: dict[str, dict[str, int]] = {}
    method_counter: dict[str, dict[str, int]] = {}
    recent_merges: list[dict[str, Any]] = []
    for row in merge_rows:
        try:
            parents = json.loads(row["models_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            parents = []
        for parent in parents:
            if isinstance(parent, str):
                _bump(parent_counter, parent)
        _bump(method_counter, row["method"] or "unknown")
        if len(recent_merges) < 10:
            recent_merges.append(
                {
                    "day": row["day"],
                    "models": parents,
                    "method": row["method"] or "unknown",
                }
            )

    training_rows = conn.execute(
        "SELECT install_id, day, base_model, dataset, status, method, steps "
        "FROM trainings WHERE day >= ? ORDER BY day DESC",
        (since,),
    ).fetchall()
    base_counter: dict[str, dict[str, int]] = {}
    dataset_counter: dict[str, dict[str, int]] = {}
    training_method_counter: dict[str, dict[str, int]] = {}
    status_counter: dict[str, dict[str, int]] = {}
    for row in training_rows:
        _bump(base_counter, row["base_model"] or "Unknown")
        _bump(dataset_counter, row["dataset"] or "Unknown")
        _bump(training_method_counter, row["method"] or "unknown")
        _bump(status_counter, row["status"] or "unknown")

    active_install_rows = conn.execute(
        """
        SELECT install_id, version, os, gpu, gpu_vendor, vram_gb, ram_gb, cpu,
               cpu_threads, backend, locale, apple_silicon
        FROM installs WHERE last_seen >= ?
        """,
        (since_epoch,),
    ).fetchall()
    gpu_counter: dict[str, dict[str, int]] = {}
    os_counter: dict[str, dict[str, int]] = {}
    backend_counter: dict[str, dict[str, int]] = {}
    ram_counter: dict[str, dict[str, int]] = {}
    vram_counter: dict[str, dict[str, int]] = {}
    version_counter: dict[str, dict[str, int]] = {}
    vendor_counter: dict[str, dict[str, int]] = {}
    thread_counter: dict[str, dict[str, int]] = {}
    locale_counter: dict[str, dict[str, int]] = {}
    for row in active_install_rows:
        _bump(gpu_counter, row["gpu"] or "Unknown")
        _bump(os_counter, row["os"] or "Unknown")
        _bump(backend_counter, (row["backend"] or "Unknown").upper())
        _bump(ram_counter, _bucket_gb(row["ram_gb"], (8, 16, 32, 64, 128)))
        _bump(vram_counter, _bucket_gb(row["vram_gb"], (4, 8, 12, 16, 24, 48)))
        _bump(version_counter, row["version"] or "unknown")
        _bump(vendor_counter, row["gpu_vendor"] or "Unknown")
        _bump(thread_counter, f"{row['cpu_threads'] or 0} threads")
        _bump(locale_counter, row["locale"] or "unknown")

    feature_rows = conn.execute("SELECT payload_json FROM features").fetchall()
    feature_totals = {key: {"installs": 0, "value": 0} for key in FEATURE_KEYS}
    for row in feature_rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in FEATURE_KEYS:
            value = _clamp_int(payload.get(key))
            if value > 0:
                feature_totals[key]["installs"] += 1
                feature_totals[key]["value"] += value

    return {
        "generated_at": int(time.time()),
        "days": days,
        "since": since,
        "totals": {
            "installs": int(install_totals["installs"]),
            "active_installs": int(install_totals["active"]),
            "new_installs": int(install_totals["new"]),
            "active_in_window": int(totals_row["active_installs"]),
            "requests": int(totals_row["requests"]),
            "prompt_tokens": int(totals_row["prompt_tokens"]),
            "completion_tokens": int(totals_row["completion_tokens"]),
            "trainings": len(training_rows),
            "merges": len(merge_rows),
            "completed_trainings": int(status_counter.get("completed", {}).get("installs", 0)),
        },
        "windows": windows,
        "series": _dense_series(series_rows, 180, today),
        "models": [
            {
                "model": row["model"],
                "uses": int(row["uses"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "installs": int(row["installs"]),
                "kinds": (row["kinds"] or "chat").split(","),
            }
            for row in model_rows
        ],
        "merges": {
            "total": len(merge_rows),
            "parents": _ranked(parent_counter),
            "methods": _ranked(method_counter),
            "recent": recent_merges,
        },
        "trainings": {
            "total": len(training_rows),
            "bases": _ranked(base_counter),
            "datasets": _ranked(dataset_counter),
            "methods": _ranked(training_method_counter),
            "status": _ranked(status_counter),
            "recent": [
                {
                    "day": row["day"],
                    "base_model": row["base_model"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "status": row["status"],
                    "steps": int(row["steps"] or 0),
                }
                for row in training_rows[:10]
            ],
        },
        "devices": {
            "gpus": _ranked(gpu_counter),
            "vendors": _ranked(vendor_counter),
            "os": _ranked(os_counter),
            "backends": _ranked(backend_counter),
            "ram": _ranked(ram_counter),
            "vram": _ranked(vram_counter),
            "cpu_threads": _ranked(thread_counter),
            "locales": _ranked(locale_counter, limit = 8),
            "versions": _ranked(version_counter),
        },
        "features": [
            {
                "key": key,
                "label": FEATURE_LABELS.get(key, key),
                "installs": feature_totals[key]["installs"],
                "value": feature_totals[key]["value"],
            }
            for key in FEATURE_KEYS
            if feature_totals[key]["installs"] > 0
        ],
    }


def list_installs(conn: sqlite3.Connection, limit: int = MAX_LIST_ROWS) -> list[dict[str, Any]]:
    limit = max(1, min(MAX_LIST_ROWS, int(limit)))
    rows = conn.execute(
        """
        SELECT i.*,
               COALESCE((SELECT SUM(requests) FROM usage_daily u
                         WHERE u.install_id = i.install_id), 0) AS requests,
               COALESCE((SELECT SUM(prompt_tokens + completion_tokens) FROM usage_daily u
                         WHERE u.install_id = i.install_id), 0) AS tokens,
               (SELECT COUNT(*) FROM trainings t WHERE t.install_id = i.install_id) AS trainings,
               (SELECT COUNT(*) FROM merges m WHERE m.install_id = i.install_id) AS merges,
               (SELECT COUNT(*) FROM model_daily m WHERE m.install_id = i.install_id) AS model_days
        FROM installs i
        ORDER BY i.last_seen DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    top_models: dict[str, str] = {}
    for model_row in conn.execute(
        """
        SELECT install_id, model, SUM(uses) AS uses
        FROM model_daily
        GROUP BY install_id, model
        ORDER BY uses DESC
        """
    ):
        top_models.setdefault(model_row["install_id"], model_row["model"])

    active_days: dict[str, int] = {
        row["install_id"]: int(row["days"])
        for row in conn.execute(
            "SELECT install_id, COUNT(DISTINCT day) AS days FROM usage_daily GROUP BY install_id"
        )
    }

    installs: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        entry["short_id"] = row["install_id"][:8]
        entry["top_model"] = top_models.get(row["install_id"])
        entry["active_days"] = active_days.get(row["install_id"], 0)
        installs.append(entry)
    return installs


def installs_csv(conn: sqlite3.Connection) -> str:
    buffer = io.StringIO()
    columns = (
        "install_id",
        "first_seen",
        "last_seen",
        "version",
        "os",
        "arch",
        "cpu",
        "cpu_cores",
        "cpu_threads",
        "ram_gb",
        "gpu",
        "vram_gb",
        "gpu_count",
        "backend",
        "locale",
        "accounts",
        "requests",
        "tokens",
        "trainings",
        "merges",
        "top_model",
    )
    writer = csv.DictWriter(buffer, fieldnames = columns, extrasaction = "ignore")
    writer.writeheader()
    for install in list_installs(conn):
        row = dict(install)
        row["first_seen"] = _iso(row["first_seen"])
        row["last_seen"] = _iso(row["last_seen"])
        writer.writerow(row)
    return buffer.getvalue()


def _iso(epoch: Any) -> str:
    try:
        return datetime.fromtimestamp(int(epoch), tz = timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OverflowError, OSError):
        return ""
