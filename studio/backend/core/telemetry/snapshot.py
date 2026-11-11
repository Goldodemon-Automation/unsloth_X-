# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build the opt-in telemetry snapshot from the local Studio database.

The snapshot is deliberately aggregates only. It reads assistant-message usage
metadata, content-free API receipts, training rows and merge manifests, and it
derives hardware classes from the same detection the System tab uses. Prompts,
replies, titles, paths, usernames, tokens and hostnames never enter it.

Model and dataset identifiers do leave the machine, because "most used model" and
"trained on what" cannot be answered without them. Settings renders the whole
payload before the first send.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
MAX_WINDOW_DAYS = 180
MAX_TRAINING_RUNS = 1000
MAX_MERGE_RUNS = 200
MAX_MODEL_CHARS = 200
MAX_TEXT_CHARS = 200


_LOCAL_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[/\\]|~[\\/])")


def _bounded(value: Any, limit: int = MAX_TEXT_CHARS) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned[:limit] if cleaned else None


def public_identifier(value: Any, *, local_label: str) -> Optional[str]:
    """A model or dataset id, with a local filesystem path reduced to a label.

    Reports promise no paths, and a user-uploaded dataset's name is often one. A
    Hub id keeps its name; an absolute or Windows path becomes an opaque label.
    """
    text = _bounded(value, MAX_MODEL_CHARS)
    if not text:
        return None
    if _LOCAL_PATH_RE.match(text) or "\\" in text:
        return local_label
    return text


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


def _day_from_ms(milliseconds: int) -> str:
    try:
        return datetime.fromtimestamp(milliseconds / 1000, tz = timezone.utc).date().isoformat()
    except (ValueError, OverflowError, OSError):
        return date.today().isoformat()


def _day_from_iso(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10]).isoformat()
    except ValueError:
        return None


def _key(install_id: str, value: str) -> str:
    return hashlib.sha256(f"{install_id}:{value}".encode("utf-8", "replace")).hexdigest()[:32]


def usage_from_metadata(metadata: Any) -> tuple[int, int]:
    """Input and output tokens from one assistant message's metadata blob.

    Mirrors the Profile tab's reading: the provider's usage chunk first, then
    llama.cpp's server timings, then the adapter's own completion count.
    """
    if not isinstance(metadata, dict):
        return 0, 0
    usage = metadata.get("contextUsage")
    usage = usage if isinstance(usage, dict) else {}
    timing = metadata.get("timing")
    timing = timing if isinstance(timing, dict) else {}
    server = metadata.get("serverTimings")
    server = server if isinstance(server, dict) else {}

    prompt = _as_int(usage.get("promptTokens")) or _as_int(server.get("prompt_n"))
    completion = _as_int(usage.get("completionTokens")) or _as_int(server.get("predicted_n"))
    if completion == 0:
        completion = _as_int(timing.get("tokenCount"))
    return prompt, completion


def model_from_metadata(metadata: Any) -> Optional[str]:
    """The model that actually answered, not necessarily the requested alias."""
    if not isinstance(metadata, dict):
        return None
    details = metadata.get("responseDetails")
    details = details if isinstance(details, dict) else {}
    usage = metadata.get("contextUsage")
    usage = usage if isinstance(usage, dict) else {}
    return _bounded(details.get("responseModelId"), MAX_MODEL_CHARS) or _bounded(
        usage.get("modelId"), MAX_MODEL_CHARS
    )


def derive_training_method(config: Any) -> str:
    """A short label for how a run trained, from the config keys trainers write."""
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except (json.JSONDecodeError, TypeError):
            config = None
    if not isinstance(config, dict):
        return "unknown"

    lora = any(
        config.get(key)
        for key in ("use_lora", "lora", "lora_rank", "lora_alpha", "r", "target_modules")
    )
    if config.get("load_in_4bit") or config.get("load_in_8bit"):
        return "qlora" if lora else "quantized"
    if lora:
        return "lora"

    for key, label in (
        ("use_dpo", "dpo"),
        ("dpo", "dpo"),
        ("use_grpo", "grpo"),
        ("grpo", "grpo"),
        ("use_ppo", "ppo"),
        ("ppo", "ppo"),
        ("orpo", "orpo"),
        ("simpo", "simpo"),
        ("use_kto", "kto"),
    ):
        if config.get(key):
            return label

    explicit = (
        config.get("method")
        or config.get("finetune_type")
        or config.get("training_type")
        or config.get("trainer_type")
    )
    if isinstance(explicit, str) and explicit.strip():
        label = explicit.strip().lower().replace("-", "_").replace(" ", "_")
        if label in ("full", "full_finetune", "full_finetuning"):
            return "full"
        return label[:32]
    if config.get("full_finetuning") or config.get("full_finetune"):
        return "full"
    return "unknown"


def _chat_rows(conn, since_ms: int) -> list[Any]:
    return conn.execute(
        """
        SELECT m.thread_id, m.role, m.metadata_json, m.created_at,
               t.model_id, t.created_at AS thread_created_at,
               t.forked_from_thread_id
        FROM chat_messages m
        LEFT JOIN chat_threads t ON t.id = m.thread_id
        WHERE m.created_at >= ? AND m.role = 'assistant'
        ORDER BY m.thread_id, m.created_at
        """,
        (since_ms,),
    ).fetchall()


def _api_rows(conn, since_ms: int) -> list[Any]:
    return conn.execute(
        """
        SELECT model, prompt_tokens, completion_tokens, created_at
        FROM api_usage_events
        WHERE created_at >= ?
        ORDER BY created_at
        """,
        (since_ms,),
    ).fetchall()


def fold_usage(conn, since_ms: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fold chat and API receipts into daily counters and daily model rows.

    Fork clones are skipped while their original is still present: forking copies
    the whole ancestry, timestamps included, so counting a clone would count the
    same response twice. Telemetry accepts the narrower undercount of a fork whose
    original was deleted rather than reproducing the full profile fold here.
    """
    usage: dict[tuple[str, str], list[int]] = {}
    models: dict[tuple[str, str, str], list[int]] = {}

    for row in _chat_rows(conn, since_ms):
        created_at = _as_int(row["created_at"])
        thread_created = _as_int(row["thread_created_at"])
        if row["forked_from_thread_id"] and created_at < thread_created:
            continue

        day = _day_from_ms(created_at)
        bucket = usage.setdefault((day, "chat"), [0, 0, 0])
        bucket[0] += 1

        metadata: Any = None
        raw = row["metadata_json"]
        if raw:
            try:
                metadata = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                metadata = None
        prompt, completion = usage_from_metadata(metadata)
        bucket[1] += prompt
        bucket[2] += completion

        model = public_identifier(
            model_from_metadata(metadata) or row["model_id"], local_label = "local model"
        )
        if model:
            model_bucket = models.setdefault((day, model, "chat"), [0, 0, 0])
            model_bucket[0] += 1
            model_bucket[1] += prompt
            model_bucket[2] += completion

    for row in _api_rows(conn, since_ms):
        created_at = _as_int(row["created_at"])
        day = _day_from_ms(created_at)
        prompt = _as_int(row["prompt_tokens"])
        completion = _as_int(row["completion_tokens"])
        bucket = usage.setdefault((day, "api"), [0, 0, 0])
        bucket[0] += 1
        bucket[1] += prompt
        bucket[2] += completion

        model = public_identifier(row["model"], local_label = "local model")
        if model:
            model_bucket = models.setdefault((day, model, "api"), [0, 0, 0])
            model_bucket[0] += 1
            model_bucket[1] += prompt
            model_bucket[2] += completion

    usage_rows = [
        {
            "day": day,
            "source": source,
            "requests": int(values[0]),
            "prompt_tokens": int(values[1]),
            "completion_tokens": int(values[2]),
        }
        for (day, source), values in sorted(usage.items())
        if values[0] > 0
    ]
    model_rows = [
        {
            "day": day,
            "model": model,
            "kind": kind,
            "uses": int(values[0]),
            "prompt_tokens": int(values[1]),
            "completion_tokens": int(values[2]),
        }
        for (day, model, kind), values in sorted(models.items())
        if values[0] > 0
    ]
    return usage_rows, model_rows


def _training_runs(conn, install_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, status, model_name, dataset_name, config_json,
               started_at, final_step
        FROM training_runs
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (MAX_TRAINING_RUNS,),
    ).fetchall()
    runs: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row["id"] or "")
        if not run_id:
            continue
        config: Any = None
        if row["config_json"]:
            try:
                config = json.loads(row["config_json"])
            except (json.JSONDecodeError, TypeError):
                config = None
        if isinstance(config, dict):
            dataset = public_identifier(
                config.get("dataset"), local_label = "local dataset"
            ) or public_identifier(config.get("dataset_name"), local_label = "local dataset")
            epochs = config.get("num_train_epochs")
        else:
            dataset = None
            epochs = None
        try:
            epoch_value = round(float(epochs), 3) if epochs is not None else 0.0
        except (TypeError, ValueError):
            epoch_value = 0.0
        runs.append(
            {
                "run_key": _key(install_id, run_id),
                "day": _day_from_iso(row["started_at"]),
                "base_model": public_identifier(row["model_name"], local_label = "local model"),
                "dataset": dataset
                or public_identifier(row["dataset_name"], local_label = "local dataset"),
                "status": _bounded(row["status"], 32),
                "method": derive_training_method(config),
                "steps": _as_int(row["final_step"]),
                "epochs": max(0.0, min(100000.0, epoch_value)),
            }
        )
    return runs


def _merge_runs(install_id: str) -> list[dict[str, Any]]:
    from core.advanced.merge_history import list_merge_runs
    from utils.paths import outputs_root

    entries = list_merge_runs(outputs_root(), limit = MAX_MERGE_RUNS)
    merges: list[dict[str, Any]] = []
    for entry in entries:
        parents = [
            public_identifier(parent, local_label = "local model")
            for parent in (entry.get("models") or [])
        ]
        parents = [parent for parent in parents if parent][:8]
        if len(parents) < 2:
            continue
        identity = str(entry.get("run_id") or entry.get("name") or "")
        if not identity:
            continue
        merges.append(
            {
                "merge_key": _key(install_id, identity),
                "day": _day_from_iso(entry.get("created_at")),
                "models": parents,
                "method": _bounded(entry.get("method"), 32) or "unknown",
            }
        )
    return merges


def _count(conn, sql: str, params: tuple = ()) -> int:
    try:
        row = conn.execute(sql, params).fetchone()
        return _as_int(row[0])
    except Exception:
        return 0


def _feature_counts(conn, merges: int) -> dict[str, int]:
    return {
        "chat_threads": _count(conn, "SELECT COUNT(*) FROM chat_threads"),
        "chat_messages": _count(conn, "SELECT COUNT(*) FROM chat_messages"),
        "trainings": _count(conn, "SELECT COUNT(*) FROM training_runs"),
        "completions": _count(
            conn, "SELECT COUNT(*) FROM training_runs WHERE status = 'completed'"
        ),
        "api_requests": _count(conn, "SELECT COUNT(*) FROM api_usage_events"),
        "merges": merges,
        "merges_successful": merges,
    }


def _gpu_vendor(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered or "tesla" in lowered:
        return "NVIDIA"
    if "amd" in lowered or "radeon" in lowered or "instinct" in lowered:
        return "AMD"
    if "intel" in lowered or "arc" in lowered:
        return "Intel"
    if "apple" in lowered or "m1" in lowered or "m2" in lowered or "m3" in lowered or "m4" in lowered:
        return "Apple"
    return None


def _device() -> dict[str, Any]:
    import psutil

    from utils.hardware import get_device, get_gpu_summary, is_apple_silicon

    gpu = get_gpu_summary() or {}
    gpu_name = _bounded(gpu.get("gpu_name"), 160)
    backend = get_device()
    return {
        "os": _bounded(platform.system(), 48),
        "os_version": _bounded(platform.release(), 64),
        "arch": _bounded(platform.machine(), 32),
        "cpu": _bounded(
            platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or platform.machine(),
            160,
        ),
        "cpu_cores": _as_int(psutil.cpu_count(logical = False)),
        "cpu_threads": _as_int(psutil.cpu_count(logical = True)),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "gpu": gpu_name,
        "gpu_vendor": _gpu_vendor(gpu_name),
        "vram_gb": round(float(gpu.get("vram_total_gb") or 0), 2),
        "gpu_count": 1 if gpu_name else 0,
        "backend": backend.value if backend is not None else "cpu",
        "apple_silicon": bool(is_apple_silicon()),
    }


def _version(package: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(package)[:24]
    except (PackageNotFoundError, Exception):  # noqa: B014 -- metadata can raise anything
        return None


def _machine() -> dict[str, Any]:
    try:
        import locale as _locale

        language = _locale.getlocale()[0] or ""
    except Exception:
        language = ""
    return {
        "locale": _bounded(language or os.environ.get("LANG", ""), 32),
        "python": _bounded(platform.python_version(), 24),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "accounts": 0,
    }


def _safe(call, fallback, label: str):
    try:
        return call()
    except Exception as error:
        logger.debug("telemetry snapshot: %s failed: %s", label, error)
        return fallback


def _studio_version() -> Optional[str]:
    from utils.studio_version import get_studio_version

    return _bounded(get_studio_version(), 64)


def build_snapshot(install_id: str, *, window_days: int = MAX_WINDOW_DAYS) -> dict[str, Any]:
    """The full report for one install. Partial data is better than no report."""
    window_days = max(1, min(MAX_WINDOW_DAYS, int(window_days)))
    since = date.today() - timedelta(days = window_days - 1)
    since_ms = int(
        datetime(since.year, since.month, since.day, tzinfo = timezone.utc).timestamp() * 1000
    )

    usage: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    trainings: list[dict[str, Any]] = []
    features: dict[str, int] = {}

    try:
        from storage.studio_db import get_connection

        conn = get_connection()
        try:
            usage, models = _safe(lambda: fold_usage(conn, since_ms), ([], []), "usage")
            trainings = _safe(lambda: _training_runs(conn, install_id), [], "trainings")
            merges = _safe(lambda: _merge_runs(install_id), [], "merges")
            features = _safe(
                lambda: _feature_counts(conn, len(merges)),
                {},
                "features",
            )
        finally:
            conn.close()
    except Exception as error:
        logger.debug("telemetry snapshot: local history unavailable: %s", error)
        merges = []

    return {
        "schema_version": SCHEMA_VERSION,
        "install_id": install_id,
        "sent_at": int(datetime.now(tz = timezone.utc).timestamp()),
        "version": _safe(_studio_version, None, "version"),
        "window_days": window_days,
        "device": _safe(_device, {}, "device"),
        "machine": _safe(_machine, {}, "machine"),
        "usage": usage,
        "models": models,
        "trainings": trainings,
        "merges": merges,
        "features": features,
    }
