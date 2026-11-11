# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Store-level tests: idempotence, windows, rankings, devices and CSV."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from telemetry import store  # noqa: E402


def day(offset: int = 0) -> str:
    return (date.today() - timedelta(days = offset)).isoformat()


def make_report(install_id: str = "abcdefgh-0001", *, base: str = "unsloth/Llama-3.2-1B") -> dict:
    return {
        "install_id": install_id,
        "version": "v0.1.39-beta",
        "window_days": 180,
        "device": {
            "os": "Windows",
            "os_version": "11",
            "arch": "AMD64",
            "cpu": "AMD Ryzen 9 5900X",
            "cpu_cores": 12,
            "cpu_threads": 24,
            "ram_gb": 64,
            "gpu": "NVIDIA GeForce RTX 4090",
            "gpu_vendor": "NVIDIA",
            "vram_gb": 24,
            "gpu_count": 1,
            "backend": "cuda",
            "apple_silicon": False,
        },
        "machine": {
            "locale": "en",
            "python": "3.13.17",
            "torch": "2.9.0",
            "transformers": "5.0.0",
            "accounts": 1,
        },
        "usage": [
            {"day": day(0), "source": "chat", "requests": 4, "prompt_tokens": 1000, "completion_tokens": 2000},
            {"day": day(1), "source": "api", "requests": 2, "prompt_tokens": 300, "completion_tokens": 400},
            {"day": day(8), "source": "chat", "requests": 1, "prompt_tokens": 50, "completion_tokens": 60},
        ],
        "models": [
            {"day": day(0), "model": "unsloth/Qwen3-8B", "kind": "chat", "uses": 10, "prompt_tokens": 1000, "completion_tokens": 2000},
            {"day": day(0), "model": "unsloth/Qwen3-8B", "kind": "api", "uses": 2, "prompt_tokens": 100, "completion_tokens": 200},
            {"day": day(1), "model": "meta-llama/Llama-3.1-8B", "kind": "chat", "uses": 3, "prompt_tokens": 200, "completion_tokens": 200},
        ],
        "trainings": [
            {
                "run_key": "run-1",
                "day": day(0),
                "base_model": base,
                "dataset": "tatsu-lab/alpaca",
                "status": "completed",
                "method": "qlora",
                "steps": 120,
                "epochs": 1.0,
            }
        ],
        "merges": [
            {
                "merge_key": "merge-1",
                "day": day(0),
                "models": ["unsloth/Qwen3-8B", "meta-llama/Llama-3.1-8B"],
                "method": "linear",
            },
            {
                "merge_key": "merge-2",
                "day": day(2),
                "models": ["unsloth/Qwen3-8B", "google/gemma-3-4b"],
                "method": "slerp",
            },
        ],
        "features": {
            "chat_threads": 12,
            "trainings": 1,
            "merges": 2,
            "api_requests": 2,
        },
    }


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "telemetry.sqlite3"
    with store.transaction(path) as conn:
        store.ingest(conn, make_report())
    return path


def test_ingest_replaces_covered_days_instead_of_double_counting(db):
    with store.transaction(db) as conn:
        first = store.overview(conn, days = 2)
        store.ingest(conn, make_report())
        second = store.overview(conn, days = 2)
    assert first["totals"]["prompt_tokens"] == second["totals"]["prompt_tokens"] == 1300
    assert first["totals"]["completion_tokens"] == second["totals"]["completion_tokens"] == 2400
    assert second["totals"]["requests"] == 6


def test_a_corrected_day_overwrites_the_old_count(db):
    with store.transaction(db) as conn:
        corrected = make_report()
        corrected["usage"][0]["prompt_tokens"] = 5
        corrected["usage"][0]["completion_tokens"] = 7
        store.ingest(conn, corrected)
        overview = store.overview(conn, days = 2)
    assert overview["totals"]["prompt_tokens"] == 305
    assert overview["totals"]["completion_tokens"] == 407


def test_windows_are_rolling_and_ordered(db):
    with store.transaction(db) as conn:
        windows = {row["days"]: row for row in store.overview(conn, days = 30)["windows"]}
    assert [row["days"] for row in windows.values()] == [2, 4, 9, 12, 30, 180]
    assert windows[2]["prompt_tokens"] == 1300
    assert windows[4]["prompt_tokens"] == 1300
    assert windows[9]["prompt_tokens"] == 1350
    assert windows[30]["prompt_tokens"] == 1350
    assert windows[180]["prompt_tokens"] == 1350


def test_models_rank_by_uses_and_carry_kinds(db):
    with store.transaction(db) as conn:
        overview = store.overview(conn, days = 30)
    top = overview["models"][0]
    assert top["model"] == "unsloth/Qwen3-8B"
    assert top["uses"] == 12
    assert set(top["kinds"]) == {"chat", "api"}
    assert top["prompt_tokens"] == 1100


def test_training_ranks_base_datasets_methods_and_status(db):
    with store.transaction(db) as conn:
        training = store.overview(conn, days = 30)["trainings"]
    assert training["total"] == 1
    assert training["bases"][0]["name"] == "unsloth/Llama-3.2-1B"
    assert training["datasets"][0]["name"] == "tatsu-lab/alpaca"
    assert training["methods"][0]["name"] == "qlora"
    assert training["status"][0]["name"] == "completed"
    assert training["recent"][0]["steps"] == 120


def test_merge_parents_count_once_per_run(db):
    with store.transaction(db) as conn:
        merges = store.overview(conn, days = 30)["merges"]
    parents = {row["name"]: row["installs"] for row in merges["parents"]}
    assert parents["unsloth/Qwen3-8B"] == 2
    assert parents["meta-llama/Llama-3.1-8B"] == 1
    methods = {row["name"]: row["installs"] for row in merges["methods"]}
    assert methods == {"linear": 1, "slerp": 1}
    assert merges["total"] == 2


def test_device_specs_are_bucketed_and_ranked(db):
    with store.transaction(db) as conn:
        devices = store.overview(conn, days = 30)["devices"]
    assert devices["gpus"][0]["name"] == "NVIDIA GeForce RTX 4090"
    assert devices["vendors"][0]["name"] == "NVIDIA"
    assert devices["os"][0]["name"] == "Windows"
    assert devices["backends"][0]["name"] == "CUDA"
    assert devices["ram"][0]["name"] == "≤ 64 GB"
    assert devices["vram"][0]["name"] == "≤ 24 GB"
    assert devices["versions"][0]["name"] == "v0.1.39-beta"


def test_features_only_keep_known_nonzero_counters(db):
    with store.transaction(db) as conn:
        features = {row["key"]: row for row in store.overview(conn, days = 30)["features"]}
    assert features["chat_threads"]["value"] == 12
    assert features["trainings"]["installs"] == 1
    assert "models_loaded" not in features


def test_installs_list_and_csv_include_the_top_model(db):
    with store.transaction(db) as conn:
        installs = store.list_installs(conn)
        body = store.installs_csv(conn)
    assert len(installs) == 1
    assert installs[0]["short_id"] == "abcdefgh"
    assert installs[0]["top_model"] == "unsloth/Qwen3-8B"
    assert installs[0]["active_days"] == 3
    assert "abcdefgh-0001" in body
    assert "unsloth/Qwen3-8B" in body


def test_invalid_days_are_dropped(db):
    with store.transaction(db) as conn:
        dirty = make_report()
        dirty["usage"].append(
            {"day": "not-a-day", "source": "chat", "requests": 99, "prompt_tokens": 9, "completion_tokens": 9}
        )
        dirty["usage"].append(
            {"day": day(0), "source": "bogus", "requests": 99, "prompt_tokens": 9, "completion_tokens": 9}
        )
        store.ingest(conn, dirty)
        overview = store.overview(conn, days = 30)
    assert overview["totals"]["requests"] == 7


def test_a_second_install_joins_the_rankings(tmp_path):
    path = tmp_path / "telemetry.sqlite3"
    with store.transaction(path) as conn:
        store.ingest(conn, make_report("abcdefgh-0001"))
        other = make_report("ijklmnop-0002")
        other["device"]["gpu"] = "Apple M4 Max"
        other["device"]["gpu_vendor"] = "Apple"
        other["device"]["ram_gb"] = 128
        other["device"]["vram_gb"] = 0
        other["trainings"][0]["dataset"] = "openai/gsm8k"
        store.ingest(conn, other)
        overview = store.overview(conn, days = 30)
    assert overview["totals"]["installs"] == 2
    assert overview["totals"]["active_installs"] == 2
    datasets = {row["name"]: row["installs"] for row in overview["trainings"]["datasets"]}
    assert datasets == {"tatsu-lab/alpaca": 1, "openai/gsm8k": 1}
    ram = {row["name"]: row["installs"] for row in overview["devices"]["ram"]}
    assert ram["≤ 64 GB"] == 1 and ram["≤ 128 GB"] == 1
