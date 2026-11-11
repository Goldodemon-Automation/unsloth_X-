# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in telemetry: settings, payload parsing and the send gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.telemetry import reporter, snapshot  # noqa: E402
from utils import telemetry_settings  # noqa: E402


class _FakeSettingsStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def get(self, key, fallback = None):
        return self.values.get(key, fallback)

    def upsert(self, values, **_kwargs):
        self.values.update(values)
        return values


@pytest.fixture()
def store(monkeypatch):
    from storage import studio_db

    fake = _FakeSettingsStore()
    monkeypatch.setattr(studio_db, "get_app_setting", fake.get)
    monkeypatch.setattr(studio_db, "upsert_app_settings", fake.upsert)
    return fake


def test_telemetry_is_off_by_default(store):
    settings = telemetry_settings.public_settings()
    assert settings["enabled"] is False
    assert settings["endpoint"] == telemetry_settings.DEFAULT_ENDPOINT
    assert settings["has_ingest_key"] is False


def test_install_id_is_random_stable_and_accepted_by_the_collector(store):
    first = telemetry_settings.get_install_id()
    second = telemetry_settings.get_install_id()
    assert first == second
    assert len(first) == 32
    assert first.isalnum()


def test_enabling_without_an_endpoint_is_refused(store):
    with pytest.raises(ValueError, match = "endpoint"):
        telemetry_settings.update_telemetry_settings(enabled = True)


def test_endpoint_must_be_http_and_carry_no_credentials(store):
    with pytest.raises(ValueError):
        telemetry_settings.normalize_endpoint("ftp://collector.example")
    with pytest.raises(ValueError):
        telemetry_settings.normalize_endpoint("https://user:pass@collector.example/report")
    assert (
        telemetry_settings.normalize_endpoint("https://collector.example/api/v1/report")
        == "https://collector.example/api/v1/report"
    )


def test_a_saved_endpoint_and_key_are_masked_in_the_public_view(store):
    telemetry_settings.update_telemetry_settings(
        endpoint = "https://collector.example/api/v1/report",
        ingest_key = "shared-secret",
        enabled = True,
    )
    public = telemetry_settings.public_settings()
    assert public["enabled"] is True
    assert public["has_ingest_key"] is True
    assert "shared-secret" not in json.dumps(public)
    assert public["dashboard_url"] == "https://collector.example/"
    assert telemetry_settings.get_telemetry_settings()["ingest_key"] == "shared-secret"


def test_clearing_the_endpoint_while_enabled_is_refused(store):
    telemetry_settings.update_telemetry_settings(
        endpoint = "https://collector.example/api/v1/report",
        enabled = True,
    )
    with pytest.raises(ValueError, match = "endpoint"):
        telemetry_settings.update_telemetry_settings(endpoint = "")


def test_should_send_honours_the_switch_interval_and_failure_backoff(store):
    now = 1_800_000_000
    assert telemetry_settings.should_send(now) is False

    telemetry_settings.update_telemetry_settings(
        endpoint = "https://collector.example/api/v1/report",
        enabled = True,
    )
    assert telemetry_settings.should_send(now) is True

    telemetry_settings.record_send(True, now = now)
    assert telemetry_settings.should_send(now + 60) is False
    assert telemetry_settings.should_send(now + 6 * 3600) is True

    telemetry_settings.record_send(False, "Collector answered HTTP 500.", now = now + 6 * 3600)
    assert telemetry_settings.should_send(now + 6 * 3600 + 60) is False
    assert telemetry_settings.should_send(now + 6 * 3600 + 16 * 60) is True
    assert telemetry_settings.get_telemetry_settings()["last_sent_at"] == now


def test_resetting_the_install_id_breaks_the_join(store):
    first = telemetry_settings.get_install_id()
    second = telemetry_settings.reset_install_id()
    assert first != second
    assert telemetry_settings.get_install_id() == second


def test_usage_parsing_matches_the_profile_fold_fallbacks():
    assert snapshot.usage_from_metadata(
        {"contextUsage": {"promptTokens": 11, "completionTokens": 22}}
    ) == (11, 22)
    assert snapshot.usage_from_metadata(
        {"serverTimings": {"prompt_n": 7, "predicted_n": 9}}
    ) == (7, 9)
    assert snapshot.usage_from_metadata({"timing": {"tokenCount": 5}}) == (0, 5)
    assert snapshot.usage_from_metadata(None) == (0, 0)


def test_local_paths_become_opaque_labels():
    assert (
        snapshot.public_identifier(r"C:\Users\nutti\models\qwen", local_label = "local model")
        == "local model"
    )
    assert (
        snapshot.public_identifier("/home/nutti/models/qwen", local_label = "local model")
        == "local model"
    )
    assert (
        snapshot.public_identifier(r"models\qwen", local_label = "local model")
        == "local model"
    )
    assert (
        snapshot.public_identifier("unsloth/Qwen3-8B", local_label = "local model")
        == "unsloth/Qwen3-8B"
    )


def test_model_resolution_prefers_the_answerer_over_the_request():
    metadata = {
        "responseDetails": {"responseModelId": "qwen3-8b-here"},
        "contextUsage": {"modelId": "requested-alias"},
    }
    assert snapshot.model_from_metadata(metadata) == "qwen3-8b-here"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"load_in_4bit": True, "use_lora": True}, "qlora"),
        ({"use_lora": True, "lora_rank": 16}, "lora"),
        ({"full_finetuning": True}, "full"),
        ({"use_dpo": True}, "dpo"),
        ({"training_type": "LoRA"}, "lora"),
        ({"training_type": "full-finetune"}, "full"),
        ({"method": "GRPO"}, "grpo"),
        ({}, "unknown"),
        ("not json", "unknown"),
    ],
)
def test_training_method_derivation(config, expected):
    assert snapshot.derive_training_method(config) == expected


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class _FakeConn:
    """Routes the two fold queries to canned rows, keyed by the table they name."""

    def __init__(self, chat_rows, api_rows):
        self.chat_rows = chat_rows
        self.api_rows = api_rows

    def execute(self, sql, _params = ()):
        return _FakeCursor(self.api_rows if "api_usage_events" in sql else self.chat_rows)


def test_fold_usage_counts_models_days_and_skips_fork_clones():
    now_ms = 1_800_000_000_000
    chat = [
        {
            "thread_id": "t1",
            "role": "assistant",
            "metadata_json": json.dumps(
                {
                    "contextUsage": {"promptTokens": 10, "completionTokens": 20},
                    "responseDetails": {"responseModelId": "unsloth/Qwen3-8B"},
                }
            ),
            "created_at": now_ms,
            "model_id": "requested-alias",
            "thread_created_at": now_ms - 1000,
            "forked_from_thread_id": None,
        },
        {
            "thread_id": "t2",
            "role": "assistant",
            "metadata_json": json.dumps(
                {"contextUsage": {"promptTokens": 100, "completionTokens": 200}}
            ),
            "created_at": now_ms - 5000,
            "model_id": "meta-llama/Llama-3.1-8B",
            "thread_created_at": now_ms,
            "forked_from_thread_id": "t1",
        },
    ]
    api = [
        {
            "model": "unsloth/Qwen3-8B",
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "created_at": now_ms,
        }
    ]
    usage, models = snapshot.fold_usage(_FakeConn(chat, api), now_ms - 86_400_000)

    chat_usage = next(row for row in usage if row["source"] == "chat")
    assert chat_usage["requests"] == 1
    assert chat_usage["prompt_tokens"] == 10
    assert chat_usage["completion_tokens"] == 20
    api_usage = next(row for row in usage if row["source"] == "api")
    assert api_usage["requests"] == 1

    chat_models = [row for row in models if row["kind"] == "chat"]
    assert len(chat_models) == 1
    assert chat_models[0]["model"] == "unsloth/Qwen3-8B"
    assert chat_models[0]["uses"] == 1


def test_a_disabled_reporter_never_touches_the_network(store, monkeypatch):
    called = []
    monkeypatch.setattr(
        reporter,
        "_post",
        lambda *args: called.append(args) or (True, ""),
    )
    result = reporter.send_snapshot(force = True)
    assert result["skipped"] is True
    assert result["ok"] is False
    assert called == []


def test_reporter_posts_the_payload_and_records_success(store, monkeypatch):
    telemetry_settings.update_telemetry_settings(
        endpoint = "https://collector.example/api/v1/report",
        ingest_key = "ingest",
        enabled = True,
    )
    seen = {}

    def fake_post(endpoint, payload, ingest_key):
        seen["endpoint"] = endpoint
        seen["payload"] = payload
        seen["key"] = ingest_key
        return True, ""

    monkeypatch.setattr(reporter, "_post", fake_post)
    monkeypatch.setattr(
        reporter,
        "build_snapshot",
        lambda install_id, window_days = 180: {
            "install_id": install_id,
            "usage": [],
            "models": [],
            "trainings": [],
            "merges": [],
        },
    )
    result = reporter.send_snapshot(force = True)
    assert result["ok"] is True
    assert seen["endpoint"] == "https://collector.example/api/v1/report"
    assert seen["key"] == "ingest"
    assert telemetry_settings.get_telemetry_settings()["last_sent_at"] is not None
