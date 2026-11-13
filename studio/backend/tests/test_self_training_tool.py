# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.inference import self_training
from utils import self_training_settings


def test_setting_defaults_off_when_nothing_is_stored(monkeypatch):
    monkeypatch.setattr("storage.studio_db.get_app_setting", lambda key, default: None)
    assert self_training_settings.get_self_training_enabled() is False


def _hardware(kind: str = "cpu", device: str = "cpu") -> dict:
    target = SimpleNamespace(
        kind = kind,
        device = device,
        name = "Test accelerator" if kind != "cpu" else "Test CPU",
        vendor = "test",
        memory_bytes = 8 * 1024**3 if kind != "cpu" else None,
    )
    return {
        "cpu": "Test CPU",
        "cpu_cores": 8,
        "system_ram_gb": 16.0,
        "free_ram_gb": 12.0,
        "accelerator": target.name if kind != "cpu" else None,
        "vram_gb": 8.0 if kind != "cpu" else None,
        "device": device,
        "device_kind": kind,
        "vendor": "test",
        "bitsandbytes": False,
        "torch_available": True,
        "available_targets": [],
        "default_target": "cpu",
        "_target": target,
        "_probe": SimpleNamespace(),
    }


def test_tool_is_unavailable_until_owner_enables_it(monkeypatch):
    monkeypatch.setattr(self_training, "is_owner_context", lambda: False)
    monkeypatch.setattr(self_training, "get_self_training_enabled", lambda: True)
    assert self_training.self_training_available() is False

    monkeypatch.setattr(self_training, "is_owner_context", lambda: True)
    monkeypatch.setattr(self_training, "get_self_training_enabled", lambda: False)
    assert self_training.self_training_available() is False

    monkeypatch.setattr(self_training, "get_self_training_enabled", lambda: True)
    assert self_training.self_training_available() is True


def test_plan_uses_hardware_objective_and_explicit_step_count():
    plan = self_training.plan_self_training(
        "Write better Python and TypeScript code",
        method = "lora",
        max_steps = 12,
        hardware = _hardware("dgpu", "cuda:0"),
    )

    assert plan["method"] == "lora"
    assert plan["settings"]["max_steps"] == 12
    assert plan["datasets"][0]["repo_id"] == "HuggingFaceH4/CodeAlpaca_20K"
    assert plan["hardware"]["device"] == "cuda:0"
    assert plan["runnable"] is True
    assert plan["training_request"]["training_type"] == "LoRA/QLoRA"


def test_auto_picks_qlora_on_a_cpu_host():
    plan = self_training.plan_self_training(
        "Improve general instruction following",
        hardware = _hardware(),
    )

    assert plan["method"] == "qlora"
    assert plan["training_request"]["load_in_4bit"] is True
    assert plan["training_request"]["training_preset"] == "qlora"


def test_cpu_rejects_a_method_the_trainer_cannot_run():
    with pytest.raises(self_training.SelfTrainingError, match = "only QLoRA and CoRA"):
        self_training.plan_self_training(
            "Improve general instruction following",
            method = "lora",
            hardware = _hardware(),
        )


def test_cmt_plan_collects_two_datasets(monkeypatch):
    monkeypatch.setattr("core.labs.feature_enabled", lambda _feature: True)
    plan = self_training.plan_self_training(
        "Improve code reasoning",
        method = "cmt",
        max_steps = 8,
        hardware = _hardware("dgpu", "cuda:0"),
    )

    assert plan["method"] == "cmt"
    assert len(plan["datasets"]) == 2
    assert len({entry["repo_id"] for entry in plan["datasets"]}) == 2


def test_execute_starts_a_training_job(monkeypatch):
    captured = {}

    monkeypatch.setattr(self_training, "self_training_available", lambda: True)
    monkeypatch.setattr(self_training, "_deferred_inference_active", lambda: False)
    monkeypatch.setattr(self_training, "plan_self_training", lambda *args, **kwargs: {
        "objective": "Improve general instruction following",
        "method": "qlora",
        "method_label": "QLoRA",
        "base_model": "unsloth/Qwen3-0.6B",
        "datasets": [{"repo_id": "tatsu-lab/alpaca", "split": "train"}],
        "settings": {"max_steps": 5},
        "hardware": {"device": "cpu"},
        "runnable": True,
        "blocked_reason": None,
        "training_request": {"model_name": "unsloth/Qwen3-0.6B"},
        "quality_note": "not evaluated",
    })

    def fake_start(request):
        captured["request"] = request
        return {"kind": "training", "job_id": "job-123", "status": "starting", "message": "queued"}

    monkeypatch.setattr(self_training, "_start_training", fake_start)

    result = json.loads(
        self_training.execute_self_training_tool(
            {"objective": "Improve general instruction following", "max_steps": 5}
        )
    )

    assert captured["request"]["model_name"] == "unsloth/Qwen3-0.6B"
    assert result["job_id"] == "job-123"
    assert result["kind"] == "training"
    assert result["plan"]["method"] == "qlora"


def test_execute_defers_start_while_chat_still_holds_the_model(monkeypatch):
    monkeypatch.setattr(self_training, "self_training_available", lambda: True)
    monkeypatch.setattr(self_training, "_deferred_inference_active", lambda: True)
    monkeypatch.setattr(self_training, "_defer_start", lambda plan: (
        "pending-123",
        {"pending_id": "pending-123", "status": "queued", "plan": plan, "job": None},
    ))
    monkeypatch.setattr(self_training, "plan_self_training", lambda *args, **kwargs: {
        "objective": "Improve general instruction following",
        "method": "qlora",
        "method_label": "QLoRA",
        "base_model": "unsloth/Qwen3-0.6B",
        "datasets": [{"repo_id": "tatsu-lab/alpaca", "split": "train"}],
        "settings": {"max_steps": 5},
        "hardware": {"device": "cpu"},
        "runnable": True,
        "blocked_reason": None,
        "training_request": {"model_name": "unsloth/Qwen3-0.6B"},
        "quality_note": "not evaluated",
    })

    result = json.loads(
        self_training.execute_self_training_tool(
            {"objective": "Improve general instruction following"}
        )
    )

    assert result["status"] == "queued"
    assert result["pending_id"] == "pending-123"
    assert result["job_id"] is None
    assert "chat turn finishes" in result["message"]


def test_execute_refuses_when_the_owner_switch_is_off(monkeypatch):
    monkeypatch.setattr(self_training, "self_training_available", lambda: False)

    with pytest.raises(self_training.SelfTrainingError, match = "owner must enable"):
        self_training.execute_self_training_tool({"objective": "Improve something"})


def test_gguf_cannot_be_a_training_base():
    with pytest.raises(self_training.SelfTrainingError, match = "GGUF"):
        self_training.plan_self_training(
            "Improve general instruction following",
            base_model = "unsloth/Qwen3-0.6B-GGUF",
            hardware = _hardware(),
        )
