# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json

import pytest

from core.inference import studio_access


def test_all_studio_tools_are_registered_once():
    from core.inference.tools import ALL_TOOLS

    names = [tool["function"]["name"] for tool in ALL_TOOLS]
    for name in studio_access.STUDIO_ACCESS_TOOL_NAMES:
        assert names.count(name) == 1


def test_all_studio_tools_are_gated(monkeypatch):
    monkeypatch.setattr(studio_access, "self_training_available", lambda: False)

    for name in ("labs", "merge_models", "automated_training", "training"):
        with pytest.raises(studio_access.StudioAccessError, match = "Studio access is off"):
            studio_access.execute_studio_access_tool(name, {"action": "state"})


def test_labs_state_reads_the_real_route(monkeypatch):
    monkeypatch.setattr(studio_access, "self_training_available", lambda: True)
    monkeypatch.setattr("routes.labs.read_labs", lambda: {"enabled": True, "features": []})

    result = json.loads(studio_access.execute_labs_tool({"action": "state"}))

    assert result["enabled"] is True


def test_merge_plan_builds_the_route_request(monkeypatch):
    monkeypatch.setattr(studio_access, "self_training_available", lambda: True)
    captured = {}

    def fake_plan(request):
        captured["models"] = request.models
        return {"models": request.models, "tier": "cpu"}

    monkeypatch.setattr("routes.advanced.plan_merge", fake_plan)

    result = json.loads(
        studio_access.execute_merge_tool(
            {
                "action": "plan",
                "models": ["org/a", "org/b"],
                "method": "ties",
            }
        )
    )

    assert captured == {"models": ["org/a", "org/b"]}
    assert result["tier"] == "cpu"


def test_automated_training_plan_uses_the_labs_engine(monkeypatch):
    monkeypatch.setattr(studio_access, "self_training_available", lambda: True)
    monkeypatch.setattr(
        "routes.labs.plan_automated_training",
        lambda request: {"runnable": True, "rounds": [request.spec["rounds"]]},
    )

    result = json.loads(
        studio_access.execute_automated_training_tool(
            {
                "action": "plan",
                "spec": {
                    "source": "org/base",
                    "datasets": [{"repo_id": "org/data"}],
                    "rounds": 1,
                },
            }
        )
    )

    assert result["runnable"] is True
    assert result["rounds"] == [1]


def test_training_runs_route_is_async_and_reaches_the_tool(monkeypatch):
    monkeypatch.setattr(studio_access, "self_training_available", lambda: True)

    async def fake_runs(**kwargs):
        return {"runs": [{"id": "run-1"}], "total": 1, "limit": kwargs["limit"]}

    monkeypatch.setattr("routes.training_history.list_training_runs", fake_runs)

    result = json.loads(
        studio_access.execute_training_tool({"action": "runs", "limit": 7})
    )

    assert result["total"] == 1
    assert result["limit"] == 7


def test_training_start_strips_secrets_and_runs_directly(monkeypatch):
    monkeypatch.setattr(studio_access, "self_training_available", lambda: True)
    monkeypatch.setattr(studio_access, "_deferred_inference_active", lambda: False)
    captured = {}

    def fake_start(config):
        captured.update(config)
        return {"kind": "training", "job_id": "job-1", "status": "queued"}

    monkeypatch.setattr(studio_access, "_start_training", fake_start)

    result = json.loads(
        studio_access.execute_training_tool(
            {
                "action": "start",
                "config": {
                    "model_name": "unsloth/Qwen3-0.6B",
                    "training_type": "LoRA/QLoRA",
                    "format_type": "auto",
                    "hf_dataset": "tatsu-lab/alpaca",
                    "hf_token": "must-not-reach-the-route",
                },
            }
        )
    )

    assert result["job_id"] == "job-1"
    assert captured["model_name"] == "unsloth/Qwen3-0.6B"
    assert "hf_token" not in captured


def test_labs_specs_drop_embedded_tokens(monkeypatch):
    monkeypatch.setattr(studio_access, "self_training_available", lambda: True)
    captured = {}

    def fake_plan(request):
        captured.update(request.spec)
        return {"runnable": True}

    monkeypatch.setattr("routes.labs.plan_cmt", fake_plan)

    studio_access.execute_labs_tool(
        {
            "action": "cmt_plan",
            "spec": {
                "source": "org/base",
                "datasets": [{"repo_id": "org/data", "hf_token": "secret"}],
                "hf_token": "secret",
            },
        }
    )

    assert "hf_token" not in captured
    assert "hf_token" not in captured["datasets"][0]


def test_unknown_action_is_reported(monkeypatch):
    monkeypatch.setattr(studio_access, "self_training_available", lambda: True)

    with pytest.raises(studio_access.StudioAccessError, match = "action must be one of"):
        studio_access.execute_labs_tool({"action": "not-a-real-action"})
