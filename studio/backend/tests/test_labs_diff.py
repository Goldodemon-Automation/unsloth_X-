# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run diff: the fields that differ between two checkpoints, and nothing invented."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.labs import artifacts, benchmarks, diff

LLAMA = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 256,
    "tie_word_embeddings": True,
}


def _checkpoint(directory: Path, manifest: dict | None = None, config: dict | None = None) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "config.json").write_text(
        json.dumps(config or LLAMA), encoding = "utf-8"
    )
    (directory / "model.safetensors").write_bytes(b"\0" * 64)
    if manifest is not None:
        (directory / "run_manifest.json").write_text(
            json.dumps(manifest), encoding = "utf-8"
        )
    return directory


def _manifest(*, run_id: str, learning_rate: float, steps: int, dataset: str) -> dict:
    return {
        "schema_version": 1,
        "stage": "train",
        "run_id": run_id,
        "created_at_iso": "2026-11-01T00:00:00Z",
        "studio_version": "0.9.0",
        "config": {"learning_rate": learning_rate, "max_steps": steps},
        "inputs": {"dataset": {"value": dataset}},
        "artifacts": {"output_dir": "/somewhere"},
    }


def _outputs(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "outputs"
    root.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(artifacts, "root", lambda: root)
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    return root


def test_compare_names_the_manifest_fields_that_moved(tmp_path, monkeypatch):
    root = _outputs(tmp_path, monkeypatch)
    left = _checkpoint(
        root / "run-a", _manifest(run_id = "a", learning_rate = 2e-4, steps = 120, dataset = "/data/one.jsonl")
    )
    right = _checkpoint(
        root / "run-b", _manifest(run_id = "b", learning_rate = 1e-4, steps = 240, dataset = "/data/one.jsonl")
    )

    report = diff.compare(str(left), str(right))
    changes = {item["field"]: item for item in report["changes"]}

    assert changes["config.learning_rate"]["left"] == 2e-4
    assert changes["config.learning_rate"]["right"] == 1e-4
    assert changes["config.learning_rate"]["group"] == "config"
    assert changes["config.max_steps"]["left"] == 120
    assert "inputs.dataset.value" not in changes
    assert report["same_count"] > 0
    assert report["changed_count"] == len(report["changes"]) >= 2
    assert report["truncated"] is False

    facts = {item["field"]: item for item in report["facts"]}
    assert facts["weight_bytes"]["changed"] is False
    assert facts["params"]["changed"] is False


def test_a_weight_size_change_shows_in_the_facts(tmp_path, monkeypatch):
    root = _outputs(tmp_path, monkeypatch)
    left = _checkpoint(root / "run-a")
    right = _checkpoint(root / "run-b")
    (right / "model-00002.safetensors").write_bytes(b"\0" * 128)

    report = diff.compare(str(left), str(right))
    facts = {item["field"]: item for item in report["facts"]}

    assert facts["weight_bytes"]["changed"] is True
    assert facts["weight_bytes"]["right"] > facts["weight_bytes"]["left"]


def test_a_folder_without_a_manifest_is_compared_on_its_facts_only(tmp_path, monkeypatch):
    root = _outputs(tmp_path, monkeypatch)
    left = _checkpoint(root / "run-a", _manifest(run_id = "a", learning_rate = 2e-4, steps = 1, dataset = "/d.jsonl"))
    right = _checkpoint(root / "run-b")

    report = diff.compare(str(left), str(right))

    assert report["right"]["has_manifest"] is False
    assert report["changes"] == []
    assert any("only the checkpoint facts" in note for note in report["notes"])
    assert {item["field"] for item in report["facts"]}


def test_comparing_a_folder_with_itself_reports_nothing_but_says_so(tmp_path, monkeypatch):
    root = _outputs(tmp_path, monkeypatch)
    only = _checkpoint(root / "run-a", _manifest(run_id = "a", learning_rate = 2e-4, steps = 1, dataset = "/d.jsonl"))

    report = diff.compare(str(only), str(only))

    assert report["changes"] == []
    assert report["changed_count"] == 0
    assert any("same folder" in note for note in report["notes"])
    assert any("same run id" in note for note in report["notes"])


def test_a_path_outside_outputs_is_refused(tmp_path, monkeypatch):
    _outputs(tmp_path, monkeypatch)
    outside = _checkpoint(tmp_path / "elsewhere", _manifest(run_id = "x", learning_rate = 1e-4, steps = 1, dataset = "/d.jsonl"))

    with pytest.raises(PermissionError):
        diff.compare(str(outside), str(outside))


def test_the_latest_benchmark_for_each_side_is_compared(tmp_path, monkeypatch):
    root = _outputs(tmp_path, monkeypatch)
    left = _checkpoint(root / "run-a", _manifest(run_id = "a", learning_rate = 2e-4, steps = 1, dataset = "/d.jsonl"))
    right = _checkpoint(root / "run-b", _manifest(run_id = "b", learning_rate = 2e-4, steps = 1, dataset = "/d.jsonl"))

    for path, score in ((left, 0.40), (right, 0.46)):
        benchmarks.save_record(
            job_id = path.name,
            artifact = str(path),
            baseline = "org/base",
            report = {
                "baseline_scores": {"mmlu": 0.30},
                "candidate_scores": {"mmlu": score},
                "comparison": {"aggregate_delta": score - 0.30},
            },
            suites = ["mmlu"],
            limit = None,
            batch_size = 1,
            duration_seconds = 1.0,
            output_dir = str(tmp_path / "runs" / path.name),
        )

    report = diff.compare(str(left), str(right))
    section = report["benchmark"]

    assert section["left"]["candidate_scores"]["mmlu"] == pytest.approx(0.40)
    assert section["right"]["candidate_scores"]["mmlu"] == pytest.approx(0.46)
    assert section["delta"][0]["suite"] == "mmlu"
    assert section["aggregate_delta"] == pytest.approx(0.06)
    assert section["left"]["vs_pinned"]["status"] == "unpinned"
