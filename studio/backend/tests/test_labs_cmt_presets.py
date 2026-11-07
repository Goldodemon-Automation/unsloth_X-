# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Named presets for the CMT backend engine (additive, off by default)."""

import json
from pathlib import Path

import pytest

from core.labs import cmt

PRESET_IDS = ["quick-check", "balanced", "code-heavy", "reasoning-heavy"]


def _model(tmp_path: Path) -> Path:
    directory = tmp_path / "tiny"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 512,
                "intermediate_size": 2048,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
            }
        ),
        encoding = "utf-8",
    )
    (directory / "model.safetensors").write_bytes(b"not a real checkpoint")
    return directory


def _jsonl(tmp_path: Path, name: str, rows: int = 40, text: str = "hello world") -> Path:
    path = tmp_path / name
    with path.open("w", encoding = "utf-8") as stream:
        for index in range(rows):
            stream.write(json.dumps({"text": f"{text} {index}"}) + "\n")
    return path


def _spec(tmp_path: Path, **extra):
    model = _model(tmp_path) if not (tmp_path / "tiny").exists() else tmp_path / "tiny"
    first = _jsonl(tmp_path, "one.jsonl")
    second = _jsonl(tmp_path, "two.jsonl", text = "another example")
    spec = {
        "source": str(model),
        "datasets": [
            {"path": str(first), "name": "one"},
            {"path": str(second), "name": "two"},
        ],
    }
    spec.update(extra)
    return spec


def test_presets_exposes_four_named_presets_with_the_expected_shape():
    listed = cmt.presets()
    assert [preset["id"] for preset in listed] == PRESET_IDS
    for preset in listed:
        assert set(preset) == {"id", "label", "description", "values"}
        assert isinstance(preset["label"], str) and preset["label"]
        assert isinstance(preset["description"], str) and preset["description"]
        assert isinstance(preset["values"], dict) and preset["values"]


def test_presets_returns_copies_so_callers_cannot_mutate_module_state():
    listed = cmt.presets()
    listed[0]["values"]["max_steps"] = 999_999
    listed[0]["label"] = "mutated"
    assert cmt.presets()[0]["values"]["max_steps"] != 999_999
    assert cmt.presets()[0]["label"] != "mutated"


@pytest.mark.parametrize("preset_id", PRESET_IDS)
def test_each_preset_values_survive_normalise(tmp_path, preset_id):
    request = cmt._normalise(_spec(tmp_path, preset = preset_id))
    assert request["preset"] == preset_id
    expected = next(p for p in cmt.presets() if p["id"] == preset_id)["values"]
    for key, value in expected.items():
        if key == "eval_fraction":
            # eval_fraction is per dataset; the preset provides the dataset default.
            assert all(entry["eval_fraction"] == value for entry in request["datasets"])
        else:
            assert request[key] == value


def test_explicit_spec_fields_beat_preset_values(tmp_path):
    request = cmt._normalise(
        _spec(tmp_path, preset = "quick-check", max_steps = 7, learning_rate = 3e-4)
    )
    assert request["preset"] == "quick-check"
    assert request["max_steps"] == 7
    assert request["learning_rate"] == 3e-4
    # Untouched fields still come from the preset.
    assert request["patience"] == cmt.PRESETS["quick-check"]["values"]["patience"]


def test_explicit_dataset_eval_fraction_beats_the_preset(tmp_path):
    spec = _spec(tmp_path, preset = "balanced")
    spec["datasets"][0]["eval_fraction"] = 0.25
    request = cmt._normalise(spec)
    assert request["datasets"][0]["eval_fraction"] == 0.25
    assert request["datasets"][1]["eval_fraction"] == 0.10


def test_unknown_preset_raises_and_lists_the_valid_ids(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        cmt._normalise(_spec(tmp_path, preset = "turbo"))
    message = str(excinfo.value)
    assert "turbo" in message
    for preset_id in PRESET_IDS:
        assert preset_id in message


def test_normalise_without_a_preset_is_unchanged(tmp_path):
    request = cmt._normalise(_spec(tmp_path))
    assert "preset" not in request
    baseline = {
        "source": str(tmp_path / "tiny"),
        "mode": "cpt",
        "added_params": None,
        "method": "depth",
        "insert_strategy": "even",
        "max_steps": 100,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_seq_length": 2048,
        "learning_rate": 2e-4,
        "eval_every": 20,
        "patience": 4,
        "seed": 3407,
        "max_train_rows": cmt.MAX_TRAIN_ROWS,
    }
    assert set(request) == set(baseline) | {"datasets"}
    for key, value in baseline.items():
        assert request[key] == value
    assert len(request["datasets"]) == 2


def test_plan_with_a_preset_still_reports_runnable(tmp_path):
    report = cmt.plan(_spec(tmp_path, preset = "reasoning-heavy"))
    assert report["runnable"] is True
    assert report["training"]["max_steps"] == 150
    assert report["training"]["patience"] == 5


# ---------------------------------------------------------------------------
# PRESET_IDS / preset_values: the machine-readable view of the same table
# ---------------------------------------------------------------------------


def test_preset_ids_is_derived_from_the_preset_keys():
    assert cmt.PRESET_IDS == tuple(PRESET_IDS)
    assert tuple(cmt.PRESETS) == cmt.PRESET_IDS
    assert tuple(preset["id"] for preset in cmt.presets()) == cmt.PRESET_IDS


def test_preset_values_returns_a_copy_that_cannot_mutate_the_table():
    values = cmt.preset_values("balanced")
    assert values == cmt.PRESETS["balanced"]["values"]
    values["max_steps"] = 999_999
    values["smuggled"] = True
    assert cmt.PRESETS["balanced"]["values"]["max_steps"] == 100
    assert "smuggled" not in cmt.PRESETS["balanced"]["values"]
    assert cmt.preset_values("balanced")["max_steps"] == 100
    assert "smuggled" not in cmt.preset_values("balanced")
    # Copies are per call, not shared between callers.
    assert cmt.preset_values("balanced") is not cmt.preset_values("balanced")


def test_preset_values_unknown_id_raises_and_lists_every_valid_id():
    with pytest.raises(ValueError) as excinfo:
        cmt.preset_values("turbo")
    message = str(excinfo.value)
    assert "turbo" in message
    for preset_id in cmt.PRESET_IDS:
        assert preset_id in message


def test_preset_values_accepts_a_padded_id():
    assert cmt.preset_values("  code-heavy  ") == cmt.PRESETS["code-heavy"]["values"]


@pytest.mark.parametrize("preset_id", PRESET_IDS)
def test_presets_and_preset_values_agree_for_every_id(preset_id):
    listed = cmt.presets()
    payload = next(preset for preset in listed if preset["id"] == preset_id)
    values = cmt.preset_values(preset_id)
    # Same field set and same numbers, including the per-dataset eval fraction.
    assert payload["values"] == values
    assert set(payload) == {"id", "label", "description", "values"}
    assert set(payload["values"]) == set(cmt.PRESETS[preset_id]["values"])
    assert payload["label"] == cmt.PRESETS[preset_id]["label"]
    assert payload["description"] == cmt.PRESETS[preset_id]["description"]
    # Agreement is by value, not by a shared mutable object.
    assert payload["values"] is not values


@pytest.mark.parametrize("preset_id", PRESET_IDS)
def test_normalise_merges_the_same_values_preset_values_reports(tmp_path, preset_id):
    request = cmt._normalise(_spec(tmp_path, preset = preset_id))
    values = cmt.preset_values(preset_id)
    assert request["preset"] == preset_id
    for key, value in values.items():
        if key == "eval_fraction":
            assert all(entry["eval_fraction"] == value for entry in request["datasets"])
        else:
            assert request[key] == value
