# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Janitor rules: a dry run that keeps the newest, the young and the pinned."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from core.labs import benchmarks, housekeeping


def _run(root: Path, name: str, days_old: float, size: int = 16) -> Path:
    directory = root / name
    (directory / "weights").mkdir(parents = True)
    (directory / "config.json").write_text("{}", encoding = "utf-8")
    (directory / "weights" / "model.safetensors").write_bytes(b"x" * size)
    stamp = time.time() - days_old * 86_400
    os.utime(directory, (stamp, stamp))
    return directory


def _workspace(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    outputs = tmp_path / "outputs"
    exports = tmp_path / "exports"
    outputs.mkdir(parents = True)
    exports.mkdir(parents = True)
    monkeypatch.setattr(housekeeping, "_paths", lambda: {"outputs": outputs, "exports": exports})
    monkeypatch.setattr(housekeeping, "_running_outputs", lambda: [])
    monkeypatch.setattr(benchmarks, "pinned_artifacts", lambda: set())
    return outputs, exports


def test_the_preview_keeps_the_newest_and_anything_younger_than_the_rule(tmp_path, monkeypatch):
    outputs, _ = _workspace(tmp_path, monkeypatch)
    newest = _run(outputs, "run-a", days_old = 1)
    second = _run(outputs, "run-b", days_old = 2)
    old = _run(outputs, "run-c", days_old = 30)
    older = _run(outputs, "run-d", days_old = 40)

    preview = housekeeping.preview_rules(
        {"keep_newest": 2, "min_age_days": 7, "roots": ["outputs"]}
    )

    kept = {entry["name"] for entry in preview["kept"]}
    offered = {entry["name"] for entry in preview["candidates"]}
    assert kept == {"run-a", "run-b"}
    assert offered == {"run-c", "run-d"}
    assert str(newest) and str(second) and str(old) and str(older)
    per_run = 16 + len("{}")
    assert preview["total_bytes"] == per_run * 2
    assert preview["rules"]["enabled"] is True


def test_a_young_folder_outside_the_keep_window_is_protected(tmp_path, monkeypatch):
    outputs, _ = _workspace(tmp_path, monkeypatch)
    _run(outputs, "run-a", days_old = 40)
    _run(outputs, "run-b", days_old = 3)
    _run(outputs, "run-c", days_old = 2)

    preview = housekeeping.preview_rules(
        {"keep_newest": 1, "min_age_days": 7, "roots": ["outputs"]}
    )

    # The newest folder is kept by position, the next one only by its age.
    assert {entry["name"] for entry in preview["kept"]} == {"run-c"}
    assert {entry["name"] for entry in preview["candidates"]} == {"run-a"}
    reasons = {Path(item["path"]).name: item["reason"] for item in preview["protected"]}
    assert "younger" in reasons["run-b"].lower()


def test_a_pinned_checkpoint_is_never_offered(tmp_path, monkeypatch):
    outputs, _ = _workspace(tmp_path, monkeypatch)
    _run(outputs, "run-a", days_old = 1)
    _run(outputs, "run-b", days_old = 40)
    pinned = _run(outputs, "run-c", days_old = 50)
    monkeypatch.setattr(
        benchmarks, "pinned_artifacts", lambda: {housekeeping._normalized(str(pinned))}
    )

    preview = housekeeping.preview_rules(
        {"keep_newest": 1, "min_age_days": 7, "roots": ["outputs"]}
    )

    assert {entry["name"] for entry in preview["candidates"]} == {"run-b"}
    reasons = {Path(item["path"]).name: item["reason"] for item in preview["protected"]}
    assert "pinned" in reasons["run-c"].lower()


def test_roots_limit_what_a_rule_touches(tmp_path, monkeypatch):
    outputs, exports = _workspace(tmp_path, monkeypatch)
    _run(outputs, "run-a", days_old = 1)
    _run(outputs, "run-b", days_old = 40)
    _run(exports, "bundle-a", days_old = 1)
    _run(exports, "bundle-b", days_old = 40)

    preview = housekeeping.preview_rules(
        {"keep_newest": 1, "min_age_days": 7, "roots": ["outputs"]}
    )

    assert {entry["name"] for entry in preview["candidates"]} == {"run-b"}


def test_apply_removes_exactly_the_preview_and_keeps_the_rest(tmp_path, monkeypatch):
    outputs, _ = _workspace(tmp_path, monkeypatch)
    newest = _run(outputs, "run-a", days_old = 1)
    old = _run(outputs, "run-b", days_old = 40)

    result = housekeeping.apply_rules(
        {"keep_newest": 1, "min_age_days": 7, "roots": ["outputs"]}
    )

    assert [item["path"] for item in result["result"]["deleted"]] == [str(old)]
    assert result["result"]["freed_bytes"] > 0
    assert newest.is_dir() and not old.exists()


def test_switched_off_rules_refuse_to_apply_but_still_preview(tmp_path, monkeypatch):
    outputs, _ = _workspace(tmp_path, monkeypatch)
    _run(outputs, "run-a", days_old = 1)
    _run(outputs, "run-b", days_old = 40)
    rules = {"enabled": False, "keep_newest": 1, "min_age_days": 7, "roots": ["outputs"]}

    preview = housekeeping.preview_rules(rules)
    assert preview["candidates"]

    with pytest.raises(ValueError):
        housekeeping.apply_rules(rules)


def test_rules_round_trip_and_bad_values_are_refused(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)

    assert housekeeping.read_rules()["keep_newest"] == 3
    saved = housekeeping.save_rules({"keep_newest": 5, "min_age_days": 14})
    assert saved["keep_newest"] == 5
    assert housekeeping.read_rules()["min_age_days"] == 14
    assert housekeeping.rules_path().is_file()

    for payload in (
        {"keep_newest": -1},
        {"min_age_days": 99_999},
        {"roots": ["everywhere"]},
        {"roots": []},
        {"enabled": "yes"},
    ):
        with pytest.raises(ValueError):
            housekeeping.save_rules(payload)

    # A hand-edited file that no longer validates falls back to the defaults
    # instead of taking the page down.
    housekeeping.rules_path().write_text('{"keep_newest": "many"}', encoding = "utf-8")
    assert housekeeping.read_rules() == housekeeping._DEFAULT_RULES
