# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The config.json parse cache: one parse per file version, never a stale answer.

``model_config.py`` parses model config.json for the capability probes, the fit
check, the picker and the planner, and the same file was parsed several times
per page interaction. These tests pin the cache contract: an unchanged file
parses once, a rewrite re-parses, invalidate() forces a read, the bound evicts,
a Hub repo answers per (repo, revision), and a missing or unreadable config
raises on every call instead of being answered from a stale entry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.models import model_config  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_config_cache():
    model_config.invalidate()
    yield
    model_config.invalidate()


def _count_parses(monkeypatch) -> dict:
    """Count calls to the read+parse step (it runs only on a cache miss)."""
    calls = {"n": 0}
    real = model_config._parse_config_file

    def counting(config_path):
        calls["n"] += 1
        return real(config_path)

    monkeypatch.setattr(model_config, "_parse_config_file", counting)
    return calls


def _write_model(tmp_path: Path, name: str, payload: dict) -> Path:
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(payload), encoding = "utf-8")
    return model_dir


def test_two_reads_of_an_unchanged_file_parse_once(tmp_path, monkeypatch):
    model_dir = _write_model(tmp_path, "model", {"model_type": "llama", "hidden_size": 4})
    parses = _count_parses(monkeypatch)

    first = model_config._read_raw_config_json(str(model_dir))
    second = model_config._read_raw_config_json(str(model_dir))

    assert first == second == {"model_type": "llama", "hidden_size": 4}
    assert parses["n"] == 1, "an unchanged config.json must parse once, not once per call"


def test_returned_dicts_are_independent(tmp_path, monkeypatch):
    """A caller that edits what it read must not poison the next answer."""
    model_dir = _write_model(tmp_path, "model", {"model_type": "llama"})
    _count_parses(monkeypatch)

    first = model_config._read_raw_config_json(str(model_dir))
    first["model_type"] = "mutated"

    assert model_config._read_raw_config_json(str(model_dir)) == {"model_type": "llama"}


def test_a_rewritten_file_re_parses(tmp_path, monkeypatch):
    model_dir = _write_model(tmp_path, "model", {"model_type": "llama"})
    parses = _count_parses(monkeypatch)

    assert model_config._read_raw_config_json(str(model_dir)) == {"model_type": "llama"}
    # A different size guarantees a different identity even at coarse mtime resolution.
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "mamba", "hidden_size": 8}), encoding = "utf-8"
    )

    assert model_config._read_raw_config_json(str(model_dir)) == {
        "model_type": "mamba",
        "hidden_size": 8,
    }
    assert parses["n"] == 2, "the rewritten config.json must be re-read"


def test_invalidate_forces_a_read(tmp_path, monkeypatch):
    model_dir = _write_model(tmp_path, "model", {"model_type": "llama"})
    parses = _count_parses(monkeypatch)

    model_config._read_raw_config_json(str(model_dir))
    model_config.invalidate(str(model_dir))
    model_config._read_raw_config_json(str(model_dir))
    assert parses["n"] == 2, "invalidate(path) must drop that file's entry"

    model_config._read_raw_config_json(str(model_dir))
    model_config.invalidate()
    model_config._read_raw_config_json(str(model_dir))
    assert parses["n"] == 3, "invalidate() must drop every entry"


def test_the_bound_evicts(tmp_path, monkeypatch):
    parses = _count_parses(monkeypatch)
    bound = model_config._CONFIG_CACHE_MAX_ENTRIES
    dirs = [
        _write_model(tmp_path, f"m{i:04d}", {"model_type": f"m{i:04d}"}) for i in range(bound + 1)
    ]

    for model_dir in dirs:
        model_config._read_raw_config_json(str(model_dir))
    assert parses["n"] == bound + 1

    model_config._read_raw_config_json(str(dirs[0]))  # evicted: parses again
    model_config._read_raw_config_json(str(dirs[-1]))  # still cached: parses once total
    assert parses["n"] == bound + 2, "the cache must stay bounded: oldest evicted, newest kept"


def test_a_missing_file_raises_and_is_not_cached(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    missing.mkdir()
    parses = _count_parses(monkeypatch)

    for _ in range(2):
        with pytest.raises(FileNotFoundError):
            model_config._read_raw_config_json(str(missing))

    assert parses["n"] == 2, "a raise must never be answered from the cache"


def test_an_unreadable_file_raises_and_is_not_cached(tmp_path, monkeypatch):
    broken = _write_model(tmp_path, "broken", {})
    (broken / "config.json").write_text("{not json", encoding = "utf-8")
    parses = _count_parses(monkeypatch)

    for _ in range(2):
        with pytest.raises(json.JSONDecodeError):
            model_config._read_raw_config_json(str(broken))

    assert parses["n"] == 2, "an unreadable config must keep raising, never replay a stale answer"


def test_hub_reads_answer_per_repo_and_revision(monkeypatch, tmp_path):
    """A Hub config.json is keyed (repo, revision): one key fetches and parses
    once, a pinned revision is its own key."""
    import huggingface_hub

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_type": "llama"}), encoding = "utf-8")
    downloads = []

    def _download(**kwargs):
        downloads.append(kwargs)
        return str(config_path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.setattr("utils.hf_probe.hf_file_definitely_absent", lambda *_a, **_k: False)
    parses = _count_parses(monkeypatch)

    first = model_config._read_raw_config_json("org/model")
    second = model_config._read_raw_config_json("org/model")
    assert first == second == {"model_type": "llama"}
    assert len(downloads) == 1, "one (repo, revision) key must fetch once"
    assert parses["n"] == 1

    model_config._read_raw_config_json("org/model", revision = "commit-a")
    assert len(downloads) == 2
    assert downloads[-1]["revision"] == "commit-a", "a pinned revision is its own key"
