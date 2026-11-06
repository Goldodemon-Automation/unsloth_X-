# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest

from core.labs import benchmarks


def _report() -> dict:
    return {
        "baseline_scores": {"mmlu": 0.40, "gsm8k": 0.12},
        "candidate_scores": {"mmlu": 0.44, "gsm8k": 0.11, "extra": 0.3},
        "comparison": {
            "entries": [
                {"suite": "mmlu", "baseline": 0.40, "candidate": 0.44, "delta": 0.04}
            ],
            "aggregate_delta": 0.015,
            "verdict": "pareto-positive",
            "improved": ["mmlu"],
            "regressions": ["gsm8k"],
            "flat": [],
        },
        "verdict": "pareto-positive",
        "errors": ["one suite timed out"],
        "warnings": [],
    }


def test_save_and_list_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    record = benchmarks.save_record(
        job_id = "run-1",
        artifact = str(tmp_path / "outputs" / "merged"),
        baseline = "org/base",
        report = _report(),
        suites = ["mmlu", "gsm8k"],
        limit = 50,
        batch_size = 4,
        duration_seconds = 12.5,
        output_dir = str(tmp_path / "runs" / "run-1"),
    )
    assert len(record["id"]) == 32
    assert record["candidate_scores"]["mmlu"] == pytest.approx(0.44)
    assert record["verdict"] == "pareto-positive"
    assert record["errors"] == ["one suite timed out"]

    listed = benchmarks.list_records()
    assert [item["id"] for item in listed] == [record["id"]]
    assert benchmarks.find_record(record["id"])["baseline_name"] == "base"

    assert benchmarks.list_records(artifact = "elsewhere") == []
    assert benchmarks.delete_record(record["id"]) is True
    assert benchmarks.list_records() == []
    assert benchmarks.delete_record(record["id"]) is False


def test_a_failed_write_leaves_no_half_written_record(tmp_path, monkeypatch):
    """A full disk must not strand a truncated record that the list silently skips."""
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    from core.advanced import policy as loading_policy

    def no_space(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(loading_policy.os, "fsync", no_space)

    with pytest.raises(loading_policy.DiskFullError):
        benchmarks.save_record(
            job_id = "run-1",
            artifact = str(tmp_path / "merged"),
            baseline = "org/base",
            report = _report(),
            suites = ["mmlu"],
            limit = 50,
            batch_size = 1,
            duration_seconds = 1.0,
            output_dir = str(tmp_path / "runs"),
        )

    directory = benchmarks.records_root()
    assert directory.is_dir()
    assert list(directory.glob("*.json")) == []
    assert list(directory.glob(".*")) == []


def test_save_rejects_a_report_without_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    with pytest.raises(ValueError):
        benchmarks.save_record(
            job_id = "run-1",
            artifact = str(tmp_path / "a"),
            baseline = "b",
            report = {"errors": ["lm-eval failed"]},
            suites = ["mmlu"],
            limit = None,
            batch_size = 8,
            duration_seconds = 1.0,
            output_dir = str(tmp_path / "run"),
        )


def test_record_ids_are_checked_before_a_path_is_built(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    with pytest.raises(ValueError):
        benchmarks.delete_record("../../escape")
    with pytest.raises(ValueError):
        benchmarks.find_record("not-hex!")


def test_create_run_dir_stays_under_the_benchmark_root(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    first = benchmarks.create_run_dir()
    second = benchmarks.create_run_dir()
    assert first.is_dir() and second.is_dir()
    assert first != second
    assert Path(first).parent == benchmarks.runs_root()


# ---------------------------------------------------------------------------
# Pinned baselines
# ---------------------------------------------------------------------------


def _scored(
    monkeypatch, tmp_path, artifact: Path, scores: dict[str, float], job: str
) -> dict:
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    return benchmarks.save_record(
        job_id = job,
        artifact = str(artifact),
        baseline = "org/base",
        report = {
            "baseline_scores": {name: 0.5 for name in scores},
            "candidate_scores": scores,
            "comparison": {},
            "verdict": "equal",
        },
        suites = list(scores),
        limit = None,
        batch_size = 1,
        duration_seconds = 1.0,
        output_dir = str(tmp_path / "runs" / job),
    )


def test_a_pin_turns_one_record_into_the_reference_for_its_checkpoint(tmp_path, monkeypatch):
    artifact = tmp_path / "merged"
    artifact.mkdir()
    baseline = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.50}, "baseline")

    entry = benchmarks.pin_record(baseline["id"])
    assert entry["record_id"] == baseline["id"]
    assert [pin["record_id"] for pin in benchmarks.list_pins()] == [baseline["id"]]
    assert benchmarks.list_pins()[0]["available"] is True
    assert benchmarks.pinned_artifacts() == {benchmarks._pin_key(str(artifact))}
    assert benchmarks.verdict(benchmarks.find_record(baseline["id"]))["status"] == "pinned"

    worse = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.40}, "worse")
    better = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.60}, "better")
    flat = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.498}, "flat")
    mixed = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.90, "gsm8k": 0.10}, "mixed")

    # 0.01 absolute is 1% of 0.50, so a 0.002 move is noise, not a regression.
    assert benchmarks.verdict(worse)["status"] == "regressed"
    assert benchmarks.verdict(better)["status"] == "improved"
    assert benchmarks.verdict(flat)["status"] == "flat"
    assert benchmarks.verdict(flat)["suites"][0]["delta"] == pytest.approx(-0.002)
    # gsm8k has no pinned score to move against, so only mmlu decides.
    assert benchmarks.verdict(mixed)["status"] == "improved"
    assert benchmarks.verdict(worse)["pinned_record_id"] == baseline["id"]


def test_two_suites_that_move_in_opposite_directions_are_mixed(tmp_path, monkeypatch):
    artifact = tmp_path / "merged"
    artifact.mkdir()
    baseline = _scored(
        monkeypatch, tmp_path, artifact, {"mmlu": 0.50, "gsm8k": 0.50}, "baseline"
    )
    benchmarks.pin_record(baseline["id"])
    record = _scored(
        monkeypatch, tmp_path, artifact, {"mmlu": 0.60, "gsm8k": 0.40}, "mixed"
    )

    result = benchmarks.verdict(record)
    assert result["status"] == "mixed"
    assert result["improved"] == ["mmlu"]
    assert result["regressions"] == ["gsm8k"]
    assert result["aggregate_delta"] == pytest.approx(0.0)


def test_annotate_marks_the_pin_and_the_verdict_on_every_record(tmp_path, monkeypatch):
    artifact = tmp_path / "merged"
    artifact.mkdir()
    baseline = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.50}, "baseline")
    benchmarks.pin_record(baseline["id"])
    later = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.30}, "later")

    annotated = {
        item["id"]: item for item in benchmarks.annotate(benchmarks.list_records())
    }
    assert annotated[baseline["id"]]["pin"]["is_self"] is True
    assert annotated[baseline["id"]]["vs_pinned"]["status"] == "pinned"
    assert annotated[later["id"]]["pin"]["is_self"] is False
    assert annotated[later["id"]]["vs_pinned"]["status"] == "regressed"


def test_annotate_reads_a_shared_pinned_record_once(tmp_path, monkeypatch):
    """One pin, many records: the list must not re-read the baseline per record.

    A history for a single checkpoint can hold dozens of records that all point
    at the same pin, and the list route annotates every one of them. The pinned
    record is the same file for all of them, so it is read once, not N times.
    """
    artifact = tmp_path / "merged"
    artifact.mkdir()
    baseline = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.50}, "baseline")
    benchmarks.pin_record(baseline["id"])
    for index in range(5):
        _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.40}, f"later-{index}")

    records = benchmarks.list_records(limit = 200)
    pinned_path = benchmarks.records_root() / f"{baseline['id']}.json"
    original = benchmarks._read
    reads: list[Path] = []

    def counting(path):
        reads.append(Path(path))
        return original(path)

    monkeypatch.setattr(benchmarks, "_read", counting)
    annotated = benchmarks.annotate(records)

    # The pin file and the shared pinned record, each read exactly once.
    assert len(reads) == 2
    assert reads.count(pinned_path) == 1
    later = [item for item in annotated if item["id"] != baseline["id"]]
    assert len(later) == 5
    assert all(item["vs_pinned"]["status"] == "regressed" for item in later)


def test_unpin_and_delete_keep_the_pin_file_honest(tmp_path, monkeypatch):
    artifact = tmp_path / "merged"
    artifact.mkdir()
    baseline = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.50}, "baseline")
    benchmarks.pin_record(baseline["id"])

    assert benchmarks.unpin(baseline["id"]) is True
    assert benchmarks.unpin(baseline["id"]) is False
    assert benchmarks.list_pins() == []

    benchmarks.pin_record(baseline["id"])
    assert benchmarks.delete_record(baseline["id"]) is True
    assert benchmarks.list_pins() == []
    assert benchmarks.pinned_artifacts() == set()


def test_a_stale_pin_is_reported_rather_than_scored(tmp_path, monkeypatch):
    artifact = tmp_path / "merged"
    artifact.mkdir()
    baseline = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.50}, "baseline")
    benchmarks.pin_record(baseline["id"])
    record = _scored(monkeypatch, tmp_path, artifact, {"mmlu": 0.50}, "later")

    # A record removed behind the pin's back (the prune path) must not be
    # scored as if it were still there.
    (benchmarks.records_root() / f"{baseline['id']}.json").unlink()

    assert benchmarks.verdict(record)["status"] == "missing"
    assert benchmarks.list_pins()[0]["available"] is False


def test_pinning_rejects_an_unknown_record(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        benchmarks.pin_record("a" * 32)
    with pytest.raises(ValueError):
        benchmarks.unpin("../../escape")
