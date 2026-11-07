# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
import json
import os
import time

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


# ---------------------------------------------------------------------------
# Picking the best of several runs
# ---------------------------------------------------------------------------


def _plain(name: str, candidate_scores: dict) -> dict:
    return {"id": name, "candidate_scores": candidate_scores}


def test_record_score_means_the_candidate_scores():
    record = _plain("a", {"mmlu": 0.40, "gsm8k": 0.20})
    assert benchmarks._record_score(record) == pytest.approx(0.30)


def test_record_score_is_none_without_a_usable_number():
    assert benchmarks._record_score({"candidate_scores": {}}) is None
    assert benchmarks._record_score({"candidate_scores": {"mmlu": "high"}}) is None
    assert benchmarks._record_score({}) is None
    assert benchmarks._record_score({"candidate_scores": None}) is None


def test_record_score_ignores_non_finite_and_non_numeric_values():
    record = {
        "candidate_scores": {
            "good": 0.50,
            "nan": float("nan"),
            "inf": float("inf"),
            "label": "n/a",
        }
    }
    # Only the finite number counts, so the mean stays 0.50 and not NaN.
    assert benchmarks._record_score(record) == pytest.approx(0.50)
    assert benchmarks._record_score({"candidate_scores": {"nan": float("nan")}}) is None


def test_rank_records_sorts_by_score_descending():
    low = _plain("low", {"mmlu": 0.10})
    high = _plain("high", {"mmlu": 0.90})
    mid = _plain("mid", {"mmlu": 0.50})
    ranked = benchmarks.rank_records([low, high, mid])
    assert [record["id"] for record in ranked] == ["high", "mid", "low"]


def test_rank_records_puts_unscored_records_last():
    scored = _plain("scored", {"mmlu": 0.10})
    blank = _plain("blank", {})
    missing = {"id": "missing"}
    ranked = benchmarks.rank_records([blank, scored, missing])
    assert [record["id"] for record in ranked] == ["scored", "blank", "missing"]


def test_rank_records_keeps_the_input_order_for_ties():
    first = _plain("first", {"mmlu": 0.50})
    second = _plain("second", {"gsm8k": 0.50})
    third = _plain("third", {"a": 0.25, "b": 0.75})
    ranked = benchmarks.rank_records([first, second, third])
    assert [record["id"] for record in ranked] == ["first", "second", "third"]


def test_rank_records_does_not_mutate_the_input():
    records = [_plain("low", {"mmlu": 0.10}), _plain("high", {"mmlu": 0.90})]
    snapshot = [dict(record) for record in records]
    ranked = benchmarks.rank_records(records)
    assert records == snapshot
    assert [record["id"] for record in records] == ["low", "high"]
    assert ranked is not records
    assert ranked[0] is records[1]


def test_best_record_picks_the_top_and_none_for_empty():
    records = [_plain("low", {"mmlu": 0.10}), _plain("high", {"mmlu": 0.90})]
    assert benchmarks.best_record(records)["id"] == "high"
    assert benchmarks.best_record([]) is None
    assert benchmarks.best_record(iter(records))["id"] == "high"


# ---------------------------------------------------------------------------
# Finding a checkpoint's records, including its step folders
# ---------------------------------------------------------------------------


def _age(record: dict, seconds_ago: float) -> Path:
    """Give one saved record a known mtime so "newest" is not a filesystem race."""
    path = benchmarks.records_root() / f"{record['id']}.json"
    stamp = time.time() - seconds_ago
    os.utime(path, (stamp, stamp))
    return path


def test_newest_for_prefers_the_exact_checkpoint_over_a_newer_step_record(tmp_path, monkeypatch):
    run = tmp_path / "outputs" / "run"
    run.mkdir(parents = True)
    exact = _scored(monkeypatch, tmp_path, run, {"mmlu": 0.50}, "exact")
    step = _scored(monkeypatch, tmp_path, run / "checkpoint-500", {"mmlu": 0.90}, "step")

    # The step record is newer, so a plain newest-first list would pick it.
    _age(exact, 120)
    _age(step, 1)

    assert [item["id"] for item in benchmarks.list_records(limit = 200)] == [
        step["id"],
        exact["id"],
    ]
    assert benchmarks.newest_for(str(run))["id"] == exact["id"]
    under = benchmarks.records_under(str(run))
    assert [item["id"] for item in under] == [step["id"], exact["id"]]


def test_records_under_uses_a_directory_boundary_not_a_string_prefix(tmp_path, monkeypatch):
    run = tmp_path / "out" / "run"
    run.mkdir(parents = True)
    here = _scored(monkeypatch, tmp_path, run, {"mmlu": 0.50}, "here")
    stale_sibling = _scored(
        monkeypatch, tmp_path, tmp_path / "out" / "run-2", {"mmlu": 0.90}, "sibling"
    )
    deeper_sibling = _scored(
        monkeypatch, tmp_path, tmp_path / "out" / "run-20" / "checkpoint-1", {"mmlu": 0.70}, "sib2"
    )
    step = _scored(monkeypatch, tmp_path, run / "checkpoint-500", {"mmlu": 0.60}, "step")
    nested = _scored(
        monkeypatch, tmp_path, run / "checkpoint-500" / "export", {"mmlu": 0.40}, "nested"
    )

    found = {item["id"] for item in benchmarks.records_under(str(run))}
    assert found == {here["id"], step["id"], nested["id"]}
    assert stale_sibling["id"] not in found
    assert deeper_sibling["id"] not in found
    # A trailing separator and a different separator style name the same folder.
    assert {item["id"] for item in benchmarks.records_under(str(run) + os.sep)} == found


def test_path_key_normalizes_backslashes_separators_and_case():
    assert benchmarks._path_key("C:\\work\\out\\run\\") == benchmarks._path_key(
        "C:/work/out/run"
    )
    assert benchmarks._path_key("/out/run//") == benchmarks._path_key("/out/run")
    assert benchmarks._path_key("C:\\work\\out\\run") == os.path.normcase(
        "C:/work/out/run".replace("/", "\\")
    ).replace("\\", "/")
    assert benchmarks._path_key("") == ""
    assert benchmarks._path_key("/") == "/"


def test_records_under_and_newest_for_accept_backslash_paths(tmp_path, monkeypatch):
    run = tmp_path / "out" / "run"
    run.mkdir(parents = True)
    step = _scored(monkeypatch, tmp_path, run / "checkpoint-500", {"mmlu": 0.60}, "step")
    _age(step, 5)
    recorded = str(step["artifact"])
    typed = recorded.replace("/", "\\") + "\\"

    assert benchmarks.newest_for(typed)["id"] == step["id"]
    assert [item["id"] for item in benchmarks.records_under(typed)] == [step["id"]]
    # A backslash typed for the step's parent still reaches the step record.
    parent = str(run).replace("/", "\\")
    assert benchmarks.newest_for(parent)["id"] == step["id"]


def test_records_under_is_newest_first_and_respects_the_limit(tmp_path, monkeypatch):
    run = tmp_path / "out" / "run"
    run.mkdir(parents = True)
    oldest = _scored(monkeypatch, tmp_path, run / "checkpoint-100", {"mmlu": 0.10}, "oldest")
    newest = _scored(monkeypatch, tmp_path, run / "checkpoint-300", {"mmlu": 0.30}, "newest")
    middle = _scored(monkeypatch, tmp_path, run / "checkpoint-200", {"mmlu": 0.20}, "middle")
    _age(oldest, 300)
    _age(newest, 10)
    _age(middle, 100)

    assert [item["id"] for item in benchmarks.records_under(str(run))] == [
        newest["id"],
        middle["id"],
        oldest["id"],
    ]
    assert [item["id"] for item in benchmarks.records_under(str(run), limit = 2)] == [
        newest["id"],
        middle["id"],
    ]
    assert benchmarks.newest_for(str(run))["id"] == newest["id"]


def test_newest_for_is_none_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    # No record directory at all yet.
    assert benchmarks.records_under(str(tmp_path / "out" / "run")) == []
    assert benchmarks.newest_for(str(tmp_path / "out" / "run")) is None

    elsewhere = _scored(monkeypatch, tmp_path, tmp_path / "out" / "other", {"mmlu": 0.5}, "other")
    _age(elsewhere, 1)
    assert benchmarks.records_under(str(tmp_path / "out" / "run")) == []
    assert benchmarks.newest_for(str(tmp_path / "out" / "run")) is None
    assert benchmarks.newest_for("") is None


def test_records_under_keeps_unscored_records(tmp_path, monkeypatch):
    """A location query reports where the evidence is, scored or not."""
    monkeypatch.setattr(benchmarks, "outputs", lambda: tmp_path)
    directory = benchmarks.records_root()
    directory.mkdir(parents = True)
    unscored = {
        "id": "f" * 32,
        "artifact": str(tmp_path / "out" / "run" / "checkpoint-500"),
        "created_at": "2026-01-01T00:00:00Z",
        "candidate_scores": {},
    }
    (directory / f"{unscored['id']}.json").write_text(json.dumps(unscored), encoding = "utf-8")

    found = benchmarks.records_under(str(tmp_path / "out" / "run"))
    assert [item["id"] for item in found] == [unscored["id"]]
    assert benchmarks.newest_for(str(tmp_path / "out" / "run"))["id"] == unscored["id"]
    assert benchmarks._record_score(found[0]) is None


def test_records_under_does_not_mutate_the_stored_record(tmp_path, monkeypatch):
    run = tmp_path / "out" / "run"
    run.mkdir(parents = True)
    record = _scored(monkeypatch, tmp_path, run / "checkpoint-500", {"mmlu": 0.60}, "step")
    path = benchmarks.records_root() / f"{record['id']}.json"
    before = path.read_bytes()

    first = benchmarks.records_under(str(run))
    first[0]["artifact"] = "tampered"
    first[0]["candidate_scores"]["mmlu"] = 99.0

    assert path.read_bytes() == before
    again = benchmarks.records_under(str(run))
    assert again[0]["artifact"] == record["artifact"]
    assert again[0]["candidate_scores"] == {"mmlu": 0.60}
    assert benchmarks.newest_for(str(run))["artifact"] == record["artifact"]

