# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Automated training plans without hardware, and runs on two small fakes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.labs import automated_training as at


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset(tmp_path: Path, name: str, category: str = "general") -> dict:
    return {"path": str(tmp_path / f"{name}.jsonl"), "name": name, "category": category}


def _spec(tmp_path: Path, datasets: int | list = 4, **overrides) -> dict:
    categories = ("code", "syntax", "reasoning", "general")
    if isinstance(datasets, list):
        pool = datasets
    else:
        pool = [
            _dataset(tmp_path, f"data-{index}", categories[index % len(categories)])
            for index in range(datasets)
        ]
    spec = {"source": "unsloth/tiny-base", "datasets": pool}
    spec.update(overrides)
    return spec


def _trainer(calls: list, *, fail=(), scores=None):
    def train(step, output_dir, progress):
        calls.append(step)
        if step["label"] in fail:
            raise RuntimeError(f"train boom {step['label']}")
        Path(output_dir).mkdir(parents = True, exist_ok = True)
        return {"output_dir": output_dir}

    return train


def _benchmark(scores: dict, *, fail=(), calls: list | None = None):
    def benchmark(step, artifact, progress):
        if calls is not None:
            calls.append(step["label"])
        if step["label"] in fail:
            raise RuntimeError(f"benchmark boom {step['label']}")
        return {
            "score": scores.get(step["label"]),
            "record_id": None if scores.get(step["label"]) is None else f"rec-{step['label']}",
        }

    return benchmark


# ---------------------------------------------------------------------------
# Constants and presets
# ---------------------------------------------------------------------------


def test_constants_and_presets_shape():
    assert (at.MIN_DATASETS, at.MAX_DATASETS) == (2, 8)
    assert (at.MIN_RUNS, at.MAX_RUNS) == (2, 72)
    assert at.MAX_ROUNDS == 12
    assert at.METHODS == ("cmt", "cpt", "fft")
    assert at.CATEGORIES == ("code", "syntax", "reasoning", "general")
    assert issubclass(at.AutomatedTrainingError, ValueError)

    entries = at.presets()
    ids = [entry["id"] for entry in entries]
    assert 3 <= len(entries) <= 4
    assert set(ids) == {"quick-check", "balanced", "deep", "reasoning-heavy"}
    for entry in entries:
        assert set(entry) == {"id", "label", "description", "spec"}
        assert entry["label"] and entry["description"]
        spec = entry["spec"]
        assert at.MIN_DATASETS <= spec["datasets_per_round"] <= at.MAX_DATASETS
        assert at.MIN_RUNS <= spec["runs_per_round"] <= at.MAX_RUNS
        assert 1 <= spec["rounds"] <= at.MAX_ROUNDS
        assert at.MIN_RUNS <= spec["continue_runs"] <= at.MAX_RUNS


def test_presets_are_fresh_copies():
    first = at.presets()
    first[0]["spec"]["rounds"] = 99
    assert at.presets()[0]["spec"]["rounds"] != 99


# ---------------------------------------------------------------------------
# normalise
# ---------------------------------------------------------------------------


def test_normalise_fills_every_default(tmp_path):
    normalised = at.normalise(_spec(tmp_path, datasets=2))

    assert normalised["source"] == "unsloth/tiny-base"
    assert normalised["methods"] == ["cmt", "cpt", "fft"]
    assert normalised["datasets_per_round"] == 4
    assert normalised["runs_per_round"] == 8
    assert normalised["rounds"] == 2
    assert normalised["continue_runs"] == 4
    assert normalised["seed"] == 3407
    assert normalised["steps_per_run"] == 100
    assert normalised["timeout_seconds"] == 0
    assert normalised["benchmark"] == {"suites": [], "limit": 8, "batch_size": 1}
    assert normalised["preset"] is None
    assert normalised["training_kwargs"] == {}
    assert all(entry["eval_fraction"] == pytest.approx(0.10) for entry in normalised["datasets"])


def test_normalise_rejects_dataset_pool_outside_two_to_eight(tmp_path):
    with pytest.raises(at.AutomatedTrainingError, match="at least 2"):
        at.normalise(_spec(tmp_path, datasets=1))
    with pytest.raises(at.AutomatedTrainingError, match="at most 8"):
        at.normalise(_spec(tmp_path, datasets=9))


@pytest.mark.parametrize(
    "field,value",
    [
        ("datasets_per_round", 1),
        ("datasets_per_round", 9),
        ("runs_per_round", 1),
        ("runs_per_round", 73),
        ("continue_runs", 1),
        ("continue_runs", 73),
        ("rounds", 0),
        ("rounds", 13),
        ("steps_per_run", 0),
    ],
)
def test_normalise_rejects_out_of_bounds_numbers(tmp_path, field, value):
    with pytest.raises(at.AutomatedTrainingError):
        at.normalise(_spec(tmp_path, **{field: value}))


def test_normalise_methods_and_dataset_entries(tmp_path):
    with pytest.raises(at.AutomatedTrainingError, match="Unknown method"):
        at.normalise(_spec(tmp_path, methods=["cmt", "dpo"]))
    with pytest.raises(at.AutomatedTrainingError, match="at least one"):
        at.run(
            _spec(tmp_path, datasets=2, datasets_per_round=2, methods=[]),
            output_dir=tmp_path / "out",
        )

    empty_methods = at.normalise(_spec(tmp_path, methods=[]))
    assert empty_methods["methods"] == []

    with pytest.raises(at.AutomatedTrainingError, match="category"):
        at.normalise(_spec(tmp_path, datasets=[_dataset(tmp_path, "a", "poetry"), _dataset(tmp_path, "b")]))
    with pytest.raises(at.AutomatedTrainingError, match="path or a Hub"):
        at.normalise({"source": "base", "datasets": [{"name": "a"}, {"name": "b"}]})
    with pytest.raises(at.AutomatedTrainingError, match="both a path"):
        at.normalise(
            {
                "source": "base",
                "datasets": [
                    {"path": "a.jsonl", "repo_id": "owner/a"},
                    {"path": "b.jsonl"},
                ],
            }
        )


def test_normalise_applies_a_preset_as_a_base_layer(tmp_path):
    normalised = at.normalise(_spec(tmp_path, preset="quick-check", rounds=3))

    assert normalised["preset"] == "quick-check"
    assert normalised["datasets_per_round"] == 2
    assert normalised["runs_per_round"] == 3
    assert normalised["rounds"] == 3  # the user value wins over the preset
    assert normalised["continue_runs"] == 2

    with pytest.raises(at.AutomatedTrainingError, match="Unknown preset"):
        at.normalise(_spec(tmp_path, preset="does-not-exist"))


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def test_plan_reports_blocked_reasons(tmp_path):
    too_small = at.plan(_spec(tmp_path, datasets=3, datasets_per_round=4))
    assert too_small["runnable"] is False
    assert "3" in too_small["blocked_reason"] and "4" in too_small["blocked_reason"]
    assert too_small["rounds"] == []

    no_methods = at.plan(_spec(tmp_path, methods=[]))
    assert no_methods["runnable"] is False
    assert "method" in no_methods["blocked_reason"].lower()

    bad_method = at.plan(_spec(tmp_path, methods=["dpo"]))
    assert bad_method["runnable"] is False
    assert "Unknown method" in bad_method["blocked_reason"]


def test_plan_lays_out_cmt_compatible_runs(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=6,
        methods=["cmt", "cpt"],
        datasets_per_round=4,
        runs_per_round=5,
        rounds=2,
        continue_runs=3,
        steps_per_run=30,
    )
    report = at.plan(spec)

    assert report["runnable"] is True
    assert report["blocked_reason"] is None
    assert len(report["rounds"]) == 2
    assert report["projection"] == {
        "total_train_runs": (5 + 3) * 2,
        "datasets_per_round": 4,
        "rounds": 2,
        "timeout_seconds": 0,
    }
    assert report["presets"] == at.presets()
    assert any("Hardware is not checked" in warning for warning in report["warnings"])

    for round_plan in report["rounds"]:
        assert set(round_plan) == {"index", "datasets", "runs", "continue_runs"}
        assert len(round_plan["datasets"]) == 4
        assert round_plan["continue_runs"] == 3
        assert len(round_plan["runs"]) == 5
        for run in round_plan["runs"]:
            assert set(run) == {"index", "method", "datasets", "steps", "spec"}
            assert run["method"] in at.METHODS
            assert run["steps"] == 30
            assert 2 <= len(run["datasets"]) <= 4
            emitted = run["spec"]
            assert emitted["source"] == spec["source"]
            assert emitted["mode"] == "cpt"
            assert emitted["max_steps"] == 30
            assert len(emitted["datasets"]) == len(run["datasets"])
            assert {entry["name"] for entry in emitted["datasets"]} == {
                entry["name"] for entry in run["datasets"]
            }
            for entry in emitted["datasets"]:
                assert set(entry) == {"path", "repo_id", "name", "category", "eval_fraction"}


def test_plan_samples_every_category_and_is_seed_deterministic(tmp_path):
    spec = _spec(tmp_path, datasets=8, datasets_per_round=4, runs_per_round=6, rounds=2)
    first = at.plan(spec)
    second = at.plan(spec)
    other = at.plan({**spec, "seed": 999})

    assert first["rounds"] == second["rounds"]
    assert first["rounds"] != other["rounds"]

    pool_categories = set(at.CATEGORIES)
    for round_plan in first["rounds"]:
        sampled = [
            entry["category"]
            for entry in at.normalise(spec)["datasets"]
            if entry["name"] in round_plan["datasets"]
        ]
        assert set(sampled) == pool_categories


def test_plan_warns_but_does_not_block_on_missing_adapters(tmp_path):
    report = at.plan(_spec(tmp_path, datasets=2, datasets_per_round=2, methods=["cpt", "fft"]))
    assert report["runnable"] is True
    assert any("CPT" in warning and "FFT" in warning for warning in report["warnings"])


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_trains_benchmarks_and_returns_the_best(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=3,
        continue_runs=2,
        rounds=2,
        steps_per_run=5,
    )
    calls: list = []
    benchmark_calls: list = []
    scores = {
        "run-01-cmt": 0.50,
        "run-02-cmt": 0.90,
        "run-03-cmt": 0.70,
        "continue-01": 0.10,
        "continue-02": 0.20,
    }
    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        train=_trainer(calls),
        benchmark=_benchmark(scores, calls=benchmark_calls),
    )

    assert summary["train_runs"] == (3 + 2) * 2
    assert summary["benchmark_runs"] == (3 + 2) * 2
    assert summary["failed_runs"] == 0
    assert len(summary["rounds"]) == 2
    assert summary["best"]["score"] == pytest.approx(0.90)
    assert summary["best"]["method"] == "cmt"
    assert summary["best"]["round"] == 2
    assert summary["best"]["output_dir"].endswith("run-02-cmt")
    assert len(calls) == len(benchmark_calls) == 10
    assert Path(summary["best"]["output_dir"]).is_dir()


def test_run_prefers_a_score_over_none(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=2,
        continue_runs=2,
        rounds=1,
    )
    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        train=_trainer([]),
        benchmark=_benchmark(
            {"run-01-cmt": None, "run-02-cmt": 0.4, "continue-01": None, "continue-02": None}
        ),
    )
    assert summary["best"]["output_dir"].endswith("run-02-cmt")
    assert summary["best"]["score"] == pytest.approx(0.4)


def test_run_keeps_the_earliest_candidate_on_a_tie(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=3,
        continue_runs=2,
        rounds=1,
    )
    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        train=_trainer([]),
        benchmark=_benchmark(
            {
                "run-01-cmt": 0.5,
                "run-02-cmt": 0.5,
                "run-03-cmt": 0.5,
                "continue-01": 0.5,
                "continue-02": 0.5,
            }
        ),
    )
    assert summary["best"]["output_dir"].endswith("run-01-cmt")


def test_continue_training_starts_from_the_round_winner(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=3,
        continue_runs=3,
        rounds=1,
    )
    calls: list = []
    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        train=_trainer(calls),
        benchmark=_benchmark(
            {
                "run-01-cmt": 0.1,
                "run-02-cmt": 0.9,
                "run-03-cmt": 0.5,
                "continue-01": 0.05,
                "continue-02": 0.05,
                "continue-03": 0.05,
            }
        ),
    )
    winner = summary["rounds"][0]["runs"][1]
    assert winner["label"] == "run-02-cmt"
    continued = [step for step in calls if step["kind"] == "continue"]
    assert len(continued) == 3
    for step in continued:
        assert step["method"] == winner["method"]
        assert step["spec"]["source"] == winner["output_dir"]
        assert [entry["name"] for entry in step["spec"]["datasets"]] == winner["datasets"]
        assert step["spec"]["mode"] == "cpt"


def test_one_failed_run_does_not_kill_the_round(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=4,
        continue_runs=2,
        rounds=1,
    )
    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        train=_trainer([], fail={"run-02-cmt"}),
        benchmark=_benchmark(
            {
                "run-01-cmt": 0.3,
                "run-02-cmt": 0.99,
                "run-03-cmt": 0.6,
                "run-04-cmt": 0.4,
                "continue-01": 0.2,
                "continue-02": 0.2,
            },
            fail={"run-03-cmt"},
        ),
    )
    round_record = summary["rounds"][0]
    assert summary["best"]["output_dir"].endswith("run-03-cmt") is False
    assert summary["best"]["score"] == pytest.approx(0.4)
    assert summary["failed_runs"] == 2
    assert {failure["stage"] for failure in round_record["failures"]} == {"train", "benchmark"}
    assert summary["train_runs"] == 5
    assert summary["benchmark_runs"] == 4
    assert len(round_record["winner"]) > 0


def test_every_failed_candidate_raises_naming_the_round(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=2,
        rounds=1,
    )

    def broken(step, output_dir, progress):
        raise RuntimeError("no accelerator")

    with pytest.raises(at.AutomatedTrainingError, match="round 1"):
        at.run(spec, output_dir=tmp_path / "out", train=broken, benchmark=_benchmark({}))


def test_default_trainer_requires_an_adapter_for_cpt_and_fft(tmp_path):
    spec = _spec(tmp_path, datasets=2, datasets_per_round=2, methods=["cpt"], runs_per_round=2)
    with pytest.raises(at.AutomatedTrainingError, match="CPT"):
        at.run(spec, output_dir=tmp_path / "out")


def test_default_adapters_run_cmt_and_warn_about_missing_benchmark(tmp_path, monkeypatch):
    from core.labs import cmt

    seen: list = []

    def fake_cmt_run(run_spec, *, output_dir, progress = None, token = None):
        seen.append((run_spec, output_dir))
        Path(output_dir).mkdir(parents = True, exist_ok = True)
        return {"output_dir": output_dir}

    monkeypatch.setattr(cmt, "run", fake_cmt_run)
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=2,
        continue_runs=2,
        rounds=1,
    )
    summary = at.run(spec, output_dir=tmp_path / "out")

    assert len(seen) == 4
    assert all(run_spec["mode"] == "cpt" for run_spec, _ in seen)
    assert summary["best"]["score"] is None
    assert any("benchmark" in warning.lower() for warning in summary["warnings"])


def test_manifest_is_written_under_the_output_directory(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=2,
        continue_runs=2,
        rounds=1,
    )
    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        train=_trainer([]),
        benchmark=_benchmark({"run-01-cmt": 0.2, "run-02-cmt": 0.1, "continue-01": 0.0, "continue-02": 0.0}),
    )
    manifest_path = tmp_path / "out" / "run_manifest.json"
    assert manifest_path.is_file()
    assert summary["manifest_path"] == str(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding = "utf-8"))

    assert payload["stage"] == "automated_training"
    assert payload["schema_version"] >= 1
    assert payload["run_id"]
    assert payload["spec"]["source"] == spec["source"]
    assert len(payload["rounds"]) == 1
    assert len(payload["winners"]) == 1
    assert payload["totals"]["train_runs"] == 4
    assert payload["totals"]["benchmark_runs"] == 4
    assert payload["winners"][0]["score"] == pytest.approx(0.2)


def test_progress_reports_every_stage(tmp_path):
    stages: list[str] = []
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=2,
        continue_runs=2,
        rounds=1,
    )
    at.run(
        spec,
        output_dir=tmp_path / "out",
        progress=lambda payload: stages.append(payload["stage"]),
        train=_trainer([]),
        benchmark=_benchmark({"run-01-cmt": 0.2, "run-02-cmt": 0.1, "continue-01": 0.0, "continue-02": 0.0}),
    )
    assert {
        "round-start",
        "training",
        "benchmark",
        "pick",
        "continue",
        "round-done",
        "done",
    } <= set(stages)
    assert stages[-1] == "done"


def test_keyboard_interrupt_is_never_swallowed(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=2,
        rounds=1,
    )

    def interrupted(step, output_dir, progress):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        at.run(
            spec,
            output_dir=tmp_path / "out",
            train=interrupted,
            benchmark=_benchmark({}),
        )


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_should_stop_cancels_after_n_runs_and_records_the_manifest(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=4,
        continue_runs=2,
        rounds=2,
    )
    calls: list = []
    stages: list[str] = []
    scores = {"run-01-cmt": 0.4, "run-02-cmt": 0.8}

    def should_stop() -> bool:
        # Two runs are enough: the third scheduling attempt is refused.
        return len(calls) >= 2

    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        progress=lambda payload: stages.append(payload["stage"]),
        train=_trainer(calls),
        benchmark=_benchmark(scores),
        should_stop=should_stop,
    )

    assert summary["cancelled"] is True
    assert len(calls) == 2
    assert summary["train_runs"] == 2
    assert summary["benchmark_runs"] == 2

    # The round that finished is kept, with a winner; the second round never ran.
    assert [record["index"] for record in summary["rounds"]] == [1]
    assert len(summary["rounds"][0]["continue"]) == 0
    assert summary["best"]["score"] == pytest.approx(0.8)
    assert summary["best"]["round"] == 1

    assert stages[-1] == "cancelled"
    assert "done" not in stages
    assert "continue" not in stages

    payload = json.loads(Path(summary["manifest_path"]).read_text(encoding = "utf-8"))
    assert payload["cancelled"] is True
    assert payload["totals"]["train_runs"] == 2
    assert len(payload["rounds"]) == 1
    assert len(payload["winners"]) == 1
    assert payload["artifacts"]["best"]["round"] == 1


def test_cancelling_before_the_first_run_still_writes_a_manifest(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=2,
        rounds=2,
    )
    calls: list = []
    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        train=_trainer(calls),
        benchmark=_benchmark({}),
        should_stop=lambda: True,
    )

    assert calls == []
    assert summary["cancelled"] is True
    assert summary["rounds"] == []
    assert summary["best"] is None
    assert summary["train_runs"] == 0
    assert summary["benchmark_runs"] == 0

    manifest_path = tmp_path / "out" / "run_manifest.json"
    assert summary["manifest_path"] == str(manifest_path)
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding = "utf-8"))
    assert payload["cancelled"] is True
    assert payload["rounds"] == []
    assert payload["winners"] == []
    assert payload["totals"] == {
        "train_runs": 0,
        "benchmark_runs": 0,
        "failed_runs": 0,
        "rounds": 0,
    }


def test_should_stop_is_not_called_after_completion(tmp_path):
    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=2,
        continue_runs=2,
        rounds=1,
    )
    events: list[str] = []

    def train(step, output_dir, progress):
        events.append("train")
        Path(output_dir).mkdir(parents = True, exist_ok = True)
        return {"output_dir": output_dir}

    def should_stop() -> bool:
        events.append("check")
        return False

    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        train=train,
        benchmark=_benchmark({}),
        should_stop=should_stop,
    )

    assert summary["cancelled"] is False
    assert summary["train_runs"] == 4
    # One check between rounds plus one before each of the four runs, and none
    # after the last run finished.
    assert events.count("check") == 5
    assert events[-1] == "train"


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


def test_timeout_seconds_defaults_to_zero_and_threads_into_step_specs(tmp_path):
    assert at.normalise(_spec(tmp_path, datasets=2))["timeout_seconds"] == 0

    spec = _spec(
        tmp_path,
        datasets=2,
        datasets_per_round=2,
        methods=["cmt"],
        runs_per_round=2,
        continue_runs=2,
        rounds=1,
        timeout_seconds=90,
    )
    calls: list = []
    summary = at.run(
        spec,
        output_dir=tmp_path / "out",
        train=_trainer(calls),
        benchmark=_benchmark({"run-01-cmt": 0.3, "run-02-cmt": 0.2}),
    )

    assert summary["cancelled"] is False
    assert len(calls) == 4
    assert all(step["spec"]["timeout_seconds"] == 90 for step in calls)
    assert all(step["spec"]["mode"] == "cpt" for step in calls)
    assert all("max_steps" in step["spec"] and "seed" in step["spec"] for step in calls)
    # The engine records the value in the manifest spec, but never enforces it.
    payload = json.loads(Path(summary["manifest_path"]).read_text(encoding = "utf-8"))
    assert payload["spec"]["timeout_seconds"] == 90


def test_timeout_seconds_reaches_the_plan_projection_and_rejects_negatives(tmp_path):
    report = at.plan(_spec(tmp_path, datasets=4, rounds=2, timeout_seconds=120))
    assert report["runnable"] is True
    assert report["projection"]["timeout_seconds"] == 120

    blocked = at.plan(_spec(tmp_path, datasets=3, datasets_per_round=4, timeout_seconds=120))
    assert blocked["runnable"] is False
    assert blocked["projection"]["timeout_seconds"] == 120

    assert at.plan(_spec(tmp_path, datasets=4))["projection"]["timeout_seconds"] == 0

    with pytest.raises(at.AutomatedTrainingError, match="timeout_seconds"):
        at.normalise(_spec(tmp_path, datasets=2, timeout_seconds=-1))