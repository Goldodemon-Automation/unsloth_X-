# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Automated training: a plan of rounds that picks its own next checkpoint.

The user supplies a pool of datasets and a starting checkpoint. Each round then:

1. samples a category-balanced subset of the pool,
2. spreads a user-chosen number of runs over those datasets and the selected
   methods (CMT, CPT and FFT, where FFT means full fine-tuning),
3. trains and benchmarks every run,
4. continue-trains the round winner a few more times and re-benchmarks it,
5. keeps the best checkpoint of the round.

Nothing here loads a weight, picks a GPU or touches the network. Planning is
pure arithmetic and seeded sampling, so the same spec always produces the same
rounds. Execution takes the trainer and benchmark as callables: the real
adapters are imported lazily, which keeps this module importable on a CPU-only
host and makes the whole loop testable with two small fakes.

A failed run is a data point, not a crash: it is recorded, skipped, and the
round still picks a winner. Only a round in which *every* candidate failed --
or an interrupt from the caller -- stops the controller.

Stopping early is a caller decision, not an error: ``run(should_stop = ...)`` is
polled before every run is scheduled and between rounds, and once it says stop
the controller keeps the rounds it finished, writes a manifest marked
``cancelled``, and returns normally.

A cancelled or interrupted run can be continued: ``run(resume_from = ...)`` reads
that run's ``run_manifest.json``, keeps its finished rounds, its winner and its
run counters, and only executes the rounds the earlier run never got to. Without
``resume_from`` nothing changes -- the controller runs every planned round from
scratch.

``timeout_seconds`` is carried by the spec and copied into every step spec so an
adapter can bound its own wait. The controller never enforces it -- adapters own
the waiting -- and ``0`` means "no engine-level timeout".
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

MIN_DATASETS = 2
MAX_DATASETS = 8
MIN_RUNS = 2
MAX_RUNS = 72
MAX_ROUNDS = 12
METHODS = ("cmt", "cpt", "fft")
CATEGORIES = ("code", "syntax", "reasoning", "general")

DEFAULT_DATASETS_PER_ROUND = 4
DEFAULT_RUNS_PER_ROUND = 8
DEFAULT_ROUNDS = 2
DEFAULT_CONTINUE_RUNS = 4
DEFAULT_SEED = 3407
DEFAULT_STEPS_PER_RUN = 100
DEFAULT_EVAL_FRACTION = 0.10
MIN_STEPS = 1
MAX_STEPS = 100_000
MAX_SEED = 2**31 - 1
DEFAULT_TIMEOUT_SECONDS = 0  # 0 means "the adapter decides", not "time out now"
MAX_TIMEOUT_SECONDS = 7 * 24 * 60 * 60  # a week; long enough to never be the reason

Progress = Callable[[dict[str, Any]], None]
Trainer = Callable[[dict[str, Any], str, Optional[Progress]], Mapping[str, Any]]
Benchmarker = Callable[[dict[str, Any], Mapping[str, Any], Optional[Progress]], Mapping[str, Any]]

_METHOD_LABELS = {"cmt": "CMT", "cpt": "CPT", "fft": "FFT"}


class AutomatedTrainingError(ValueError):
    """A spec, plan or round that cannot be run as written."""


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "quick-check",
        "label": "Quick check",
        "description": "One short round: does any method beat the base on a small sample?",
        "spec": {
            "datasets_per_round": 2,
            "runs_per_round": 3,
            "rounds": 1,
            "continue_runs": 2,
            "steps_per_run": 40,
            "benchmark": {"suites": [], "limit": 4, "batch_size": 1},
        },
    },
    {
        "id": "balanced",
        "label": "Balanced",
        "description": "A few rounds over every category, with a short continuation pass.",
        "spec": {
            "datasets_per_round": 4,
            "runs_per_round": 8,
            "rounds": 3,
            "continue_runs": 4,
            "steps_per_run": 100,
            "benchmark": {"suites": [], "limit": 8, "batch_size": 1},
        },
    },
    {
        "id": "deep",
        "label": "Deep",
        "description": "Long rounds over most of the pool, with many continuation runs.",
        "spec": {
            "datasets_per_round": 6,
            "runs_per_round": 24,
            "rounds": 6,
            "continue_runs": 12,
            "steps_per_run": 250,
            "benchmark": {"suites": [], "limit": 16, "batch_size": 2},
        },
    },
    {
        "id": "reasoning-heavy",
        "label": "Reasoning heavy",
        "description": "More runs and a reasoning benchmark suite, still balanced by category.",
        "spec": {
            "datasets_per_round": 4,
            "runs_per_round": 12,
            "rounds": 4,
            "continue_runs": 6,
            "steps_per_run": 150,
            "benchmark": {"suites": ["reasoning"], "limit": 12, "batch_size": 1},
        },
    },
)


def presets() -> list[dict[str, Any]]:
    """The named starting points, as fresh copies a caller may edit."""
    return copy.deepcopy(list(_PRESETS))


def _preset(preset_id: str) -> dict[str, Any]:
    for entry in _PRESETS:
        if entry["id"] == preset_id:
            return entry
    known = ", ".join(entry["id"] for entry in _PRESETS)
    raise AutomatedTrainingError(f"Unknown preset {preset_id!r}. Choose one of: {known}.")


# ---------------------------------------------------------------------------
# Request normalisation
# ---------------------------------------------------------------------------


def _integer(
    value: Any,
    field: str,
    *,
    minimum: int,
    maximum: int,
    default: int,
) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise AutomatedTrainingError(f"{field} must be a whole number.") from error
    if parsed < minimum or parsed > maximum:
        raise AutomatedTrainingError(f"{field} must be between {minimum} and {maximum}.")
    return parsed


def _fraction(value: Any, field: str) -> float:
    if value is None or value == "":
        return DEFAULT_EVAL_FRACTION
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AutomatedTrainingError(f"{field} must be a number.") from error
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 0.5:
        raise AutomatedTrainingError(f"{field} must be between 0 and 0.5.")
    return parsed


def _methods(value: Any) -> list[str]:
    if value is None:
        return list(METHODS)
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise AutomatedTrainingError("methods must be a list of 'cmt', 'cpt' or 'fft'.")
    chosen: list[str] = []
    for raw in value:
        method = str(raw or "").strip().lower()
        if method not in METHODS:
            raise AutomatedTrainingError(
                f"Unknown method {raw!r}; choose from {', '.join(METHODS)}."
            )
        if method not in chosen:
            chosen.append(method)
    return chosen


def _dataset_name(raw: Mapping[str, Any], path: str, repo_id: str, index: int) -> str:
    name = str(raw.get("name") or "").strip()
    if name:
        return name
    if path:
        return str(Path(path).stem or f"dataset-{index}")
    return repo_id.split("/")[-1] or f"dataset-{index}"


def _datasets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise AutomatedTrainingError("datasets must be a list of 2 to 8 dataset entries.")
    if len(value) < MIN_DATASETS:
        raise AutomatedTrainingError(f"Automated training needs at least {MIN_DATASETS} datasets.")
    if len(value) > MAX_DATASETS:
        raise AutomatedTrainingError(f"Automated training takes at most {MAX_DATASETS} datasets.")

    entries: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, Mapping):
            raise AutomatedTrainingError(f"Dataset {index} is not an object.")
        path = str(raw.get("path") or "").strip()
        repo_id = str(raw.get("repo_id") or "").strip()
        if path and repo_id:
            raise AutomatedTrainingError(
                f"Dataset {index} names both a path and a Hub id; give it one of the two."
            )
        if not path and not repo_id:
            raise AutomatedTrainingError(f"Dataset {index} needs a path or a Hub dataset id.")
        category = str(raw.get("category") or "general").strip().lower()
        if category not in CATEGORIES:
            raise AutomatedTrainingError(
                f"Dataset {index} category {category!r} is not one of {', '.join(CATEGORIES)}."
            )
        name = _dataset_name(raw, path, repo_id, index)
        if name in used_names:
            name = f"{name}-{index}"
        used_names.add(name)
        entries.append(
            {
                "path": path,
                "repo_id": repo_id or None,
                "name": name,
                "category": category,
                "eval_fraction": _fraction(raw.get("eval_fraction"), f"Dataset {index} eval fraction"),
            }
        )
    return entries


def _benchmark(value: Any) -> dict[str, Any]:
    raw = {} if value is None else value
    if not isinstance(raw, Mapping):
        raise AutomatedTrainingError("benchmark must be an object.")
    suites = raw.get("suites")
    if suites is None:
        suites = []
    if isinstance(suites, str) or not isinstance(suites, (list, tuple)):
        raise AutomatedTrainingError("benchmark suites must be a list of suite names.")
    return {
        "suites": [str(suite) for suite in suites],
        "limit": _integer(raw.get("limit"), "benchmark limit", minimum=1, maximum=1000, default=8),
        "batch_size": _integer(
            raw.get("batch_size"), "benchmark batch_size", minimum=1, maximum=512, default=1
        ),
    }


def _training_kwargs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AutomatedTrainingError("training_kwargs must be an object.")
    return dict(value)


def normalise(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate *spec* and fill every default the controller needs."""
    if not isinstance(spec, Mapping):
        raise AutomatedTrainingError("The automated-training spec must be an object.")

    raw = dict(spec)
    preset_id: Optional[str] = None
    requested_preset = raw.get("preset")
    if requested_preset not in (None, ""):
        preset_id = str(requested_preset).strip()
        entry = _preset(preset_id)
        # A preset is a base layer: every value the user wrote still wins.
        for key, value in entry["spec"].items():
            raw.setdefault(key, value)

    source = str(raw.get("source") or "").strip()
    if not source:
        raise AutomatedTrainingError("Automated training needs a base checkpoint path or Hub id.")

    return {
        "source": source,
        "datasets": _datasets(raw.get("datasets")),
        "methods": _methods(raw.get("methods")),
        "datasets_per_round": _integer(
            raw.get("datasets_per_round"),
            "datasets_per_round",
            minimum=MIN_DATASETS,
            maximum=MAX_DATASETS,
            default=DEFAULT_DATASETS_PER_ROUND,
        ),
        "runs_per_round": _integer(
            raw.get("runs_per_round"),
            "runs_per_round",
            minimum=MIN_RUNS,
            maximum=MAX_RUNS,
            default=DEFAULT_RUNS_PER_ROUND,
        ),
        "rounds": _integer(
            raw.get("rounds"), "rounds", minimum=1, maximum=MAX_ROUNDS, default=DEFAULT_ROUNDS
        ),
        "continue_runs": _integer(
            raw.get("continue_runs"),
            "continue_runs",
            minimum=MIN_RUNS,
            maximum=MAX_RUNS,
            default=DEFAULT_CONTINUE_RUNS,
        ),
        "seed": _integer(
            raw.get("seed"), "seed", minimum=0, maximum=MAX_SEED, default=DEFAULT_SEED
        ),
        "steps_per_run": _integer(
            raw.get("steps_per_run"),
            "steps_per_run",
            minimum=MIN_STEPS,
            maximum=MAX_STEPS,
            default=DEFAULT_STEPS_PER_RUN,
        ),
        # Threaded to the adapter through the step spec; never enforced here.
        "timeout_seconds": _integer(
            raw.get("timeout_seconds"),
            "timeout_seconds",
            minimum=0,
            maximum=MAX_TIMEOUT_SECONDS,
            default=DEFAULT_TIMEOUT_SECONDS,
        ),
        "benchmark": _benchmark(raw.get("benchmark")),
        "preset": preset_id,
        "training_kwargs": _training_kwargs(raw.get("training_kwargs")),
    }


# ---------------------------------------------------------------------------
# Planning (no weights, no I/O)
# ---------------------------------------------------------------------------


def _present_categories(pool: Sequence[Mapping[str, Any]]) -> list[str]:
    seen = {str(entry["category"]) for entry in pool}
    return [category for category in CATEGORIES if category in seen]


def _sample_datasets(
    pool: Sequence[Mapping[str, Any]],
    count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Sample *count* datasets, covering every present category when it fits.

    The guarantee is the one the plan promises: with at least two categories in
    the pool, a round big enough to name them all names at least one dataset
    from each. The rest of the slots are filled uniformly from what is left, so
    a large pool still gets variety instead of always the same first entries.
    """
    categories = _present_categories(pool)
    warnings: list[str] = []
    indices = list(range(len(pool)))
    picks: list[int] = []

    if len(categories) >= 2 and count >= len(categories):
        for category in categories:
            members = [index for index in indices if pool[index]["category"] == category]
            picks.append(rng.choice(members))
    elif len(categories) >= 2:
        warnings.append(
            f"This round samples {count} datasets but the pool has {len(categories)} categories, "
            "so not every category can be represented."
        )

    remaining = [index for index in indices if index not in picks]
    rng.shuffle(remaining)
    picks.extend(remaining[: max(0, count - len(picks))])
    rng.shuffle(picks)
    return [dict(pool[index]) for index in picks], warnings


def _method_sequence(methods: Sequence[str], count: int, rng: random.Random) -> list[str]:
    if not methods:
        return []
    offset = rng.randrange(len(methods))
    return [methods[(offset + index) % len(methods)] for index in range(count)]


def _run_datasets(sampled: Sequence[Mapping[str, Any]], run_index: int) -> list[dict[str, Any]]:
    """A deterministic CMT-sized subset: at least two, at most the whole sample."""
    span = len(sampled)
    if span <= MIN_DATASETS:
        return [dict(entry) for entry in sampled]
    sizes = (MIN_DATASETS, min(3, span), span)
    size = sizes[run_index % len(sizes)]
    start = run_index % span
    return [dict(sampled[(start + offset) % span]) for offset in range(size)]


def _run_spec(
    normalised: Mapping[str, Any],
    datasets: Sequence[Mapping[str, Any]],
    *,
    source: str,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    """A spec a CMT adapter can pass straight to ``cmt.run``."""
    spec = dict(normalised["training_kwargs"])
    spec.update(
        {
            "source": source,
            "mode": "cpt",
            "datasets": [dict(entry) for entry in datasets],
            "max_steps": steps,
            "seed": seed,
            "timeout_seconds": int(normalised.get("timeout_seconds") or 0),
        }
    )
    return spec


def _build_round(
    normalised: Mapping[str, Any], round_index: int, rng: random.Random
) -> dict[str, Any]:
    pool = normalised["datasets"]
    sampled, _ = _sample_datasets(pool, normalised["datasets_per_round"], rng)
    methods = _method_sequence(normalised["methods"], normalised["runs_per_round"], rng)
    runs: list[dict[str, Any]] = []
    for offset, method in enumerate(methods):
        index = offset + 1
        datasets = _run_datasets(sampled, offset)
        seed = rng.randrange(0, MAX_SEED + 1)
        runs.append(
            {
                "index": index,
                "method": method,
                "datasets": datasets,
                "steps": normalised["steps_per_run"],
                "spec": _run_spec(
                    normalised,
                    datasets,
                    source=normalised["source"],
                    steps=normalised["steps_per_run"],
                    seed=seed,
                ),
            }
        )
    return {
        "index": round_index,
        "datasets": [str(entry["name"]) for entry in sampled],
        "runs": runs,
        "continue_runs": normalised["continue_runs"],
    }


def _plan_from(normalised: Mapping[str, Any]) -> dict[str, Any]:
    pool = normalised["datasets"]
    warnings: list[str] = []

    if not normalised["methods"]:
        return _blocked("Select at least one training method.", normalised, warnings)
    if len(pool) < normalised["datasets_per_round"]:
        return _blocked(
            f"The pool holds {len(pool)} datasets but each round needs "
            f"{normalised['datasets_per_round']}.",
            normalised,
            warnings,
        )

    categories = _present_categories(pool)
    if len(categories) >= 2 and normalised["datasets_per_round"] < len(categories):
        warnings.append(
            f"Each round samples {normalised['datasets_per_round']} of {len(categories)} categories; "
            "raise datasets_per_round to keep every category in every round."
        )
    if len(categories) == 1:
        warnings.append(
            f"The pool only has the {categories[0]!r} category, so balancing has nothing to mix."
        )
    missing_adapters = [method for method in normalised["methods"] if method != "cmt"]
    if missing_adapters:
        names = ", ".join(_METHOD_LABELS.get(method, method.upper()) for method in missing_adapters)
        warnings.append(
            f"{names} runs need adapters supplied to run(); the default trainer only runs CMT."
        )
    warnings.append(
        "Hardware is not checked at plan time: every method runs wherever its adapter runs, "
        "so a plan is runnable on a CPU-only host even when it would be slow."
    )

    rounds = [
        _build_round(normalised, index, random.Random(f"{normalised['seed']}:{index}"))
        for index in range(1, normalised["rounds"] + 1)
    ]
    return {
        "runnable": True,
        "blocked_reason": None,
        "rounds": rounds,
        "warnings": warnings,
        "projection": {
            "total_train_runs": (normalised["runs_per_round"] + normalised["continue_runs"])
            * normalised["rounds"],
            "datasets_per_round": normalised["datasets_per_round"],
            "rounds": normalised["rounds"],
            "timeout_seconds": normalised["timeout_seconds"],
        },
        "presets": presets(),
    }


def _blocked(
    reason: str, normalised: Mapping[str, Any], warnings: list[str]
) -> dict[str, Any]:
    return {
        "runnable": False,
        "blocked_reason": reason,
        "rounds": [],
        "warnings": list(warnings),
        "projection": {
            "total_train_runs": 0,
            "datasets_per_round": int(normalised.get("datasets_per_round") or 0),
            "rounds": 0,
            "timeout_seconds": int(normalised.get("timeout_seconds") or 0),
        },
        "presets": presets(),
    }


def plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Lay out every round, dataset and run without loading a weight."""
    try:
        normalised = normalise(spec)
    except AutomatedTrainingError as error:
        return _blocked(str(error), {}, [])
    return _plan_from(normalised)


# ---------------------------------------------------------------------------
# Run adapters
# ---------------------------------------------------------------------------


def _default_train(
    step: Mapping[str, Any], output_dir: str, progress: Optional[Progress]
) -> Mapping[str, Any]:
    method = str(step.get("method") or "").strip().lower()
    if method != "cmt":
        raise AutomatedTrainingError(
            f"No training adapter was supplied for {_METHOD_LABELS.get(method, method.upper())}. "
            "The default trainer only runs CMT; pass train=... for CPT or FFT."
        )
    from core.labs import cmt

    return cmt.run(step["spec"], output_dir=output_dir, progress=progress)


def _default_benchmark(
    step: Mapping[str, Any], artifact: Mapping[str, Any], progress: Optional[Progress]
) -> Mapping[str, Any]:
    return {"score": None, "record_id": None}


def _emit(progress: Optional[Progress], stage: str, message: str, **extra: Any) -> None:
    if progress is None:
        return
    progress({"stage": stage, "message": message, **extra})


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _rank_key(candidate: Mapping[str, Any]) -> tuple[bool, float, int]:
    score = candidate.get("score")
    is_missing = score is None
    return (is_missing, 0.0 if is_missing else -float(score), int(candidate["order"]))


def _better(candidate: Mapping[str, Any], current: Optional[Mapping[str, Any]]) -> bool:
    if current is None:
        return True
    return _rank_key(candidate) < _rank_key(current)


def _score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise AutomatedTrainingError("A benchmark score must be a number or null.") from error
    if not math.isfinite(parsed):
        raise AutomatedTrainingError("A benchmark score must be finite.")
    return parsed


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _candidate_view(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": candidate["kind"],
        "label": candidate["label"],
        "index": candidate["index"],
        "method": candidate["method"],
        "datasets": [str(entry["name"]) for entry in candidate["datasets"]],
        "steps": candidate["steps"],
        "output_dir": candidate["output_dir"],
        "score": candidate["score"],
        "record_id": candidate["record_id"],
        "round": candidate["round"],
    }


def _best_view(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": candidate.get("output_dir"),
        "score": candidate.get("score"),
        "method": candidate.get("method"),
        "round": candidate.get("round"),
    }


def _failure_view(failure: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": failure["kind"],
        "label": failure["label"],
        "index": failure["index"],
        "method": failure["method"],
        "datasets": [str(entry["name"]) for entry in failure["datasets"]],
        "stage": failure["stage"],
        "error": failure["error"],
        "round": failure["round"],
    }


# ---------------------------------------------------------------------------
# Resuming a previous run
# ---------------------------------------------------------------------------

_RESUME_MANIFEST_NAME = "run_manifest.json"


def _count(value: Any) -> int:
    """A counter read from JSON, clamped to something a total can hold."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _previous_round_records(
    payload: Mapping[str, Any], warnings: list[str]
) -> list[dict[str, Any]]:
    """The finished round records of a previous manifest, in their own order."""
    raw = payload.get("rounds")
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append(
            "The previous run manifest has a 'rounds' block that is not a list, so no round "
            "was reused; every round runs again."
        )
        return []

    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, entry in enumerate(raw, start = 1):
        if not isinstance(entry, Mapping):
            warnings.append(f"Ignoring previous round record {position}: it is not an object.")
            continue
        try:
            index = int(entry.get("index"))
        except (TypeError, ValueError):
            warnings.append(f"Ignoring previous round record {position}: it has no usable index.")
            continue
        if index in seen:
            warnings.append(
                f"Ignoring a duplicate record for round {index} in the previous manifest."
            )
            continue
        seen.add(index)
        records.append(copy.deepcopy(dict(entry)))
    return records


def _previous_winners(
    payload: Mapping[str, Any], warnings: list[str]
) -> list[dict[str, Any]]:
    raw = payload.get("winners")
    if raw is None:
        return []
    if not isinstance(raw, list):
        warnings.append(
            "The previous run manifest has a 'winners' block that is not a list, so its winners "
            "were not carried forward."
        )
        return []
    return [copy.deepcopy(dict(entry)) for entry in raw if isinstance(entry, Mapping)]


def _previous_best(payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The best checkpoint a previous manifest names, wherever it recorded it."""
    for value in (payload.get("best"), _nested(payload.get("artifacts"), "best")):
        if isinstance(value, Mapping) and value:
            return copy.deepcopy(dict(value))
    return None


def _nested(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _comparison_candidate(best: Mapping[str, Any]) -> dict[str, Any]:
    """A previous best reshaped into what ``_rank_key`` reads.

    ``order`` is -1, lower than the 0 of the first candidate of any resumed
    round, so a new candidate has to *beat* the previous score: an equal score
    keeps the checkpoint the earlier run already produced.
    """
    candidate = dict(best)
    try:
        score = candidate.get("score")
        candidate["score"] = None if score is None else float(score)
    except (TypeError, ValueError):
        candidate["score"] = None
    candidate.setdefault("kind", "previous")
    candidate.setdefault("label", str(candidate.get("output_dir") or "previous run"))
    candidate.setdefault("method", None)
    candidate.setdefault("index", 0)
    candidate.setdefault("datasets", [])
    candidate.setdefault("steps", 0)
    candidate.setdefault("record_id", None)
    candidate.setdefault("round", None)
    candidate.setdefault("output_dir", "")
    candidate["order"] = -1
    return candidate


def _load_resume(
    resume_from: str | Path, warnings: list[str]
) -> Optional[dict[str, Any]]:
    """Read the manifest of the run being continued.

    A missing, unreadable or malformed manifest is not an error: the run simply
    starts from scratch, and the reason travels back in ``warnings`` so the
    caller can see why nothing was reused.
    """
    directory = Path(resume_from).expanduser()
    path = directory
    if path.name != _RESUME_MANIFEST_NAME:
        path = directory / _RESUME_MANIFEST_NAME
    if not path.is_file():
        warnings.append(
            f"Resume source {str(path)!r} has no run manifest, so every round runs from scratch."
        )
        return None
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError) as error:
        warnings.append(
            f"Resume source {str(path)!r} could not be read as JSON "
            f"({type(error).__name__}: {error}); every round runs from scratch."
        )
        return None
    if not isinstance(payload, Mapping):
        warnings.append(
            f"Resume source {str(path)!r} is not a JSON object, so every round runs from scratch."
        )
        return None

    rounds = _previous_round_records(payload, warnings)
    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        if totals is not None:
            warnings.append(
                "The previous run manifest has no usable 'totals' block, so its run counters "
                "start at zero."
            )
        totals = {}
    return {
        "path": str(path),
        "rounds": rounds,
        "winners": _previous_winners(payload, warnings),
        "best": _previous_best(payload),
        "totals": {
            "train_runs": _count(totals.get("train_runs")),
            "benchmark_runs": _count(totals.get("benchmark_runs")),
            "failed_runs": _count(totals.get("failed_runs")),
        },
        "finished": {int(record["index"]) for record in rounds},
    }


def _is_interrupt(error: BaseException) -> bool:
    return isinstance(error, (KeyboardInterrupt, SystemExit))


def _stop_requested(should_stop: Optional[Callable[[], bool]]) -> bool:
    """Whether the caller asked to stop. No callback means "keep going"."""
    if should_stop is None:
        return False
    return bool(should_stop())


def run(
    spec: Mapping[str, Any],
    *,
    output_dir: str | Path,
    progress: Optional[Progress] = None,
    train: Optional[Trainer] = None,
    benchmark: Optional[Benchmarker] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    resume_from: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Execute the planned rounds and return the best checkpoint of each.

    ``should_stop`` is polled before every run is scheduled and between rounds
    (and before every continuation run). Once it returns ``True`` nothing new
    starts: the rounds that finished are kept, the manifest is written with
    ``"cancelled": True``, a ``cancelled`` progress stage is emitted, and the
    summary is returned normally. A cancellation is a result, not an error, so
    it never becomes an exception -- while interrupts raised by an adapter (or
    by the callback itself) still propagate untouched.

    ``resume_from`` names a previous output directory. Its ``run_manifest.json``
    supplies the rounds that already finished, the best checkpoint so far and
    the run counters; those rounds are skipped and everything else is appended
    to them, so a run can be continued instead of repeated. A resumed round can
    only take the best checkpoint over when it scores better than the previous
    best. A missing or malformed manifest is ignored (with a warning) and every
    round runs from scratch, which is also what happens when ``resume_from`` is
    not given at all.
    """
    normalised = normalise(spec)
    planned = _plan_from(normalised)
    if not planned["runnable"]:
        raise AutomatedTrainingError(planned.get("blocked_reason") or "This plan cannot run.")

    if train is None:
        missing = [method for method in normalised["methods"] if method != "cmt"]
        if missing:
            names = ", ".join(_METHOD_LABELS.get(method, method.upper()) for method in missing)
            raise AutomatedTrainingError(
                f"No training adapter was supplied for {names}. Pass train=... to run those methods."
            )
        trainer: Trainer = _default_train
    else:
        trainer = train

    using_default_benchmark = benchmark is None
    scorer: Benchmarker = _default_benchmark if benchmark is None else benchmark

    out = Path(output_dir).expanduser()
    out.mkdir(parents = True, exist_ok = True)

    warnings: list[str] = []
    if using_default_benchmark:
        warnings.append(
            "No benchmark adapter was supplied, so every candidate scored null; the round winner "
            "falls back to the earliest successful run."
        )

    previous = _load_resume(resume_from, warnings) if resume_from is not None else None
    finished_rounds: set[int] = set() if previous is None else set(previous["finished"])
    previous_best = None if previous is None else previous["best"]
    # ``None`` until a previous best exists: without one there is nothing to
    # compare a round winner against, so the last winner stays the answer.
    resumed_best: Optional[dict[str, Any]] = (
        None if previous_best is None else _comparison_candidate(previous_best)
    )

    train_runs = 0 if previous is None else previous["totals"]["train_runs"]
    benchmark_runs = 0 if previous is None else previous["totals"]["benchmark_runs"]
    failed_runs = 0 if previous is None else previous["totals"]["failed_runs"]
    round_records: list[dict[str, Any]] = [] if previous is None else list(previous["rounds"])
    winners: list[dict[str, Any]] = [] if previous is None else list(previous["winners"])
    cancelled = False
    total_rounds = len(planned["rounds"])

    if previous is not None:
        skipped = sum(
            1 for round_plan in planned["rounds"] if int(round_plan["index"]) in finished_rounds
        )
        _emit(
            progress,
            "resume",
            f"Resuming from {previous['path']}: skipping {skipped} of {total_rounds} planned "
            f"round(s) already recorded, keeping {len(round_records)} finished round(s).",
            source = previous["path"],
            skipped = skipped,
            total = total_rounds,
            rounds = len(round_records),
        )

    for round_plan in planned["rounds"]:
        round_index = int(round_plan["index"])
        if round_index in finished_rounds:
            continue
        if _stop_requested(should_stop):
            cancelled = True
            break

        round_dir = out / f"round-{round_index:02d}"
        round_dir.mkdir(parents = True, exist_ok = True)
        _emit(
            progress,
            "round-start",
            f"Round {round_index}/{total_rounds}: {len(round_plan['datasets'])} datasets, "
            f"{len(round_plan['runs'])} runs.",
            round = round_index,
            total = total_rounds,
        )

        candidates: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        order = 0

        for entry in round_plan["runs"]:
            if _stop_requested(should_stop):
                cancelled = True
                break
            label = f"run-{int(entry['index']):02d}-{entry['method']}"
            step = {
                **entry,
                "label": label,
                "kind": "run",
                "round": round_index,
            }
            step_dir = round_dir / label
            step_dir.mkdir(parents = True, exist_ok = True)
            _emit(
                progress,
                "training",
                f"Round {round_index}: training {label} on "
                f"{', '.join(str(item['name']) for item in step['datasets'])}.",
                round = round_index,
                label = label,
                method = step["method"],
            )
            trained = _train_once(trainer, step, step_dir, progress, failures)
            if trained is None:
                continue
            train_runs += 1
            _emit(
                progress,
                "benchmark",
                f"Round {round_index}: benchmarking {label}.",
                round = round_index,
                label = label,
                method = step["method"],
            )
            scored = _benchmark_once(scorer, step, trained, progress, failures)
            if scored is None:
                continue
            benchmark_runs += 1
            candidates.append(
                _candidate(step, trained, scored, order = order, kind = "run", label = label)
            )
            order += 1

        if not candidates:
            failed_runs += len(failures)
            if cancelled:
                # Stopped before this round produced anything: nothing to rank and
                # nothing to report as a failure, so the round is simply not recorded.
                break
            raise AutomatedTrainingError(
                f"Every candidate in round {round_index} failed; the run has no winner."
            )

        winner = candidates[0]
        for candidate in candidates[1:]:
            if _better(candidate, winner):
                winner = candidate
        _emit(
            progress,
            "pick",
            f"Round {round_index}: best of {len(candidates)} candidates is {winner['label']} "
            f"({_format_score(winner['score'])}).",
            round = round_index,
            label = winner["label"],
            score = winner["score"],
        )

        base_winner = winner
        continued: list[dict[str, Any]] = []
        for continue_index in range(1, normalised["continue_runs"] + 1):
            if _stop_requested(should_stop):
                # A cancellation skips the continuation runs; the round still keeps
                # the winner it already picked.
                cancelled = True
                break
            label = f"continue-{continue_index:02d}"
            step = _continue_step(
                normalised,
                base_winner,
                round_index = round_index,
                label = label,
                index = continue_index,
            )
            step_dir = round_dir / label
            step_dir.mkdir(parents = True, exist_ok = True)
            _emit(
                progress,
                "continue",
                f"Round {round_index}: continue-training {label} from {base_winner['label']}.",
                round = round_index,
                label = label,
                method = step["method"],
            )
            trained = _train_once(trainer, step, step_dir, progress, failures)
            if trained is None:
                continue
            train_runs += 1
            _emit(
                progress,
                "benchmark",
                f"Round {round_index}: re-benchmarking {label}.",
                round = round_index,
                label = label,
                method = step["method"],
            )
            scored = _benchmark_once(scorer, step, trained, progress, failures)
            if scored is None:
                continue
            benchmark_runs += 1
            candidate = _candidate(
                step, trained, scored, order = order, kind = "continue", label = label
            )
            order += 1
            continued.append(candidate)
            if _better(candidate, winner):
                winner = candidate

        failed_runs += len(failures)
        winner_view = _candidate_view(winner)
        round_records.append(
            {
                "index": round_index,
                "datasets": list(round_plan["datasets"]),
                "runs": [
                    _candidate_view(candidate) for candidate in candidates
                ],
                "continue": [_candidate_view(candidate) for candidate in continued],
                "failures": [_failure_view(failure) for failure in failures],
                "continue_runs": normalised["continue_runs"],
                "winner": winner_view,
            }
        )
        winners.append({"round": round_index, **winner_view})
        if resumed_best is not None and _better(winner, resumed_best):
            # Only a round that beats the previous best takes the lead over.
            resumed_best = winner
        _emit(
            progress,
            "round-done",
            f"Round {round_index} won by {winner['label']} ({_format_score(winner['score'])}).",
            round = round_index,
            label = winner["label"],
            score = winner["score"],
        )
        if cancelled:
            break

    best_summary: Optional[dict[str, Any]] = None
    if resumed_best is not None:
        best_summary = _best_view(resumed_best)
    elif winners:
        best_summary = _best_view(winners[-1])
    totals = {
        "train_runs": train_runs,
        "benchmark_runs": benchmark_runs,
        "failed_runs": failed_runs,
        "rounds": len(round_records),
    }
    manifest_path = _write_manifest(
        out,
        normalised = normalised,
        rounds = round_records,
        winners = winners,
        best = best_summary or {},
        totals = totals,
        warnings = warnings,
        cancelled = cancelled,
    )

    if cancelled:
        _emit(
            progress,
            "cancelled",
            f"Automated training cancelled after {len(round_records)} of {total_rounds} rounds; "
            "no further runs were scheduled.",
            rounds = len(round_records),
            total = total_rounds,
            best = best_summary,
        )
    else:
        finished_best = best_summary or {}
        _emit(
            progress,
            "done",
            f"Automated training finished; the best checkpoint is "
            f"{finished_best.get('output_dir')} ({_format_score(finished_best.get('score'))}).",
            best = best_summary,
        )
    return {
        "output_dir": str(out),
        "rounds": round_records,
        "best": best_summary,
        "train_runs": train_runs,
        "benchmark_runs": benchmark_runs,
        "failed_runs": failed_runs,
        "cancelled": cancelled,
        "warnings": warnings,
        "manifest_path": str(manifest_path),
    }


def _train_once(
    trainer: Trainer,
    step: Mapping[str, Any],
    step_dir: Path,
    progress: Optional[Progress],
    failures: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    try:
        result = trainer(dict(step), str(step_dir), progress)
        if not isinstance(result, Mapping):
            raise AutomatedTrainingError("The training adapter did not return an object.")
        produced = str(result.get("output_dir") or "").strip()
        if not produced:
            raise AutomatedTrainingError("The training adapter returned no output_dir.")
    except Exception as error:  # noqa: BLE001 - a failed run is recorded, not raised
        if _is_interrupt(error):
            raise
        failures.append(_failure(step, "train", error))
        return None
    return {"output_dir": produced, "result": dict(result)}


def _benchmark_once(
    scorer: Benchmarker,
    step: Mapping[str, Any],
    trained: Mapping[str, Any],
    progress: Optional[Progress],
    failures: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    try:
        artifact = dict(trained["result"])
        artifact.setdefault("output_dir", trained["output_dir"])
        result = scorer(dict(step), artifact, progress)
        if not isinstance(result, Mapping):
            raise AutomatedTrainingError("The benchmark adapter did not return an object.")
        score = _score(result.get("score"))
        record_id = result.get("record_id")
        record_id = None if record_id is None else str(record_id)
    except Exception as error:  # noqa: BLE001 - a failed benchmark is skipped
        if _is_interrupt(error):
            raise
        failures.append(_failure(step, "benchmark", error))
        return None
    return {"score": score, "record_id": record_id}


def _failure(step: Mapping[str, Any], stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "kind": step["kind"],
        "label": step["label"],
        "index": int(step["index"]),
        "method": str(step["method"]),
        "datasets": [dict(entry) for entry in step["datasets"]],
        "stage": stage,
        "error": f"{type(error).__name__}: {error}",
        "round": int(step["round"]),
    }


def _candidate(
    step: Mapping[str, Any],
    trained: Mapping[str, Any],
    scored: Mapping[str, Any],
    *,
    order: int,
    kind: str,
    label: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "index": int(step["index"]),
        "method": str(step["method"]),
        "datasets": [dict(entry) for entry in step["datasets"]],
        "steps": int(step["steps"]),
        "output_dir": str(trained["output_dir"]),
        "score": scored["score"],
        "record_id": scored["record_id"],
        "round": int(step["round"]),
        "order": order,
    }


def _continue_step(
    normalised: Mapping[str, Any],
    winner: Mapping[str, Any],
    *,
    round_index: int,
    label: str,
    index: int,
) -> dict[str, Any]:
    datasets = [dict(entry) for entry in winner["datasets"]]
    return {
        "index": index,
        "method": winner["method"],
        "datasets": datasets,
        "steps": normalised["steps_per_run"],
        "label": label,
        "kind": "continue",
        "round": round_index,
        "spec": _run_spec(
            normalised,
            datasets,
            source=str(winner["output_dir"]),
            steps=normalised["steps_per_run"],
            seed = (normalised["seed"] + round_index * 101 + index) % (MAX_SEED + 1),
        ),
    }


def _format_score(score: Any) -> str:
    return "no score" if score is None else f"score {float(score):.4f}"


def _write_manifest(
    output_dir: Path,
    *,
    normalised: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
    winners: Sequence[Mapping[str, Any]],
    best: Mapping[str, Any],
    totals: Mapping[str, Any],
    warnings: Sequence[str],
    cancelled: bool = False,
) -> Path:
    from core.advanced import run_manifest

    manifest = run_manifest.build_manifest(
        "automated_training",
        config = {
            "spec": dict(normalised),
            "methods": list(normalised["methods"]),
            "steps_per_run": normalised["steps_per_run"],
            "benchmark": dict(normalised["benchmark"]),
        },
        seeds = {"seed": normalised["seed"]},
        inputs = {"source": {"source": normalised["source"]}},
        artifacts = {
            "output_dir": str(output_dir),
            "best": dict(best),
            "winners": list(winners),
        },
    )
    manifest.update(
        {
            "spec": dict(normalised),
            "rounds": list(rounds),
            "winners": list(winners),
            "totals": dict(totals),
            "warnings": list(warnings),
            "cancelled": bool(cancelled),
        }
    )
    return run_manifest.write_manifest(output_dir, manifest)


__all__ = [
    "AutomatedTrainingError",
    "CATEGORIES",
    "MAX_DATASETS",
    "MAX_ROUNDS",
    "MAX_RUNS",
    "METHODS",
    "MIN_DATASETS",
    "MIN_RUNS",
    "normalise",
    "plan",
    "presets",
    "run",
]