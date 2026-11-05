# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Labs API: the experimental-features switch, the Artifacts studio and the
Pipelines planner.

Every route past the switch asks :func:`core.labs.require` first, so turning an
experiment off closes its endpoint as well as its page. That is the whole point
of a server-side flag: a hidden button is not a gate.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from auth import policy
from auth.authentication import get_current_subject
from core import labs
from core.advanced import repair
from core.labs import (
    arch_swap,
    artifacts,
    autoscale,
    benchmarks,
    cmt,
    diff,
    housekeeping,
    lineage,
    pipelines,
    surgery,
)

# Router-level, like routes.advanced: a hidden page is not a gate, and neither is a
# GET that never asked who is calling. Every route below inherits this, so an
# unauthenticated request cannot read artifact paths, run manifests or the disk
# inventory even when the feature is switched on.
router = APIRouter(dependencies = [Depends(get_current_subject)])


async def _require_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


class LabsFeature(BaseModel):
    id: str
    label: str
    description: str
    status: str
    studio: bool
    enabled: bool


class LabsState(BaseModel):
    enabled: bool
    features: list[LabsFeature]
    note: str


class LabsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = None
    features: Optional[dict[str, bool]] = None


class PipelinePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]


class PipelineSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]


class RepairPresetRequest(BaseModel):
    """The shrink repair preset: a shrunken checkpoint and its base."""

    model_config = ConfigDict(extra="forbid")

    source: str
    base: str
    suites: list[str] = Field(default_factory = lambda: ["mmlu"])
    tolerance: float = 0.02
    steps: int = 120
    batch_size: int = 1
    sequence_length: int = 2048
    learning_rate: float = 5e-5


class BenchmarkRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: str = Field(..., min_length = 1)
    baseline: str = Field(..., min_length = 1)
    suites: list[str] = Field(..., min_length = 1)
    limit: Optional[int] = Field(None, gt = 0, le = 100_000)
    batch_size: int = Field(8, gt = 0, le = 1024)
    timeout_seconds: int = Field(3600, gt = 0, le = 86_400)


class HousekeepingDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(..., min_length = 1, max_length = 200)
    confirm: bool = False


class JanitorRulesUpdate(BaseModel):
    """Any subset of the janitor rules; the rest is filled from the saved file."""

    model_config = ConfigDict(extra="forbid")

    enabled: Optional[bool] = None
    keep_newest: Optional[int] = Field(None, ge = 0, le = 100)
    min_age_days: Optional[int] = Field(None, ge = 0, le = 3650)
    roots: Optional[list[str]] = None


class CmtPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]


class CmtRunRequest(CmtPlanRequest):
    output_dir: Optional[str] = None


class AutoScaleRequest(CmtPlanRequest):
    """The model, the datasets it will train on, and any size override."""

    output_dir: Optional[str] = None


class SurgeryRequest(BaseModel):
    """The checkpoint and the prune/upcycle knobs, as the panel sends them."""

    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]
    output_dir: Optional[str] = None


class ArchSwapRequest(BaseModel):
    """A source checkpoint, a target checkpoint, and the surplus-block policy."""

    model_config = ConfigDict(extra="forbid")

    spec: dict[str, Any]
    output_dir: Optional[str] = None


_LABS_NOTE = (
    "Labs experiments are unfinished. They can change shape or break between "
    "updates, and an artifact they write is not covered by the stable contract."
)


def _state() -> LabsState:
    state = labs.describe_state()
    return LabsState(
        enabled = bool(state["enabled"]),
        features = [LabsFeature(**feature) for feature in state["features"]],
        note = _LABS_NOTE,
    )


def _gate(feature: str) -> None:
    try:
        labs.require(feature)
    except labs.LabsDisabled as error:
        raise HTTPException(status_code = 403, detail = str(error)) from error


def _translate(error: Exception) -> HTTPException:
    if isinstance(error, PermissionError):
        return HTTPException(status_code = 403, detail = str(error))
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code = 404, detail = str(error))
    return HTTPException(status_code = 400, detail = str(error))


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


@router.get("", response_model = LabsState)
def read_labs() -> LabsState:
    """The switch and every registered experiment. Readable by any signed-in account."""
    return _state()


@router.put("", response_model = LabsState, dependencies = [Depends(_require_owner)])
def update_labs(payload: LabsUpdate) -> LabsState:
    """Owner-only: enable Labs, or switch one experiment off."""
    if payload.enabled is None and payload.features is None:
        raise HTTPException(status_code = 400, detail = "Nothing to update.")
    try:
        labs.set_state(enabled = payload.enabled, features = payload.features)
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code = 500, detail = str(error)) from error
    return _state()


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@router.get("/artifacts")
def list_artifacts(limit: int = Query(default = 50, ge = 1, le = 200)) -> dict[str, Any]:
    """Checkpoints this Studio wrote, newest first, with their legal next steps."""
    _gate("artifacts")
    try:
        records = artifacts.scan(limit = limit)
        root = str(artifacts.root())
    except Exception as error:  # noqa: BLE001 - reported as an HTTP failure with its reason
        raise _translate(error) from error
    return {"artifacts": records, "root": root}


@router.get("/artifacts/detail")
def artifact_detail(path: str) -> dict[str, Any]:
    """One artifact. The raw manifest rides along only while that experiment is on."""
    _gate("artifacts")
    try:
        record = artifacts.find(path)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if labs.feature_enabled("manifest-viewer") and record.get("manifest_path"):
        try:
            record["raw_manifest"] = artifacts.read_manifest(path)["manifest"]
        except Exception as error:  # noqa: BLE001 - the record is still useful without it
            record["manifest_error"] = str(error)
    return record


@router.get("/artifacts/diff")
def artifact_diff(left: str, right: str) -> dict[str, Any]:
    """Field-level differences between two checkpoints this Studio wrote."""
    _gate("artifacts")
    try:
        return diff.compare(left, right)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.get("/artifacts/manifest")
def artifact_manifest(path: str) -> dict[str, Any]:
    """The raw run manifest beside one artifact."""
    _gate("manifest-viewer")
    try:
        return artifacts.read_manifest(path)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


@router.get("/lineage")
def run_lineage() -> dict[str, Any]:
    """The manifest graph under the Studio outputs folder, roots first."""
    _gate("lineage")
    try:
        return lineage.scan()
    except Exception as error:  # noqa: BLE001 - reported as an HTTP failure with its reason
        raise _translate(error) from error


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


@router.post("/pipelines/plan", dependencies = [Depends(_require_owner)])
def plan_pipeline(request: PipelinePlanRequest) -> dict[str, Any]:
    """Validate and price a chain from configs only. Starts nothing."""
    _gate("pipelines")
    try:
        return pipelines.preflight(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.get("/pipelines/presets")
def pipeline_presets() -> dict[str, Any]:
    """Preset chains the planner can build without the UI hard-coding them."""
    _gate("pipelines")
    return {"presets": [repair.describe_preset()]}


@router.post("/pipelines/preset", dependencies = [Depends(_require_owner)])
def plan_preset(request: RepairPresetRequest) -> dict[str, Any]:
    """Build the shrink repair chain and price it. Starts nothing."""
    _gate("pipelines")
    try:
        spec = pipelines.build_preset(
            repair.PRESET_ID,
            request.source,
            request.base,
            suites = request.suites,
            tolerance = request.tolerance,
            steps = request.steps,
            batch_size = request.batch_size,
            sequence_length = request.sequence_length,
            learning_rate = request.learning_rate,
        )
        return {"spec": spec, "plan": pipelines.preflight(spec)}
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.get("/pipelines")
def list_pipelines() -> dict[str, Any]:
    """Saved pipeline plans, newest first."""
    _gate("pipelines")
    try:
        return {"pipelines": pipelines.list_pipelines(), "root": str(pipelines.root())}
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.post("/pipelines", dependencies = [Depends(_require_owner)])
def save_pipeline(request: PipelineSaveRequest) -> dict[str, Any]:
    """Validate a chain, then save it. A blocked chain still saves, with the block recorded."""
    _gate("pipelines")
    try:
        return {"pipeline": pipelines.save_pipeline(request.spec)}
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.delete("/pipelines/{pipeline_id}", dependencies = [Depends(_require_owner)])
def delete_pipeline(pipeline_id: str) -> dict[str, Any]:
    """Delete one saved pipeline plan. The checkpoints it referenced are untouched."""
    _gate("pipelines")
    try:
        deleted = pipelines.delete_pipeline(pipeline_id)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if not deleted:
        raise HTTPException(status_code = 404, detail = f"No pipeline {pipeline_id!r}.")
    return {"deleted": pipeline_id}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@router.get("/benchmarks")
def list_benchmarks(
    limit: int = Query(default = 50, ge = 1, le = 200),
    artifact: Optional[str] = None,
) -> dict[str, Any]:
    """Recorded eval runs, newest first, optionally for one checkpoint."""
    _gate("benchmarks")
    try:
        records = benchmarks.annotate(
            benchmarks.list_records(limit = limit, artifact = artifact)
        )
        pins = benchmarks.list_pins()
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    return {"records": records, "pins": pins}


@router.post("/benchmarks/run", dependencies = [Depends(_require_owner)])
def run_benchmark(request: BenchmarkRunRequest) -> dict[str, Any]:
    """Score an artifact against a baseline and keep the report."""
    _gate("benchmarks")
    from core.advanced import eval_harness, eval_runner, run_manifest

    try:
        artifact = artifacts.find(request.artifact)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    baseline = request.baseline.strip()
    if not baseline:
        raise HTTPException(status_code = 400, detail = "A baseline model is required.")
    try:
        plan = eval_harness.build_eval_plan(
            request.suites,
            baseline_model = baseline,
            candidate_model = artifact["path"],
            limit = request.limit,
            batch_size = request.batch_size,
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    availability = eval_harness.harness_available()
    if not availability["available"]:
        raise HTTPException(status_code = 409, detail = availability["install_hint"])

    run_dir = benchmarks.create_run_dir()
    started = time.monotonic()

    def work() -> dict[str, Any]:
        result = eval_runner.run_plan(
            plan,
            output_dir = str(run_dir),
            timeout = request.timeout_seconds,
        )
        record = benchmarks.save_record(
            job_id = run_dir.name,
            artifact = artifact["path"],
            baseline = baseline,
            report = result,
            suites = request.suites,
            limit = request.limit,
            batch_size = request.batch_size,
            duration_seconds = round(time.monotonic() - started, 1),
            output_dir = str(run_dir),
        )
        manifest = run_manifest.build_manifest(
            "benchmark",
            config = {
                "suites": list(request.suites),
                "limit": request.limit,
                "batch_size": request.batch_size,
            },
            inputs = {
                "baseline": run_manifest.checkpoint_digest(baseline),
                "candidate": run_manifest.checkpoint_digest(artifact["path"]),
            },
            artifacts = {
                "output_dir": str(run_dir),
                "record_id": record["id"],
                "verdict": record["verdict"],
                "aggregate_delta": record["comparison"].get("aggregate_delta"),
            },
        )
        manifest_path = run_manifest.write_manifest(run_dir, manifest)
        result["record"] = record
        result["run_manifest"] = str(manifest_path)
        return result

    from routes.advanced import _start_job

    return _start_job(
        "eval",
        work,
        output_dir = str(run_dir),
        sources = [baseline, artifact["path"]],
    )


@router.post("/benchmarks/{record_id}/pin", dependencies = [Depends(_require_owner)])
def pin_benchmark(record_id: str) -> dict[str, Any]:
    """Make one record the baseline later runs are measured against."""
    _gate("benchmarks")
    try:
        entry = benchmarks.pin_record(record_id)
        return {"pin": entry, "pins": benchmarks.list_pins()}
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.delete("/benchmarks/{record_id}/pin", dependencies = [Depends(_require_owner)])
def unpin_benchmark(record_id: str) -> dict[str, Any]:
    """Drop the pin this record holds. The record itself is kept."""
    _gate("benchmarks")
    try:
        removed = benchmarks.unpin(record_id)
        pins = benchmarks.list_pins()
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if not removed:
        raise HTTPException(status_code = 404, detail = f"No pin on record {record_id!r}.")
    return {"unpinned": record_id, "pins": pins}


@router.delete("/benchmarks/{record_id}", dependencies = [Depends(_require_owner)])
def delete_benchmark(record_id: str) -> dict[str, Any]:
    """Forget one benchmark record. The checkpoint itself is untouched."""
    _gate("benchmarks")
    try:
        deleted = benchmarks.delete_record(record_id)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if not deleted:
        raise HTTPException(status_code = 404, detail = f"No benchmark record {record_id!r}.")
    return {"deleted": record_id}


# ---------------------------------------------------------------------------
# CMT (experimental capacity matching)
# ---------------------------------------------------------------------------


def _cmt_output_dir(source: str, prefix: str = "cmt") -> Path:
    from utils.paths.storage_roots import outputs_root

    name = Path(source.rstrip("/\\")).name or "model"
    safe = "".join(character if character.isalnum() or character in "-_." else "-" for character in name)
    return outputs_root() / f"{prefix}-{safe[:80]}-{uuid.uuid4().hex[:6]}"


@router.post("/cmt/plan", dependencies = [Depends(_require_owner)])
def plan_cmt(request: CmtPlanRequest) -> dict[str, Any]:
    """Size the expansion and split every dataset, without loading a weight."""
    _gate("cmt")
    try:
        return cmt.plan(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.post("/cmt/run", dependencies = [Depends(_require_owner)])
def run_cmt(request: CmtRunRequest) -> dict[str, Any]:
    """Run the whole experiment: split, expand, freeze, train the new blocks, score."""
    _gate("cmt")
    try:
        report = cmt.plan(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if not report["runnable"]:
        raise HTTPException(
            status_code = 400,
            detail = report.get("blocked_reason") or "This CMT plan cannot run.",
        )
    output = (
        Path(request.output_dir).expanduser()
        if request.output_dir and request.output_dir.strip()
        else _cmt_output_dir(report["source"])
    )

    def work(progress) -> dict[str, Any]:
        return cmt.run(
            request.spec,
            output_dir = str(output),
            progress = progress,
        )

    from routes.advanced import _start_job

    return _start_job(
        "cmt",
        work,
        output_dir = str(output),
        sources = [report["source"]],
        with_progress = True,
    )


# ---------------------------------------------------------------------------
# Auto-scale (grow before training, then leave training to the chosen method)
# ---------------------------------------------------------------------------


@router.post("/auto-scale/plan", dependencies = [Depends(_require_owner)])
def plan_auto_scale(request: AutoScaleRequest) -> dict[str, Any]:
    """Size the growth the datasets justify. Loads no weight and starts nothing."""
    _gate("auto-scale")
    try:
        return autoscale.plan(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.post("/auto-scale/run", dependencies = [Depends(_require_owner)])
def run_auto_scale(request: AutoScaleRequest) -> dict[str, Any]:
    """Expand the checkpoint and save it; the returned path is a Training model."""
    _gate("auto-scale")
    try:
        report = autoscale.plan(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if not report["runnable"]:
        raise HTTPException(
            status_code = 400,
            detail = report.get("blocked_reason") or "This scale plan cannot run.",
        )
    output = (
        Path(request.output_dir).expanduser()
        if request.output_dir and request.output_dir.strip()
        else _cmt_output_dir(report["model"], prefix = "scaled")
    )

    def work(progress) -> dict[str, Any]:
        return autoscale.run(
            request.spec,
            output_dir = str(output),
            progress = progress,
        )

    expansion = report.get("expansion") or {}
    planned_base = float(expansion.get("base_params") or 0)
    planned_target = float(expansion.get("target_params") or 0)
    growth_ratio = (
        planned_target / planned_base if planned_base > 0 and planned_target > 0 else None
    )

    from routes.advanced import _start_job

    return _start_job(
        "autoscale",
        work,
        output_dir = str(output),
        sources = [report["model"]],
        with_progress = True,
        growth_ratio = growth_ratio,
    )


# ---------------------------------------------------------------------------
# Model surgery (prune, then upcycle to routed experts)
# ---------------------------------------------------------------------------


@router.post("/surgery/plan", dependencies = [Depends(_require_owner)])
def plan_surgery(request: SurgeryRequest) -> dict[str, Any]:
    """Owner-only: price the prune and the upcycle. Loads no weight, writes nothing.

    A plan reads config and tensor headers from whatever local path the spec
    names, so it is as much a filesystem reader as the run it precedes. It sits
    behind the same owner gate for the same reason the other plans do: a guest
    account cannot see another account's checkpoints through it.
    """
    _gate("surgery")
    try:
        return surgery.plan(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.post("/surgery/run", dependencies = [Depends(_require_owner)])
def run_surgery(request: SurgeryRequest) -> dict[str, Any]:
    """Owner-only: write the pruned (and optionally upcycled) checkpoint."""
    _gate("surgery")
    try:
        report = surgery.plan(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if not report["runnable"]:
        raise HTTPException(
            status_code = 400,
            detail = report.get("blocked_reason") or "This surgery plan cannot run.",
        )
    output = (
        Path(request.output_dir).expanduser()
        if request.output_dir and request.output_dir.strip()
        else _cmt_output_dir(report["model"], prefix = "surgery")
    )

    def work(progress) -> dict[str, Any]:
        return surgery.run(request.spec, output_dir = str(output), progress = progress)

    from routes.advanced import _start_job

    return _start_job(
        "surgery",
        work,
        output_dir = str(output),
        sources = [report["model"]],
        with_progress = True,
    )


# ---------------------------------------------------------------------------
# Arch swap (tensor mapping between architectures)
# ---------------------------------------------------------------------------


@router.post("/arch-swap/plan", dependencies = [Depends(_require_owner)])
def plan_arch_swap(request: ArchSwapRequest) -> dict[str, Any]:
    """Owner-only: the role-by-role mapping a swap would perform, and its cost.

    Both source and target are local paths in the spec, so the plan is a reader
    of two folders that may not belong to the caller; the run is owner-only, and
    the plan reaches the same folders.
    """
    _gate("arch-swap")
    try:
        return arch_swap.plan(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.post("/arch-swap/run", dependencies = [Depends(_require_owner)])
def run_arch_swap(request: ArchSwapRequest) -> dict[str, Any]:
    """Owner-only: write the mapped checkpoint into the target's layout."""
    _gate("arch-swap")
    try:
        report = arch_swap.plan(request.spec)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
    if not report["runnable"]:
        raise HTTPException(
            status_code = 400,
            detail = report.get("blocked_reason") or "This swap cannot run.",
        )
    output = (
        Path(request.output_dir).expanduser()
        if request.output_dir and request.output_dir.strip()
        else _cmt_output_dir(report["source"], prefix = "swap")
    )

    def work(progress) -> dict[str, Any]:
        return arch_swap.run(request.spec, output_dir = str(output), progress = progress)

    from routes.advanced import _start_job

    return _start_job(
        "arch-swap",
        work,
        output_dir = str(output),
        sources = [report["source"]],
        with_progress = True,
    )


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------


@router.get("/housekeeping")
def scan_housekeeping() -> dict[str, Any]:
    """Sizes under the Studio outputs, exports and Hugging Face caches."""
    _gate("housekeeping")
    try:
        return housekeeping.scan()
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.post("/housekeeping/delete", dependencies = [Depends(_require_owner)])
def delete_housekeeping(request: HousekeepingDeleteRequest) -> dict[str, Any]:
    """Owner-only: remove the folders the scan offered."""
    _gate("housekeeping")
    try:
        return housekeeping.delete(request.paths, confirm = request.confirm)
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.get("/housekeeping/rules")
def read_janitor_rules() -> dict[str, Any]:
    """The saved retention rules, or the defaults before anything was saved."""
    _gate("housekeeping")
    try:
        return {"rules": housekeeping.read_rules()}
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.put("/housekeeping/rules", dependencies = [Depends(_require_owner)])
def update_janitor_rules(request: JanitorRulesUpdate) -> dict[str, Any]:
    """Owner-only: save the rules. They delete nothing until they are applied."""
    _gate("housekeeping")
    try:
        merged = {**housekeeping.read_rules(), **request.model_dump(exclude_none = True)}
        return {"rules": housekeeping.save_rules(merged)}
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


def _rule_overrides(request: JanitorRulesUpdate) -> dict[str, Any]:
    return {**housekeeping.read_rules(), **request.model_dump(exclude_none = True)}


@router.post("/housekeeping/rules/preview")
def preview_janitor_rules(request: JanitorRulesUpdate) -> dict[str, Any]:
    """What the rules would remove right now, including unsaved edits. Deletes nothing."""
    _gate("housekeeping")
    try:
        return housekeeping.preview_rules(_rule_overrides(request))
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error


@router.post("/housekeeping/rules/apply", dependencies = [Depends(_require_owner)])
def apply_janitor_rules(request: JanitorRulesUpdate) -> dict[str, Any]:
    """Owner-only: remove exactly what the preview offered."""
    _gate("housekeeping")
    try:
        return housekeeping.apply_rules(_rule_overrides(request))
    except Exception as error:  # noqa: BLE001
        raise _translate(error) from error
