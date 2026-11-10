# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which device a training run will use, and what that device can actually do.

The registry (`core.training.device_targets`) answers *what is on this machine*; this route adds the
two things only a session can supply: the target the user has chosen, and the memory budget described
in words a person can act on. Both are reads of already-measured state -- `TargetProbe.detect()` is the
one probe, and it is on a worker thread because a wedged driver blocks.

Auth-gated like `/api/system`, and for the same reason: the payload fingerprints the host (device
names, memory sizes). The Train page polls it, so the probe is cached for a minute and nothing here
starts a runtime or loads a model. It does reach torch indirectly -- `TargetProbe.detect()` asks
torch which accelerators answer, which is the whole point -- so the work is bounded and reused, not
absent. On a host with an Intel GPU it asks one more thing of torch, through `utils.hardware.xpu`,
which GPU build the wheel actually carries: that read is cached with the probe, not run per poll.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import asdict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth import policy
from auth.authentication import get_current_subject
from core.training.device_targets import (
    TrainingTarget,
    TrainingTargetError,
    TargetProbe,
    resolve_training_target,
    training_target_payload,
)
from loggers import get_logger
from utils import compute_target_settings
from utils.hardware import accelerator_budget

router = APIRouter()
logger = get_logger(__name__)

# The Train page polls /targets, and one payload runs a torch device probe, psutil twice and six
# runtime PATH scans (~30 ms here). Hardware does not change between two polls a second apart, so the
# answer is reused for the same minute the hardware inventory uses; a POST invalidates it so a
# deliberate change is reflected at once.
_PROBE_TTL_S = 60.0
# `fn` is the probe callable the entry was built from: a cache keyed only on time would serve a
# swapped-in probe's answer after something replaced `_probe` (a test, or a host that installs a
# backend mid-session), so identity is part of the key.
_probe_cache: dict = {"at": 0.0, "probe": None, "npu": None, "gpu_runtime": None, "fn": None}
# A plain lock, not asyncio's: the payload is built on a worker thread (`asyncio.to_thread`), so an
# async lock could not be acquired here. It makes the cache single-flight -- two Train pages opening
# together used to run two full probes and publish whichever finished last.
_probe_lock = threading.Lock()


def _invalidate_probe_cache() -> None:
    with _probe_lock:
        _probe_cache["at"] = 0.0
        _probe_cache["probe"] = None
        _probe_cache["npu"] = None
        _probe_cache["gpu_runtime"] = None
        _probe_cache["fn"] = None


class TargetChoice(BaseModel):
    """A requested target. Free-form on purpose: the registry owns the accepted spellings and its
    error names them, which is one list of ids rather than two that can drift."""

    target: str = Field(min_length = 1, max_length = 64)


def _probe() -> TargetProbe:
    return TargetProbe.detect()


def _has_intel_gpu(probe: TargetProbe) -> bool:
    """Whether this host has the Intel GPU row that needs explaining.

    Asked of the probe rather than of the finished payload: the registry publishes exactly one GPU row
    per `probe.gpus` entry and copies its `vendor` straight through, so the two lists cannot disagree,
    and the torch read can be skipped before it happens on a machine whose only accelerator is NVIDIA.
    """
    gpus = getattr(probe, "gpus", None)
    if not isinstance(gpus, (list, tuple)):
        return False
    return any(
        str(gpu.get("vendor") or "").strip().lower() == "intel"
        for gpu in gpus
        if isinstance(gpu, dict)
    )


# Not "PyTorch is not installed": that is the sentence this field exists to correct, and a failed
# runtime read has no evidence about the wheel at all. Kept as a constant for the same reason
# `utils/hardware/xpu.py` keeps its notes as constants -- it is user-facing text, and a typo there is
# a bug rather than a formatting change.
_GPU_RUNTIME_UNKNOWN_NOTE = (
    "The Intel GPU runtime could not be read here, so whether PyTorch can train "
    "on this GPU is unknown."
)


def _gpu_runtime_unknown() -> dict:
    """The runtime status for a host this route could not read.

    The same five keys as `xpu_torch_status`, so a UI that renders `gpu_runtime.note` never meets a
    missing key, and a note that claims nothing about which wheel is installed: the probe failed, and
    that is the whole truth here. Saying "PyTorch is not installed" would be the exact mistake this
    payload exists to avoid.
    """
    return {
        "available": False,
        "build": "unknown",
        "device_count": 0,
        "driver": None,
        "note": _GPU_RUNTIME_UNKNOWN_NOTE,
    }


def _xpu_torch_status() -> dict:
    """What the installed PyTorch means for the Intel GPU, as `utils.hardware.xpu` words it.

    Module-level so a test can replace the read, and imported inside the call for the same reason the
    NPU capability is: an install without the module must still serve the targets that do exist.
    `xpu_torch_status` is built not to raise, but a probe that does -- a wedged driver, a stand-in that
    answers with the wrong shape -- must cost a `gpu_runtime` field, not the whole GET, because the
    Train page cannot render without it.
    """
    try:
        from utils.hardware.xpu import xpu_torch_status
        status = dict(xpu_torch_status())
    except Exception as exc:  # noqa: BLE001 - a status read must never fail the page
        logger.debug("Intel GPU runtime status unavailable: %s", exc)
        return _gpu_runtime_unknown()
    for key, value in _gpu_runtime_unknown().items():
        # A partial answer keeps the field's contract: five keys, always.
        status.setdefault(key, value)
    return status


def _cached_probe_state() -> tuple[TargetProbe, dict, dict | None]:
    """(probe, NPU capability, Intel GPU runtime) from one probe pass, reused inside the TTL.

    The capability used to re-run the NPU detection the probe had already done, so a single GET
    detected NPUs twice and re-scanned PATH for every runtime. The runtime status is the third torch
    read, so it is cached with the other two and invalidated with them; it is only taken on a host that
    has the Intel GPU row it describes, and it is None on every other host -- the UI must be able to
    tell "no Intel GPU to explain" from "an Intel GPU whose status is unknown".
    """
    now = time.monotonic()
    cached = _probe_cache["probe"]
    if (
        cached is not None
        and _probe_cache["fn"] is _probe
        and now - _probe_cache["at"] < _PROBE_TTL_S
    ):
        return cached, _probe_cache["npu"], _probe_cache["gpu_runtime"]
    with _probe_lock:
        # Re-read under the lock: a second caller may have rebuilt it while this one waited.
        now = time.monotonic()
        cached = _probe_cache["probe"]
        if cached is not None and now - _probe_cache["at"] < _PROBE_TTL_S:
            return cached, _probe_cache["npu"], _probe_cache["gpu_runtime"]
        probe = _probe()
        npu = _npu_capability()
        gpu_runtime = _xpu_torch_status() if _has_intel_gpu(probe) else None
        _probe_cache.update(at = now, probe = probe, npu = npu, gpu_runtime = gpu_runtime, fn = _probe)
        return probe, npu, gpu_runtime


def _budget_for(target: TrainingTarget, probe: TargetProbe) -> dict:
    integrated = target.kind == "igpu"
    # On an integrated GPU the number torch reports is the *floor*, not the budget: an Xe part
    # commonly reserves 512 MB while the driver allocates well past it out of system RAM.
    dedicated = (
        accelerator_budget.carveout_bytes(getattr(probe, "gpus", ()))
        if integrated
        else target.memory_bytes
    )
    return accelerator_budget.budget_report(
        integrated = integrated,
        dedicated_bytes = dedicated,
        system_memory_bytes = probe.system_memory_bytes,
    )


def _npu_capability() -> dict:
    """The NPU's training capability, or an unmeasured stand-in when the probe module cannot answer.

    Imported here rather than at module scope so an installation without the NPU detection module
    still serves the four targets that exist on every machine.
    """
    try:
        from utils.hardware.npu import npu_training_capability
        return npu_training_capability()
    except Exception as exc:  # noqa: BLE001 - a capability read must never fail the page
        logger.debug("NPU capability unavailable: %s", exc)
        return {
            "mode": "unavailable",
            "backward_pass": False,
            "device": None,
            "tasks": [],
            "providers": [],
            "reasons": ["not_measured"],
            "notes": "This build could not read the NPU, so training will not use it.",
        }


def _resolve_primary(
    probe: TargetProbe | None = None,
) -> tuple[str, TrainingTarget, str | None]:
    """(requested, resolved, fallback_reason) for this session's stored compute target.

    The only place a stored choice turns into a device, because two routes name that device and they
    cannot be allowed to disagree: `/targets` publishes it as `resolved` under the picker, and
    `/npu/assist` hands it to the planner, which puts it in every sentence the NPU card shows. When the
    assist route resolved `auto` for itself, a machine with a dGPU and a stored `cpu` got "the backward
    pass runs on the discrete GPU" directly under a picker reading CPU.

    The settings read happens here, per call, so a choice made in another tab is answered on the next
    poll while the hardware facts behind it stay cached. A remembered choice this machine can no longer
    serve (an eGPU was unplugged) falls back to the default and returns the reason, so neither route can
    strand the page and neither can resolve differently under the same failure.
    """
    stored = compute_target_settings.get_selected_compute_target()
    try:
        return stored, resolve_training_target(stored, probe), None
    except TrainingTargetError as exc:
        logger.info("Stored compute target %r is unavailable here: %s", stored, exc)
        return stored, resolve_training_target(None, probe), str(exc)


def _assist_inputs() -> tuple[dict, TrainingTarget | None]:
    """(NPU capability, primary training device) for the assist preview, in one worker-thread pass.

    Both come from `_cached_probe_state()`, so the preview's `capability` is the same measured answer
    `/targets` publishes as `npu` rather than a second NPU detection pass, and the device comes from
    `_resolve_primary()`, so it is the device the picker is showing.

    A machine this route cannot probe costs the card its facts, not the card itself: the capability read
    falls back to its own stand-in and the primary device comes back None, which the route words as the
    CPU -- the one device every installation has. `/targets` answers the same failure with a 500 because
    it has a page to render; a preview has nothing to fail about.
    """
    try:
        probe, cap, _gpu_runtime = _cached_probe_state()
    except Exception as exc:  # noqa: BLE001 - a probe that raises still leaves the guarded capability
        logger.debug("Compute probe unavailable for the assist plan: %s", exc)
        return _npu_capability(), None
    try:
        return cap, _resolve_primary(probe)[1]
    except Exception as exc:  # noqa: BLE001 - a plan is still answerable without a primary device
        logger.debug("Primary training device unavailable for the assist plan: %s", exc)
        return cap, None


def _payload() -> dict:
    probe, npu, gpu_runtime = _cached_probe_state()
    payload = training_target_payload(probe)
    stored, resolved, fallback_reason = _resolve_primary(probe)
    if fallback_reason is not None:
        # A choice this machine can no longer serve must not strand the page: resolve to the default,
        # keep `requested` showing what was asked for, and say why it did not win.
        payload["fallback_reason"] = fallback_reason
    payload["requested"] = stored
    payload["resolved"] = resolved.id
    payload["budget"] = _budget_for(resolved, probe)
    payload["npu"] = npu
    # The registry can only say `torch_missing` for a GPU it cannot see training, and that token reads
    # as "PyTorch is not installed" -- false on the hybrid host that has torch and has it as a CUDA
    # build. This is the sentence that names the wheel actually installed and the remedy, taken from
    # the one module that answers the question, and only when there is an Intel GPU row to explain.
    payload["gpu_runtime"] = gpu_runtime
    return payload


@router.get("/targets")
async def list_compute_targets(current_subject: str = Depends(get_current_subject)) -> dict:
    """Every device on this machine with its capability, plus the resolved choice.

    Deliberately includes unsupported devices: the Train page has to show that the machine has an
    NPU it cannot train on, which is different information from not having one.
    """
    return await asyncio.to_thread(_payload)


@router.post("/target", dependencies = [Depends(policy.require_owner)])
async def choose_compute_target(
    choice: TargetChoice, current_subject: str = Depends(get_current_subject)
) -> dict:
    """Validate a chosen target, remember it, and answer with what it resolves to."""
    try:
        resolved = await asyncio.to_thread(resolve_training_target, choice.target, None)
    except TrainingTargetError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None
    try:
        stored = await asyncio.to_thread(
            compute_target_settings.set_selected_compute_target, choice.target
        )
        _invalidate_probe_cache()
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None
    payload = training_target_payload()
    return {
        "requested": stored,
        "resolved": resolved.id,
        "target": asdict(resolved),
        "targets": payload["targets"],
    }


@router.get("/npu/assist")
async def npu_assist_plan(
    eval_enabled: bool = False,
    qat: bool = False,
    requested: str = "auto",
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """What the NPU would do for a run on this machine, in the words the UI shows.

    Read-only preview: it plans, it does not start anything. `requested="full"` is answered with a
    refusal that names the reason, which is the whole point of exposing the mode -- a user who asks
    for NPU-only training deserves to be told no vendor ships a backward pass, not to be silently
    put on the assist path.

    The device every sentence names is this session's stored compute target, resolved through the same
    function `/targets` uses, so the card cannot contradict the picker above it. `requested` here is the
    NPU mode, not that target: the two answers travel together in one response as `mode` and `resolved`.
    """
    from core.training.npu_assist import plan_npu_assist

    cap, primary = await asyncio.to_thread(_assist_inputs)
    plan = await asyncio.to_thread(
        plan_npu_assist,
        cap,
        requested = requested,
        # resolve_training_target always returns a target (the CPU row is always supported), so None
        # only means the registry itself failed; treating that as the CPU is the honest fallback.
        primary_kind = getattr(primary, "kind", "cpu"),
        eval_enabled = eval_enabled,
        qat = qat,
    )
    return {
        "enabled": plan.enabled,
        "mode": plan.mode,
        "tasks": list(plan.tasks),
        "providers": list(plan.providers),
        "refused_reason": plan.refused_reason,
        "notes": list(plan.notes),
        "capability": cap,
        # The device the sentences above name, spelled exactly as `/targets` spells it. Published so a
        # UI (and a test) can check the card against the picker instead of trusting the prose.
        "resolved": getattr(primary, "id", None),
    }
