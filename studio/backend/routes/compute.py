# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which device a training run will use, and what that device can actually do.

The registry (`core.training.device_targets`) answers *what is on this machine*; this route adds the
two things only a session can supply: the target the user has chosen, and the memory budget described
in words a person can act on. Both are reads of already-measured state -- `TargetProbe.detect()` is the
one probe, and it is on a worker thread because a wedged driver blocks.

Auth-gated like `/api/system`, and for the same reason: the payload fingerprints the host (device
names, memory sizes). It is also polled by the Train page, so every handler here must stay a cheap
read; nothing in this module starts a runtime, loads a model or imports torch.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

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


class TargetChoice(BaseModel):
    """A requested target. Free-form on purpose: the registry owns the accepted spellings and its
    error names them, which is one list of ids rather than two that can drift."""

    target: str = Field(min_length = 1, max_length = 64)


def _probe() -> TargetProbe:
    return TargetProbe.detect()


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


def _payload() -> dict:
    probe = _probe()
    payload = training_target_payload(probe)
    stored = compute_target_settings.get_selected_compute_target()
    try:
        resolved = resolve_training_target(stored, probe)
    except TrainingTargetError as exc:
        # A remembered choice that this machine can no longer serve (an eGPU was unplugged) must not
        # strand the page: fall back to the default and say so with the stored value still visible.
        logger.info("Stored compute target %r is unavailable here: %s", stored, exc)
        resolved = resolve_training_target(None, probe)
        payload["fallback_reason"] = str(exc)
    payload["requested"] = stored
    payload["resolved"] = resolved.id
    payload["budget"] = _budget_for(resolved, probe)
    payload["npu"] = _npu_capability()
    return payload


@router.get("/targets")
async def list_compute_targets(current_subject: str = Depends(get_current_subject)) -> dict:
    """Every device on this machine with its capability, plus the resolved choice.

    Deliberately includes unsupported devices: the Train page has to show that the machine has an
    NPU it cannot train on, which is different information from not having one.
    """
    return await asyncio.to_thread(_payload)


@router.post("/target")
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
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None
    payload = training_target_payload()
    return {
        "requested": stored,
        "resolved": resolved.id,
        "target": asdict(resolved),
        "targets": payload["targets"],
    }
