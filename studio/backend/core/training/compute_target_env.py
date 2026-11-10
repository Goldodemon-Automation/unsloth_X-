# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn the run's chosen compute target into the environment the worker must have before torch loads.

This runs in the training subprocess, first thing, because every value it writes is read at import
time by torch, SYCL, the allocator and Unsloth's own device module: setting them after `import torch`
changes nothing. It is a separate module from the worker so it can be tested without importing the
worker's torch/transformers stack, and so the parent process can reason about the same mapping.

The target is a *request*. Everything written here is either a default the loader reads or a pin that
cannot fail, because a job that dies in this function leaves the user a stack trace and no progress
bar -- strictly worse than a job whose log names the device it actually used.
"""

from __future__ import annotations

import os
from typing import MutableMapping, Optional

# Registry ids this module understands. Kept as a literal rather than imported from
# device_targets so a subprocess that cannot build a probe can still honour a target.
COMPUTE_TARGET_KINDS = ("cpu", "igpu", "dgpu", "npu")

# The library-level override each target needs. Two targets write nothing:
#   "dgpu" -- auto-detection already picks the discrete card, and forcing "cuda" would break an AMD
#             host, where the same card reports as "hip".
#   "igpu" -- the Studio picks the card through CUDA_VISIBLE_DEVICES/ZE_AFFINITY_MASK, and unsloth's
#             "igpu" pin *raises* when no integrated backend answers. A run started while the iGPU was
#             visible, whose child then masks it, would die with a stack trace instead of the warning
#             this module promises; the shared-memory flags below are the part it actually needs.
_DEVICE_OVERRIDES = {
    "cpu": "cpu",
    "npu": "npu",
}


def target_kind(value: object) -> Optional[str]:
    """The registry id in a stored/requested target, or None.

    A resolved *device string* ("xpu:0", "cuda:1") is accepted and mapped to its kind, because the
    parent stores the id but an older queued job may carry the device it resolved to.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if candidate in COMPUTE_TARGET_KINDS:
        return candidate
    if candidate == "auto" or not candidate:
        return None
    base = candidate.split(":", 1)[0]
    if base in ("xpu", "cuda", "mps"):
        # Which of the two an accelerator is cannot be told from the device string alone; the
        # integrated flag is what decides, and the caller passes it.
        return "dgpu"
    if base in ("npu",):
        return "npu"
    if base in ("cpu",):
        return "cpu"
    return None


def apply_compute_target_environment(
    config: dict,
    environ: Optional[MutableMapping[str, str]] = None,
) -> dict:
    """Write the target's environment and return what was set, for the run's setup line.

    ``config`` keys read here: ``compute_target`` (registry id or device string),
    ``compute_target_vendor`` ("intel"/"amd"/"nvidia"/"other") and ``compute_target_integrated``
    (bool). Only the first is required; the other two refine the allocator defaults.

    The returned mapping is the env keys that were set. The one exception is the ``"__error__"`` key,
    present only when the allocator defaults could not be computed at all: the caller logs it, and
    the job continues, because a run without those hints is slower rather than wrong.
    """
    target_environ = os.environ if environ is None else environ
    kind = target_kind(config.get("compute_target"))
    if kind is None:
        # "auto" or absent: detection decides, and there is nothing to override. Reported as empty
        # rather than guessed at, because guessing "dgpu" would set a discrete card's allocator
        # defaults on a machine that turned out to be an APU.
        return {}

    vendor = str(config.get("compute_target_vendor") or "other").strip().lower()
    integrated = bool(config.get("compute_target_integrated")) or kind == "igpu"
    if integrated and kind == "dgpu":
        # A resolved device string ("xpu:0") cannot say whether the accelerator is integrated; the
        # flag is the only thing that can, and an APU needs the integrated defaults.
        kind = "igpu"

    applied: dict[str, str] = {}
    override = _DEVICE_OVERRIDES.get(kind)
    if override is not None and not target_environ.get("UNSLOTH_TRAINING_DEVICE"):
        target_environ["UNSLOTH_TRAINING_DEVICE"] = override
        applied["UNSLOTH_TRAINING_DEVICE"] = override

    try:
        from utils.hardware.accelerator_budget import apply_accelerator_environment
        applied.update(
            apply_accelerator_environment(
                kind,
                vendor = vendor,
                integrated = integrated,
                environ = target_environ,
            )
        )
    except Exception as exc:  # noqa: BLE001 - defaults are an optimisation, not a precondition
        # A run without the allocator hints is slower, never wrong; failing the job over them would
        # turn a missing optional module into a lost training run.
        applied["__error__"] = str(exc)
    return applied


def describe_compute_target(config: dict) -> str:
    """One sentence for the setup log naming where the run will train.

    Written for a person reading a log at 2am: it says the device, and it says the cost when the
    device is the CPU, because that is the difference between an hour and two days.
    """
    kind = target_kind(config.get("compute_target"))
    if kind is None:
        return "Compute target: auto (detected device)."
    vendor = str(config.get("compute_target_vendor") or "").strip().lower()
    label = {
        "cpu": "CPU (no accelerator: expect a CPU-scale run)",
        "igpu": "integrated GPU (shared memory)",
        "dgpu": "discrete GPU",
        "npu": "NPU",
    }[kind]
    named = f"{vendor} {label}" if vendor and vendor != "other" else label
    return f"Compute target: {named}."
