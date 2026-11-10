# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How much memory an accelerator can actually reach, and how to make it reach it.

Two problems live here, and they share one cause: an integrated GPU is not a small discrete GPU.

**Budget.** ``torch.xpu``/``torch.cuda`` report the *carve-out* on an integrated GPU -- the slice the
BIOS or the driver reserved, often 512 MB, sometimes 4 GB. The driver can allocate well past it out
of system RAM, so a fit check that trusts ``total_memory`` refuses models the machine runs happily.
The real ceiling is the RAM that can be taken without starving the OS; the carve-out is a floor, not
a ceiling, so the budget is the larger of the two. :func:`usable_accelerator_memory_bytes`
is that rule, and it is the only place the fraction is defined.

**Environment.** Allocators fragment. On a discrete card that wastes VRAM; on a unified-memory
device the same fragmentation can push pages into system RAM and slow the whole run, or fail an
allocation the driver would have served. ``expandable_segments`` lets the caching allocator grow
into the shared pool instead of refusing a block. The rest of :func:`accelerator_environment` is
the same kind of decision: settings that only ever help the device they name, applied with
``setdefault`` so a user's own value always wins.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping, MutableMapping, Optional

# Fraction of system RAM an integrated GPU may use. Half is what Intel's driver advertises as the
# shared ceiling on Windows, and AMD's APUs report a similar figure through GTT; 0.6 leaves the OS,
# the dataset pipeline and the UI the rest of a machine that has no separate pool to fall back on.
# Below 0.5 the budget stops describing what the driver allows; above 0.7 a background allocation
# can commit the machine into swap mid-run, which is slower than refusing at fit time.
SHARED_MEMORY_FRACTION = 0.6

# Which environment keys each target sets, documented where they are set rather than in a table, so
# the reason sits beside the value.
_INTEL_XPU_KEYS = (
    "PYTORCH_XPU_ALLOC_CONF",
    "SYCL_CACHE_PERSISTENT",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
)
_AMD_HIP_KEYS = ("PYTORCH_HIP_ALLOC_CONF", "TORCH_BLAS_PREFER_HIPBLASLT")
_CUDA_KEYS = ("PYTORCH_CUDA_ALLOC_CONF",)
# compile switches the loaders read; a device with no Inductor backend must clear all three.
_COMPILE_KEYS = ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE", "UNSLOTH_COMPILE_DISABLE")


def shared_accelerator_budget(
    system_memory_bytes: Optional[int],
    dedicated_bytes: Optional[int] = None,
    fraction: float = SHARED_MEMORY_FRACTION,
) -> Optional[int]:
    """Bytes an accelerator may hold on a unified-memory host.

    Returns the larger of what it already owns and its share of system RAM, because the driver will
    not take a reservation away from the card: a 2 GB carve-out on a 16 GB laptop is still 2 GB even
    when the fraction would allow 9.6 GB, and the card is never limited to less than it reserved.
    ``None`` when system memory is unknown -- unknown is not zero, and a caller that treats it as
    zero refuses every model.
    """
    if system_memory_bytes is None or system_memory_bytes <= 0:
        # Unmeasured system memory does not erase a measured reservation: the card still holds what
        # it holds, and a caller that has only the carve-out should plan against it.
        return int(dedicated_bytes) if dedicated_bytes else None
    share = int(system_memory_bytes * max(0.0, min(1.0, fraction)))
    if dedicated_bytes is not None and dedicated_bytes > share:
        return int(dedicated_bytes)
    return share


def usable_accelerator_memory_bytes(
    *,
    integrated: bool,
    dedicated_bytes: Optional[int],
    system_memory_bytes: Optional[int],
    fraction: float = SHARED_MEMORY_FRACTION,
) -> Optional[int]:
    """The memory a training run may plan against.

    A discrete card's own total is the answer, private and fixed. An integrated one answers with
    :func:`shared_accelerator_budget`, since its ``total_memory`` is the carve-out and not the
    ceiling. CPU answers with system RAM: it has no other pool, and the fit check still has to
    refuse a model that cannot load.
    """
    if not integrated:
        return dedicated_bytes
    return shared_accelerator_budget(
        system_memory_bytes,
        dedicated_bytes = dedicated_bytes,
        fraction = fraction,
    )


def accelerator_environment(
    kind: str,
    *,
    vendor: str = "other",
    integrated: bool = False,
    platform: Optional[str] = None,
) -> dict[str, str]:
    """Environment keys to set before a run starts on this target.

    Pure: returns what it would set and touches nothing, so a caller can log the decision and a test
    can read it. Every value here is a default -- :func:`apply_accelerator_environment` uses
    ``setdefault`` -- because a user who typed a value into their environment has a reason.
    """
    target = (kind or "").strip().lower()
    platform = platform or sys.platform
    env: dict[str, str] = {}

    if target == "cpu":
        # Nothing to configure for the CPU that utils.cpu_threads does not already own. The compile
        # switches matter though: Dynamo has no CPU backend worth the graph breaks on a decoder, and
        # an Unsloth checkout imported beside a CPU run would otherwise leave patches armed.
        for key in _COMPILE_KEYS:
            env[key] = "1"
        return env

    if target == "npu":
        # No vendor's NPU runtime accepts a compiled decoder graph, and Unsloth's patches target
        # accelerator kernels that do not exist here; the assist path runs stock transformers.
        for key in _COMPILE_KEYS:
            env[key] = "1"
        env["UNSLOTH_NPU_DEVICE"] = "1"
        return env

    if target == "igpu":
        if platform == "darwin":
            # Metal owns its own allocator and reads none of these keys; setting them would be
            # noise in the job's setup line that describes nothing on the machine.
            return {}
        if vendor == "intel":
            # Xe's allocator is the one that needs this most: with a 512 MB carve-out and a shared
            # pool behind it, the default segment split fails allocations the driver would have
            # served out of GTT.
            env["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
            # Without it the SYCL kernel cache is recompiled per process, which on an integrated
            # part is minutes of the first step rather than seconds.
            env["SYCL_CACHE_PERSISTENT"] = "1"
            # Immediate command lists cut per-kernel launch latency, which dominates a small
            # integrated part far more than a discrete card, where the queue hides it.
            env["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
        elif vendor == "amd":
            env["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
            # hipBLASLt has no tuned kernel for the smallest GEMM shapes an iGPU run produces and
            # falls back anyway; leaving it on costs a dispatch check per call.
            env["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
        # Integrated parts share memory with the host, so a run must not also pin host memory for
        # fast transfers: there is no separate pool for the pin to be faster than. Unsloth's loader
        # reads this flag before it decides to pin.
        env["UNSLOTH_SKIP_PIN_MEMORY"] = "1"
        env["UNSLOTH_IGPU"] = "1"
        if integrated:
            # Inductor's XPU/HIP backends are unreliable enough on integrated parts that a 30-60
            # step compile is not repaid inside a typical iGPU run.
            for key in _COMPILE_KEYS:
                env[key] = "1"
        return env

    # Discrete accelerators: nothing to change. The stock allocator split reuses freed blocks sooner
    # than expandable segments do, and a single job on a whole card is not the memory-pressure case
    # that setting exists for. Named here so the absence is a decision, not an oversight.
    return env


def apply_accelerator_environment(
    kind: str,
    *,
    vendor: str = "other",
    integrated: bool = False,
    environ: Optional[MutableMapping[str, str]] = None,
    platform: Optional[str] = None,
) -> dict[str, str]:
    """Set the defaults from :func:`accelerator_environment`, returning only what was actually set.

    The return value is what a job's setup line should log: a value already present in the
    environment is not ours to report as applied.
    """
    target_environ = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    for key, value in accelerator_environment(
        kind,
        vendor = vendor,
        integrated = integrated,
        platform = platform,
    ).items():
        if target_environ.get(key) is not None and target_environ.get(key) != "":
            continue
        target_environ[key] = value
        applied[key] = value
    return applied


def carveout_bytes(gpus: object) -> Optional[int]:
    """The reservation the driver reports for an integrated GPU, from probe entries shaped
    ``{"integrated": bool, "memory_bytes": int | None}``.

    Shared by the compute route and the fit check so the two cannot disagree about which number is
    the carve-out; the caller passes whatever its probe produced and gets None when nothing says.
    """
    for gpu in gpus or ():  # type: ignore[union-attr]
        try:
            if not gpu.get("integrated"):
                continue
            memory = gpu.get("memory_bytes")
        except AttributeError:
            continue
        if memory:
            return int(memory)
    return None


def budget_report(
    *,
    integrated: bool,
    dedicated_bytes: Optional[int],
    system_memory_bytes: Optional[int],
    fraction: float = SHARED_MEMORY_FRACTION,
) -> dict:
    """JSON-ready view of the budget decision, with the sentence the UI shows.

    The sentence exists because "0.5 GB" beside a model that needs 3 GB reads as a refusal on a
    machine with 16 GB of RAM, unless it also says the card shares that RAM.
    """
    usable = usable_accelerator_memory_bytes(
        integrated = integrated,
        dedicated_bytes = dedicated_bytes,
        system_memory_bytes = system_memory_bytes,
        fraction = fraction,
    )
    if integrated and usable is not None:
        notes = (
            f"This GPU shares system memory: {_gb(usable)} GB of the machine's "
            f"{_gb(system_memory_bytes)} GB is usable for training, "
            f"{int(round(fraction * 100))}% of it."
        )
    elif dedicated_bytes is not None:
        notes = f"This GPU has {_gb(dedicated_bytes)} GB of its own memory."
    else:
        notes = "This device's memory could not be measured; the fit check will estimate."
    return {
        "integrated": bool(integrated),
        "dedicated_bytes": dedicated_bytes,
        "system_memory_bytes": system_memory_bytes,
        "shared_fraction": fraction,
        "usable_bytes": usable,
        "notes": notes,
    }


def _gb(value: Optional[int]) -> str:
    return f"{(value or 0) / 1024**3:.1f}"


# Re-exported for callers that want the key lists without reaching into the module's private names.
INTEGRATED_ENV_KEYS: Mapping[str, tuple] = {
    "intel": _INTEL_XPU_KEYS,
    "amd": _AMD_HIP_KEYS,
    "nvidia": _CUDA_KEYS,
}
