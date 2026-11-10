# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Decides how a CPU training run uses the machine it lands on.

A CPU run is fifty to a hundred times the wall-clock of a GPU run, and that gap is mostly decided
by settings nobody sets: how many threads feed the matrix multiplies, which attention
implementation runs, how many dataloader workers exist, whether activations are recomputed, and
which precision the frozen base holds. ``plan_cpu_training`` turns host measurements into those
answers so the rules live in one testable place and the loader only applies them. Nothing here
imports torch and nothing here touches the machine: every unknown or nonsensical measurement
degrades to the conservative answer instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass

# The dtype rule defers to the training module's fit helper so this planner and the loader's
# estimate block can never disagree about what "does not fit" means. Importing the sibling costs
# nothing: ``core.training.__init__`` imports it before this module can load, and it pulls in no
# torch at module scope.
from .training import cpu_fp32_weights_fit


@dataclass(frozen = True)
class CpuTrainingPlan:
    """The settings a CPU training run should start with.

    ``reasons`` holds one plain sentence per non-obvious choice so the loader can log why the
    machine ended up with these numbers.
    """

    torch_threads: int  # argument for torch.set_num_threads
    interop_threads: int  # argument for torch.set_num_interop_threads; always 1
    attn_implementation: str  # "sdpa" | "eager"
    dataloader_workers: int  # 0 on the spawn platforms, a small count on Linux
    gradient_checkpointing: bool  # trade recomputation for activation memory
    dtype: str  # "float32" | "bfloat16"
    reasons: tuple[str, ...]


def plan_cpu_training(
    *,
    usable_cpus: int | None,
    available_memory_bytes: int | None,
    fp32_weight_bytes: int | None,
    platform: str | None = None,
    supports_sdpa: bool = True,
) -> CpuTrainingPlan:
    """Build the CPU training plan from host measurements.

    Every measurement may be None ("unknown"), and every rule below then picks the conservative
    answer: one thread, zero dataloader workers, eager attention, fp32, no checkpointing.
    """
    reasons: list[str] = []

    # Treat zero and negative measurements as unknown: they describe no real host and must not
    # steer a rule into dividing by zero or comparing against nothing.
    memory_known = available_memory_bytes is not None and available_memory_bytes > 0
    weights_known = fp32_weight_bytes is not None and fp32_weight_bytes > 0
    cpus_known = usable_cpus is not None and usable_cpus > 0

    # Threads: one per physical core. No portable physical-core count exists (WMI, sysfs and
    # sysctl all disagree), so physical cores are approximated as half the usable CPUs whenever
    # there are more than two: SMT siblings add little to a GEMM and cost cache contention. A
    # machine with two CPUs or fewer is taken at face value, because halving it would leave the
    # matrix multiplies single-threaded for no measurable win. The count is clamped to at least
    # one thread and never more than the usable CPUs.
    if not cpus_known:
        torch_threads = 1
        reasons.append("The CPU count is unknown, so the run keeps a single thread.")
    else:
        torch_threads = usable_cpus // 2 if usable_cpus > 2 else usable_cpus
        torch_threads = max(1, min(torch_threads, usable_cpus))
        if usable_cpus > 2:
            reasons.append(
                "One thread per physical core is approximated as half the usable CPUs because "
                "SMT siblings add little to a GEMM and cost cache contention."
            )

    # Memory clamp on the thread count, the way utils.cpu_threads clamps OpenBLAS: fewer threads
    # share the same weights, but each thread keeps its own per-thread buffers, so once the fp32
    # weights take more than 60% of the available memory those buffers decide between a run and
    # an OOM kill. A memory-bound GEMM loses little from half the threads.
    if (
        memory_known
        and weights_known
        and fp32_weight_bytes * 10 > available_memory_bytes * 6
    ):
        torch_threads = max(1, torch_threads // 2)
        reasons.append(
            "The thread count is halved because the fp32 weights take more than 60% of the "
            "available memory and every thread keeps its own buffers."
        )

    # Inter-op threads: always 1. torch sizes its intra-op and inter-op pools independently, so a
    # core-sized inter-op pool beside a core-sized intra-op pool oversubscribes the machine
    # (the pools multiply, not add), and a CPU training run has exactly one forward graph to
    # schedule at a time anyway.
    interop_threads = 1
    reasons.append(
        "The inter-op pool stays at one thread because a second core-sized pool beside the "
        "intra-op one would oversubscribe the machine."
    )

    # Attention: SDPA runs the fused kernel torch ships instead of materializing the full
    # attention matrix in plain tensor math; only a stack without SDPA support falls back to
    # eager. An unknown capability is falsy and degrades to the safe fallback.
    if supports_sdpa:
        attn_implementation = "sdpa"
        reasons.append("SDPA is the fused attention kernel torch ships for this stack.")
    else:
        attn_implementation = "eager"
        reasons.append("Eager attention is the fallback for a stack without SDPA support.")

    # Dataloader workers: zero on win32 and darwin, where spawned workers re-import the training
    # module (torch and unsloth_zoo with it) and the spawn hazards are not worth it on one
    # machine. Linux forks cheaply, so a few workers can collate the next batch while the CPU
    # computes this one; the cap at four keeps the collate workers off the GEMM threads.
    if platform in ("win32", "darwin"):
        dataloader_workers = 0
        reasons.append(
            "Dataloader workers stay at zero because spawned workers re-import the training "
            "module and the spawn platforms are not worth the hazard."
        )
    elif isinstance(platform, str) and platform.startswith("linux") and cpus_known:
        dataloader_workers = min(4, max(1, usable_cpus // 4))
        reasons.append(
            "Linux forks dataloader workers cheaply, so a few collate the next batch while the "
            "CPU computes this one."
        )
    else:
        # An unknown platform or unknown CPU count keeps the spawn-platform answer: zero workers
        # can never hurt a run.
        dataloader_workers = 0
        reasons.append(
            "Dataloader workers stay at zero because the platform or the CPU count is unknown."
        )

    # Gradient checkpointing: on once the fp32 weights take over 40% of the available memory.
    # Recomputing activations trades compute for memory, and a CPU run that starts swapping is
    # slower than one that recomputes. Unknown measurements keep it off.
    gradient_checkpointing = (
        memory_known and weights_known and fp32_weight_bytes * 10 > available_memory_bytes * 4
    )
    if gradient_checkpointing:
        reasons.append(
            "Gradient checkpointing trades recomputation for memory because a run that starts "
            "swapping is slower than one that recomputes."
        )

    # Dtype: cpu_fp32_weights_fit is the single fit rule (the fp32 weights plus a 2 GiB
    # activation headroom inside the available RAM), the same one _load_cpu_model applied before
    # this module existed. The loader now asks this plan instead of deciding locally, so this
    # module is the single place that decides the CPU load dtype and the two cannot drift.
    if not cpu_fp32_weights_fit(fp32_weight_bytes, available_memory_bytes):
        dtype = "bfloat16"
        reasons.append(
            "The frozen base loads in bfloat16 because the fp32 weights cannot fit in the "
            "available memory."
        )
    else:
        dtype = "float32"

    return CpuTrainingPlan(
        torch_threads = torch_threads,
        interop_threads = interop_threads,
        attn_implementation = attn_implementation,
        dataloader_workers = dataloader_workers,
        gradient_checkpointing = gradient_checkpointing,
        dtype = dtype,
        reasons = tuple(reasons),
    )
