# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the CPU training tuning plan.

The planner is a pure function of its measurements and this host has no torch, so every rule is
exercised directly: the SMT thread approximation and its clamps, the memory-driven thread
halving, the platform dataloader-worker policy, the checkpointing and dtype thresholds, the
frozen shape of the plan, and the sentence quality of the recorded reasons.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.training.cpu_tuning import plan_cpu_training

GB = 1024**3


def plan(**measurements):
    """Plan with the measurements the test does not care about marked unknown."""
    unknown = {
        "usable_cpus": None,
        "available_memory_bytes": None,
        "fp32_weight_bytes": None,
    }
    unknown.update(measurements)
    return plan_cpu_training(**unknown)


def test_threads_are_one_per_approximate_physical_core():
    # SMT siblings add little to a GEMM, so the usable CPU count is halved past two CPUs.
    assert plan(usable_cpus = 16).torch_threads == 8
    assert plan(usable_cpus = 8).torch_threads == 4
    assert plan(usable_cpus = 3).torch_threads == 1


def test_thread_count_is_clamped_to_at_least_one_and_never_more_than_the_cpus():
    # A 2-CPU box is taken at face value; halving it would leave the GEMM single-threaded.
    assert plan(usable_cpus = 1).torch_threads == 1
    assert plan(usable_cpus = 2).torch_threads == 2
    assert plan(usable_cpus = 17).torch_threads == 8
    assert plan(usable_cpus = -4).torch_threads == 1


def test_memory_pressure_halves_the_thread_count_and_adds_a_reason():
    # 7 GB of fp32 weights in 10 GB of RAM is over 60%: per-thread buffers decide the run's fate.
    tuned = plan(
        usable_cpus = 16,
        available_memory_bytes = 10 * GB,
        fp32_weight_bytes = 7 * GB,
    )
    assert tuned.torch_threads == 4
    assert any("60%" in reason for reason in tuned.reasons)
    # Exactly 60% does not clamp.
    at_boundary = plan(
        usable_cpus = 16,
        available_memory_bytes = 10 * GB,
        fp32_weight_bytes = 6 * GB,
    )
    assert at_boundary.torch_threads == 8


def test_the_memory_clamp_never_drops_the_run_below_one_thread():
    # 3 usable CPUs halve to 1 thread, which the clamp must then halve to 1, not 0.
    tuned = plan(
        usable_cpus = 3,
        available_memory_bytes = 10 * GB,
        fp32_weight_bytes = 7 * GB,
    )
    assert tuned.torch_threads == 1


def test_unknown_inputs_yield_the_conservative_plan():
    conservative = plan_cpu_training(
        usable_cpus = None,
        available_memory_bytes = None,
        fp32_weight_bytes = None,
        platform = None,
        supports_sdpa = None,
    )
    assert conservative.torch_threads == 1
    assert conservative.interop_threads == 1
    assert conservative.attn_implementation == "eager"
    assert conservative.dataloader_workers == 0
    assert conservative.gradient_checkpointing is False
    assert conservative.dtype == "float32"


def test_interop_threads_stay_at_one_on_every_machine():
    assert plan(usable_cpus = 16).interop_threads == 1
    assert plan().interop_threads == 1


def test_attention_falls_back_to_eager_without_sdpa_support():
    assert plan(supports_sdpa = True).attn_implementation == "sdpa"
    assert plan(supports_sdpa = False).attn_implementation == "eager"


def test_dataloader_workers_are_zero_on_the_spawn_platforms():
    # Spawned workers would re-import the training module; the hazard is not worth it here.
    assert plan(usable_cpus = 16, platform = "win32").dataloader_workers == 0
    assert plan(usable_cpus = 16, platform = "darwin").dataloader_workers == 0


def test_dataloader_workers_scale_on_linux_but_stay_capped_and_floored():
    assert plan(usable_cpus = 32, platform = "linux").dataloader_workers == 4
    assert plan(usable_cpus = 16, platform = "linux").dataloader_workers == 4
    assert plan(usable_cpus = 8, platform = "linux").dataloader_workers == 2
    assert plan(usable_cpus = 4, platform = "linux").dataloader_workers == 1
    # An unknown CPU count keeps the conservative zero even on Linux.
    assert plan(usable_cpus = None, platform = "linux").dataloader_workers == 0


def test_gradient_checkpointing_flips_just_above_forty_percent():
    # Over 40% of the RAM in fp32 weights the activations are recomputed instead of kept.
    at_boundary = plan(available_memory_bytes = 10 * GB, fp32_weight_bytes = 4 * GB)
    over_boundary = plan(
        available_memory_bytes = 10 * GB,
        fp32_weight_bytes = 4 * GB + 1,
    )
    assert at_boundary.gradient_checkpointing is False
    assert over_boundary.gradient_checkpointing is True
    # Unknown memory keeps checkpointing off.
    assert plan(fp32_weight_bytes = 7 * GB).gradient_checkpointing is False


def test_dtype_flips_to_bfloat16_only_when_fp32_cannot_fit():
    # The fit rule is cpu_fp32_weights_fit: the fp32 weights plus a 2 GiB activation headroom
    # inside the available RAM, the same rule the loader applied before this module existed.
    fits = plan(available_memory_bytes = 10 * GB, fp32_weight_bytes = 8 * GB - 1)
    does_not_fit = plan(available_memory_bytes = 10 * GB, fp32_weight_bytes = 8 * GB + 1)
    assert fits.dtype == "float32"
    assert does_not_fit.dtype == "bfloat16"
    assert plan(available_memory_bytes = 4 * GB, fp32_weight_bytes = 6 * GB).dtype == "bfloat16"
    # Unknown measurements keep fp32.
    assert plan().dtype == "float32"


def test_the_plan_is_frozen():
    tuned = plan(usable_cpus = 8)
    with pytest.raises(FrozenInstanceError):
        tuned.torch_threads = 2  # type: ignore[misc]


def test_every_reason_is_a_plain_sentence():
    rich = plan_cpu_training(
        usable_cpus = 16,
        available_memory_bytes = 10 * GB,
        fp32_weight_bytes = 7 * GB,
        platform = "win32",
        supports_sdpa = False,
    )
    conservative = plan_cpu_training(
        usable_cpus = None,
        available_memory_bytes = None,
        fp32_weight_bytes = None,
        platform = None,
        supports_sdpa = None,
    )
    assert rich.reasons
    for tuned in (rich, conservative):
        for reason in tuned.reasons:
            assert " " in reason
            assert "_" not in reason
