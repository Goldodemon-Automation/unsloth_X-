# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The integrated-GPU memory budget and the allocator defaults that go with it.

The budget half is pinned because getting it wrong is expensive in both directions: trusting
``total_memory`` on an iGPU refuses models that run, and ignoring the carve-out below the shared
fraction would plan against RAM the driver will not hand over. The environment half is pinned
because the ``setdefault`` contract is what stops a run from overwriting a value the user chose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.hardware import accelerator_budget as ab  # noqa: E402

GB = 1024**3


def test_a_carveout_above_the_shared_share_wins():
    # An 8 GB reservation on a 10 GB machine stays 8 GB even though 60% of RAM is 6: the driver does
    # not take back memory it has already reserved for the card.
    budget = ab.shared_accelerator_budget(10 * GB, dedicated_bytes = 8 * GB)

    assert budget == 8 * GB


def test_a_carveout_below_the_shared_share_is_a_floor_not_a_ceiling():
    # The common laptop case: 512 MB reserved, and the figure torch reports.
    budget = ab.shared_accelerator_budget(16 * GB, dedicated_bytes = int(0.5 * GB))

    assert budget == int(16 * GB * 0.6)
    assert budget > int(0.5 * GB)


def test_the_shared_fraction_is_the_documented_default():
    assert ab.SHARED_MEMORY_FRACTION == 0.6
    assert ab.shared_accelerator_budget(10 * GB) == int(6 * GB)


def test_unknown_system_memory_is_none_rather_than_zero():
    # None means "unmeasured"; zero would make every fit check refuse every model.
    assert ab.shared_accelerator_budget(None) is None
    assert ab.shared_accelerator_budget(0) is None
    assert ab.shared_accelerator_budget(-1) is None


def test_a_dedicated_figure_survives_an_unmeasured_system():
    assert ab.shared_accelerator_budget(None, dedicated_bytes = 2 * GB) == 2 * GB


def test_the_fraction_is_clamped_to_a_sane_range():
    # A configured 0.0 or 4.0 must not produce an empty budget or an over-commit.
    assert ab.shared_accelerator_budget(10 * GB, fraction = 0.0) == 0
    assert ab.shared_accelerator_budget(10 * GB, fraction = 4.0) == 10 * GB


def test_a_discrete_card_plans_against_its_own_memory_only():
    usable = ab.usable_accelerator_memory_bytes(
        integrated = False,
        dedicated_bytes = 24 * GB,
        system_memory_bytes = 64 * GB,
    )

    assert usable == 24 * GB


def test_a_discrete_card_with_no_reading_stays_unknown():
    usable = ab.usable_accelerator_memory_bytes(
        integrated = False,
        dedicated_bytes = None,
        system_memory_bytes = 64 * GB,
    )

    assert usable is None


def test_an_integrated_gpu_plans_against_the_shared_budget():
    usable = ab.usable_accelerator_memory_bytes(
        integrated = True,
        dedicated_bytes = int(0.5 * GB),
        system_memory_bytes = 32 * GB,
    )

    assert usable == int(32 * GB * 0.6)


def test_the_report_says_that_an_integrated_gpu_shares_system_memory():
    report = ab.budget_report(
        integrated = True,
        dedicated_bytes = int(0.5 * GB),
        system_memory_bytes = 16 * GB,
    )

    assert report["usable_bytes"] == int(16 * GB * 0.6)
    # The sentence is the user-facing half: a bare 0.5 GB reads as a refusal.
    assert "shares system memory" in report["notes"]
    assert "60%" in report["notes"]


def test_the_report_states_a_discrete_cards_own_memory():
    report = ab.budget_report(
        integrated = False,
        dedicated_bytes = 12 * GB,
        system_memory_bytes = 64 * GB,
    )

    assert report["usable_bytes"] == 12 * GB
    assert "12.0 GB" in report["notes"]


def test_an_unmeasured_device_says_so_instead_of_quoting_zero():
    report = ab.budget_report(
        integrated = False,
        dedicated_bytes = None,
        system_memory_bytes = None,
    )

    assert report["usable_bytes"] is None
    assert "could not be measured" in report["notes"]


def test_cpu_clears_the_compile_switches():
    env = ab.accelerator_environment("cpu")

    assert env == {
        "TORCHDYNAMO_DISABLE": "1",
        "TORCH_COMPILE_DISABLE": "1",
        "UNSLOTH_COMPILE_DISABLE": "1",
    }


def test_npu_clears_the_compile_switches_and_marks_the_device():
    env = ab.accelerator_environment("npu")

    assert env["UNSLOTH_NPU_DEVICE"] == "1"
    assert env["TORCHDYNAMO_DISABLE"] == "1"


def test_intel_igpu_gets_the_shared_pool_allocator():
    env = ab.accelerator_environment("igpu", vendor = "intel", integrated = True, platform = "win32")

    assert env["PYTORCH_XPU_ALLOC_CONF"] == "expandable_segments:True"
    assert env["SYCL_CACHE_PERSISTENT"] == "1"
    assert env["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] == "1"
    # Unified memory: pinning host memory buys nothing and costs RAM the run needs.
    assert env["UNSLOTH_SKIP_PIN_MEMORY"] == "1"
    assert env["UNSLOTH_IGPU"] == "1"
    # An integrated part does not repay a Dynamo compile.
    assert env["TORCHDYNAMO_DISABLE"] == "1"


def test_amd_apu_gets_hip_allocator_but_not_hipblaslt():
    env = ab.accelerator_environment("igpu", vendor = "amd", integrated = True, platform = "linux")

    assert env["PYTORCH_HIP_ALLOC_CONF"] == "expandable_segments:True"
    assert env["TORCH_BLAS_PREFER_HIPBLASLT"] == "0"


def test_apple_igpu_sets_nothing():
    env = ab.accelerator_environment("igpu", vendor = "apple", integrated = True, platform = "darwin")

    assert env == {}


def test_a_discrete_card_is_left_alone():
    assert ab.accelerator_environment("dgpu", vendor = "nvidia", integrated = False) == {}
    assert ab.accelerator_environment("cuda", vendor = "nvidia") == {}


def test_applying_never_overwrites_a_user_value():
    environ = {"PYTORCH_XPU_ALLOC_CONF": "max_split_size_mb:512"}

    applied = ab.apply_accelerator_environment(
        "igpu", vendor = "intel", integrated = True, environ = environ, platform = "linux"
    )

    assert environ["PYTORCH_XPU_ALLOC_CONF"] == "max_split_size_mb:512"
    # Only what this call actually set is reported, or a job's setup line would claim to have
    # applied somebody else's setting.
    assert "PYTORCH_XPU_ALLOC_CONF" not in applied
    assert applied["UNSLOTH_IGPU"] == "1"


def test_an_empty_user_value_is_treated_as_unset():
    environ = {"SYCL_CACHE_PERSISTENT": ""}

    applied = ab.apply_accelerator_environment(
        "igpu", vendor = "intel", integrated = True, environ = environ, platform = "linux"
    )

    assert environ["SYCL_CACHE_PERSISTENT"] == "1"
    assert applied["SYCL_CACHE_PERSISTENT"] == "1"


def test_applying_to_the_real_environment_is_reversible_for_tests(monkeypatch):
    for key in ("UNSLOTH_IGPU", "UNSLOTH_SKIP_PIN_MEMORY", "PYTORCH_XPU_ALLOC_CONF"):
        monkeypatch.delenv(key, raising = False)

    applied = ab.apply_accelerator_environment("igpu", vendor = "intel", integrated = True)

    # The default platform is this host's, so the Intel branch is what got applied here.
    assert applied
    assert "UNSLOTH_IGPU" in ab.os.environ


def test_every_documented_key_list_is_reachable():
    # The mapping exists so a caller can ask which keys a vendor owns without running the function.
    assert ab.INTEGRATED_ENV_KEYS["intel"] is ab._INTEL_XPU_KEYS
    assert ab.INTEGRATED_ENV_KEYS["amd"] is ab._AMD_HIP_KEYS
    assert ab.INTEGRATED_ENV_KEYS["nvidia"] is ab._CUDA_KEYS


@pytest.mark.parametrize("kind", ["", "  ", "GPU", "qualcomm", "mps"])
def test_unknown_targets_set_nothing_rather_than_guessing(kind):
    assert ab.accelerator_environment(kind) == {}

def test_a_cpu_budget_says_ram_and_uses_the_free_figure():
    report = ab.budget_report(
        integrated = False,
        dedicated_bytes = 40 * GB,
        system_memory_bytes = 64 * GB,
        kind = "cpu",
    )

    assert report["usable_bytes"] == 40 * GB
    # The sentence used to read "This GPU has 40.0 GB of its own memory" for a CPU.
    assert "GPU" not in report["notes"]
    assert "CPU" in report["notes"]
    assert "40.0 GB" in report["notes"]


def test_the_carveout_follows_the_gpu_the_target_names():
    gpus = [
        {"index": 0, "integrated": True, "memory_bytes": int(0.5 * GB)},
        {"index": 1, "integrated": True, "memory_bytes": 2 * GB},
    ]

    # Two integrated parts: the first one's reservation must not size a run aimed at the second.
    assert ab.carveout_bytes(gpus) == int(0.5 * GB)
    assert ab.carveout_bytes(gpus, index = 1) == 2 * GB
    assert ab.carveout_bytes(gpus, index = 2) is None
