# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The compute target becomes the worker's environment before torch loads.

Pinned here because these values are read at import time: a test that only checked the mapping
function would pass while the run still imported torch on auto-detect and ignored the user's choice.
The `auto` case is pinned too -- guessing a discrete card for an empty target would set a discrete
card's allocator defaults on an APU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.training import compute_target_env as env_mod  # noqa: E402


def test_a_cpu_target_pins_the_library_to_the_cpu():
    environ = {}

    applied = env_mod.apply_compute_target_environment({"compute_target": "cpu"}, environ = environ)

    assert environ["UNSLOTH_TRAINING_DEVICE"] == "cpu"
    assert applied["UNSLOTH_TRAINING_DEVICE"] == "cpu"
    # Compile has no payoff on a decoder on the CPU, and the switches must be set before torch.
    assert environ["TORCHDYNAMO_DISABLE"] == "1"


def test_an_npu_target_pins_the_library_to_the_npu():
    environ = {}

    env_mod.apply_compute_target_environment({"compute_target": "npu"}, environ = environ)

    assert environ["UNSLOTH_TRAINING_DEVICE"] == "npu"
    assert environ["UNSLOTH_NPU_DEVICE"] == "1"


def test_an_igpu_target_carries_its_shared_memory_flag():
    environ = {}

    env_mod.apply_compute_target_environment(
        {"compute_target": "igpu", "compute_target_vendor": "intel"},
        environ = environ,
    )

    assert environ["UNSLOTH_TRAINING_DEVICE"] == "igpu"
    assert environ["UNSLOTH_IGPU"] == "1"
    assert environ["UNSLOTH_SKIP_PIN_MEMORY"] == "1"


def test_a_discrete_target_overrides_nothing():
    environ = {}

    applied = env_mod.apply_compute_target_environment(
        {"compute_target": "dgpu", "compute_target_vendor": "nvidia"},
        environ = environ,
    )

    # Auto-detection already picks the discrete card; a forced "cuda" would break an AMD host where
    # that same card reports as "hip".
    assert "UNSLOTH_TRAINING_DEVICE" not in environ
    assert applied == {}


def test_auto_sets_nothing_at_all():
    environ = {}

    for target in ("auto", None, "", "   "):
        assert env_mod.apply_compute_target_environment({"compute_target": target}, environ = environ) == {}

    assert environ == {}


def test_a_user_set_override_is_not_replaced():
    environ = {"UNSLOTH_TRAINING_DEVICE": "xpu"}

    env_mod.apply_compute_target_environment({"compute_target": "cpu"}, environ = environ)

    # The environment the job was launched with wins over the request, so a debugging override
    # survives a config that says otherwise.
    assert environ["UNSLOTH_TRAINING_DEVICE"] == "xpu"


def test_allocation_defaults_still_respect_a_user_value():
    environ = {"PYTORCH_XPU_ALLOC_CONF": "max_split_size_mb:256"}

    env_mod.apply_compute_target_environment(
        {"compute_target": "igpu", "compute_target_vendor": "intel"},
        environ = environ,
    )

    assert environ["PYTORCH_XPU_ALLOC_CONF"] == "max_split_size_mb:256"


def test_a_resolved_device_string_names_its_kind():
    assert env_mod.target_kind("xpu:0") == "dgpu"
    assert env_mod.target_kind("cuda:1") == "dgpu"
    assert env_mod.target_kind("npu:0") == "npu"
    assert env_mod.target_kind("cpu") == "cpu"
    assert env_mod.target_kind("IGPU") == "igpu"
    assert env_mod.target_kind("auto") is None
    assert env_mod.target_kind(None) is None
    assert env_mod.target_kind("vulkan") is None


def test_an_integrated_accelerator_named_as_a_device_string_is_treated_as_integrated():
    environ = {}

    # An older queued job stores the resolved device string, and the flag is what makes it an APU.
    env_mod.apply_compute_target_environment(
        {
            "compute_target": "xpu:0",
            "compute_target_vendor": "intel",
            "compute_target_integrated": True,
        },
        environ = environ,
    )

    assert environ["UNSLOTH_IGPU"] == "1"


def test_a_missing_budget_module_degrades_instead_of_failing_the_run():
    environ = {}

    real_import = __import__

    def refuse(name, *args, **kwargs):
        if name == "utils.hardware.accelerator_budget":
            raise ImportError("budget module unavailable in this install")
        return real_import(name, *args, **kwargs)

    import builtins

    builtins_import = builtins.__import__
    builtins.__import__ = refuse
    try:
        applied = env_mod.apply_compute_target_environment(
            {"compute_target": "cpu"}, environ = environ
        )
    finally:
        builtins.__import__ = builtins_import

    # The device override still happened: that is the part the run cannot do without.
    assert environ["UNSLOTH_TRAINING_DEVICE"] == "cpu"
    assert applied["__error__"]


def test_importing_the_module_writes_nothing():
    # The parent process imports this module to describe a queued job; it must not acquire a device
    # override on behalf of a job it is not running. A fresh interpreter is the only way to see the
    # import's own effects, since this process has already imported it.
    import os
    import subprocess

    code = (
        "import os, sys; sys.path.insert(0, %r); "
        "import core.training.compute_target_env; "
        "print(os.environ.get('UNSLOTH_TRAINING_DEVICE', '<unset>'))"
        % str(_BACKEND)
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_TRAINING_DEVICE"},
        timeout = 120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "<unset>"


def test_the_setup_line_names_the_device_and_its_cost():
    assert "CPU" in env_mod.describe_compute_target({"compute_target": "cpu"})
    # The CPU sentence carries the warning: an hour and two days is the difference it names.
    assert "slower" in env_mod.describe_compute_target({"compute_target": "cpu"}) or (
        "CPU-scale" in env_mod.describe_compute_target({"compute_target": "cpu"})
    )
    assert "integrated GPU" in env_mod.describe_compute_target({"compute_target": "igpu"})
    assert "discrete GPU" in env_mod.describe_compute_target(
        {"compute_target": "dgpu", "compute_target_vendor": "nvidia"}
    )
    assert env_mod.describe_compute_target({}).startswith("Compute target: auto")
