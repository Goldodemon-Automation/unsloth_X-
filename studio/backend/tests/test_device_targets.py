# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The compute-target registry: what each device can do, which one wins, and why.

Two properties are load-bearing. A CPU-only host must resolve a real target instead of raising --
that is what makes CPU training representable at all -- and an NPU must never be the automatic
choice, because it cannot run a backward pass and every automatic run would start by refusing
itself. The rest of the file pins the capability rules that the fit check and the UI both read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.training.device_targets import (  # noqa: E402
    TargetProbe,
    TrainingTargetError,
    default_training_target,
    list_training_targets,
    looks_integrated,
    resolve_training_target,
    training_target_payload,
)

GB = 1024**3


def _probe(**overrides) -> TargetProbe:
    base = dict(
        cpu_name = "AMD Ryzen 9 7950X",
        cpu_cores = 32,
        system_memory_bytes = 64 * GB,
        torch_available = True,
        torch_device = "cpu",
        gpus = [],
        npu = None,
        bnb_available = False,
        triton_available = False,
        flash_attn_available = False,
    )
    base.update(overrides)
    return TargetProbe(**base)


def _gpu(**overrides) -> dict:
    base = dict(
        index = 0,
        name = "NVIDIA GeForce RTX 4090",
        vendor = "nvidia",
        integrated = False,
        memory_bytes = 24 * GB,
        arch = "sm_89",
        unified = False,
    )
    base.update(overrides)
    return base


def test_a_cpu_only_host_gets_one_supported_target_and_it_is_the_default():
    probe = _probe(torch_available = False, torch_device = None)

    targets = list_training_targets(probe)

    assert [target.id for target in targets] == ["cpu"]
    assert targets[0].supported is True
    assert default_training_target(probe).id == "cpu"


def test_a_cpu_only_host_now_resolves_cpu_instead_of_raising():
    # The regression this module exists for: before it, a host with no accelerator had no way to
    # name the device it was about to train on.
    probe = _probe(torch_available = False, torch_device = None)

    assert resolve_training_target("cpu", probe).device == "cpu"
    assert resolve_training_target("auto", probe).kind == "cpu"
    assert resolve_training_target(None, probe).kind == "cpu"


def test_cpu_capabilities_are_fp32_only():
    target = list_training_targets(_probe())[0]

    assert target.capabilities["bf16"] is False
    assert target.capabilities["bnb4bit"] is False
    assert target.capabilities["triton"] is False
    assert target.capabilities["grad_checkpoint"] is True
    assert target.recommended["dtype"] == "float32"
    # A CPU run's ceiling is system RAM.
    assert target.memory_bytes == 64 * GB


def test_an_integrated_gpu_sorts_before_a_discrete_one():
    probe = _probe(
        gpus = [
            _gpu(index = 0, name = "NVIDIA GeForce RTX 4090"),
            _gpu(
                index = 0,
                name = "Intel(R) Arc(TM) Graphics",
                vendor = "intel",
                integrated = True,
                memory_bytes = int(0.5 * GB),
                unified = True,
            ),
        ]
    )

    kinds = [target.kind for target in list_training_targets(probe)]

    assert kinds == ["cpu", "igpu", "dgpu"]


def test_an_integrated_intel_part_is_xpu_and_has_no_four_bit():
    probe = _probe(
        torch_device = "xpu",
        gpus = [
            _gpu(
                name = "Intel(R) Arc(TM) Graphics",
                vendor = "intel",
                integrated = True,
                memory_bytes = int(0.5 * GB),
                unified = True,
            )
        ],
    )

    target = resolve_training_target("igpu", probe)

    assert target.device == "xpu:0"
    assert target.capabilities["bnb4bit"] is False
    assert target.capabilities["bf16"] is True
    assert target.recommended["dtype"] == "bfloat16"


def test_an_amd_apu_gets_the_shared_memory_budget_not_the_carveout():
    probe = _probe(
        torch_device = "hip",
        gpus = [
            _gpu(
                name = "AMD Radeon(TM) Graphics",
                vendor = "amd",
                integrated = True,
                memory_bytes = int(0.5 * GB),
                arch = "gfx1036:sramecc+",
                unified = True,
            )
        ],
    )

    target = resolve_training_target("igpu", probe)

    # 60% of 64 GB, not the 512 MB the driver reserved: the reservation is the floor.
    assert target.memory_bytes == int(64 * GB * 0.6)
    # gfx103x claims bf16 it cannot run, so the recommendation drops to fp16.
    assert target.capabilities["bf16"] is False
    assert target.recommended["dtype"] == "float16"


def test_a_discrete_card_keeps_its_own_memory_and_four_bit():
    probe = _probe(
        torch_device = "cuda",
        gpus = [_gpu()],
        bnb_available = True,
        triton_available = True,
        flash_attn_available = True,
    )

    target = default_training_target(probe)

    assert target.kind == "dgpu"
    assert target.memory_bytes == 24 * GB
    assert target.capabilities["bnb4bit"] is True
    assert target.capabilities["flash_attn"] is True
    assert target.device == "cuda:0"


def test_a_present_npu_is_listed_but_never_supported_or_automatic():
    probe = _probe(
        gpus = [_gpu()],
        bnb_available = True,
        npu = {
            "present": True,
            "enabled": True,
            "supported": True,
            "primary": {"name": "AMD XDNA2", "vendor": "amd"},
        },
    )

    targets = list_training_targets(probe)
    npu = [target for target in targets if target.kind == "npu"]

    assert npu and npu[0].supported is False
    assert "no_backward_pass" in npu[0].reasons
    # The GPU is still the automatic answer.
    assert default_training_target(probe).kind == "dgpu"


def test_asking_for_the_npu_explains_the_refusal():
    probe = _probe(
        npu = {"present": True, "enabled": False, "supported": False, "primary": {"name": "XDNA2"}}
    )

    with pytest.raises(TrainingTargetError) as caught:
        resolve_training_target("npu", probe)

    message = str(caught.value)
    assert "no_backward_pass" in message
    assert "UNSLOTH_STUDIO_ENABLE_NPU" in message


def test_an_absent_npu_is_not_listed_at_all():
    probe = _probe(npu = {"present": False, "enabled": True, "supported": False})

    assert [target.kind for target in list_training_targets(probe)] == ["cpu"]


def test_an_unmeasured_npu_is_not_listed_either():
    # None means "this install cannot read NPUs", which must not render as a device row.
    assert [target.kind for target in list_training_targets(_probe(npu = None))] == ["cpu"]


def test_resolution_accepts_ids_device_strings_and_messy_input():
    probe = _probe(
        torch_device = "xpu",
        gpus = [
            _gpu(
                index = 1,
                name = "Intel(R) Arc(TM) A770 Graphics",
                vendor = "intel",
                integrated = False,
                memory_bytes = 16 * GB,
            )
        ],
    )

    assert resolve_training_target(" DGpu ", probe).kind == "dgpu"
    assert resolve_training_target("xpu:1", probe).kind == "dgpu"
    assert resolve_training_target("AUTO", probe).kind == "dgpu"


def test_an_unknown_target_names_what_is_available():
    probe = _probe()

    with pytest.raises(TrainingTargetError) as caught:
        resolve_training_target("tpu", probe)

    message = str(caught.value)
    assert "tpu" in message
    assert "cpu" in message


def test_an_absent_device_kind_says_the_machine_has_none():
    with pytest.raises(TrainingTargetError, match = "no dgpu device"):
        resolve_training_target("dgpu", _probe())


def test_a_missing_torch_keeps_the_gpu_row_but_marks_it_unsupported():
    probe = _probe(
        torch_available = False,
        torch_device = None,
        gpus = [_gpu(name = "NVIDIA GeForce RTX 3060", vendor = "nvidia", memory_bytes = 12 * GB)],
    )

    target = [item for item in list_training_targets(probe) if item.kind == "dgpu"][0]

    assert target.supported is False
    assert target.reasons == ("torch_missing",)
    # The default falls back to the CPU instead of the unusable card.
    assert default_training_target(probe).kind == "cpu"


def test_the_payload_is_json_serialisable():
    probe = _probe(
        gpus = [_gpu()],
        bnb_available = True,
        npu = {"present": True, "enabled": True, "supported": True, "primary": {"name": "XDNA2"}},
    )

    payload = training_target_payload(probe)

    encoded = json.dumps(payload)
    assert '"default": "dgpu"' in encoded
    assert isinstance(payload["targets"][0]["reasons"], list)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Intel(R) Arc(TM) Graphics", True),
        ("Intel(R) Arc(TM) A770 Graphics", False),
        ("AMD Radeon(TM) Graphics", True),
        ("AMD Radeon RX 7900 XTX", False),
        ("NVIDIA GeForce RTX 4090", False),
        ("Apple M4 Pro", True),
    ],
)
def test_integrated_detection_by_name(name, expected):
    assert looks_integrated(name, vendor = "other") is expected


def test_detect_runs_on_this_host_without_torch():
    probe = TargetProbe.detect()

    targets = list_training_targets(probe)

    assert targets[0].kind == "cpu"
    assert targets[0].supported is True
    # This machine has no torch installed, which is exactly the case the CPU row exists for.
    assert isinstance(probe.torch_available, bool)

def test_a_partial_torch_read_leaves_no_usable_gpu_row():
    # A driver that answers for device 0 and then throws must not leave a dGPU row whose `device` fell
    # through to "cpu" while `supported` still said yes: the fit check would size a card it cannot read.
    probe = _probe(torch_device = None, gpus = [])

    kinds = [target.kind for target in list_training_targets(probe)]

    assert kinds == ["cpu"]
    assert default_training_target(probe).kind == "cpu"


def test_an_intel_discrete_card_is_not_a_unified_memory_device():
    probe = _probe(
        torch_device = "xpu",
        gpus = [
            _gpu(index = 0, name = "Intel(R) Arc(TM) A770 Graphics", vendor = "intel",
                 integrated = False, memory_bytes = 16 * GB, arch = "dg2", unified = False)
        ],
    )

    target = resolve_training_target("dgpu", probe)

    # Only an integrated part shares the host's memory; the payload used to claim both at once.
    assert target.unified_memory is False
    assert target.memory_bytes == 16 * GB


def test_a_cpu_plans_against_free_ram_not_the_installed_total():
    probe = _probe(system_memory_bytes = 64 * GB, system_memory_available_bytes = 40 * GB)

    cpu = list_training_targets(probe)[0]

    assert cpu.memory_bytes == 40 * GB
