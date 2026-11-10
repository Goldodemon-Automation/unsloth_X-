# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The registry's capabilities must answer the loader's question, not a cheaper one.

Two dicts describe one device: `unsloth.device_type.device_capabilities()` (the loader's view) and
`core.training.device_targets._capabilities()` (the UI's view). They share `CAPABILITY_KEYS` on
purpose, and the user-visible risk is a 4-bit option offered on a host whose bitsandbytes has no
working native kernels: the wizard promises the path and the load then refuses it. These tests pin
the shared keys, the loader-owned 4-bit gate, and the one deliberate difference -- the NPU row
describes *training* while the library's dict describes the device -- so a later edit cannot drift
them apart again.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.training.device_targets import (  # noqa: E402
    TargetProbe,
    list_training_targets,
)

_REPO_ROOT = _BACKEND.parents[1]
_DEVICE_TYPE_PATH = _REPO_ROOT / "unsloth" / "device_type.py"
GB = 1024**3


def _capability_keys() -> tuple:
    """Read CAPABILITY_KEYS without importing `unsloth`, which imports torch."""
    tree = ast.parse(_DEVICE_TYPE_PATH.read_text(encoding = "utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "CAPABILITY_KEYS":
                return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"CAPABILITY_KEYS is missing from {_DEVICE_TYPE_PATH}")


CAPABILITY_KEYS = _capability_keys()


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


def _npu() -> dict:
    return {
        "present": True,
        "enabled": True,
        "supported": True,
        "primary": {"name": "AMD XDNA2", "vendor": "amd"},
    }


def _fake_loader(monkeypatch, *, allow_bitsandbytes: bool) -> None:
    """Install a stand-in `unsloth.device_type` carrying only the gate the registry reads."""
    package = types.ModuleType("unsloth")
    package.__path__ = []
    module = types.ModuleType("unsloth.device_type")
    module.ALLOW_BITSANDBYTES = allow_bitsandbytes
    monkeypatch.setitem(sys.modules, "unsloth", package)
    monkeypatch.setitem(sys.modules, "unsloth.device_type", module)


def _all_rows() -> list:
    probe = _probe(
        gpus = [
            _gpu(),
            _gpu(
                index = 0,
                name = "Intel(R) Arc(TM) Graphics",
                vendor = "intel",
                integrated = True,
                memory_bytes = int(0.5 * GB),
                unified = True,
            ),
        ],
        npu = _npu(),
    )
    return list_training_targets(probe)


def _capabilities_for(probe: TargetProbe, kind: str) -> dict:
    return [target for target in list_training_targets(probe) if target.kind == kind][0].capabilities


def test_every_device_row_carries_exactly_the_library_capability_keys():
    rows = _all_rows()

    assert {target.kind for target in rows} == {"cpu", "igpu", "dgpu", "npu"}
    for target in rows:
        assert tuple(target.capabilities) == CAPABILITY_KEYS


def test_a_host_whose_loader_refuses_four_bit_is_not_offered_four_bit(monkeypatch):
    # The regression: find_spec says the package is installed, but its native kernels are dead, so
    # the loader clears ALLOW_BITSANDBYTES and refuses a 4-bit load mid-run.
    _fake_loader(monkeypatch, allow_bitsandbytes = False)
    probe = _probe(gpus = [_gpu()], bnb_available = True)

    caps = _capabilities_for(probe, "dgpu")

    assert caps["bnb4bit"] is False


def test_a_host_with_working_native_kernels_is_offered_four_bit(monkeypatch):
    _fake_loader(monkeypatch, allow_bitsandbytes = True)
    probe = _probe(gpus = [_gpu()], bnb_available = True)

    caps = _capabilities_for(probe, "dgpu")

    assert caps["bnb4bit"] is True


def test_gradient_checkpointing_is_a_training_answer_not_a_device_answer():
    rows = {target.kind: target.capabilities["grad_checkpoint"] for target in _all_rows()}

    # An NPU has autograd as a device but no backward pass a training run can use, so the training
    # answer is False even though the library's per-device dict says True.
    assert rows["npu"] is False
    assert rows["cpu"] is True
    assert rows["igpu"] is True
    assert rows["dgpu"] is True


def test_every_capability_value_is_a_flat_boolean():
    for target in _all_rows():
        for key, value in target.capabilities.items():
            # The UI renders these as availability, so None or a string would silently read as
            # "available" in some clients and "unavailable" in others.
            assert type(value) is bool, f"{target.kind}.{key} is {value!r}"
