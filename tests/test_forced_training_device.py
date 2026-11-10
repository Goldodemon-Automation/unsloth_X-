# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""``UNSLOTH_TRAINING_DEVICE`` is how a host without an accelerator gets a real device type.

Two properties matter and both are pinned here. First, the override is the *only* way to reach
``"cpu"``/``"npu"``: an accelerator that fails to probe must still raise, because Unsloth's kernels
assume one exists and a silent CPU fallback replaces a loud failure with a run that is 50-100x
slower. Second, ``igpu`` resolves to whichever accelerator backend is actually usable -- an
integrated GPU differs from a discrete one in memory, not in kernels -- and refuses rather than
quietly training on the CPU when no integrated backend answers.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_TYPE_PATH = REPO_ROOT / "unsloth" / "device_type.py"
CUDA_PROPERTIES = types.SimpleNamespace(
    name = "NVIDIA B200",
    total_memory = 16 * 1024**3,
    major = 10,
    minor = 0,
)


def _fake_torch(
    *,
    properties = CUDA_PROPERTIES,
    hip_version = None,
    cuda_available = False,
    xpu_backend = None,
    npu_backend = None,
    mps_available = None,
):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_available,
        device_count = lambda: 1,
        get_device_properties = lambda _index: properties,
        get_device_name = lambda _index: "",
        empty_cache = lambda: None,
        current_device = lambda: 0,
    )
    torch.version = types.SimpleNamespace(cuda = "12.8", hip = hip_version, xpu = "2026.1")
    if xpu_backend is not None:
        torch.xpu = xpu_backend
    if npu_backend is not None:
        torch.npu = npu_backend
    if mps_available is not None:
        torch.backends = types.SimpleNamespace(
            mps = types.SimpleNamespace(is_available = lambda: mps_available)
        )
    return torch


def _load_device_type(
    monkeypatch,
    torch_module,
    *,
    forced = None,
    allow_cpu = False,
    mlx_available = False,
):
    """Import `unsloth/device_type.py` against a stand-in torch, as the module's own tests do."""
    for name, value in (
        ("UNSLOTH_TRAINING_DEVICE", forced),
        ("UNSLOTH_ALLOW_CPU", "1" if allow_cpu else None),
    ):
        if value is None:
            monkeypatch.delenv(name, raising = False)
        else:
            monkeypatch.setenv(name, value)

    package_name = "_forced_device_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(DEVICE_TYPE_PATH.parent)]
    monkeypatch.setitem(sys.modules, package_name, package)

    bnb_availability = types.ModuleType(f"{package_name}.bnb_availability")
    bnb_availability.native_kernels_ready = lambda *_args, **_kwargs: True
    monkeypatch.setitem(sys.modules, bnb_availability.__name__, bnb_availability)

    zoo = types.ModuleType("unsloth_zoo")
    zoo.__path__ = []
    zoo_utils = types.ModuleType("unsloth_zoo.utils")
    zoo_utils.Version = Version
    zoo_mlx = types.ModuleType("unsloth_zoo.mlx")
    zoo_mlx.is_mlx_available = lambda: mlx_available
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.utils", zoo_utils)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx", zoo_mlx)

    bitsandbytes = types.ModuleType("bitsandbytes")
    bitsandbytes.__version__ = "0.49.2"
    monkeypatch.setitem(sys.modules, "bitsandbytes", bitsandbytes)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    module_name = f"{package_name}.device_type"
    spec = importlib.util.spec_from_file_location(module_name, DEVICE_TYPE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_cpu_override_reports_cpu_on_a_host_with_no_accelerator(monkeypatch):
    device_type = _load_device_type(monkeypatch, _fake_torch(), forced = "cpu")

    assert device_type.DEVICE_TYPE == "cpu"
    assert device_type.DEVICE_TYPE_TORCH == "cpu"
    # One device and one process: there is no second CPU to shard a job across.
    assert device_type.DEVICE_COUNT == 1


def test_npu_override_selects_the_npu_even_beside_a_usable_cuda(monkeypatch):
    # The point of the override is that the user's choice wins over probing order; otherwise a
    # laptop with an eGPU could never put a small job on its NPU.
    npu = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        get_device_properties = lambda _index: types.SimpleNamespace(name = "XDNA2"),
        empty_cache = lambda: None,
        current_device = lambda: 0,
    )
    device_type = _load_device_type(
        monkeypatch,
        _fake_torch(cuda_available = True, npu_backend = npu),
        forced = "npu",
    )

    assert device_type.DEVICE_TYPE == "npu"
    assert device_type.DEVICE_COUNT == 1


def test_igpu_override_resolves_to_the_integrated_accelerator(monkeypatch):
    xpu = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        get_device_properties = lambda _index: types.SimpleNamespace(
            name = "Intel Arc Graphics", total_memory = 8 * 1024**3
        ),
        empty_cache = lambda: None,
        current_device = lambda: 0,
    )
    device_type = _load_device_type(monkeypatch, _fake_torch(xpu_backend = xpu), forced = "igpu")

    # Same kernels as a discrete card: the difference is the shared memory budget, which the
    # caller sizes for, so the device type stays the accelerator's own.
    assert device_type.DEVICE_TYPE == "xpu"


def test_igpu_override_refuses_instead_of_falling_back_to_cpu(monkeypatch):
    with pytest.raises(RuntimeError, match = "no integrated GPU backend is usable"):
        _load_device_type(monkeypatch, _fake_torch(), forced = "igpu")

    # The message has to name the ways out, or the user is left with a run that will not start.
    with pytest.raises(RuntimeError, match = "UNSLOTH_TRAINING_DEVICE=cpu"):
        _load_device_type(monkeypatch, _fake_torch(), forced = "igpu")


def test_an_unknown_override_names_the_accepted_values(monkeypatch):
    with pytest.raises(ValueError) as caught:
        _load_device_type(monkeypatch, _fake_torch(), forced = "vulkan")

    message = str(caught.value)
    assert "vulkan" in message
    for accepted in ("cpu", "igpu", "npu", "cuda", "xpu", "hip", "mps"):
        assert accepted in message


def test_without_the_override_a_missing_accelerator_still_raises(monkeypatch):
    # Unpinned on purpose: this is the upstream contract the override must not weaken.
    with pytest.raises(NotImplementedError, match = "does not currently work on this device"):
        _load_device_type(monkeypatch, _fake_torch())


def test_auto_and_blank_spellings_mean_auto_detect(monkeypatch):
    for spelling in ("auto", "", "  AUTO  "):
        with pytest.raises(NotImplementedError):
            _load_device_type(monkeypatch, _fake_torch(), forced = spelling)


def test_the_override_is_case_and_space_insensitive(monkeypatch):
    device_type = _load_device_type(monkeypatch, _fake_torch(), forced = "  CPU  ")

    assert device_type.DEVICE_TYPE == "cpu"


def test_cpu_capabilities_are_fp32_only(monkeypatch):
    device_type = _load_device_type(monkeypatch, _fake_torch(), forced = "cpu")

    caps = device_type.device_capabilities()

    assert set(caps) == set(device_type.CAPABILITY_KEYS)
    assert caps["bf16"] is False
    assert caps["fp16"] is False
    assert caps["bnb4bit"] is False
    assert caps["triton"] is False
    assert caps["flash_attn"] is False
    assert caps["compile"] is False
    # Gradient checkpointing is plain autograd and works anywhere; a CPU run needs it most.
    assert caps["grad_checkpoint"] is True


def test_npu_capabilities_report_no_usable_training_path(monkeypatch):
    npu = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        get_device_properties = lambda _index: types.SimpleNamespace(name = "XDNA2"),
        empty_cache = lambda: None,
        current_device = lambda: 0,
    )
    device_type = _load_device_type(monkeypatch, _fake_torch(npu_backend = npu), forced = "npu")

    caps = device_type.device_capabilities()

    # Nothing an NPU vendor exposes today runs a backward pass, so every training feature is off.
    assert caps["bnb4bit"] is False
    assert caps["triton"] is False
    assert caps["compile"] is False
    assert caps["flash_attn"] is False


def test_rdna2_bf16_is_withheld_on_hip(monkeypatch):
    properties = types.SimpleNamespace(
        name = "AMD Radeon Graphics",
        total_memory = 8 * 1024**3,
        gcnArchName = "gfx1036:sramecc+:xnack-",
    )
    device_type = _load_device_type(
        monkeypatch,
        _fake_torch(properties = properties, hip_version = "6.3", cuda_available = True),
    )

    caps = device_type.device_capabilities()

    # gfx10 claims bf16 and then dies in Triton's dot with no Python exception (issue 7922).
    assert caps["bf16"] is False
    assert caps["fp16"] is True


def test_capabilities_accept_an_explicit_device_type(monkeypatch):
    device_type = _load_device_type(monkeypatch, _fake_torch(cuda_available = True), forced = "cpu")

    # The registry describes targets the process is not running on, so the argument must win over
    # the module's own DEVICE_TYPE rather than being ignored.
    assert device_type.device_capabilities("cpu")["triton"] is False
    assert device_type.device_capabilities("cuda")["triton"] is True
