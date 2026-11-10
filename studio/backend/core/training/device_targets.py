# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The devices a training run can use, as one explicit list with one honest capability per device.

Before this module the answer was spread across a coarse verdict (`DeviceType` plus a global
`CHAT_ONLY` flag) and a pile of `DEVICE_TYPE == "cuda"` string compares, which had two consequences
worth naming. A host with no accelerator had no *representable* choice at all, so even the CPU recipe
that already existed could not be described, sized or explained. And an integrated GPU was described
as "xpu" or "hip" with no way to say that its memory is shared with the operating system -- the single
fact that decides whether a run fits.

So: enumerate every candidate, state what each can actually execute, and keep the two decisions that
used to be implicit -- which device wins, and what to recommend on it -- in one place that a test can
read. Nothing here loads a weight, starts a runtime or imports torch unless a probe asks it to.

`TargetProbe` is injectable for exactly that reason: every capability below is a pure function of
measured facts, so the tests assert the rules against hand-built hosts instead of this machine.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

TARGET_IDS = ("cpu", "igpu", "dgpu", "npu")

# Starting points, not promises: the fit check lowers the batch before a run starts, and these are
# what the wizard shows before it has measured anything. Kept in one table because three copies of
# "8" in three call sites is how the wizard and the worker came to disagree.
_RECOMMENDED = {
    "cpu": {"batch_size": 1, "grad_accum": 16, "dtype": "float32", "gradient_checkpointing": True},
    "igpu": {"batch_size": 1, "grad_accum": 8, "dtype": "bfloat16", "gradient_checkpointing": True},
    "dgpu": {
        "batch_size": 4,
        "grad_accum": 4,
        "dtype": "bfloat16",
        "gradient_checkpointing": False,
    },
    "npu": {"batch_size": 1, "grad_accum": 1, "dtype": "float32", "gradient_checkpointing": False},
}

# Integrated parts report names like "Intel(R) Arc(TM) Graphics" with no model number, against
# "Intel(R) Arc(TM) A770 Graphics" for a card. Kept as data rather than a regex so a new part is a
# one-line change and the reason it is listed sits beside it.
_DISCRETE_HINTS = (
    "a770",
    "a750",
    "a580",
    "a380",
    "arc pro",
    "b580",
    "b570",
    "radeon rx",
    "radeon pro",
    "geforce",
    "quadro",
    "rtx",
    "gtx",
    "tesla",
    "instinct",
    "radeon vii",
)
_INTEGRATED_HINTS = (
    "iris",
    "uhd graphics",
    "hd graphics",
    "arc(tm) graphics",
    "arc graphics",
    "radeon graphics",
    "vega",
    "mali",
    "adreno",
    "apple m",
)

_VENDOR_HINTS = (
    ("nvidia", ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla", "h100", "a100")),
    ("intel", ("intel", "iris", "uhd", "arc")),
    ("amd", ("amd", "radeon", "instinct", "gfx")),
    ("apple", ("apple",)),
    ("qualcomm", ("qualcomm", "adreno", "snapdragon")),
)


class TrainingTargetError(Exception):
    """A requested target cannot be used here. The message names the target and its reasons."""


@dataclass(frozen = True)
class TrainingTarget:
    """One device, described well enough to size a run on it and to explain the choice."""

    id: str
    device: str
    kind: str
    name: str
    vendor: str
    memory_bytes: Optional[int]
    unified_memory: bool
    supported: bool
    reasons: tuple[str, ...] = ()
    capabilities: dict = field(default_factory = dict)
    recommended: dict = field(default_factory = dict)
    notes: tuple[str, ...] = ()


@dataclass
class TargetProbe:
    """Measured facts, nothing inferred. `detect()` fills it from the real host; tests build it."""

    cpu_name: Optional[str] = None
    cpu_cores: Optional[int] = None
    system_memory_bytes: Optional[int] = None
    # Free RAM beside the installed total: a CPU run competes with the OS for the difference, and the
    # trainer already loads its base against `available` (core/training/trainer.py).
    system_memory_available_bytes: Optional[int] = None
    torch_available: bool = False
    torch_device: Optional[str] = None
    gpus: list = field(default_factory = list)
    npu: Optional[dict] = None
    bnb_available: bool = False
    triton_available: bool = False
    flash_attn_available: bool = False

    @classmethod
    def detect(cls) -> "TargetProbe":
        """Fill a probe from this host. Every step is guarded: an unmeasurable fact stays None.

        Ordered cheapest first, and torch last, because importing it is the expensive part and a
        caller that only needs the CPU row should not pay for it. A probe that cannot import torch
        is a normal host (a planning-only install, a CPU laptop), not an error.
        """
        probe = cls()
        probe.cpu_name = _cpu_name()
        probe.cpu_cores = _cpu_cores()
        probe.system_memory_bytes, probe.system_memory_available_bytes = _system_memory_bytes()
        probe.torch_available = importlib.util.find_spec("torch") is not None
        probe.bnb_available = importlib.util.find_spec("bitsandbytes") is not None
        probe.triton_available = importlib.util.find_spec("triton") is not None
        probe.flash_attn_available = importlib.util.find_spec("flash_attn") is not None
        if probe.torch_available:
            probe.torch_device, probe.gpus = _torch_devices()
        if (
            probe.torch_device is None
            and sys.platform == "darwin"
            and platform.machine() == "arm64"
        ):
            # Apple Silicon: Metal is the accelerator even when torch is not importable here.
            probe.torch_device = "mps"
            probe.gpus = [
                {
                    "index": 0,
                    "name": probe.cpu_name or "Apple Silicon",
                    "vendor": "apple",
                    "integrated": True,
                    "memory_bytes": None,
                    "arch": None,
                    "unified": True,
                }
            ]
        probe.npu = _npu_probe()
        return probe


def _cpu_name() -> Optional[str]:
    try:
        name = platform.processor() or platform.uname().processor
    except Exception:
        return None
    name = (name or "").strip()
    if not name or name.lower() in ("unknown", "arm", "arm64"):
        return None
    return name


def _cpu_cores() -> Optional[int]:
    try:
        count = getattr(os, "process_cpu_count", None)
        if callable(count):
            return count()
        return os.cpu_count()
    except Exception:
        return None


def _system_memory_bytes() -> tuple[Optional[int], Optional[int]]:
    """(installed, available) bytes; either may be None when psutil cannot answer."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:
        return None, None


def _vendor_from_name(name: str) -> str:
    lowered = (name or "").lower()
    for vendor, hints in _VENDOR_HINTS:
        if any(hint in lowered for hint in hints):
            return vendor
    return "other"


def _normalized_gpu_name(name: str) -> str:
    """Lowercase, with the trademark markers the drivers put in every consumer name removed.

    "AMD Radeon(TM) Graphics" and "AMD Radeon Graphics" are the same part, and the hints below are
    matched as substrings, so "(tm)" alone would make an integrated APU read as a discrete card.
    """
    return (name or "").lower().replace("(tm)", " ").replace("(r)", " ").replace("  ", " ")


def looks_integrated(name: str, *, vendor: str = "other") -> bool:
    """Whether a GPU name reads as an integrated part.

    Name heuristics, because no torch API answers this: `torch.cuda` and `torch.xpu` both report a
    name and a memory total and nothing about how that memory is wired. The decisive evidence is the
    memory total itself, which the caller compares against the host's RAM, so this is the one signal
    used only when the totals say nothing.
    """
    lowered = _normalized_gpu_name(name)
    if any(hint in lowered for hint in _DISCRETE_HINTS):
        return False
    if vendor == "apple":
        return True
    return any(hint in lowered for hint in _INTEGRATED_HINTS)


def _torch_devices() -> tuple[Optional[str], list]:
    """The torch device type this host trains on, and one entry per visible accelerator."""
    try:
        import torch
    except Exception:
        return None, []
    gpus: list = []
    try:
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            # ROCm reuses torch.cuda, so the vendor comes from the name, not the namespace.
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                name = str(getattr(properties, "name", "") or "")
                vendor = _vendor_from_name(name)
                gpus.append(
                    {
                        "index": index,
                        "name": name,
                        "vendor": vendor,
                        "integrated": looks_integrated(name, vendor = vendor),
                        "memory_bytes": int(getattr(properties, "total_memory", 0) or 0) or None,
                        "arch": getattr(properties, "gcnArchName", None),
                        "unified": False,
                    }
                )
            return ("hip" if getattr(torch.version, "hip", None) else "cuda"), gpus
        if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            for index in range(torch.xpu.device_count()):
                properties = torch.xpu.get_device_properties(index)
                name = str(getattr(properties, "name", "") or "")
                integrated = looks_integrated(name, vendor = "intel")
                gpus.append(
                    {
                        "index": index,
                        "name": name,
                        "vendor": _vendor_from_name(name) or "intel",
                        "integrated": integrated,
                        # An Arc A770 reports its own memory: only an integrated part shares the
                        # host's. Hardcoding True here made the same payload say both.
                        "unified": integrated,
                        "memory_bytes": int(getattr(properties, "total_memory", 0) or 0) or None,
                        "arch": getattr(properties, "architecture", None),
                    }
                )
            return "xpu", gpus
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is not None and mps.is_available():
            gpus.append(
                {
                    "index": 0,
                    "name": "Apple Silicon GPU",
                    "vendor": "apple",
                    "integrated": True,
                    "memory_bytes": None,
                    "arch": None,
                    "unified": True,
                }
            )
            return "mps", gpus
    except Exception:
        # A wedged driver reports nothing. Returning the devices read before the failure would hand
        # back a dGPU row whose `device` falls through to "cpu" and which the caller still calls
        # supported -- and the fit check would then size a 24 GB card that cannot be read.
        return None, []
    return "cpu", gpus


def _npu_probe() -> Optional[dict]:
    """The NPU aggregate, or None when this install cannot read NPUs at all.

    None and `{"present": False}` mean different things: the first is "not measured", the second is
    "measured and absent". The UI says "Checking…" for the first and "no NPU" for the second.
    """
    try:
        from utils.hardware.npu import detect_npu
        return detect_npu()
    except Exception:
        return None


def _device_string(kind: str, gpu: Optional[dict], torch_device: Optional[str]) -> str:
    """The torch device the worker should use for this target."""
    if kind == "cpu" or gpu is None:
        return "cpu"
    vendor = gpu.get("vendor") or "other"
    index = int(gpu.get("index") or 0)
    if vendor == "apple" or torch_device == "mps":
        return "mps"
    if vendor == "intel" or torch_device == "xpu":
        return f"xpu:{index}"
    if torch_device in ("cuda", "hip"):
        return f"cuda:{index}"
    return "cpu"


def _igpu_budget(gpu: dict, probe: TargetProbe) -> Optional[int]:
    """Shared memory an integrated GPU may plan against.

    The number torch reports for an integrated part is the carve-out the driver reserved, not what it
    can allocate: a Xe part commonly reserves 512 MB and then serves several GB out of GTT. The
    fraction of RAM is the real ceiling, and the reservation is a floor under it.
    """
    from utils.hardware.accelerator_budget import shared_accelerator_budget
    return shared_accelerator_budget(
        probe.system_memory_bytes,
        dedicated_bytes = gpu.get("memory_bytes"),
    )


def _loader_allows_bitsandbytes(probe: TargetProbe) -> bool:
    """Whether the loader would accept a 4-bit load, asked of the loader's own module.

    `find_spec` is not enough: bitsandbytes >= 0.46 resolves its native symbols lazily, so an
    installed wheel with dead kernels imports cleanly and only raises when a kernel is called -- in
    the middle of a training step. `ALLOW_BITSANDBYTES` is the flag the loader itself reads, so the
    UI must offer 4-bit exactly when that flag is True.

    A host with no usable `unsloth` import (a planning-only install, the registry's own tests) has
    no loader gate to mirror, so the package probe is the only signal left. A library that imports
    but cannot answer is "no": unmeasured is False, and the cost is a slower path rather than a dead
    run.
    """
    try:
        from unsloth.device_type import ALLOW_BITSANDBYTES
    except ImportError:
        # `unsloth` is absent, or present but cannot import without torch. Either way there is no
        # loader gate to read here, so the probe is the only evidence available.
        return bool(probe.bnb_available)
    except Exception:  # noqa: BLE001 - a library that exists but cannot answer is not a yes
        return False
    return bool(ALLOW_BITSANDBYTES)


def _capabilities(
    kind: str,
    probe: TargetProbe,
    gpu: Optional[dict] = None,
) -> dict:
    """What this device can execute for a *training run*.

    Unmeasured is False: a False costs a slower path, a wrong True kills the run in the middle of a
    step. The `bnb4bit` answer is the loader's own gate rather than a package probe, so the UI cannot
    offer a 4-bit path that `unsloth/models/loader.py` will refuse once it starts.
    """
    caps = {
        "bf16": False,
        "fp16": False,
        "bnb4bit": False,
        "triton": False,
        "grad_checkpoint": kind in ("cpu", "igpu", "dgpu", "mps"),
        "compile": False,
        "flash_attn": False,
    }
    if kind == "cpu":
        # bf16 on a CPU is either emulated in software or needs AVX512-BF16/AMX, which no portable
        # query reports; fp32 is the honest recommendation, and gradient checkpointing is plain
        # autograd, which works anywhere and is what keeps a CPU run inside RAM.
        return caps
    if kind == "npu":
        # `grad_checkpoint` describes *training* capability, and no vendor exposes a backward pass
        # to PyTorch, so it is False here. `unsloth.device_type.device_capabilities()` says True for
        # an NPU because its dict is a per-device statement (autograd exists on the device); the two
        # answer different questions and are not meant to match on this key.
        caps["grad_checkpoint"] = False
        return caps
    arch = str((gpu or {}).get("arch") or "")
    vendor = (gpu or {}).get("vendor") or "other"
    if kind == "igpu":
        caps.update(
            bf16 = vendor == "intel" or (vendor == "amd" and _amd_arch_has_bf16(arch)),
            fp16 = True,
            # bitsandbytes ships no kernels for an integrated part: a 4-bit load there is a CPU
            # dequant per step, which is slower than fp16 and needs the memory it was meant to save.
            bnb4bit = False,
            triton = bool(probe.triton_available) and vendor in ("intel", "amd"),
            compile = False,
            flash_attn = False,
        )
        return caps
    # Discrete.
    caps.update(
        bf16 = _discrete_bf16(vendor, arch),
        fp16 = True,
        # The loader's own gate, not find_spec: an installed-but-dead bitsandbytes imports cleanly
        # and only fails when a native kernel runs, which would be offered here and then refused
        # mid-load. The loader reads ALLOW_BITSANDBYTES, so the UI has to read it too.
        bnb4bit = _loader_allows_bitsandbytes(probe),
        # A known approximation, not an oversight: find_spec proves a Triton package exists, while
        # the loader needs its kernels to work for this arch, and no cheap query answers that
        # without importing it. Only the 4-bit gate has a cheap, authoritative flag to read.
        triton = bool(probe.triton_available),
        compile = True,
        flash_attn = bool(probe.flash_attn_available) and vendor == "nvidia",
    )
    return caps


def _amd_arch_has_bf16(arch: str) -> bool:
    """gfx10 (RDNA 1/2) claims bf16 it lacks and Triton's dot then kills the process (issue 7922)."""
    return bool(arch) and not arch.split(":", 1)[0].strip().lower().startswith("gfx10")


def _discrete_bf16(vendor: str, arch: str) -> bool:
    if vendor == "nvidia":
        # Every NVIDIA card Unsloth trains on is Volta or newer; the pre-Volta cards that are not are
        # kept on fp16 by the loader's own compile gate, so this answer is about the vendor's stack.
        return True
    if vendor == "amd":
        return _amd_arch_has_bf16(arch)
    if vendor == "intel":
        return True
    return False


def _notes_for(kind: str, target_name: str, memory_bytes: Optional[int]) -> tuple[str, ...]:
    if kind == "cpu":
        return (
            "No accelerator is in use: training runs on the CPU and is roughly 50-100x slower "
            "than the same run on a GPU.",
        )
    if kind == "igpu":
        return (
            "This GPU shares system memory with the rest of the machine, so the free memory left "
            "after the operating system is the real limit.",
            "Integrated GPUs do not support 4-bit training; the run uses 16-bit or fp32.",
        )
    if kind == "npu":
        return (
            "NPUs have no backward pass, so a model cannot be trained on one.",
            "An NPU can still help a run by tokenizing data and scoring checkpoints while training "
            "runs on another device.",
        )
    if kind == "dgpu" and memory_bytes:
        return ("This GPU has its own memory, separate from system RAM.",)
    return ()


def list_training_targets(probe: Optional[TargetProbe] = None) -> list:
    """Every candidate on this host, in a stable order: CPU, integrated GPUs, discrete GPUs, NPU.

    Unsupported devices are listed too. "This machine has an NPU it cannot train on" is different
    information from "this machine has no NPU", and a UI that hides the first cannot tell a user why
    NPU training is unavailable to them.
    """
    probe = probe or TargetProbe.detect()
    targets: list = []

    cpu_caps = _capabilities("cpu", probe)
    targets.append(
        TrainingTarget(
            id = "cpu",
            device = "cpu",
            kind = "cpu",
            name = probe.cpu_name or "CPU",
            vendor = _vendor_from_name(probe.cpu_name or "") or "other",
            # Free RAM, not installed: planning against the total is how a CPU run is sized for memory
            # the operating system and the dataset pipeline already hold.
            memory_bytes = probe.system_memory_available_bytes or probe.system_memory_bytes,
            unified_memory = True,
            # The compatibility path, not a fallback: a CPU run is slow but real, and it is the only
            # device every installation has.
            supported = True,
            reasons = (),
            capabilities = cpu_caps,
            recommended = dict(_RECOMMENDED["cpu"]),
            notes = _notes_for("cpu", probe.cpu_name or "CPU", None),
        )
    )

    integrated = [gpu for gpu in probe.gpus if gpu.get("integrated")]
    discrete = [gpu for gpu in probe.gpus if not gpu.get("integrated")]
    for gpu in integrated + discrete:
        kind = "igpu" if gpu.get("integrated") else "dgpu"
        capabilities = _capabilities(kind, probe, gpu)
        recommended = dict(_RECOMMENDED[kind])
        if kind == "igpu" and not capabilities["bf16"]:
            # fp16 is the fallback, not fp32: an integrated part that lacks bf16 still has fp16, and
            # fp32 doubles the one resource the device does not have.
            recommended["dtype"] = "float16" if capabilities["fp16"] else "float32"
        memory = _igpu_budget(gpu, probe) if kind == "igpu" else gpu.get("memory_bytes")
        reasons = (
            () if probe.torch_available or gpu.get("vendor") == "apple" else ("torch_missing",)
        )
        targets.append(
            TrainingTarget(
                id = kind,
                device = _device_string(kind, gpu, probe.torch_device),
                kind = kind,
                name = gpu.get("name") or kind,
                vendor = gpu.get("vendor") or "other",
                memory_bytes = memory,
                unified_memory = bool(gpu.get("unified")) or kind == "igpu",
                supported = not reasons,
                reasons = reasons,
                capabilities = capabilities,
                recommended = recommended,
                notes = _notes_for(kind, gpu.get("name") or kind, memory),
            )
        )

    npu = probe.npu
    if npu is not None and npu.get("present"):
        target, npu_notes = _npu_target(npu)
        targets.append(
            TrainingTarget(
                id = "npu",
                device = "npu:0",
                kind = "npu",
                name = target,
                vendor = str((npu.get("primary") or {}).get("vendor") or "other"),
                memory_bytes = None,
                unified_memory = True,
                supported = False,
                reasons = npu_notes,
                capabilities = _capabilities("npu", probe),
                recommended = dict(_RECOMMENDED["npu"]),
                notes = _notes_for("npu", target, None),
            )
        )
    return targets


def _npu_target(npu: dict) -> tuple[str, tuple[str, ...]]:
    """The NPU row's name and why it cannot train.

    A present NPU is never `supported` for training, and the reason list says which fact is missing:
    the device cannot be enabled, no runtime is installed, or the vendor family is unsupported. All
    three are actionable, in that order of likelihood.
    """
    primary = npu.get("primary") or {}
    name = str(primary.get("name") or "NPU")
    reasons = ["no_backward_pass"]
    if not npu.get("enabled"):
        reasons.append("disabled")
    elif not npu.get("supported"):
        reasons.append(str(npu.get("reason") or "unsupported_family"))
    return name, tuple(reasons)


def default_training_target(probe: Optional[TargetProbe] = None) -> TrainingTarget:
    """The best supported target: a discrete GPU, then an integrated one, then the CPU.

    The NPU is never chosen automatically. It cannot train, so making it the default would mean every
    run starts by refusing itself.
    """
    targets = list_training_targets(probe)
    for kind in ("dgpu", "igpu"):
        for target in targets:
            if target.kind == kind and target.supported:
                return target
    return targets[0]


def resolve_training_target(
    requested: Optional[str], probe: Optional[TargetProbe] = None
) -> TrainingTarget:
    """The target a request names, or a `TrainingTargetError` that says why it cannot be used.

    Accepts "auto"/None, a target id, or the exact device string a previous resolution produced
    ("xpu:0"), because a queued job may have stored either.
    """
    targets = list_training_targets(probe)
    wanted = (requested or "auto").strip().lower() if isinstance(requested, str) else "auto"
    if wanted in ("", "auto"):
        return _default_from(targets)
    for target in targets:
        if wanted in (target.id, target.device.lower()):
            if not target.supported:
                raise TrainingTargetError(
                    f"The {target.id} device '{target.name}' cannot train here: "
                    f"{', '.join(target.reasons)}. " + _reason_hint(target.reasons)
                )
            return target
    available = ", ".join(sorted({target.id for target in targets}))
    if wanted in TARGET_IDS:
        # Named by id but not present: the host has no device of that kind at all.
        raise TrainingTargetError(
            f"This machine has no {wanted} device for training. Available: {available or 'none'}."
        )
    raise TrainingTargetError(
        f"'{requested}' is not a training target. Use one of: auto, {available or 'cpu'}."
    )


def _default_from(targets: list) -> TrainingTarget:
    for kind in ("dgpu", "igpu"):
        for target in targets:
            if target.kind == kind and target.supported:
                return target
    return targets[0]


def _reason_hint(reasons: tuple) -> str:
    hints = {
        "no_backward_pass": "No NPU vendor exposes a training backward pass to PyTorch.",
        "disabled": "Set UNSLOTH_STUDIO_ENABLE_NPU=1 to let Studio use it as an assist device.",
        "absent": "The device is not present.",
        "unsupported_family": "This NPU generation is not supported.",
        "torch_missing": "PyTorch is not installed in this environment.",
        "arch_unknown": "The GPU architecture could not be read, so its precision support is unknown.",
    }
    return " ".join(hints[reason] for reason in reasons if reason in hints)


def training_target_payload(probe: Optional[TargetProbe] = None) -> dict:
    """JSON-ready view for the route, with tuples turned into lists."""
    targets = list_training_targets(probe)
    resolved = _default_from(targets)
    return {
        "targets": [_target_json(target) for target in targets],
        "default": resolved.id,
        "resolved": resolved.id,
    }


def _target_json(target: TrainingTarget) -> dict:
    payload = asdict(target)
    payload["reasons"] = list(target.reasons)
    payload["notes"] = list(target.notes)
    return payload
