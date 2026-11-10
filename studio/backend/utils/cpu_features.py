# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which CPU features this host advertises, and whether any of them make bf16 genuinely fast.

On most x86 CPUs bf16 has no arithmetic of its own: torch widens to fp32, does the math, and
narrows again, so a bf16 run is slower than fp32 while halving the memory. That is why the CPU
training plan defaults to fp32. Two instruction sets change the answer -- AMX-BF16 (Sapphire
Rapids and newer) and AVX512-BF16 (Cooper Lake / Ice Lake and newer) -- and so does Apple
Silicon, where the vector units handle bf16 natively. There, bf16 is both faster than fp32 and
half the memory.

Everything here is stdlib-only and reads no host state unless asked: the feature words come from
an injected ``/proc/cpuinfo`` text in tests, and a function that cannot measure the host says so
rather than guessing.
"""

from __future__ import annotations

import platform as _host_platform
import sys
from typing import Iterable, Optional

_CPUINFO_PATH = "/proc/cpuinfo"
# x86 kernels spell the feature line "flags", arm64 kernels spell it "Features".
_CPUINFO_FLAG_KEYS = frozenset(("flags", "features"))

# Sapphire Rapids (4th gen Xeon Scalable) introduced AMX-BF16; Emerald and Granite Rapids keep it.
_AMX_BF16 = "amx_bf16"
# AVX512-BF16 arrived with Cooper Lake and Ice Lake, but it is only usable beside the AVX512
# foundation set, so the pair is what counts. The flag alone can appear on a CPU whose kernel
# masked AVX512F off (virtualized hosts mostly), and trusting it there is the wrong answer.
_AVX512_BF16 = "avx512_bf16"
_AVX512_F = "avx512f"


def cpu_flags(
    platform: str | None = None,
    cpuinfo_text: str | None = None,
) -> frozenset[str]:
    """The CPU's feature words, lowercased, from Linux's ``/proc/cpuinfo`` or from injected text.

    ``cpuinfo_text`` is the injection point for tests, so they never depend on the host's CPU.
    Without it, only Linux is read. Windows and macOS have no equivalent plain-text feature
    list that is worth a subprocess at plan time, so both return an empty frozenset -- that means
    the features are *unmeasurable*, not that the CPU has none, and callers must treat it as
    "unknown" rather than "absent".
    """
    target = sys.platform if platform is None else platform
    text = cpuinfo_text
    if text is None:
        if not (isinstance(target, str) and target.startswith("linux")):
            return frozenset()
        try:
            with open(_CPUINFO_PATH, "r", encoding = "utf-8", errors = "replace") as handle:
                text = handle.read()
        except OSError:
            return frozenset()
    return _flags_from_cpuinfo(text)


def _flags_from_cpuinfo(cpuinfo_text: str) -> frozenset[str]:
    """Union the feature words of every ``flags`` / ``Features`` line in a cpuinfo dump."""
    flags: set[str] = set()
    for line in cpuinfo_text.splitlines():
        key, separator, value = line.partition(":")
        if not separator or key.strip().lower() not in _CPUINFO_FLAG_KEYS:
            continue
        # Multi-socket and SMT dumps repeat the line per processor; the union is the host's set.
        flags.update(word.lower() for word in value.split())
    return frozenset(flags)


def _feature_words(flags: Optional[Iterable[str]]) -> frozenset[str]:
    """Normalise a caller's flags: None and a raw cpuinfo line are both handled, not raised on."""
    if flags is None:
        return frozenset()
    # A caller that passed the whole "flags : ..." value rather than a collection gets its words.
    if isinstance(flags, str):
        return frozenset(word.lower() for word in flags.split())
    try:
        return frozenset(str(flag).lower() for flag in flags)
    except TypeError:  # not iterable: an unmeasurable CPU, the caller's mistake stays harmless
        return frozenset()


def _is_apple_silicon(platform: Optional[str], machine: Optional[str]) -> bool:
    target_platform = sys.platform if platform is None else platform
    target_machine = _host_platform.machine() if machine is None else machine
    return target_platform == "darwin" and (target_machine or "").lower() in ("arm64", "aarch64")


def supports_fast_bf16(
    flags: Optional[Iterable[str]],
    platform: str | None = None,
    machine: str | None = None,
) -> bool:
    """Whether bf16 arithmetic runs in hardware on this CPU, making a bf16 run the faster one.

    True for AMX-BF16 (Sapphire Rapids, 4th gen Xeon Scalable, and newer: Emerald and Granite
    Rapids), for AVX512-BF16 together with AVX512F (Cooper Lake and Ice Lake, and newer), and
    for Apple Silicon (M1 and newer, ``darwin`` + ``arm64``). AVX512-BF16 alone is False: without
    AVX512F the instructions cannot execute. Everything else is False, including the unmeasurable
    empty flag set.
    """
    if _is_apple_silicon(platform, machine):
        return True
    features = _feature_words(flags)
    if _AMX_BF16 in features:
        return True
    return _AVX512_BF16 in features and _AVX512_F in features


def bf16_summary(
    flags: Optional[Iterable[str]],
    platform: str | None = None,
    machine: str | None = None,
) -> str:
    """One sentence a log line or the UI can show verbatim about this CPU's bf16 capability.

    Never raises: the sentence is decoration beside a training plan, so an unmeasurable or
    malformed flag set degrades to "could not be determined" instead of failing the plan. The
    wording avoids underscores so it reads as prose, not as a flag name.
    """
    try:
        features = _feature_words(flags)
        if _is_apple_silicon(platform, machine):
            return "This CPU is Apple Silicon, whose vector units run bf16 natively."
        if _AMX_BF16 in features:
            return "This CPU has AMX-BF16, so bf16 training runs on the matrix units."
        if _AVX512_BF16 in features:
            if _AVX512_F in features:
                return (
                    "This CPU has AVX512-BF16, so bf16 training runs in hardware instead of "
                    "software emulation."
                )
            return (
                "This CPU advertises AVX512-BF16 without AVX512F, so bf16 cannot run in "
                "hardware and training stays on fp32."
            )
        if not features:
            # Empty is "unmeasurable", not "none": Windows and macOS reach this branch.
            return (
                "This CPU's feature flags could not be measured, so bf16 training keeps the "
                "software path and fp32 stays the default."
            )
        return (
            "This CPU has no bf16 instructions, so bf16 would be emulated in software and fp32 "
            "is the faster choice."
        )
    except Exception:  # noqa: BLE001 -- a log line must never break a training plan
        return "This CPU's bf16 capability could not be determined."
