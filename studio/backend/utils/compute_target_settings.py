# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The training compute target the user picked, remembered across reloads.

Kept in the installation's app settings rather than in the browser so a job started from the CLI,
another tab or a second machine on the same Studio server runs where the user chose. The value is
the *requested* target ("auto" or one of the registry ids), not a resolved device string: "auto" has
to stay auto, or a laptop that later gains an eGPU would keep training on the CPU it resolved to.

Reading is total -- an unset row, a corrupt row or a stale id all answer "auto" -- because this is
read on every /api/compute/targets poll and a settings row must not be able to break the Train page.
Writing is strict, and validates against the same list the registry resolves, so a typo is refused
at the API boundary instead of silently becoming "auto".
"""

from __future__ import annotations

from typing import Optional

COMPUTE_TARGET_SETTING_KEY = "training_compute_target"

# "auto" first: it is the only value that is not a registry id, and it is the default.
VALID_COMPUTE_TARGETS = ("auto", "cpu", "igpu", "dgpu", "npu")
DEFAULT_COMPUTE_TARGET = "auto"


def normalize_compute_target(value: object) -> Optional[str]:
    """The canonical spelling of a target, or None when it is not one.

    Case and surrounding space are forgiven: the value arrives from a `select`, a URL and a stored
    row, and all three have been seen to disagree about capitalisation.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    return candidate if candidate in VALID_COMPUTE_TARGETS else None


def get_selected_compute_target() -> str:
    """The stored target, or "auto". Never raises."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(COMPUTE_TARGET_SETTING_KEY, None)
    except Exception:
        # A database that cannot answer is the same as no answer: auto-detect.
        return DEFAULT_COMPUTE_TARGET
    return normalize_compute_target(stored) or DEFAULT_COMPUTE_TARGET


def set_selected_compute_target(value: str) -> str:
    """Store a target after validating it. Raises ValueError for anything else."""
    normalized = normalize_compute_target(value)
    if normalized is None:
        raise ValueError(
            f"{value!r} is not a compute target. Use one of: {', '.join(VALID_COMPUTE_TARGETS)}."
        )
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({COMPUTE_TARGET_SETTING_KEY: normalized})
    return normalized
