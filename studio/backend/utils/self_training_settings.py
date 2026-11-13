# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted owner switch for letting models launch training jobs.

This is deliberately an installation-wide setting, like the Helper pre-cache
switch: the server must refuse the tool itself, not just hide it from the
Settings page. The tool also checks the acting account is the owner, so a
managed account can never inherit the switch.
"""

from __future__ import annotations

from typing import Any

SELF_TRAINING_SETTING_KEY = "models_can_train_other_models"
DEFAULT_SELF_TRAINING_ENABLED = False


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def get_self_training_enabled() -> bool:
    """Read the persisted switch, defaulting to off on any storage error."""
    try:
        from storage.studio_db import get_app_setting

        stored = get_app_setting(SELF_TRAINING_SETTING_KEY, None)
    except Exception:
        stored = None
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_SELF_TRAINING_ENABLED


def set_self_training_enabled(value: Any) -> bool:
    """Persist whether models may launch training jobs from a tool call."""
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("Model self-training must be true or false.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({SELF_TRAINING_SETTING_KEY: parsed})
    return parsed
