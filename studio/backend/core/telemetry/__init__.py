# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in telemetry: the local snapshot and the reporter that sends it."""

from core.telemetry.snapshot import build_snapshot

__all__ = ["build_snapshot"]
