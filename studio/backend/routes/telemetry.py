# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Telemetry settings and the manual send.

Owner-only, because the switch is installation-wide and the preview contains
model and dataset identifiers from every account on the machine. Mutations also
require an interactive UI session: an API key should not be able to turn
reporting on or push a report by hand.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from routes.provider_credentials import require_ui_session
from utils import telemetry_settings

router = APIRouter()


async def _owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


class TelemetryUpdate(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: Optional[bool] = None
    endpoint: Optional[str] = Field(None, max_length = telemetry_settings.MAX_ENDPOINT_CHARS)
    ingest_key: Optional[str] = Field(None, max_length = telemetry_settings.MAX_INGEST_KEY_CHARS)
    interval_hours: Optional[int] = Field(
        None,
        ge = telemetry_settings.MIN_INTERVAL_HOURS,
        le = telemetry_settings.MAX_INTERVAL_HOURS,
    )
    reset_install_id: bool = False


@router.get("/settings", dependencies = [Depends(_owner)])
def get_settings() -> dict[str, Any]:
    try:
        return telemetry_settings.public_settings()
    except Exception as error:
        raise HTTPException(
            status_code = 500,
            detail = f"Could not read telemetry settings: {type(error).__name__}",
        ) from error


@router.put("/settings", dependencies = [Depends(_owner)])
def update_settings(
    payload: TelemetryUpdate,
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> dict[str, Any]:
    require_ui_session(via_api_key)
    try:
        return telemetry_settings.update_telemetry_settings(
            enabled = payload.enabled,
            endpoint = payload.endpoint,
            ingest_key = payload.ingest_key,
            interval_hours = payload.interval_hours,
            reset_id = payload.reset_install_id,
        )
    except ValueError as error:
        raise HTTPException(status_code = 400, detail = str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code = 500,
            detail = f"Could not save telemetry settings: {type(error).__name__}",
        ) from error


@router.get("/preview", dependencies = [Depends(_owner)])
def preview_report() -> dict[str, Any]:
    """Exactly what the next report would contain, built fresh and not sent."""
    from core.telemetry.snapshot import build_snapshot

    try:
        settings = telemetry_settings.get_telemetry_settings()
        return build_snapshot(
            settings["install_id"],
            window_days = telemetry_settings.MAX_WINDOW_DAYS,
        )
    except Exception as error:
        raise HTTPException(
            status_code = 500,
            detail = f"Could not build the preview: {type(error).__name__}",
        ) from error


@router.post("/send", dependencies = [Depends(_owner)])
def send_now(
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> dict[str, Any]:
    require_ui_session(via_api_key)
    from core.telemetry.reporter import send_snapshot

    return send_snapshot(force = True)
