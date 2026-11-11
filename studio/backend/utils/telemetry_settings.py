# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings for opt-in usage telemetry.

Telemetry is OFF until the installation owner turns it on. The switch, the
endpoint and the optional ingest token live in ``app_settings`` beside the other
Studio settings, because collection is installation-wide and the server has to
refuse to gather data even when a stale browser still draws the switch as on.

The install id is a random UUID with no relationship to hardware, accounts or
anything else on the machine. It exists only so two reports can be told apart.
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

SETTING_KEY = "telemetry_settings"
INSTALL_ID_KEY = "telemetry_install_id"
STATE_KEY = "telemetry_send_state"

DEFAULT_INTERVAL_HOURS = 6
MIN_INTERVAL_HOURS = 1
MAX_INTERVAL_HOURS = 168
MAX_ENDPOINT_CHARS = 300
MAX_INGEST_KEY_CHARS = 512
MAX_WINDOW_DAYS = 180

DEFAULT_ENDPOINT = os.environ.get("X8_TELEMETRY_ENDPOINT", "").strip()
DEFAULT_INGEST_KEY = os.environ.get("X8_TELEMETRY_INGEST_TOKEN", "").strip()

_INSTALL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$")
_install_lock = threading.Lock()


def _read_stored() -> dict[str, Any]:
    try:
        from storage.studio_db import get_app_setting

        raw = get_app_setting(SETTING_KEY, None)
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _read_state() -> dict[str, Any]:
    try:
        from storage.studio_db import get_app_setting

        raw = get_app_setting(STATE_KEY, None)
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _clamp_interval(value: Any) -> int:
    try:
        hours = int(value)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_HOURS
    return max(MIN_INTERVAL_HOURS, min(MAX_INTERVAL_HOURS, hours))


def normalize_endpoint(value: Any) -> str:
    """An http(s) URL with no embedded credentials, or a ValueError.

    Plain http is allowed so a collector on the same machine can be tested; the
    Settings tab says so beside the field.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if len(text) > MAX_ENDPOINT_CHARS:
        raise ValueError(f"Endpoint must be at most {MAX_ENDPOINT_CHARS} characters.")
    parsed = urlsplit(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Endpoint must be an http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("Endpoint must not carry a username or password.")
    return text


def dashboard_url(endpoint: str) -> str:
    """The console origin implied by a report endpoint, for the Settings link."""
    parsed = urlsplit(endpoint)
    if not parsed.netloc:
        return ""
    path = parsed.path
    suffix = "/api/v1/report"
    if path.endswith(suffix):
        path = path[: -len(suffix)]
    return urlunsplit((parsed.scheme, parsed.netloc, path or "/", "", ""))


def get_install_id() -> str:
    """The stored random install id, creating it once if absent."""
    with _install_lock:
        try:
            from storage.studio_db import get_app_setting, upsert_app_settings

            stored = get_app_setting(INSTALL_ID_KEY, "")
        except Exception:
            stored = ""
        if isinstance(stored, str) and _INSTALL_ID_RE.fullmatch(stored.strip()):
            return stored.strip()
        generated = uuid.uuid4().hex
        try:
            upsert_app_settings({INSTALL_ID_KEY: generated}, read_back = False)
        except Exception:
            # A read-only DB still gets a usable id for this process; a later run
            # tries again rather than refusing to report for a storage hiccup.
            pass
        return generated


def reset_install_id() -> str:
    """Replace the install id, so future reports cannot be joined to past ones."""
    generated = uuid.uuid4().hex
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({INSTALL_ID_KEY: generated}, read_back = False)
    return generated


def get_telemetry_settings() -> dict[str, Any]:
    """The effective settings, including the ingest token. Never send to a client."""
    stored = _read_stored()
    endpoint = stored.get("endpoint")
    if endpoint is None:
        endpoint = DEFAULT_ENDPOINT
    ingest_key = stored.get("ingest_key")
    if ingest_key is None:
        ingest_key = DEFAULT_INGEST_KEY
    state = _read_state()
    return {
        "enabled": bool(stored.get("enabled", False)),
        "endpoint": str(endpoint or ""),
        "ingest_key": str(ingest_key or ""),
        "interval_hours": _clamp_interval(stored.get("interval_hours", DEFAULT_INTERVAL_HOURS)),
        "install_id": get_install_id(),
        "last_sent_at": state.get("last_sent_at"),
        "last_attempt_at": state.get("last_attempt_at"),
        "last_error": state.get("last_error"),
    }


def public_settings() -> dict[str, Any]:
    """The Settings tab's view: the token is replaced by whether one exists."""
    settings = get_telemetry_settings()
    endpoint = settings["endpoint"]
    return {
        "enabled": settings["enabled"],
        "endpoint": endpoint,
        "has_ingest_key": bool(settings["ingest_key"]),
        "interval_hours": settings["interval_hours"],
        "install_id": settings["install_id"],
        "last_sent_at": settings["last_sent_at"],
        "last_attempt_at": settings["last_attempt_at"],
        "last_error": settings["last_error"],
        "dashboard_url": dashboard_url(endpoint),
        "window_days": MAX_WINDOW_DAYS,
    }


def update_telemetry_settings(
    *,
    enabled: Optional[bool] = None,
    endpoint: Optional[str] = None,
    ingest_key: Optional[str] = None,
    interval_hours: Optional[int] = None,
    reset_id: bool = False,
) -> dict[str, Any]:
    """Apply one partial update, then return the public view.

    ``ingest_key=""`` clears the token; ``None`` leaves it alone. Enabling with no
    endpoint is refused rather than silently doing nothing every six hours.
    """
    from storage.studio_db import upsert_app_settings

    stored = dict(_read_stored())
    current_endpoint = stored.get("endpoint")
    if current_endpoint is None:
        current_endpoint = DEFAULT_ENDPOINT

    if endpoint is not None:
        stored["endpoint"] = normalize_endpoint(endpoint)
    if ingest_key is not None:
        if len(ingest_key) > MAX_INGEST_KEY_CHARS:
            raise ValueError(f"Ingest key must be at most {MAX_INGEST_KEY_CHARS} characters.")
        stored["ingest_key"] = ingest_key.strip()
    if interval_hours is not None:
        if not isinstance(interval_hours, int) or isinstance(interval_hours, bool):
            raise ValueError("Interval must be a whole number of hours.")
        if not MIN_INTERVAL_HOURS <= interval_hours <= MAX_INTERVAL_HOURS:
            raise ValueError(
                f"Interval must be between {MIN_INTERVAL_HOURS} and {MAX_INTERVAL_HOURS} hours."
            )
        stored["interval_hours"] = interval_hours
    effective_endpoint = stored.get("endpoint", current_endpoint)
    if stored.get("enabled") and not effective_endpoint:
        raise ValueError("Turn telemetry off before clearing the endpoint.")
    if enabled is not None:
        if enabled and not effective_endpoint:
            raise ValueError("Set an endpoint before turning telemetry on.")
        stored["enabled"] = bool(enabled)
        if enabled:
            # An old failure from a previous endpoint should not follow the new one.
            state = _read_state()
            state.pop("last_error", None)
            upsert_app_settings({STATE_KEY: state}, read_back = False)

    upsert_app_settings({SETTING_KEY: stored}, read_back = False)
    if reset_id:
        reset_install_id()
    return public_settings()


def record_send(ok: bool, error: str = "", *, now: Optional[int] = None) -> dict[str, Any]:
    """Remember the attempt, keeping the last successful time on a failure."""
    stamp = int(now if now is not None else time.time())
    from storage.studio_db import upsert_app_settings

    state = dict(_read_state())
    state["last_attempt_at"] = stamp
    if ok:
        state["last_sent_at"] = stamp
        state["last_error"] = None
    else:
        state["last_error"] = (error or "Send failed.")[:500]
    upsert_app_settings({STATE_KEY: state}, read_back = False)
    return state


def should_send(now: Optional[int] = None) -> bool:
    """Whether the background reporter should try now.

    A successful send holds for the interval. A failure is retried in fifteen
    minutes rather than every loop tick, so a collector that is down does not
    become a busy loop.
    """
    stamp = int(now if now is not None else time.time())
    try:
        settings = get_telemetry_settings()
    except Exception:
        return False
    if not settings["enabled"] or not settings["endpoint"]:
        return False
    last_attempt = settings.get("last_attempt_at")
    if isinstance(last_attempt, (int, float)) and stamp - int(last_attempt) < 15 * 60:
        return False
    last_sent = settings.get("last_sent_at")
    if isinstance(last_sent, (int, float)) and stamp - int(last_sent) < settings["interval_hours"] * 3600:
        return False
    return True
