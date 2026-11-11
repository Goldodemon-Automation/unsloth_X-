# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The opt-in reporter: one snapshot, one POST, every six hours while enabled.

Sending is intentionally boring. stdlib urllib, a bounded body, a bounded
timeout and no retries inside a single send: a collector that is down records
one failure and the next loop tick is fifteen minutes away. The reporter never
raises into startup, and it does nothing at all until Settings turns it on.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from core.telemetry.snapshot import build_snapshot
from loggers import get_logger
from utils import telemetry_settings

logger = get_logger(__name__)

USER_AGENT = "UnslothX8-Telemetry/1"
REQUEST_TIMEOUT_S = 20.0
MAX_RESPONSE_BYTES = 4096
IDLE_SECONDS = 15 * 60
FIRST_DELAY_SECONDS = 90

_stop = threading.Event()
_thread: Optional[threading.Thread] = None
_start_lock = threading.Lock()


def _post(endpoint: str, payload: dict[str, Any], ingest_key: str) -> tuple[bool, str]:
    """POST one report. Returns (ok, error) and never raises."""
    try:
        body = json.dumps(payload, separators = (",", ":"), ensure_ascii = False).encode("utf-8")
    except (TypeError, ValueError) as error:
        return False, f"Could not encode the report: {type(error).__name__}."

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if ingest_key:
        headers["Authorization"] = f"Bearer {ingest_key}"

    request = urllib_request.Request(endpoint, data = body, headers = headers, method = "POST")
    try:
        with urllib_request.urlopen(request, timeout = REQUEST_TIMEOUT_S) as response:
            status = int(getattr(response, "status", 0) or 0)
            if not 200 <= status < 300:
                return False, f"Collector answered HTTP {status}."
            try:
                response.read(MAX_RESPONSE_BYTES)
            except Exception:
                pass
            return True, ""
    except urllib_error.HTTPError as failure:
        detail = ""
        try:
            detail = failure.read(MAX_RESPONSE_BYTES).decode("utf-8", "replace").strip()
        except Exception:
            pass
        suffix = f": {detail[:200]}" if detail else ""
        return False, f"Collector answered HTTP {failure.code}{suffix}"
    except urllib_error.URLError as failure:
        return False, f"Could not reach the collector: {failure.reason}"
    except Exception as failure:
        return False, f"Send failed: {type(failure).__name__}: {failure}"


def send_snapshot(*, force: bool = False) -> dict[str, Any]:
    """Build and send one report now. ``force`` bypasses the six-hour gate, not the switch."""
    settings = telemetry_settings.get_telemetry_settings()
    if not settings["enabled"]:
        return {"ok": False, "error": "Telemetry is off. Turn it on first.", "skipped": True}
    endpoint = telemetry_settings.normalize_endpoint(settings["endpoint"])
    if not endpoint:
        return {"ok": False, "error": "No collector endpoint is set."}
    if not force and not telemetry_settings.should_send():
        return {"ok": False, "error": "Not due yet.", "skipped": True}

    payload = build_snapshot(settings["install_id"], window_days = telemetry_settings.MAX_WINDOW_DAYS)
    ok, error = _post(endpoint, payload, settings["ingest_key"])
    state = telemetry_settings.record_send(ok, error)
    if ok:
        logger.info(
            "telemetry report sent install=%s models=%d trainings=%d merges=%d",
            settings["install_id"][:8],
            len(payload.get("models") or []),
            len(payload.get("trainings") or []),
            len(payload.get("merges") or []),
        )
    else:
        logger.warning("telemetry report failed: %s", error)
    return {
        "ok": ok,
        "error": error or None,
        "attempted_at": state.get("last_attempt_at"),
        "sent_at": state.get("last_sent_at"),
    }


def _loop() -> None:
    if _stop.wait(FIRST_DELAY_SECONDS):
        return
    while not _stop.is_set():
        try:
            if telemetry_settings.should_send():
                send_snapshot()
        except Exception as error:  # noqa: BLE001 -- one bad tick must not kill the loop
            logger.debug("telemetry reporter tick failed: %s", error)
        if _stop.wait(IDLE_SECONDS):
            return


def start_reporter() -> None:
    """Start the daemon once. Repeat calls are no-ops."""
    global _thread
    with _start_lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(
            target = _loop,
            name = "x8-telemetry-reporter",
            daemon = True,
        )
        _thread.start()


def stop_reporter() -> None:
    _stop.set()
