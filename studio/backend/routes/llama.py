# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama.cpp prebuilt update endpoints -- the single main update item.

GET  /api/llama/update-status  -> is a newer prebuilt available + job state
GET  /api/llama/update-changelog -> new carried changes since the installed build
POST /api/llama/update         -> download + atomically swap to the latest

Detection reuses utils.llama_cpp_freshness; the swap reuses install_llama_prebuilt.py via
utils.llama_cpp_update. Both fail open so the UI never blocks on a missing marker / offline GitHub.

whisper.cpp updates piggyback here: the status payload carries a whisper sub-status (update_available is the
llama OR whisper union) and the apply job chains a whisper phase after the llama phase when whisper is behind,
with a per-phase breakdown in job.phases. All pre-existing top-level fields keep their shape, so older clients
keep working unchanged. audio.cpp is the third chained phase, offered when the managed runtime is not the release
this Studio pins, with its own sub-status.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from auth import policy
from auth.authentication import get_current_subject
from loggers import get_logger
from utils.llama_cpp_architectures import (
    _ARCHITECTURE_IDS,
    activate as activate_architecture,
    start_install as start_architecture_install,
)
from utils.llama_cpp_update import (
    get_architecture_status,
    get_backend_status,
    get_update_changelog,
    get_update_status,
    start_backend_switch,
    start_update,
)

logger = get_logger(__name__)
router = APIRouter()


class LlamaUpdateJob(BaseModel):
    state: str = Field("idle", description = "idle | running | success | error")
    operation: Optional[Literal["update", "switch"]] = None
    requested_backend: Optional[Literal["auto", "cpu", "cuda", "rocm", "vulkan"]] = None
    message: str = ""
    from_tag: Optional[str] = None
    to_tag: Optional[str] = None
    reload_required: Optional[bool] = None
    error: Optional[str] = None
    progress: Optional[float] = Field(None, description = "0..1 while running, 1 on success.")
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    phases: Optional[dict] = Field(
        None,
        description = (
            "Per-phase breakdown of a chained llama+whisper job "
            "(name -> state/progress/to_tag/...); None for pre-chaining jobs."
        ),
    )


class WhisperSubStatus(BaseModel):
    """The whisper piggyback inside the llama update item."""

    update_available: bool = Field(
        False, description = "True when the chained apply would run a whisper phase."
    )
    installed_tag: Optional[str] = None
    latest_tag: Optional[str] = None
    update_size_bytes: Optional[int] = None
    skip_reason: Optional[str] = Field(
        None,
        description = (
            "Why the whisper phase would be skipped "
            "(up_to_date | local_link | source_build | not_installed | ...)."
        ),
    )


class AudioCppSubStatus(WhisperSubStatus):
    """The audio.cpp phase inside the llama update item; latest_tag is the release this Studio pins."""


class LlamaUpdateStatusResponse(BaseModel):
    supported: bool = Field(
        False,
        description = "True when the install came from an Unsloth prebuilt (has a marker).",
    )
    update_available: bool = Field(
        False,
        description = (
            "True when an update would do something: llama.cpp is behind OR the "
            "whisper piggyback is behind OR audio.cpp is not the pinned release."
        ),
    )
    llama_update_available: bool = Field(
        False, description = "True when the latest llama.cpp release is newer than the install."
    )
    update_component: Optional[Literal["llama", "whisper", "audio"]] = Field(
        None,
        description = "Component whose versions the combined update banner should display.",
    )
    stale: bool = Field(
        False, description = "Update available AND install older than the staleness threshold."
    )
    installed_tag: Optional[str] = None
    latest_tag: Optional[str] = None
    published_repo: Optional[str] = None
    installed_at_utc: Optional[str] = None
    age_days: Optional[int] = None
    source_build: bool = Field(
        False, description = "True when there is no marker (source build) but a prebuilt is offered."
    )
    update_size_bytes: Optional[int] = Field(
        None, description = "Download size of the prebuilt Update would fetch, in bytes."
    )
    backend_migration_available: bool = Field(
        False,
        description = (
            "True when the install recorded an AUTOMATIC backend choice and detection "
            "now resolves elsewhere, so Update would move it. Independent of "
            "update_available: the release can be current while the backend has drifted."
        ),
    )
    from_backend: Optional[str] = Field(
        None, description = "Installed backend, when a migration is available."
    )
    to_backend: Optional[str] = Field(
        None, description = "Backend a re-applied automatic selection would install."
    )
    whisper: Optional[WhisperSubStatus] = Field(
        None, description = "Whisper piggyback sub-status; None when the probe is unavailable."
    )
    audio: Optional[AudioCppSubStatus] = Field(
        None, description = "audio.cpp phase sub-status; None when the probe is unavailable."
    )
    job: LlamaUpdateJob = Field(default_factory = LlamaUpdateJob)


class LlamaUpdateChangeLink(BaseModel):
    label: str
    url: str


class LlamaUpdateChange(BaseModel):
    summary: str
    links: list[LlamaUpdateChangeLink] = Field(default_factory = list)


class LlamaUpdateChangelogResponse(BaseModel):
    matched: bool = Field(
        False,
        description = "True when both releases were resolved and compared.",
    )
    installed_tag: Optional[str] = None
    latest_tag: Optional[str] = None
    changes: list[LlamaUpdateChange] = Field(default_factory = list)
    total_changes: int = 0
    truncated: bool = False
    release_url: Optional[str] = None
    error: Optional[str] = None


class LlamaUpdateActionResponse(BaseModel):
    started: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    job: LlamaUpdateJob = Field(default_factory = LlamaUpdateJob)


_llama_update_lock = threading.Lock()
_last_llama_update_step = -1


def _log_llama_update_progress(job: LlamaUpdateJob) -> None:
    """One llama_update_progress line per 10% step so a prebuilt update reports
    progress without a line per poll. Resyncs when a new update starts."""
    global _last_llama_update_step
    if job.state != "running" or job.progress is None:
        return
    step = int(max(0.0, min(float(job.progress), 1.0)) * 10)
    with _llama_update_lock:
        prev = _last_llama_update_step
        if step == prev:
            return
        _last_llama_update_step = step
        if step < prev:
            return
    logger.info("llama_update_progress", to_tag = job.to_tag or "", percent = step * 10)


@router.get("/update-status", response_model = LlamaUpdateStatusResponse)
async def llama_update_status(
    force_refresh: bool = Query(
        False, description = "Bypass the 24h release cache for an explicit check."
    ),
    current_subject: str = Depends(get_current_subject),
) -> LlamaUpdateStatusResponse:
    # Off the event loop: detection may probe the host and read GitHub.
    status = await asyncio.to_thread(get_update_status, force_refresh = force_refresh)
    resp = LlamaUpdateStatusResponse(**status)
    _log_llama_update_progress(resp.job)
    return resp


# Replaces the installation's llama and whisper executables for everyone, so it is owner-only.
@router.post(
    "/update",
    response_model = LlamaUpdateActionResponse,
    dependencies = [Depends(get_current_subject), Depends(policy.require_owner)],
)
async def llama_update(
    current_subject: str = Depends(get_current_subject),
) -> LlamaUpdateActionResponse:
    action = await asyncio.to_thread(start_update)
    return LlamaUpdateActionResponse(**action)


@router.get("/update-changelog", response_model = LlamaUpdateChangelogResponse)
async def llama_update_changelog(
    force_refresh: bool = Query(False, description = "Retry the exact release lookups."),
    installed_tag: Optional[str] = Query(
        None,
        max_length = 200,
        description = "Installed tag the caller is displaying; ignored unless it still matches.",
    ),
    latest_tag: Optional[str] = Query(
        None,
        max_length = 200,
        description = "Target the caller is displaying, so a newer one cached meanwhile "
        "does not retarget the comparison.",
    ),
    current_subject: str = Depends(get_current_subject),
) -> LlamaUpdateChangelogResponse:
    result = await asyncio.to_thread(
        get_update_changelog,
        force_refresh = force_refresh,
        installed_tag = installed_tag,
        latest_tag = latest_tag,
    )
    return LlamaUpdateChangelogResponse(**result)


class LlamaBackendOption(BaseModel):
    backend: str = Field(..., description = "auto | cpu | cuda | rocm | vulkan")
    available: bool = Field(
        False, description = "True when a prebuilt for this backend installs on this host."
    )
    unavailable_reason: Optional[str] = Field(
        None, description = "unavailable | no_prebuilt | error, when available is false."
    )
    resolved_backend: Optional[str] = Field(
        None, description = "For 'auto', the backend hardware detection picks right now."
    )
    release_tag: Optional[str] = None
    download_size_bytes: Optional[int] = None


class LlamaBackendStatusResponse(BaseModel):
    supported: bool = Field(
        False, description = "True when this install's backend can be switched from here."
    )
    reason: Optional[str] = Field(
        None,
        description = (
            "Why it cannot: not_installed | local_link | source_build | no_install_dir "
            "| unresolved (the backend list could not be resolved, e.g. offline)."
        ),
    )
    env_backend: Optional[str] = Field(
        None,
        description = (
            "Backend pinned by UNSLOTH_LLAMA_CPP_BACKEND / UNSLOTH_FORCE_VULKAN. It "
            "overrides a stored choice, so the picker shows it as read-only."
        ),
    )
    backend: Optional[str] = Field(None, description = "What the install runs on now.")
    backend_request: str = Field(
        "auto",
        description = (
            "The recorded choice; 'auto' means hardware detection. A name this "
            "build does not know was written by a newer Unsloth and is read-only."
        ),
    )
    selection_applied: bool = Field(
        True,
        description = (
            "False when the recorded choice is 'auto' and detection would now "
            "resolve to a different backend than the installed one."
        ),
    )
    installed_tag: Optional[str] = None
    options: list[LlamaBackendOption] = Field(default_factory = list)
    job: LlamaUpdateJob = Field(default_factory = LlamaUpdateJob)


class LlamaArchitectureJob(BaseModel):
    state: str = Field("idle", description = "idle | running | success | error")
    architecture: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    progress: Optional[float] = Field(None, description = "0..1 while running, 1 on success.")
    from_tag: Optional[str] = None
    to_tag: Optional[str] = None
    activated: Optional[bool] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class LlamaArchitectureInfo(BaseModel):
    id: str
    label: str
    repo: Optional[str] = None
    source_build: bool = Field(
        False, description = "True when this architecture has a pinned source fallback."
    )
    source_commit: Optional[str] = Field(
        None, description = "Pinned commit used for a source build, when available."
    )
    available: bool = False
    installable: bool = False
    selectable: bool = False
    installed: bool = False
    active: bool = False
    reason: Optional[str] = None
    asset: Optional[str] = None
    release_tag: Optional[str] = None
    latest_tag: Optional[str] = None
    installed_tag: Optional[str] = None
    installed_at_utc: Optional[str] = None
    download_size_bytes: Optional[int] = None
    update_available: bool = False
    binary_path: Optional[str] = None
    install_dir: Optional[str] = None
    description: str = ""


class LlamaArchitecturesStatus(BaseModel):
    active: str = "normal"
    runtime_locked: bool = False
    runtime_lock_reason: Optional[str] = None
    release_lookup_failed: bool = Field(
        False,
        description = (
            "True when at least one architecture's release lookup failed or timed out, "
            "so its tags are missing. Installed facts are still reported: the panel is "
            "degraded, not empty, and the row says release_lookup_failed where the fork "
            "has no source fallback."
        ),
    )
    profiles: list[LlamaArchitectureInfo] = Field(default_factory = list)
    job: LlamaArchitectureJob = Field(default_factory = LlamaArchitectureJob)


class LlamaArchitectureInstallRequest(BaseModel):
    architecture: str = Field(..., min_length = 1, max_length = 64)
    activate: bool = Field(
        True, description = "Switch to the installed architecture when the job succeeds."
    )


class LlamaArchitectureActionResponse(BaseModel):
    started: bool
    reason: Optional[str] = None
    message: Optional[str] = None
    job: LlamaArchitectureJob = Field(default_factory = LlamaArchitectureJob)


class LlamaBackendRequest(BaseModel):
    backend: Literal["auto", "cpu", "cuda", "rocm", "vulkan"] = Field(
        ..., description = "Backend to install. 'auto' restores hardware detection."
    )


@router.get("/backend", response_model = LlamaBackendStatusResponse)
async def llama_backend_status(
    force_refresh: bool = Query(
        False, description = "Bypass the 24h resolver cache for an explicit re-check."
    ),
    current_subject: str = Depends(get_current_subject),
) -> LlamaBackendStatusResponse:
    # Off the event loop: resolving the options runs the installer's probe.
    status = await asyncio.to_thread(get_backend_status, force_refresh = force_refresh)
    return LlamaBackendStatusResponse(**status)


@router.post(
    "/backend",
    response_model = LlamaUpdateActionResponse,
    dependencies = [Depends(get_current_subject), Depends(policy.require_owner)],
)
async def llama_backend_switch(
    request: LlamaBackendRequest, current_subject: str = Depends(get_current_subject)
) -> LlamaUpdateActionResponse:
    """Install the llama.cpp build for another backend and record the choice.

    Shares the update job, so it answers already_running rather than starting a
    second writer against the same install; callers poll /update-status for
    progress exactly as they do for an update.
    """
    logger.info("llama_backend_switch_requested", backend = request.backend)
    action = await asyncio.to_thread(start_backend_switch, request.backend)
    return LlamaUpdateActionResponse(**action)


@router.get("/architectures", response_model = LlamaArchitecturesStatus)
async def llama_architectures_status(
    force_refresh: bool = Query(
        False,
        description = (
            "Re-resolve the releases and rescan the installed trees instead of "
            "reusing the status cache. The release lookup stays bounded, so an "
            "unreachable GitHub degrades the answer instead of stalling it."
        ),
    ),
    current_subject: str = Depends(get_current_subject),
) -> LlamaArchitecturesStatus:
    """Installed, downloadable, and unavailable llama.cpp architectures.

    Polled on open and every 900 ms while an install runs, so it goes through the
    cached wrapper in utils.llama_cpp_update: repeated reads reuse the resolved
    releases and the installed-tree scan until a local fact changes, while the
    job -- the progress half -- is read fresh on every call.
    """
    # Off the event loop: the first read per key may read GitHub and the disk.
    status = await asyncio.to_thread(get_architecture_status, force_refresh = force_refresh)
    return LlamaArchitecturesStatus(**status)


# Replaces the runtime for everyone, so installing and switching stay owner-only.
@router.post(
    "/architectures/install",
    response_model = LlamaArchitectureActionResponse,
    dependencies = [Depends(get_current_subject), Depends(policy.require_owner)],
)
async def llama_architecture_install(
    request: LlamaArchitectureInstallRequest,
    current_subject: str = Depends(get_current_subject),
) -> LlamaArchitectureActionResponse:
    # Refused here rather than inside start_install, which answers started=False with a
    # reason: a caller that checks only the status code reads that 200 as "the install
    # began". The manifest is the allowlist, so the answer is 404 naming the id.
    if request.architecture not in _ARCHITECTURE_IDS:
        raise HTTPException(
            status_code = 404,
            detail = (
                f"{request.architecture!r} is not a llama.cpp architecture Unsloth can install; "
                f"known ids are {', '.join(sorted(_ARCHITECTURE_IDS))}."
            ),
        )
    logger.info(
        "llama_architecture_install_requested",
        architecture = request.architecture,
        activate = request.activate,
    )
    action = await asyncio.to_thread(
        start_architecture_install,
        request.architecture,
        activate = request.activate,
    )
    return LlamaArchitectureActionResponse(**action)


@router.post(
    "/architectures/activate",
    response_model = LlamaArchitecturesStatus,
    dependencies = [Depends(get_current_subject), Depends(policy.require_owner)],
)
async def llama_architecture_activate(
    request: LlamaArchitectureInstallRequest,
    current_subject: str = Depends(get_current_subject),
) -> LlamaArchitecturesStatus:
    """Record an installed architecture as the runtime and return the new status."""
    try:
        await asyncio.to_thread(activate_architecture, request.architecture)
    except ValueError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    # Answer through the same cached wrapper the picker polls. Activation only
    # changed the recorded setting, which is part of the cache key, so this is
    # one fresh compute and the polls that follow are hits.
    status = await asyncio.to_thread(get_architecture_status, force_refresh = False)
    return LlamaArchitecturesStatus(**status)
