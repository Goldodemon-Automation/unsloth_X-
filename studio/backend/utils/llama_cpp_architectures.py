# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install and activate llama.cpp forks from releases or pinned source.

The four forks this exposes are not interchangeable source trees: each publishes
its own asset naming, and some publish no binaries at all. This module keeps one
manifest for them, resolves the newest release through GitHub, downloads only an
asset that matches this host, verifies the digest GitHub reports when it reports
one, extracts the archive without trusting its paths, and records an install
marker beside the extracted tree. The download is bounded (GitHub hosts over
TLS, a byte cap, a whole-transfer deadline) and extraction refuses traversing,
linked, duplicated or special members, because what arrives is a third-party
binary this app then executes. Dory publishes no release, so it falls back to
a CMake build of a pinned commit when this host has the toolchain. Activation is
a single app setting; the runtime resolver in ``core.inference.llama_cpp``
consults it before the custom folder, so switching does not overwrite the user's
custom-path choice.

Nothing here pretends a fork exists. X- LLama.zig has no build source yet, so its
row says so and stays disabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Callable, Optional
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)

ACTIVE_ARCHITECTURE_SETTING_KEY = "llama_cpp_architecture"
ARCHITECTURE_MARKER_NAME = "UNSLOTH_LLAMA_ARCH_INFO.json"
DEFAULT_ARCHITECTURE = "normal"

_RELEASE_CACHE_SECONDS = 600
_RELEASE_FAILURE_SECONDS = 45
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_USER_AGENT = "Unsloth-Studio-llama-architecture-installer"
# GitHub serves release assets from a CDN, so a redirect is normal; a redirect
# anywhere else is not. Most assets carry no digest, which makes this the only
# origin check the download has.
_ALLOWED_DOWNLOAD_HOSTS = ("github.com", "githubusercontent.com")
# The digest GitHub reports is only meaningful for the algorithms hashlib can
# reproduce here. Anything else must fail the install instead of passing
# unverified, because the installer claims the bytes match what GitHub published.
_DIGEST_ALGORITHMS = ("sha256", "sha512")
# A release asset is a runtime bundle; 2 GiB is far above the largest fork build
# and small enough that a hostile or misconfigured asset cannot fill the disk.
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
# An extracted tree is only ever a runtime or a source checkout. The cap stops a
# compression bomb from expanding until the disk is full before the install fails.
_MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
_DOWNLOAD_SOCKET_TIMEOUT_SECONDS = 60
# The socket timeout above only bounds one stalled read. A server that drips one
# byte per minute keeps resetting it, so the whole transfer needs a deadline or
# the install thread (and the install lock) is held forever. An hour is long
# enough for the largest fork bundle on a slow link and still ends a drip feed.
_DOWNLOAD_DEADLINE_SECONDS = 3600
# A release document is a few hundred kilobytes at most; the cap keeps a hostile
# API answer from growing the process while it is parsed.
_MAX_API_RESPONSE_BYTES = 4 * 1024 * 1024
_API_READ_TIMEOUT_SECONDS = 8.0
_MAX_ARCHIVE_MEMBERS = 200_000
_SOURCE_CONFIGURE_TIMEOUT_SECONDS = 900
_SOURCE_BUILD_TIMEOUT_SECONDS = 3600
_SOURCE_LOG_TAIL_LINES = 40


class InstallError(RuntimeError):
    """A user-facing install failure."""


@dataclass(frozen = True)
class ArchitectureProfile:
    id: str
    label: str
    repo: Optional[str] = None
    # "prebuilt" profiles resolve a release asset; normal is the bundled runtime.
    kind: str = "prebuilt"
    description: str = ""
    # A pinned commit makes a source fallback reproducible. The Dory PR is not
    # merged, so its branch head is the only source that actually contains it.
    source_commit: Optional[str] = None


ARCHITECTURES: tuple[ArchitectureProfile, ...] = (
    ArchitectureProfile(
        id = "normal",
        label = "Normal (llama.cpp)",
        kind = "bundled",
        description = "The llama.cpp runtime Unsloth installs and updates.",
    ),
    ArchitectureProfile(
        id = "dory",
        label = "Dory (brianchmielNV/llama.cpp)",
        repo = "brianchmielNV/llama.cpp",
        description = "The Dory llama.cpp fork.",
        source_commit = "1c3cdd23787750814aa21a070dc4d6dc451af871",
    ),
    ArchitectureProfile(
        id = "bee",
        label = "Bee (BeeLlama.cpp)",
        repo = "Anbeeld/beellama.cpp",
        description = "The BeeLlama.cpp fork.",
    ),
    ArchitectureProfile(
        id = "neurall",
        label = "Neuralls llama.cpp (neurall/llama.cpp)",
        repo = "neurall/llama.cpp",
        description = "The Neuralls llama.cpp build.",
    ),
    ArchitectureProfile(
        id = "ik",
        label = "IK (ik_llama.cpp)",
        # ikawrakow/ik_llama.cpp does not attach binaries. Thireus publishes the
        # CPU/GPU builds of that fork, so this is the installable release source.
        repo = "Thireus/ik_llama.cpp",
        description = "The IK llama.cpp fork, using Thireus' prebuilt builds.",
    ),
    ArchitectureProfile(
        id = "x",
        label = "X- LLama.zig (coming soon)",
        kind = "not_ready",
        description = "X- LLama.zig has no published build yet.",
    ),
)

_BY_ID = {profile.id: profile for profile in ARCHITECTURES}
_ARCHITECTURE_IDS = frozenset(_BY_ID)

_RELEASE_CACHE: dict[str, tuple[float, Optional[dict]]] = {}
_RELEASE_FAILED_AT: dict[str, float] = {}
_RELEASE_LOOKUP_FAILED: set[str] = set()
_RELEASE_LOCK = threading.Lock()

_JOB_LOCK = threading.Lock()
_OPERATION_LOCK = threading.Lock()
_JOB: dict = {}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_job() -> dict:
    return {
        "state": "idle",
        "architecture": None,
        "message": "",
        "error": None,
        "progress": None,
        "from_tag": None,
        "to_tag": None,
        "activated": None,
        "started_at": None,
        "finished_at": None,
    }


_JOB.update(_new_job())


def _set_job(**fields) -> dict:
    with _JOB_LOCK:
        _JOB.update(fields)
        return dict(_JOB)


def get_job() -> dict:
    with _JOB_LOCK:
        return dict(_JOB)


def _profile(architecture: str) -> Optional[ArchitectureProfile]:
    return _BY_ID.get(architecture)


def _host() -> tuple[str, str]:
    if sys.platform == "win32":
        system = "windows"
    elif sys.platform == "darwin":
        system = "darwin"
    else:
        system = "linux"
    machine = platform.machine().strip().lower()
    if machine in {"x86_64", "amd64", "x64", "i686", "i386"}:
        machine = "x64"
    elif machine in {"arm64", "aarch64"}:
        machine = "arm64"
    return system, machine


def _source_build_tools() -> tuple[bool, Optional[str]]:
    """Whether this host can compile a pinned source tree.

    Windows can let CMake locate Visual Studio, so requiring cl.exe on PATH
    would disable a build that works. Unix needs a real C/C++ compiler beside
    CMake, because configure would otherwise fail only after the download.
    """
    if shutil.which("cmake") is None:
        return False, "cmake"
    if sys.platform != "win32" and not any(
        shutil.which(name) for name in ("cc", "gcc", "clang", "c++", "g++", "clang++")
    ):
        return False, "compiler"
    return True, None


def _source_tag(profile: ArchitectureProfile) -> Optional[str]:
    if not profile.source_commit:
        return None
    return f"source-{profile.source_commit[:8]}"


def _score_asset(architecture: str, name: str, system: str, machine: str) -> Optional[int]:
    """How well a release asset fits this host, or None when it does not.

    Scores are deliberately conservative and CPU-first for IK: a CPU build runs
    everywhere, while picking a CUDA/Vulkan asset would also require a driver
    and a runtime the fork does not bundle on every platform.
    """
    lower = name.lower()
    if system == "windows" and machine == "x64":
        if architecture == "neurall":
            if (
                lower.startswith("llama-")
                and "bin-win" in lower
                and "x64" in lower
                and "cuda" in lower
            ):
                return 100
        if architecture == "ik":
            if "bin-win-cpu" not in lower or "cudart" in lower:
                return None
            if "x86_64_v3-avx2" in lower:
                return 100
            if "avx2" in lower and "avx512" not in lower and "znver" not in lower:
                return 90
            if "avx2" in lower:
                return 80
    if system == "linux" and machine == "x64":
        if architecture == "neurall":
            # cudart-* is the optional CUDA runtime archive, not the engine.
            if (
                lower.startswith("llama-")
                and "bin-ubuntu" in lower
                and "x64" in lower
                and "cuda" in lower
            ):
                return 100
        if architecture == "ik":
            if "cudart" in lower or "vulkan" in lower or "cuda" in lower:
                return None
            if "bin-ubuntu-x64-avx2" in lower:
                return 100
            if "bin-ubuntu-x64" in lower and "avx2" in lower:
                return 80
    if system == "darwin" and machine == "x64" and architecture == "ik":
        if "bin-macos-x64" in lower and "avx2" in lower:
            return 100
        if "bin-macos-x64" in lower:
            return 60
    # Bee publishes CPU bundles, and Dory may eventually. Keep a conservative
    # fallback so a CPU archive becomes installable without another code path,
    # while never guessing at a GPU runtime.
    if architecture in {"dory", "bee"}:
        cpu_only = not any(
            token in lower
            for token in ("cuda", "rocm", "hip", "vulkan", "sycl", "metal")
        )
        if system == "windows" and machine == "x64":
            if "bin-win" in lower and "x64" in lower and "cudart" not in lower:
                if "cpu" in lower or cpu_only:
                    return 100
        elif system == "linux" and machine == "x64":
            if "ubuntu" in lower and "x64" in lower and "cudart" not in lower:
                if "cpu" in lower or cpu_only:
                    return 100
        elif system == "linux" and machine == "arm64" and architecture == "bee":
            if "bin-ubuntu-arm64" in lower:
                return 100
        elif system == "darwin" and machine == "arm64":
            if "macos" in lower and "arm64" in lower:
                return 100
        elif system == "darwin" and machine == "x64":
            if "macos" in lower and "x64" in lower:
                return 90
    return None


def _choose_asset(architecture: str, release: dict) -> Optional[dict]:
    system, machine = _host()
    best: Optional[tuple[int, dict]] = None
    for asset in release.get("assets") or []:
        name = asset.get("name")
        if not isinstance(name, str):
            continue
        score = _score_asset(architecture, name, system, machine)
        if score is None:
            continue
        if best is None or score > best[0]:
            best = (score, asset)
    return best[1] if best else None


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _read_json(url: str, timeout: float = _API_READ_TIMEOUT_SECONDS) -> object:
    """Read a small JSON document from the GitHub API, bounded in time and size.

    ``timeout`` bounds one socket operation, not the whole response: an endpoint
    that sends headers and then drips bytes keeps resetting it. This runs on the
    install thread, which holds the install lock, and on every Settings status
    poll, so a stalled answer would wedge both. The body is read in chunks
    against a wall-clock deadline and a size cap instead of one unbounded
    ``read()``, which also keeps a hostile answer from exhausting memory.
    """
    request = urllib.request.Request(url, headers = _github_headers())
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    received = 0
    with urllib.request.urlopen(request, timeout = timeout) as response:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Reading {url} took longer than {timeout} seconds."
                )
            chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            received += len(chunk)
            if received > _MAX_API_RESPONSE_BYTES:
                raise ValueError(
                    f"The response from {url} is larger than Unsloth will parse."
                )
            chunks.append(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


def _fetch_release(repo: str) -> Optional[dict]:
    """Newest published release, or None when the repository has no releases."""
    try:
        payload = _read_json(f"https://api.github.com/repos/{repo}/releases/latest")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if not isinstance(payload, dict):
        return None
    assets = []
    for item in payload.get("assets") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        url = item.get("browser_download_url")
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        digest = item.get("digest")
        assets.append(
            {
                "name": name,
                "url": url,
                "size": item.get("size") if isinstance(item.get("size"), int) else None,
                "digest": digest if isinstance(digest, str) else None,
            }
        )
    tag = payload.get("tag_name")
    return {
        "tag": tag if isinstance(tag, str) else None,
        "html_url": payload.get("html_url") if isinstance(payload.get("html_url"), str) else None,
        "assets": assets,
    }


def _latest_release(
    repo: str,
    *,
    force_refresh: bool = False,
    fetch: Optional[Callable[[str], Optional[dict]]] = None,
) -> Optional[dict]:
    now = time.monotonic()
    with _RELEASE_LOCK:
        cached = _RELEASE_CACHE.get(repo)
        if not force_refresh and cached and cached[0] > now:
            return cached[1]
        failed_at = _RELEASE_FAILED_AT.get(repo)
        if not force_refresh and failed_at is not None and now - failed_at < _RELEASE_FAILURE_SECONDS:
            return None
    try:
        release = (fetch or _fetch_release)(repo)
    except Exception as exc:
        logger.warning("llama architecture: release lookup failed", repo = repo, error = str(exc))
        with _RELEASE_LOCK:
            _RELEASE_FAILED_AT[repo] = now
            _RELEASE_LOOKUP_FAILED.add(repo)
        return None
    with _RELEASE_LOCK:
        _RELEASE_CACHE[repo] = (now + _RELEASE_CACHE_SECONDS, release)
        _RELEASE_FAILED_AT.pop(repo, None)
        _RELEASE_LOOKUP_FAILED.discard(repo)
    return release


def reset_caches() -> None:
    with _RELEASE_LOCK:
        _RELEASE_CACHE.clear()
        _RELEASE_FAILED_AT.clear()
        _RELEASE_LOOKUP_FAILED.clear()


def _install_root() -> Path:
    home = (os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or "").strip()
    if home:
        root = Path(home).expanduser()
    else:
        try:
            from utils.paths.storage_roots import unsloth_home

            root = unsloth_home() or (Path.home() / ".unsloth")
        except Exception:
            root = Path.home() / ".unsloth"
    return root / "llama.cpp-architectures"


def _install_dir(architecture: str) -> Path:
    return _install_root() / architecture


def _marker_path(architecture: str) -> Path:
    return _install_dir(architecture) / ARCHITECTURE_MARKER_NAME


def _read_marker(architecture: str) -> Optional[dict]:
    path = _marker_path(architecture)
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("architecture") != architecture:
        return None
    return payload


def _safe_marker_binary(architecture: str, marker: dict) -> Optional[Path]:
    root = _install_dir(architecture)
    relative = marker.get("binary_relpath")
    if isinstance(relative, str) and relative:
        try:
            candidate = (root / relative).resolve()
            if candidate.is_file() and candidate.is_relative_to(root.resolve()):
                return candidate
        except (OSError, ValueError):
            pass
    try:
        from utils.llama_cpp_path_settings import resolve_llama_server_binary

        return resolve_llama_server_binary(root)
    except Exception:
        return None


def _installed(architecture: str) -> tuple[Optional[dict], Optional[Path]]:
    marker = _read_marker(architecture)
    if marker is None:
        return None, None
    binary = _safe_marker_binary(architecture, marker)
    if binary is None:
        return marker, None
    return marker, binary


def _active_id() -> str:
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        value = run_as(OWNER, get_app_setting, ACTIVE_ARCHITECTURE_SETTING_KEY, DEFAULT_ARCHITECTURE)
    except Exception:
        return DEFAULT_ARCHITECTURE
    if isinstance(value, str) and value in _ARCHITECTURE_IDS:
        return value
    return DEFAULT_ARCHITECTURE


def active_architecture_id() -> str:
    return _active_id()


def active_architecture_directory() -> Optional[Path]:
    """The installed fork directory the runtime should use, or None for bundled.

    A deliberate architecture selection is a pin: when its marker or binary has
    gone missing this returns None *and* the caller must not fall through to a
    different custom folder. The resolver checks the marker separately so it can
    report that state instead of silently running an older runtime.
    """
    architecture = _active_id()
    if architecture == DEFAULT_ARCHITECTURE:
        return None
    marker, binary = _installed(architecture)
    if marker is None:
        return None
    if binary is None:
        return None
    return binary.parent


def _runtime_env_locked() -> Optional[str]:
    direct = (os.environ.get("LLAMA_SERVER_PATH") or "").strip()
    if direct:
        return "LLAMA_SERVER_PATH"
    custom = (os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or "").strip()
    if custom and os.environ.get("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH") != "1":
        return "UNSLOTH_LLAMA_CPP_PATH"
    return None


def _set_active(architecture: str) -> None:
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    run_as(OWNER, upsert_app_settings, {ACTIVE_ARCHITECTURE_SETTING_KEY: architecture})


def _reason_for_missing_build(
    architecture: str, release: Optional[dict], asset: Optional[dict]
) -> Optional[str]:
    profile = _profile(architecture)
    if profile is None:
        return "unknown_architecture"
    if profile.kind == "not_ready" or profile.repo is None:
        return "build_not_ready"
    if release is None:
        return "no_release"
    if asset is None:
        return "no_build_for_platform"
    return None


def _profile_payload(
    profile: ArchitectureProfile,
    *,
    active: str,
    release: Optional[dict],
    asset: Optional[dict],
) -> dict:
    marker, binary = _installed(profile.id)
    installed = binary is not None
    source_ready, _source_missing = (
        _source_build_tools() if profile.source_commit else (False, None)
    )
    source_available = bool(profile.source_commit) and source_ready
    reason: Optional[str] = None
    if profile.kind == "not_ready":
        reason = "build_not_ready"
    elif marker is not None and binary is None:
        # Checked before the prebuilt branch, which would otherwise always win: a marker whose
        # recorded binary is gone means the row used to be installed and is now broken. Saying
        # "installable" there reads as "not installed", so a deleted llama-server looked like a
        # fork the user never installed and the picker never offered to repair it.
        reason = "missing_binary"
    elif profile.kind == "prebuilt":
        if asset is not None:
            reason = None
        elif profile.source_commit:
            reason = "source_build" if source_available else "build_tools_missing"
        else:
            reason = _reason_for_missing_build(profile.id, release, asset)

    available = (
        profile.kind == "bundled"
        or asset is not None
        or source_available
        or installed
    )
    if profile.kind == "not_ready":
        available = False
    if marker is not None and binary is None:
        available = False

    installed_tag = marker.get("release_tag") if marker else None
    # If the source fallback is what this row would install, compare against the
    # pinned commit, not a release tag whose assets do not fit this machine.
    latest_tag = (
        release.get("tag")
        if release is not None and asset is not None
        else _source_tag(profile)
    )
    update_available = bool(
        installed
        and (asset is not None or profile.source_commit is not None)
        and latest_tag
        and installed_tag
        and latest_tag != installed_tag
    )
    if profile.kind == "bundled":
        installed = True
        available = True
        update_available = False

    return {
        "id": profile.id,
        "label": profile.label,
        "repo": profile.repo,
        "source_build": bool(profile.source_commit),
        "source_commit": profile.source_commit,
        "available": available,
        "installable": profile.kind == "prebuilt"
        and (asset is not None or source_available),
        "selectable": (
            profile.kind == "bundled"
            or installed
            or asset is not None
            or source_available
        ),
        "installed": installed,
        "active": active == profile.id,
        "reason": reason,
        "asset": asset.get("name") if asset else None,
        "release_tag": latest_tag,
        "latest_tag": latest_tag,
        "installed_tag": installed_tag,
        "installed_at_utc": marker.get("installed_at_utc") if marker else None,
        "download_size_bytes": asset.get("size") if asset else None,
        "update_available": update_available,
        "binary_path": str(binary) if binary else None,
        "install_dir": str(_install_dir(profile.id))
        if profile.kind == "prebuilt"
        else None,
        "description": profile.description,
    }


def get_status(*, force_refresh: bool = False) -> dict:
    """The architecture picker payload, including real release/install state.

    A release lookup failure is not reported as "no build": it keeps the row
    installable if it was already installed, and says the lookup failed, so an
    offline machine never looks like a fork without a release.
    """
    active = _active_id()
    job = get_job()
    releases: dict[str, Optional[dict]] = {}
    prebuilt = [
        profile
        for profile in ARCHITECTURES
        if profile.kind == "prebuilt" and profile.repo is not None
    ]
    # A running job is already the only useful row in the panel. Do not pay the
    # release lookup (or start four threads) on every progress poll.
    if prebuilt and job["state"] != "running":
        # One settings open should not pay four serial network timeouts. Cached
        # answers still short-circuit inside _latest_release.
        with ThreadPoolExecutor(max_workers = min(4, len(prebuilt))) as pool:
            futures = {
                pool.submit(
                    _latest_release, profile.repo, force_refresh = force_refresh
                ): profile.id
                for profile in prebuilt
            }
            for future, architecture in futures.items():
                try:
                    releases[architecture] = future.result()
                except Exception:
                    releases[architecture] = None

    profiles = []
    for profile in ARCHITECTURES:
        release = releases.get(profile.id)
        asset = _choose_asset(profile.id, release) if release else None
        payload = _profile_payload(profile, active = active, release = release, asset = asset)
        with _RELEASE_LOCK:
            lookup_failed = profile.repo in _RELEASE_LOOKUP_FAILED if profile.repo else False
        if (
            profile.kind == "prebuilt"
            and lookup_failed
            and profile.source_commit is None
        ):
            payload["reason"] = "release_lookup_failed"
        profiles.append(payload)

    env_lock = _runtime_env_locked()
    return {
        "active": active,
        "runtime_locked": env_lock is not None,
        "runtime_lock_reason": env_lock,
        "profiles": profiles,
        "job": job,
    }


def _download_host_allowed(url: str) -> bool:
    """Whether ``url`` is GitHub serving over TLS.

    An asset URL redirects to GitHub's asset CDN, so a redirect is expected and
    must be followed. A redirect anywhere else is not: whoever can answer DNS
    for a broken proxy, and every release whose author controls the redirect
    target, would otherwise choose which server hands over the runtime this
    app is about to execute. Most assets publish no digest, so the host of the
    final URL is the only remaining origin guarantee.
    """
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False
    for allowed in _ALLOWED_DOWNLOAD_HOSTS:
        if host == allowed or host.endswith("." + allowed):
            return True
    return False


def _verify_download_origin(response: object) -> None:
    """Refuse a download that landed on a host other than GitHub.

    ``urlopen`` follows redirects silently, including a downgrade to plain
    HTTP, so the final URL is checked after the connection exists and before a
    byte is written. A response without ``geturl`` only comes from an injected
    opener (tests); real urllib responses always report their final URL.
    """
    geturl = getattr(response, "geturl", None)
    if not callable(geturl):
        return
    try:
        final = geturl()
    except Exception as exc:
        raise InstallError(
            "The build download did not report which URL served it."
        ) from exc
    if not isinstance(final, str) or not _download_host_allowed(final):
        try:
            shown = urllib.parse.urlsplit(final if isinstance(final, str) else "").hostname
        except ValueError:
            shown = None
        raise InstallError(
            f"The build download was served from {shown or 'an unknown host'}, which is "
            "not GitHub. Nothing was installed."
        )


def _download_digest(digest: object) -> tuple[Optional[str], Optional[str]]:
    """The algorithm and expected hex digest of a GitHub ``digest`` field.

    GitHub reports ``sha256:<hex>`` for newer assets and nothing at all for
    older ones. A digest in an algorithm this host cannot reproduce would
    otherwise be skipped in silence, so the install would claim the bytes
    matched a published hash while nothing was compared: refuse it instead.
    """
    if not isinstance(digest, str) or not digest.strip():
        return None, None
    algorithm, separator, expected = digest.strip().partition(":")
    algorithm = algorithm.strip().lower()
    expected = expected.strip().lower()
    if not separator or algorithm not in _DIGEST_ALGORITHMS:
        raise InstallError(
            f"The release reports a {algorithm or 'malformed'} digest Unsloth cannot "
            "verify. Nothing was installed."
        )
    if len(expected) != hashlib.new(algorithm).digest_size * 2:
        raise InstallError(
            f"The release reports a malformed {algorithm.upper()} digest. "
            "Nothing was installed."
        )
    try:
        int(expected, 16)
    except ValueError:
        raise InstallError(
            f"The release reports a malformed {algorithm.upper()} digest. "
            "Nothing was installed."
        ) from None
    return algorithm, expected


def _download_asset(
    asset: dict,
    destination: Path,
    *,
    progress: Callable[[float], None],
    opener: Optional[Callable[..., object]] = None,
) -> None:
    """Fetch one release asset with a verified origin, size, and duration.

    Every field of ``asset`` comes from a GitHub API response for a third-party
    fork, and the archive it names is executed later, so the transfer is bounded
    three ways: the final URL must still be GitHub over TLS, the byte count may
    not exceed ``_MAX_DOWNLOAD_BYTES`` (GitHub's declared size is a hint, not a
    limit), and the whole transfer must finish inside
    ``_DOWNLOAD_DEADLINE_SECONDS`` so a server that drips bytes cannot pin the
    install thread and its lock open forever. The digest GitHub reports is
    compared against the bytes actually received, never against what was
    requested.
    """
    url = asset.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise InstallError("The selected release asset has no usable download URL.")
    expected_size = asset.get("size") if isinstance(asset.get("size"), int) else None
    algorithm, expected_digest = _download_digest(asset.get("digest"))
    if expected_size is not None and not 0 <= expected_size <= _MAX_DOWNLOAD_BYTES:
        raise InstallError(
            "The selected build is larger than Unsloth will download "
            f"({expected_size} bytes). Nothing was installed."
        )
    request = urllib.request.Request(url, headers = {"User-Agent": _USER_AGENT})
    hasher = hashlib.new(algorithm or "sha256")
    received = 0
    destination.parent.mkdir(parents = True, exist_ok = True)
    open_url = opener or urllib.request.urlopen
    deadline = time.monotonic() + _DOWNLOAD_DEADLINE_SECONDS
    try:
        with open_url(request, timeout = _DOWNLOAD_SOCKET_TIMEOUT_SECONDS) as response:
            _verify_download_origin(response)
            total = response.headers.get("Content-Length") if hasattr(response, "headers") else None
            total_bytes = int(total) if total and str(total).isdigit() else expected_size
            if total_bytes is not None and total_bytes > _MAX_DOWNLOAD_BYTES:
                raise InstallError(
                    "The selected build is larger than Unsloth will download. "
                    "Nothing was installed."
                )
            with destination.open("wb") as handle:
                while True:
                    if time.monotonic() >= deadline:
                        raise InstallError(
                            "The build download took longer than "
                            f"{_DOWNLOAD_DEADLINE_SECONDS} seconds. Nothing was installed."
                        )
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > _MAX_DOWNLOAD_BYTES:
                        raise InstallError(
                            "The selected build is larger than Unsloth will download. "
                            "Nothing was installed."
                        )
                    handle.write(chunk)
                    hasher.update(chunk)
                    if total_bytes:
                        progress(min(0.70, 0.08 + 0.62 * (received / total_bytes)))
    except InstallError:
        raise
    except Exception as exc:
        raise InstallError(f"Could not download the build: {exc}") from exc
    if expected_size is not None and received != expected_size:
        raise InstallError(
            f"The download was incomplete: expected {expected_size} bytes, received {received}."
        )
    if expected_digest is not None:
        actual = hasher.hexdigest()
        if actual != expected_digest:
            if algorithm == "sha256":
                raise InstallError("The downloaded build failed its SHA-256 verification.")
            raise InstallError(
                f"The downloaded build failed its {str(algorithm).upper()} verification."
            )


def _member_parts(name: object) -> tuple[str, ...]:
    """The safe, relative path parts of one archive member name.

    The archive is attacker-controlled: it is a third-party fork's release, and
    the digest GitHub reports is optional. A member named ``../../.bashrc``,
    ``/etc/cron.d/x``, ``C:\\Windows\\System32\\x.dll`` or one carrying a
    backslash-escaped traversal would otherwise be written outside the staging
    directory as the user running Studio. Refusing is safer than rewriting: no
    real llama.cpp build archive needs an absolute or traversing member, and a
    silently rewritten member is a filename the digest check never covered.
    """
    if not isinstance(name, str):
        raise InstallError("The build archive contains an entry with no name.")
    if "\x00" in name:
        raise InstallError("The build archive contains an unsafe path.")
    normalized = name.replace("\\", "/")
    # A leading slash covers POSIX absolutes and UNC shares (//server/share).
    if normalized.startswith("/"):
        raise InstallError("The build archive contains an unsafe path.")
    parts = tuple(part for part in normalized.split("/") if part not in ("", "."))
    for part in parts:
        # A colon is a drive letter or NTFS alternate data stream on Windows and
        # never legitimate in a build archive on any host.
        if part == ".." or ":" in part or PureWindowsPath(part).drive:
            raise InstallError("The build archive contains an unsafe path.")
    return parts


def _safe_target(root: Path, name: str) -> Path:
    """Resolve one archive member below ``root``, refusing anything outside.

    ``resolve()`` runs last so a path that is lexically inside the destination
    but reaches outside through an already-extracted symlink is refused too.
    """
    root_resolved = root.resolve()
    target = root_resolved.joinpath(*_member_parts(name))
    resolved = target.resolve()
    if resolved != root_resolved and not resolved.is_relative_to(root_resolved):
        raise InstallError("The build archive contains an unsafe path.")
    return resolved


def _safe_link_target(root: Path, base: Path, linkname: object) -> Path:
    """Resolve a tar link target, refusing one that leaves ``root``.

    ``TarInfo.name`` is the only thing a name check sees. Without checking
    ``linkname`` too, an archive can create ``runtime`` as a symlink to ``/`` or
    to a hardlink of a file outside the tree and then write through it, which
    plants files anywhere the user can write. A link may use ``..`` only while
    it still resolves inside the destination.
    """
    if not isinstance(linkname, str) or not linkname or "\x00" in linkname:
        raise InstallError("The build archive contains a link with no target.")
    normalized = linkname.replace("\\", "/")
    if normalized.startswith("/"):
        raise InstallError("The build archive links outside the install directory.")
    for part in normalized.split("/"):
        if ":" in part or PureWindowsPath(part).drive:
            raise InstallError("The build archive links outside the install directory.")
    root_resolved = root.resolve()
    resolved = (base / normalized).resolve()
    if resolved != root_resolved and not resolved.is_relative_to(root_resolved):
        raise InstallError("The build archive links outside the install directory.")
    return resolved


def _write_member(source, target: Path, *, limit: int) -> int:
    """Stream one member to ``target``, stopping at ``limit`` bytes.

    The member's declared size is a hint from the archive, so the real bound is
    enforced on the bytes read: a member that expands past the remaining budget
    fails instead of filling the disk.
    """
    written = 0
    with target.open("wb") as handle:
        while True:
            chunk = source.read(min(_DOWNLOAD_CHUNK_BYTES, limit + 1 - written))
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise InstallError("The build archive expands too large. Nothing was installed.")
            handle.write(chunk)
    return written


def _member_mode(mode: int, default: int) -> int:
    """The permission bits to give an extracted member.

    Setuid, setgid and sticky bits are dropped: an archive does not get to leave
    a setuid binary in the install directory.
    """
    return (mode & 0o777) or default


def _extract_zip(bundle: zipfile.ZipFile, destination: Path) -> None:
    members = bundle.infolist()
    if len(members) > _MAX_ARCHIVE_MEMBERS:
        raise InstallError("The build archive has too many files.")
    root = destination.resolve()
    written: set[Path] = set()
    total = 0
    for member in members:
        target = _safe_target(root, member.filename)
        if member.is_dir():
            target.mkdir(parents = True, exist_ok = True)
            continue
        if target == root:
            raise InstallError("The build archive contains an unsafe path.")
        kind = (member.external_attr >> 16) & 0o170000
        if kind == 0o120000:
            raise InstallError(
                "The build archive contains a symbolic link. Nothing was installed."
            )
        if kind not in (0, 0o100000, 0o40000):
            raise InstallError(
                "The build archive contains a device or other special file. "
                "Nothing was installed."
            )
        if target in written:
            raise InstallError(
                "The build archive contains duplicate files. Nothing was installed."
            )
        written.add(target)
        declared = member.file_size if member.file_size > 0 else 0
        remaining = _MAX_EXTRACTED_BYTES - total
        if declared > remaining:
            raise InstallError("The build archive expands too large. Nothing was installed.")
        target.parent.mkdir(parents = True, exist_ok = True)
        with bundle.open(member) as source:
            total += _write_member(source, target, limit = remaining)
        if sys.platform != "win32":
            try:
                os.chmod(target, _member_mode(member.external_attr >> 16, 0o644))
            except OSError:
                pass


def _extract_tar(bundle: tarfile.TarFile, destination: Path) -> None:
    members = bundle.getmembers()
    if len(members) > _MAX_ARCHIVE_MEMBERS:
        raise InstallError("The build archive has too many files.")
    root = destination.resolve()
    written: set[Path] = set()
    total = 0
    for member in members:
        target = _safe_target(root, member.name)
        if member.isdir():
            target.mkdir(parents = True, exist_ok = True)
            continue
        if target == root:
            raise InstallError("The build archive contains an unsafe path.")
        if not (member.isfile() or member.issym() or member.islnk()):
            raise InstallError(
                "The build archive contains a device or other special file. "
                "Nothing was installed."
            )
        if target in written:
            raise InstallError(
                "The build archive contains duplicate files. Nothing was installed."
            )
        written.add(target)
        if member.issym() or member.islnk():
            # A symlink is relative to its own directory, a hardlink to the root.
            link_base = target.parent if member.issym() else root
            link_target = _safe_link_target(root, link_base, member.linkname)
            target.parent.mkdir(parents = True, exist_ok = True)
            try:
                if member.issym():
                    os.symlink(member.linkname, target, target_is_directory = link_target.is_dir())
                else:
                    if not link_target.is_file():
                        raise InstallError(
                            "The build archive links to a file it does not contain. "
                            "Nothing was installed."
                        )
                    os.link(link_target, target)
            except OSError as exc:
                raise InstallError(
                    f"The build archive contains a link Unsloth cannot create: {exc}"
                ) from exc
            continue
        declared = member.size if member.size > 0 else 0
        remaining = _MAX_EXTRACTED_BYTES - total
        if declared > remaining:
            raise InstallError("The build archive expands too large. Nothing was installed.")
        source = bundle.extractfile(member)
        if source is None:
            raise InstallError(
                "The build archive contains an unreadable file. Nothing was installed."
            )
        target.parent.mkdir(parents = True, exist_ok = True)
        with source:
            total += _write_member(source, target, limit = remaining)
        if sys.platform != "win32":
            try:
                os.chmod(target, _member_mode(member.mode, 0o644))
            except OSError:
                pass


def _extract_archive(archive: Path, destination: Path) -> None:
    """Extract a downloaded build, validating every member before it is written.

    The archive is written member by member through the checks above rather than
    handed to ``extractall``: ``extractall(filter = "data")`` only exists on
    newer Pythons, its refusal arrives as a ``tarfile`` error instead of an
    ``InstallError`` the installer can report, and its absence used to fall back
    to an unvalidated ``extractall`` that honoured symlinks and device nodes.
    Validation and extraction now agree on exactly one path per member.
    """
    destination.mkdir(parents = True, exist_ok = True)
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                _extract_zip(bundle, destination)
            return
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive, "r:*") as bundle:
                _extract_tar(bundle, destination)
            return
    except InstallError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile, tarfile.TarError) as exc:
        raise InstallError(
            f"The build archive could not be extracted: {exc}"
        ) from exc
    raise InstallError("The build archive is neither a zip nor a tar archive.")


def _find_binary(root: Path) -> Optional[Path]:
    binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    candidates = [path for path in root.rglob(binary_name) if path.is_file()]
    if not candidates:
        return None
    candidates.sort(key = lambda path: (len(path.relative_to(root).parts), str(path)))
    if sys.platform != "win32":
        executable = [path for path in candidates if os.access(path, os.X_OK)]
        if executable:
            return executable[0]
    return candidates[0]


def _promote_install(
    profile: ArchitectureProfile,
    extracted: Path,
    binary: Path,
    marker_fields: dict,
    *,
    progress: Callable[[float], None],
) -> dict:
    """Move a prepared tree into the install root and write its marker."""
    install_dir = _install_dir(profile.id)
    backup = _install_root() / f".old-{profile.id}-{uuid4().hex}"
    if sys.platform != "win32":
        try:
            binary.chmod(binary.stat().st_mode | 0o111)
        except OSError:
            pass
    binary_relpath = binary.relative_to(extracted).as_posix()
    install_dir.parent.mkdir(parents = True, exist_ok = True)
    previous: Optional[Path] = None
    if install_dir.exists():
        os.replace(install_dir, backup)
        previous = backup
    try:
        os.replace(extracted, install_dir)
        marker = {
            "architecture": profile.id,
            "label": profile.label,
            "repo": profile.repo,
            "binary_relpath": binary_relpath,
            "installed_at_utc": _utcnow(),
            **marker_fields,
        }
        (install_dir / ARCHITECTURE_MARKER_NAME).write_text(
            json.dumps(marker, indent = 2), encoding = "utf-8"
        )
    except Exception:
        if install_dir.exists():
            shutil.rmtree(install_dir, ignore_errors = True)
        if previous is not None and previous.exists():
            os.replace(previous, install_dir)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors = True)
    progress(0.97)
    return marker


def _install_from_asset(
    profile: ArchitectureProfile,
    release: dict,
    asset: dict,
    *,
    progress: Callable[[float], None],
) -> dict:
    staging = _install_root() / f".staging-{profile.id}-{uuid4().hex}"
    archive = staging / "download.bin"
    extracted = staging / "extracted"
    try:
        _download_asset(asset, archive, progress = progress)
        progress(0.78)
        _extract_archive(archive, extracted)
        progress(0.90)
        binary = _find_binary(extracted)
        if binary is None:
            raise InstallError(
                "The downloaded build does not contain llama-server. It may not be a "
                "complete runtime archive."
            )
        return _promote_install(
            profile,
            extracted,
            binary,
            {
                "release_tag": release.get("tag"),
                "asset": asset.get("name"),
                "digest": asset.get("digest"),
                "source_build": False,
                "source_commit": None,
            },
            progress = progress,
        )
    finally:
        shutil.rmtree(staging, ignore_errors = True)


def _source_root(extracted: Path) -> Path:
    """The directory holding the root CMakeLists.txt in an archive."""
    candidates = [
        path.parent
        for path in extracted.rglob("CMakeLists.txt")
        if (path.parent / "ggml").is_dir()
    ]
    if not candidates:
        candidates = [path.parent for path in extracted.rglob("CMakeLists.txt")]
    if not candidates:
        raise InstallError("The Dory source archive has no CMakeLists.txt.")
    return min(candidates, key = lambda path: len(path.relative_to(extracted).parts))


def _source_build_jobs() -> int:
    raw = os.environ.get("UNSLOTH_LLAMA_BUILD_JOBS", "").strip()
    if raw:
        try:
            return max(1, min(int(raw), 32))
        except ValueError:
            pass
    return max(1, min(8, os.cpu_count() or 1))


def _log_tail(path: Path) -> str:
    try:
        lines = path.read_text(encoding = "utf-8", errors = "replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-_SOURCE_LOG_TAIL_LINES:])


def _run_logged_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout: int,
    failure: str,
) -> None:
    log_path.parent.mkdir(parents = True, exist_ok = True)
    try:
        with log_path.open("w", encoding = "utf-8", errors = "replace") as log:
            # An argument list with shell = False: the repository and commit are
            # manifest constants, and neither they nor a path in the source tree
            # may ever reach a command interpreter.
            process = subprocess.Popen(
                command,
                cwd = str(cwd),
                stdout = log,
                stderr = subprocess.STDOUT,
                text = True,
                shell = False,
            )
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    process.kill()
                    process.wait()
                    raise InstallError(f"{failure}: timed out after {timeout} seconds.")
                time.sleep(0.5)
            returncode = process.returncode
    except OSError as exc:
        raise InstallError(f"Could not start {command[0]}: {exc}") from exc
    if returncode != 0:
        detail = _log_tail(log_path)
        raise InstallError(
            f"{failure}.{chr(10) + detail if detail else ''}"
        )


def _build_log_path(architecture: str, step: str) -> Path:
    """Where a source-build log survives the staging tree being removed.

    These logs used to be written inside the staging directory, which ``_install_from_source``
    deletes in its ``finally``: the failure message told the user to read a file that no longer
    existed. Only the 40-line tail in the job error survived. Keeping them beside the install root
    makes the advice true for a build that failed minutes ago as well as the one that just failed.
    """
    directory = _install_root() / "logs"
    try:
        directory.mkdir(parents = True, exist_ok = True)
    except OSError:
        # An unwritable log directory must not fail the build it is describing.
        directory = _install_root()
    return directory / f"{architecture}-{step}.log"


def _run_source_build(
    source_root: Path,
    build_dir: Path,
    *,
    architecture: str,
    progress: Callable[[float], None],
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise InstallError(
            "Dory needs CMake to build from source. Install CMake and retry from Settings."
        )
    _set_job(message = "Configuring the Dory source build...")
    progress(0.72)
    configure_command = [
        cmake,
        "-S",
        str(source_root),
        "-B",
        str(build_dir),
        "-Wno-dev",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DGGML_NATIVE=ON",
        "-DLLAMA_CURL=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if sys.platform == "win32":
        # Matches setup.ps1: the static build otherwise links both MSVC runtimes.
        configure_command.append("-DCMAKE_EXE_LINKER_FLAGS=/NODEFAULTLIB:LIBCMT")
    _run_logged_command(
        configure_command,
        cwd = source_root,
        log_path = _build_log_path(architecture, "configure"),
        timeout = _SOURCE_CONFIGURE_TIMEOUT_SECONDS,
        failure = "Dory CMake configure failed",
    )
    _set_job(message = "Compiling Dory source; this can take several minutes...")
    progress(0.78)
    _run_logged_command(
        [
            cmake,
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-server",
            "--parallel",
            str(_source_build_jobs()),
        ],
        cwd = source_root,
        log_path = _build_log_path(architecture, "build"),
        timeout = _SOURCE_BUILD_TIMEOUT_SECONDS,
        failure = "Dory CMake build failed",
    )
    progress(0.95)


def _install_from_source(
    profile: ArchitectureProfile,
    *,
    progress: Callable[[float], None],
) -> dict:
    if not profile.source_commit or not profile.repo:
        raise InstallError("This architecture has no pinned source build.")
    source_ready, missing = _source_build_tools()
    if not source_ready:
        raise InstallError(
            f"Dory needs {missing or 'build tools'} to build from source. "
            "Install it and retry from Settings."
        )
    staging = _install_root() / f".staging-{profile.id}-{uuid4().hex}"
    archive = staging / "source.bin"
    extracted = staging / "source"
    try:
        _download_asset(
            {
                "url": f"https://github.com/{profile.repo}/archive/{profile.source_commit}.tar.gz",
                "size": None,
                "digest": None,
            },
            archive,
            progress = progress,
        )
        progress(0.72)
        _extract_archive(archive, extracted)
        source_root = _source_root(extracted)
        _run_source_build(
            source_root,
            source_root / "build",
            architecture = profile.id,
            progress = progress,
        )
        binary = _find_binary(source_root)
        if binary is None:
            raise InstallError(
                "Dory built but did not produce llama-server. The CMake build log is at "
                f"{_build_log_path(profile.id, 'build')}."
            )
        # Install the runtime beside its shared libraries, not the fetched source
        # tree and CMake objects. Static builds need only the binary, but copying
        # its output directory also keeps macOS Metal or Windows DLL outputs whole.
        package = staging / "package"
        shutil.copytree(binary.parent, package / "bin", dirs_exist_ok = True)
        package_binary = package / "bin" / binary.name
        _set_job(message = "Installing the compiled Dory runtime...")
        return _promote_install(
            profile,
            package,
            package_binary,
            {
                "release_tag": _source_tag(profile),
                "asset": None,
                "digest": None,
                "source_build": True,
                "source_commit": profile.source_commit,
            },
            progress = progress,
        )
    finally:
        shutil.rmtree(staging, ignore_errors = True)


def _run_install(profile: ArchitectureProfile, activate: bool) -> None:
    from_tag = None
    try:
        marker, _binary = _installed(profile.id)
        from_tag = marker.get("release_tag") if marker else None
        _set_job(
            state = "running",
            architecture = profile.id,
            message = f"Looking up the latest {profile.label} build...",
            progress = 0.02,
            from_tag = from_tag,
        )
        if profile.repo is None:
            raise InstallError("This architecture has no published build source yet.")
        release = _latest_release(profile.repo, force_refresh = True)
        asset = _choose_asset(profile.id, release) if release is not None else None

        def progress(value: float) -> None:
            _set_job(progress = max(0.0, min(float(value), 0.99)))

        if asset is not None:
            _set_job(
                message = f"Downloading {asset['name']}...",
                to_tag = release.get("tag") if release else None,
                progress = 0.08,
            )
            marker = _install_from_asset(profile, release or {}, asset, progress = progress)
        elif profile.source_commit:
            source_ready, missing = _source_build_tools()
            if not source_ready:
                raise InstallError(
                    f"Dory needs {missing or 'build tools'} to build from source. "
                    "Install it and retry from Settings."
                )
            _set_job(
                message = "Downloading the pinned Dory source...",
                to_tag = _source_tag(profile),
                progress = 0.05,
            )
            marker = _install_from_source(profile, progress = progress)
        else:
            raise InstallError(
                f"{profile.label} has no published release assets yet. Nothing was installed."
            )
        installed_tag = marker.get("release_tag")
        activated = False
        activation_error = None
        if activate:
            try:
                _set_active(profile.id)
                activated = True
            except Exception as exc:
                activation_error = str(exc)
                logger.warning(
                    "llama architecture: install succeeded but activation failed",
                    architecture = profile.id,
                    error = activation_error,
                )
        suffix = "" if activated or not activate else f" Could not switch to it: {activation_error}"
        _set_job(
            state = "success",
            message = (
                f"Built and installed {profile.label} from source.{suffix}"
                if asset is None
                else f"Installed {profile.label}.{suffix}"
            ),
            progress = 1.0,
            to_tag = installed_tag,
            activated = activated if activate else None,
            finished_at = _utcnow(),
        )
    except Exception as exc:
        message = str(exc) if isinstance(exc, InstallError) else "The architecture install failed."
        logger.warning(
            "llama architecture: install failed",
            architecture = profile.id,
            error = str(exc),
        )
        _set_job(
            state = "error",
            message = message,
            error = str(exc),
            progress = None,
            from_tag = from_tag,
            finished_at = _utcnow(),
        )
    finally:
        try:
            _OPERATION_LOCK.release()
        except RuntimeError:
            pass


def start_install(architecture: str, *, activate: bool = True) -> dict:
    profile = _profile(architecture)
    if profile is None:
        return {
            "started": False,
            "reason": "unknown_architecture",
            "message": f"{architecture!r} is not a llama.cpp architecture Unsloth can install.",
            "job": get_job(),
        }
    if profile.kind == "bundled":
        return {
            "started": False,
            "reason": "bundled",
            "message": "Normal uses the bundled llama.cpp runtime; it does not need an install.",
            "job": get_job(),
        }
    if profile.kind == "not_ready" or profile.repo is None:
        return {
            "started": False,
            "reason": "build_not_ready",
            "message": f"{profile.label} has no published build yet.",
            "job": get_job(),
        }
    env_lock = _runtime_env_locked()
    if env_lock:
        return {
            "started": False,
            "reason": "environment_override",
            "message": f"The llama.cpp path is managed by the {env_lock} environment variable.",
            "job": get_job(),
        }
    if not _OPERATION_LOCK.acquire(blocking = False):
        return {
            "started": False,
            "reason": "already_running",
            "message": "Another llama.cpp install or switch is already running.",
            "job": get_job(),
        }
    _set_job(**_new_job())
    job = _set_job(
        state = "running",
        architecture = profile.id,
        message = f"Preparing the {profile.label} build...",
        progress = 0.0,
        started_at = _utcnow(),
    )
    thread = threading.Thread(
        target = _run_install,
        args = (profile, activate),
        name = f"llama-architecture-{profile.id}",
        daemon = True,
    )
    try:
        thread.start()
    except Exception:
        _OPERATION_LOCK.release()
        raise
    return {"started": True, "reason": None, "message": None, "job": job}


def activate(architecture: str) -> dict:
    """Record ``architecture`` as the runtime the resolver should use.

    Runs under the same install lock as ``start_install``. Without it, a switch
    can land between the two directory renames in ``_promote_install``: the
    install directory then briefly holds no marker, so a switch either fails
    with "install this first" while the fork is in fact installed, or records a
    tree that is about to be replaced by the one still being promoted. Refusing
    while another install or switch is in flight is the honest answer, and it is
    the same reason ``start_install`` refuses.
    """
    if architecture not in _ARCHITECTURE_IDS:
        raise ValueError(f"{architecture!r} is not a llama.cpp architecture.")
    env_lock = _runtime_env_locked()
    if env_lock:
        raise ValueError(f"The llama.cpp path is managed by the {env_lock} environment variable.")
    if not _OPERATION_LOCK.acquire(blocking = False):
        raise ValueError(
            "Another llama.cpp install or switch is already running. Retry when it finishes."
        )
    try:
        if architecture == DEFAULT_ARCHITECTURE:
            _set_active(DEFAULT_ARCHITECTURE)
        else:
            marker, binary = _installed(architecture)
            if marker is None or binary is None:
                raise ValueError("Install this architecture before switching to it.")
            _set_active(architecture)
    finally:
        _OPERATION_LOCK.release()
    # Outside the lock: get_status may resolve releases over the network.
    return get_status()


def _reset_for_tests() -> None:
    """Clear module state between hermetic tests."""
    with _JOB_LOCK:
        _JOB.clear()
        _JOB.update(_new_job())
    with _RELEASE_LOCK:
        _RELEASE_CACHE.clear()
        _RELEASE_FAILED_AT.clear()
    try:
        while _OPERATION_LOCK.locked():
            _OPERATION_LOCK.release()
    except RuntimeError:
        pass
