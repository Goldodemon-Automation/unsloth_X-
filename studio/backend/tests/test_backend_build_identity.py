# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""This fork's backend must be distinguishable from the upstream package it shares a name with.

The fork's wheel is `unsloth` at the same base version as the upstream release it is cut from, so
`pip install "unsloth>=2026.10.6"` can be satisfied by PyPI and silently install the upstream
package -- whose backend has no `/api/labs`, no `/api/settings/self-training` and no
`/api/llama/architectures`. The UI then shows "API endpoint not found" on exactly those rows.

Two things stop that: the fork's version carries `+x8.N` (a PEP 440 local version, which cannot be
published to PyPI, so no index can satisfy it), and the backend reports which build it is on both
health endpoints so the mismatch is visible instead of inferred.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _fork_version() -> str:
    match = re.search(
        r'__version__ = "([^"]+)"', (REPO / "unsloth" / "_version.py").read_text(encoding = "utf-8")
    )
    assert match, "unsloth/_version.py has no __version__ literal"
    return match.group(1)


def test_the_shipped_version_carries_the_fork_local_segment():
    version = _fork_version()
    assert "+x8" in version, (
        "the backend version lost its fork marker, so an index can satisfy it with the upstream "
        f"package of the same base version; got {version!r}"
    )
    # PEP 440: the base is X.Y.Z, the local segment is everything after '+'. The base has to keep
    # matching the upstream release this tree is cut from, or the installer floor stops describing it.
    base = version.split("+", 1)[0]
    assert re.fullmatch(r"\d+\.\d+\.\d+", base), base
    assert re.fullmatch(r"x8(\.\d+)+", version.split("+", 1)[1]), version


def test_the_fork_marker_is_a_local_version_an_index_cannot_serve():
    """A local segment is the point: PyPI rejects it, so it can only come from a shipped file."""
    version = _fork_version()
    local = version.split("+", 1)[1]
    # PEP 440 normalises local versions to lowercase alphanumerics separated by dots/dashes/underscores.
    assert re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", local), local
    assert local == local.lower()


def test_the_backend_reports_which_build_it_is():
    import main

    assert main.backend_build_kind(f"{_fork_version()}") == "x8"
    assert main.backend_build_kind("2026.10.6") == "upstream"
    assert main.backend_build_kind("2026.11.4") == "upstream"
    assert main.backend_build_kind("dev") == "dev"
    assert main.backend_build_kind("") == "upstream"


def test_the_installer_refuses_a_venv_that_is_not_the_fork_build():
    """The guard is in install.ps1; assert it exists and names both cases it must catch."""
    ps1 = (REPO / "install.ps1").read_text(encoding = "utf-8")
    assert "Bundled Unsloth backend wheel is missing from this build" in ps1, (
        "a desktop build with no bundled wheel must fail instead of falling through to PyPI"
    )
    assert "Installed backend is not the X8 build" in ps1, (
        "the installer must read the installed version back, not assume the wheel landed"
    )
    assert re.search(r"\-notmatch '\\\+x8'", ps1), "the version check must look for the +x8 marker"
    # And it must only demand the wheel when the desktop shell is the caller.
    assert "UNSLOTH_DESKTOP_BACKEND_VERSION" in ps1
