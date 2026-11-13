# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which llama-server Studio actually runs, and what happens when that one is broken.

``core.inference.llama_cpp.LlamaCppBackend._find_llama_server_binary`` decides
the backend binary. Its precedence, highest first, is:

  1. ``LLAMA_SERVER_PATH``            environment: a direct path to a binary
  2. ``UNSLOTH_LLAMA_CPP_PATH``       environment: a llama.cpp folder
  3. the architecture activated in Settings (an installed llama.cpp fork)
  4. the custom llama.cpp folder selected in Settings
  5. the bundled / auto-downloaded runtime (``$UNSLOTH_HOME/llama.cpp``)
  6. legacy in-tree ``./llama.cpp``, then ``llama-server`` on ``PATH``, then
     the legacy extracted ``./bin/llama-server``

The two Settings pins (3 and 4) differ only in what a broken install does. A
custom folder is a tree the user maintains, so a llama-server that has gone
missing there stops the search -- silently running a different runtime while
Settings still shows that folder is the bug #5941 fixed. An activated
architecture is a tree this app installs and replaces, so an unusable one is
reported, the remedy is named, and the search continues with the next step:
returning None there failed every load with "run ``unsloth studio setup``",
advice that neither repairs a fork install nor clears the pin, so the same
failure came back on the next load.

Each test below states the failure mode it prevents. Nothing here downloads,
spawns a process or touches the network: every install root is a ``tmp_path``,
the settings store is a dict, and ``PATH`` is stubbed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.llama_cpp as llama_cpp  # noqa: E402
import storage.studio_db as studio_db  # noqa: E402
import utils.llama_cpp_architectures as arch  # noqa: E402
import utils.llama_cpp_path_settings as path_settings  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

FORK = "ik"


class _Runtime:
    """A throwaway install layout plus the two settings the resolver reads.

    ``UNSLOTH_HOME`` is the master root, so the bundled runtime is
    ``<root>/llama.cpp`` and the fork installs land under
    ``<root>/studio/llama.cpp-architectures`` -- exactly the layout the
    installer and the resolver agree on, with nothing of the developer's own
    ``~/.unsloth`` reachable.
    """

    def __init__(self, root: Path, store: dict):
        self.root = root
        self.store = store

    @property
    def bundled_root(self) -> Path:
        return self.root / "llama.cpp"

    @property
    def install_root(self) -> Path:
        return self.root / "studio" / "llama.cpp-architectures"

    def binary(self, directory: Path, *, layout: str = "build", platform = None) -> Path:
        """Write a llama-server in one of the layouts the resolver searches."""
        name = path_settings.llama_server_binary_name(platform)
        if layout == "root":
            binary = directory / name
        else:
            binary = directory / "build" / "bin" / name
        binary.parent.mkdir(parents = True, exist_ok = True)
        binary.write_bytes(b"stub llama-server")
        binary.chmod(0o755)
        return binary

    def bundled(self, *, layout: str = "build", platform = None) -> Path:
        return self.binary(self.bundled_root, layout = layout, platform = platform)

    def custom(
        self, name: str = "custom", *, layout: str = "build", platform = None
    ) -> tuple[Path, Path]:
        """A custom llama.cpp folder and the llama-server inside it."""
        directory = self.root / name
        return directory, self.binary(directory, layout = layout, platform = platform)

    def fork(self, architecture: str = FORK, *, layout: str = "build", platform = None) -> Path:
        """Install a fork the way the installer leaves it: tree plus marker.

        The marker records ``binary_relpath`` relative to the install directory,
        which is what makes a half-written tree distinguishable from a whole one.
        """
        install_dir = self.install_root / architecture
        assert arch._install_dir(architecture) == install_dir, (
            "the test layout has to be the one utils.llama_cpp_architectures uses, "
            f"got {arch._install_dir(architecture)}"
        )
        binary = self.binary(install_dir, layout = layout, platform = platform)
        (install_dir / arch.ARCHITECTURE_MARKER_NAME).write_text(
            json.dumps(
                {
                    "architecture": architecture,
                    "label": architecture,
                    "binary_relpath": binary.relative_to(install_dir).as_posix(),
                }
            ),
            encoding = "utf-8",
        )
        return binary

    def select_architecture(self, architecture: str) -> None:
        self.store[arch.ACTIVE_ARCHITECTURE_SETTING_KEY] = architecture

    def select_custom_folder(self, directory = None) -> None:
        self.store[path_settings.CUSTOM_LLAMA_CPP_PATH_SETTING_KEY] = (
            None if directory is None else str(directory)
        )


class _RecordingLogger:
    """Stands in for the module logger, which is a structlog bound logger rather
    than a stdlib one -- caplog never sees it, so an assertion against caplog
    would hold however loudly the resolver warned."""

    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append(str(message) % args if args else str(message))

    def __getattr__(self, name):
        def sink(message = "", *args, **kwargs):
            return None

        return sink


class _ProbeCounter:
    """Counts filesystem probes, so "no re-scan" is a number, not a timing."""

    def __init__(self, root: Path):
        self.root = root
        self.reset()

    def reset(self) -> None:
        self.is_file = 0
        self.access = 0
        self.tree_walks = 0

    @property
    def total(self) -> int:
        return self.is_file + self.access

    def under(self, path) -> bool:
        try:
            return Path(os.path.abspath(path)).is_relative_to(os.path.abspath(self.root))
        except (OSError, TypeError, ValueError):
            return False


@pytest.fixture()
def runtime(monkeypatch, tmp_path) -> _Runtime:
    root = tmp_path / "master"
    root.mkdir()
    monkeypatch.setenv("UNSLOTH_HOME", str(root))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(root / "studio"))
    for name in (
        "LLAMA_SERVER_PATH",
        "UNSLOTH_LLAMA_CPP_PATH",
        path_settings.MANAGED_LLAMA_CPP_PATH_MARKER,
    ):
        monkeypatch.delenv(name, raising = False)
    # A llama-server on the developer's PATH would outrank nothing here but would
    # be found by the last steps, and a stray cmake would make a source fork look
    # installable.
    monkeypatch.setattr(llama_cpp.shutil, "which", lambda _name: None)
    store: dict = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: store.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda values: store.update(values) or values
    )
    arch.reset_caches()
    arch._reset_for_tests()
    yield _Runtime(root, store)
    arch.reset_caches()
    arch._reset_for_tests()


@pytest.fixture()
def recorder(monkeypatch) -> _RecordingLogger:
    recording = _RecordingLogger()
    monkeypatch.setattr(llama_cpp, "logger", recording)
    return recording


@pytest.fixture()
def probes(monkeypatch, runtime) -> _ProbeCounter:
    """Count every probe that reaches the temp install tree.

    ``os.walk``/``rglob``/``glob`` are counted separately: a resolver that
    searched the tree for llama-server would answer the same way and cost
    whatever the tree happens to hold, which is what this pins against.
    """
    counter = _ProbeCounter(runtime.root)
    real_is_file = Path.is_file
    real_access = os.access
    real_walk = os.walk
    real_rglob = Path.rglob
    real_glob = Path.glob

    def counting_is_file(self):
        if counter.under(self):
            counter.is_file += 1
        return real_is_file(self)

    def counting_access(path, mode, *args, **kwargs):
        if counter.under(path):
            counter.access += 1
        return real_access(path, mode, *args, **kwargs)

    def counting_walk(top, *args, **kwargs):
        counter.tree_walks += 1
        return real_walk(top, *args, **kwargs)

    def counting_rglob(self, pattern, *args, **kwargs):
        counter.tree_walks += 1
        return real_rglob(self, pattern, *args, **kwargs)

    def counting_glob(self, pattern, *args, **kwargs):
        counter.tree_walks += 1
        return real_glob(self, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", counting_is_file)
    monkeypatch.setattr(os, "access", counting_access)
    monkeypatch.setattr(os, "walk", counting_walk)
    monkeypatch.setattr(Path, "rglob", counting_rglob)
    monkeypatch.setattr(Path, "glob", counting_glob)
    return counter


def test_the_documented_order_is_environment_fork_custom_then_bundled(runtime, monkeypatch):
    """The whole precedence chain, walked one step at a time.

    Prevents a future edit from reordering the steps silently: the environment
    overrides outrank everything (a service unit or a launch flag is the
    operator's last word), the activated fork outranks the custom folder (so
    choosing a fork does not overwrite that choice), and the bundled runtime is
    only reached when neither pin is set.
    """
    bundled = runtime.bundled()
    assert LlamaCppBackend._find_llama_server_binary() == str(bundled)

    custom_dir, custom_binary = runtime.custom()
    runtime.select_custom_folder(custom_dir)
    assert LlamaCppBackend._find_llama_server_binary() == str(custom_binary)

    fork_binary = runtime.fork()
    runtime.select_architecture(FORK)
    assert LlamaCppBackend._find_llama_server_binary() == str(fork_binary)

    env_directory = runtime.root / "env-llama"
    env_binary = runtime.binary(env_directory)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(env_directory))
    assert LlamaCppBackend._find_llama_server_binary() == str(env_binary)

    direct = runtime.root / "direct-llama-server"
    direct.write_bytes(b"stub llama-server")
    direct.chmod(0o755)
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(direct))
    assert LlamaCppBackend._find_llama_server_binary() == str(direct)

    monkeypatch.delenv("LLAMA_SERVER_PATH")
    assert LlamaCppBackend._find_llama_server_binary() == str(env_binary)

    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH")
    assert LlamaCppBackend._find_llama_server_binary() == str(fork_binary)


def test_activating_a_fork_outranks_the_custom_folder_and_deactivating_returns(runtime):
    """Switching a fork on and off moves the binary, and never rewrites the folder.

    Prevents two halves of the picker from disagreeing: activating a fork must
    take effect for the next load while the stored custom folder stays exactly
    as the user saved it, and going back to Normal must return to that folder
    rather than to the bundled runtime.
    """
    custom_dir, custom_binary = runtime.custom()
    runtime.select_custom_folder(custom_dir)
    fork_binary = runtime.fork()

    runtime.select_architecture(FORK)
    assert LlamaCppBackend._find_llama_server_binary() == str(fork_binary)
    assert (
        runtime.store[path_settings.CUSTOM_LLAMA_CPP_PATH_SETTING_KEY] == str(custom_dir)
    )

    runtime.select_architecture(arch.DEFAULT_ARCHITECTURE)
    assert LlamaCppBackend._find_llama_server_binary() == str(custom_binary)


def test_a_deleted_fork_binary_falls_back_to_the_next_runtime_and_names_the_remedy(
    runtime, recorder
):
    """A fork whose binary was deleted is reported, then the search continues.

    Prevents the dead end this replaced: the resolver returned None, so every
    load raised "no executable llama.cpp runtime ... run ``unsloth studio
    setup``" -- advice that neither reinstalls the fork nor clears the pin, so
    the next load failed identically and the working runtime sitting beside it
    was never used. The next step in the order wins, and the warning says which
    architecture broke and how to fix it.
    """
    bundled = runtime.bundled()
    custom_dir, custom_binary = runtime.custom()
    runtime.select_custom_folder(custom_dir)
    fork_binary = runtime.fork()
    runtime.select_architecture(FORK)

    fork_binary.unlink()

    assert LlamaCppBackend._find_llama_server_binary() == str(custom_binary)
    assert recorder.warnings, "a fork that cannot be used has to be reported"
    warning = recorder.warnings[-1]
    assert FORK in warning
    assert "Settings" in warning and "Normal" in warning, (
        f"the warning has to name the remedy, got: {warning!r}"
    )

    # With no custom folder left, the bundled runtime is the next step.
    runtime.select_custom_folder(None)
    assert LlamaCppBackend._find_llama_server_binary() == str(bundled)


def test_a_fork_binary_without_its_execute_bit_falls_back_and_names_the_remedy(
    runtime, recorder, monkeypatch
):
    """A fork llama-server that cannot be executed is not the one that runs.

    Prevents a load from dying on "permission denied" from a tree the app owns:
    an install that lost its execute bit (an archive extracted without modes, a
    repair that dropped it) is reported with the remedy and the next runtime is
    used, instead of returning None and blaming a missing runtime.
    """
    monkeypatch.setattr(llama_cpp.sys, "platform", "linux")
    fork_binary = runtime.fork(platform = "linux")
    bundled = runtime.bundled(platform = "linux")
    runtime.select_architecture(FORK)

    real_access = os.access
    monkeypatch.setattr(
        os,
        "access",
        lambda path, mode, *args, **kwargs: (
            False if str(path) == str(fork_binary) else real_access(path, mode, *args, **kwargs)
        ),
    )

    assert LlamaCppBackend._find_llama_server_binary() == str(bundled)
    warning = recorder.warnings[-1]
    assert str(fork_binary) in warning, f"the warning has to name the binary: {warning!r}"
    assert "Settings" in warning and "Normal" in warning, (
        f"the warning has to name the remedy, got: {warning!r}"
    )


def test_a_half_written_fork_install_is_not_installed_and_is_not_used(runtime, recorder, monkeypatch):
    """A marker without the binary it records is not an install.

    Prevents a half-written tree (an interrupted extract, a partially deleted
    install) from being launched: the recorded relpath is what makes the tree an
    install, so a missing binary there means "not installed" for the picker and
    "not usable" for the resolver, which falls through instead of running a
    llama-server out of a tree that is still being written.
    """
    runtime.select_architecture(FORK)
    install_dir = runtime.install_root / FORK
    install_dir.mkdir(parents = True, exist_ok = True)
    marker = {
        "architecture": FORK,
        "label": FORK,
        "binary_relpath": "build/bin/llama-server",
    }
    (install_dir / arch.ARCHITECTURE_MARKER_NAME).write_text(
        json.dumps(marker), encoding = "utf-8"
    )
    bundled = runtime.bundled()

    # The manifest agrees: a marker with no binary is not an installed fork.
    assert arch._installed(FORK) == (marker, None)
    assert arch.active_architecture_directory() is None

    monkeypatch.setattr(arch, "_latest_release", lambda *args, **kwargs: None)
    status = arch.get_status()
    row = next(profile for profile in status["profiles"] if profile["id"] == FORK)
    assert row["installed"] is False
    assert row["available"] is False
    # The reason the picker renders as "The installed build is missing
    # llama-server. Reinstall it." -- not a network problem, which is what the
    # row said before the marker was checked first.
    assert row["reason"] == "missing_binary"

    assert LlamaCppBackend._find_llama_server_binary() == str(bundled)
    assert recorder.warnings, "a fork that cannot be used has to be reported"


def test_a_settings_read_failure_degrades_to_the_default_runtime(runtime, monkeypatch):
    """A settings database that cannot be read must not take discovery down.

    Prevents a locked or half-migrated settings DB from turning every model load
    into an exception: both Settings pins become unreadable, so the resolver
    degrades to the bundled runtime, which is the state a fresh install is in.
    """
    runtime.fork()
    runtime.select_architecture(FORK)
    bundled = runtime.bundled()

    def unreadable(*args, **kwargs):
        raise RuntimeError("settings database is locked")

    monkeypatch.setattr(studio_db, "get_app_setting", unreadable)

    assert LlamaCppBackend._find_llama_server_binary() == str(bundled)


def test_a_failing_architecture_read_does_not_crash_the_resolver(runtime, recorder, monkeypatch):
    """A broken architecture module degrades to the default, with a warning.

    Prevents a failure inside the architecture lookup (a settings read that
    raises, an import that does not resolve) from propagating out of discovery
    and failing the load: it is reported, treated as "no fork selected", and the
    documented order continues.
    """
    runtime.fork()
    runtime.select_architecture(FORK)
    bundled = runtime.bundled()

    def unreadable():
        raise RuntimeError("llama_cpp_architectures is unavailable")

    monkeypatch.setattr(arch, "active_architecture_id", unreadable)

    assert LlamaCppBackend._find_llama_server_binary() == str(bundled)
    assert any("search order" in warning for warning in recorder.warnings), (
        f"the failure has to be reported, got: {recorder.warnings!r}"
    )


def test_repeated_resolution_is_stable_and_does_not_rescan_the_tree(runtime, probes):
    """Resolution is a fixed set of candidate probes, not a search of the tree.

    Prevents two regressions at once: a resolver that answered differently on
    the next call (so a load used a different binary than the status page
    reported), and one that walked the install tree every call, whose cost grows
    with the number of files in a fork install instead of staying bounded by the
    layouts it knows. Both are measured by counting probes, never by timing.
    """
    fork_binary = runtime.fork()
    custom_dir, _custom_binary = runtime.custom()
    runtime.select_custom_folder(custom_dir)
    runtime.select_architecture(FORK)

    probes.reset()
    first = LlamaCppBackend._find_llama_server_binary()
    first_probes = probes.total

    probes.reset()
    second = LlamaCppBackend._find_llama_server_binary()
    second_probes = probes.total

    assert first == second == str(fork_binary)
    assert first_probes == second_probes, "resolution must not accumulate state between calls"
    assert probes.tree_walks == 0, "resolution must not walk the install tree"
    assert first_probes <= 24, f"expected a bounded candidate probe, counted {first_probes}"

    # 25 more files under the install, in a tree the resolver has no layout for:
    # a scan of the tree would pay for every one of them, the fixed candidate
    # list pays for none. (Only the files directly beside the winning binary are
    # listed, by the GPU-capability check -- a bounded cost reported separately.)
    nested = fork_binary.parent.parent / "share" / "deep"
    nested.mkdir(parents = True)
    for index in range(25):
        (nested / f"junk-{index}.bin").write_bytes(b"x")
    probes.reset()
    assert LlamaCppBackend._find_llama_server_binary() == str(fork_binary)
    assert probes.tree_walks == 0, "resolution must not walk the install tree"
    assert probes.total == first_probes, (
        f"a file under the install grew the probe count ({first_probes} -> "
        f"{probes.total}): resolution is scanning the tree"
    )
