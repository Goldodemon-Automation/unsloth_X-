# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the llama.cpp fork installer."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import warnings
import zipfile
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.llama_cpp_architectures as arch  # noqa: E402


@pytest.fixture(autouse = True)
def _isolated_install(monkeypatch, tmp_path):
    arch.reset_caches()
    arch._reset_for_tests()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    for name in ("LLAMA_SERVER_PATH", "UNSLOTH_LLAMA_CPP_PATH", "UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH"):
        monkeypatch.delenv(name, raising = False)
    yield
    arch.reset_caches()
    arch._reset_for_tests()


_RELEASE_URL = "https://github.com/o/r/releases/download/t/a.zip"


def _release_asset(*, size = None, digest = None, url: str = _RELEASE_URL) -> dict:
    """A release asset as the installer receives it from the GitHub API."""
    return {"url": url, "size": size, "digest": digest}


def _asset(name: str, size: int = 123) -> dict:
    return {
        "name": name,
        "url": f"https://example.test/{name}",
        "size": size,
        "digest": None,
    }


def _tar_member(name: str, *, kind: str = tarfile.REGTYPE, linkname: str = "") -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    return info


def _write_tar(path: Path, members, mode: str = "w") -> Path:
    """Write a tar from (TarInfo, payload or None) pairs."""
    with tarfile.open(path, mode) as bundle:
        for info, payload in members:
            if payload is None:
                bundle.addfile(info)
                continue
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    return path


class _Opener:
    """A stand-in for urlopen that reports a chosen final URL and body."""

    def __init__(self, body: bytes, *, final_url: str = _RELEASE_URL):
        self._body = body
        self._final_url = final_url
        self.calls = 0

    def __call__(self, _request, timeout = None):
        self.calls += 1
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def geturl(self):
        return self._final_url

    def read(self, _size = -1):
        body, self._body = self._body, b""
        return body


class _DripOpener(_Opener):
    """A server that never stalls and never finishes: one byte per read."""

    def read(self, _size = -1):
        self.calls += 1
        if self.calls > 500:
            raise AssertionError("the download loop never gave up on a drip feed")
        return b"x"


class _FakeClock:
    """Monotonic clock that advances on every read, without sleeping."""

    def __init__(self, step: float = 25.0):
        self._now = 0.0
        self._step = step

    def monotonic(self) -> float:
        self._now += self._step
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += seconds


def _install_stub(architecture: str) -> Path:
    install_dir = arch._install_dir(architecture)
    binary = install_dir / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True, exist_ok = True)
    binary.write_text("stub", encoding = "utf-8")
    (install_dir / arch.ARCHITECTURE_MARKER_NAME).write_text(
        '{"architecture":"%s","binary_relpath":"build/bin/llama-server"}' % architecture,
        encoding = "utf-8",
    )
    return binary


def test_neuralls_windows_picks_the_combined_runtime_not_cudart(monkeypatch):
    monkeypatch.setattr(arch, "_host", lambda: ("windows", "x64"))
    release = {
        "tag": "b15065",
        "assets": [
            _asset("cudart-llama-b15065-bin-win-cuda-13.4-x64.zip"),
            _asset("llama-b15065-bin-win-cuda-13.4-vulkan-hip-x64.zip"),
        ],
    }

    chosen = arch._choose_asset("neurall", release)

    assert chosen is not None
    assert chosen["name"].endswith("vulkan-hip-x64.zip")


def test_a_future_dory_windows_cpu_asset_is_installable(monkeypatch):
    monkeypatch.setattr(arch, "_host", lambda: ("windows", "x64"))
    release = {
        "tag": "b1",
        "assets": [
            _asset("llama-b1-bin-win-cuda-12-x64.zip"),
            _asset("llama-b1-bin-win-cpu-x64.zip"),
        ],
    }

    chosen = arch._choose_asset("dory", release)

    assert chosen is not None
    assert chosen["name"] == "llama-b1-bin-win-cpu-x64.zip"


def test_bee_linux_prefers_its_cpu_bundle_over_gpu_bundles(monkeypatch):
    assert arch._profile("bee").repo == "Anbeeld/beellama.cpp"
    monkeypatch.setattr(arch, "_host", lambda: ("linux", "x64"))
    release = {
        "tag": "v0.4.9",
        "assets": [
            _asset("beellama-v0.4.9-bin-ubuntu-cuda-13.3-x64.tar.gz"),
            _asset("beellama-v0.4.9-bin-ubuntu-rocm-7.2-x64.tar.gz"),
            _asset("beellama-v0.4.9-bin-ubuntu-vulkan-x64.tar.gz"),
            _asset("beellama-v0.4.9-bin-ubuntu-x64.tar.gz"),
        ],
    }

    chosen = arch._choose_asset("bee", release)

    assert chosen is not None
    assert chosen["name"] == "beellama-v0.4.9-bin-ubuntu-x64.tar.gz"


def test_ik_linux_never_picks_a_gpu_asset(monkeypatch):
    monkeypatch.setattr(arch, "_host", lambda: ("linux", "x64"))
    release = {
        "tag": "main-b5658",
        "assets": [
            _asset("ik_llama-main-b5658-bin-ubuntu-vulkan-x64-avx2.zip"),
            _asset("ik_llama-main-b5658-bin-ubuntu-x64-avx2.zip"),
        ],
    }

    chosen = arch._choose_asset("ik", release)

    assert chosen is not None
    assert chosen["name"] == "ik_llama-main-b5658-bin-ubuntu-x64-avx2.zip"


def test_dory_falls_back_to_pinned_source_without_a_release(monkeypatch):
    monkeypatch.setattr(arch, "_active_id", lambda: "normal")
    monkeypatch.setattr(arch, "_latest_release", lambda repo, force_refresh = False: None)
    monkeypatch.setattr(arch, "_source_build_tools", lambda: (True, None))

    status = arch.get_status()
    by_id = {profile["id"]: profile for profile in status["profiles"]}

    assert by_id["dory"]["reason"] == "source_build"
    assert by_id["dory"]["available"] is True
    assert by_id["dory"]["installable"] is True
    assert by_id["dory"]["source_build"] is True
    assert by_id["dory"]["source_commit"] == "1c3cdd23787750814aa21a070dc4d6dc451af871"
    assert by_id["x"]["reason"] == "build_not_ready"
    assert by_id["x"]["selectable"] is False


def test_dory_source_build_is_blocked_when_cmake_is_missing(monkeypatch):
    monkeypatch.setattr(arch, "_active_id", lambda: "normal")
    monkeypatch.setattr(arch, "_latest_release", lambda repo, force_refresh = False: None)
    monkeypatch.setattr(arch, "_source_build_tools", lambda: (False, "cmake"))

    status = arch.get_status()
    dory = next(profile for profile in status["profiles"] if profile["id"] == "dory")

    assert dory["reason"] == "build_tools_missing"
    assert dory["available"] is False
    assert dory["installable"] is False
    assert dory["selectable"] is False


def test_download_rejects_a_bad_sha256(tmp_path):
    payload = b"not the build"

    class _Response:
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, _size):
            nonlocal payload
            chunk, payload = payload, b""
            return chunk

    with pytest.raises(arch.InstallError, match = "SHA-256"):
        arch._download_asset(
            {
                "url": "https://example.test/build.zip",
                "size": len(payload),
                "digest": "sha256:" + "0" * 64,
            },
            tmp_path / "build.zip",
            progress = lambda _value: None,
            opener = lambda _request, timeout = 60: _Response(),
        )


def test_extraction_rejects_a_path_that_escapes_the_directory(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "no")

    with pytest.raises(arch.InstallError, match = "unsafe path"):
        arch._extract_archive(archive, tmp_path / "out")

    assert not (tmp_path / "escaped.txt").exists()


def test_extraction_rejects_a_backslash_traversal_and_a_drive_letter(tmp_path):
    """A zip built on Windows can name members with backslashes or drive letters.

    Only the extracting layer interprets a name, and a backslash is a separator
    only on Windows: on a POSIX host ``..\\..\\escaped`` is one literal file
    name, while on Windows it is the same traversal, and ``C:\\Windows\\...``
    addresses the real system directory instead of the install directory. Both
    are refused on every host rather than rewritten into something else.
    """
    for name in ("..\\..\\escaped.txt", "C:\\Windows\\System32\\planted.dll"):
        archive = tmp_path / "win.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(name, "no")
        with pytest.raises(arch.InstallError, match = "unsafe path"):
            arch._extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()
    assert not list((tmp_path / "out").glob("*Windows*"))


def test_extraction_rejects_an_absolute_member_instead_of_rewriting_it(tmp_path):
    """An absolute member used to be silently rewritten under the destination.

    Rewriting is worse than refusing: the file lands somewhere the archive did
    not name, so what the user ends up with is a path no check ever agreed to.
    """
    archive = tmp_path / "absolute.tar"
    _write_tar(archive, [(_tar_member("/etc/planted.txt"), b"no")])

    with pytest.raises(arch.InstallError, match = "unsafe path"):
        arch._extract_archive(archive, tmp_path / "out")

    assert not (tmp_path / "out" / "etc").exists()


def test_extraction_rejects_a_tar_symlink_that_escapes(tmp_path):
    """A tar member name is not the whole attack surface: the link target is.

    An archive can publish ``runtime`` as a symlink to ``..`` and then a second
    member ``runtime/planted``, so a name-only check passes while the write
    lands outside the staging directory as the user running Studio.
    """
    archive = tmp_path / "symlink.tar"
    _write_tar(
        archive,
        [
            (_tar_member("runtime", kind = tarfile.SYMTYPE, linkname = ".."), None),
            (_tar_member("runtime/planted.txt"), b"no"),
        ],
    )

    with pytest.raises(arch.InstallError, match = "links outside"):
        arch._extract_archive(archive, tmp_path / "out")

    assert not (tmp_path / "planted.txt").exists()


def test_extraction_rejects_an_absolute_symlink_in_a_tar(tmp_path):
    """An absolute link target points at the host, not at the archive."""
    archive = tmp_path / "absolute_link.tar"
    link = "/etc" if sys.platform != "win32" else "C:\\Windows"
    _write_tar(archive, [(_tar_member("hooks", kind = tarfile.SYMTYPE, linkname = link), None)])

    with pytest.raises(arch.InstallError, match = "links outside"):
        arch._extract_archive(archive, tmp_path / "out")


def test_extraction_rejects_a_tar_hardlink_that_escapes(tmp_path):
    """A hardlink target is relative to the archive root and equally trusted."""
    archive = tmp_path / "hardlink.tar"
    _write_tar(
        archive,
        [
            (
                _tar_member("shadow", kind = tarfile.LNKTYPE, linkname = "../../../../etc/shadow"),
                None,
            )
        ],
    )

    with pytest.raises(arch.InstallError, match = "links outside"):
        arch._extract_archive(archive, tmp_path / "out")


def test_extraction_never_hands_members_to_extractall(tmp_path, monkeypatch):
    """Validation and extraction must agree on exactly one path per member.

    Both stdlib extractors re-derive their own destination, and ``zipfile``
    quietly rewrites absolute names. Extracting member by member through the
    validated path is what makes the check authoritative.
    """

    def _refuse(*_args, **_kwargs):
        raise AssertionError("extraction must not delegate to extractall")

    monkeypatch.setattr(tarfile.TarFile, "extractall", _refuse)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", _refuse)

    archive = tmp_path / "build.tar"
    _write_tar(archive, [(_tar_member("build/bin/llama-server"), b"stub")])
    arch._extract_archive(archive, tmp_path / "from_tar")
    assert (tmp_path / "from_tar" / "build" / "bin" / "llama-server").read_bytes() == b"stub"

    zip_archive = tmp_path / "build.zip"
    with zipfile.ZipFile(zip_archive, "w") as bundle:
        bundle.writestr("build/bin/llama-server", "stub")
    arch._extract_archive(zip_archive, tmp_path / "from_zip")
    assert (tmp_path / "from_zip" / "build" / "bin" / "llama-server").read_text() == "stub"


def test_extraction_rejects_a_link_escape_without_tarfile_filters(tmp_path, monkeypatch):
    """tarfile's ``filter`` argument only exists from Python 3.11.4 onwards.

    The installer used to fall back to a bare ``extractall`` when the keyword
    was missing, which honoured symlinks and device nodes with no checks at all.
    On such a Python the only thing standing between the archive and the user's
    filesystem is this module's own validation.
    """
    real_extractall = tarfile.TarFile.extractall

    def _old_extractall(self, path = ".", members = None, *, numeric_owner = False, **kwargs):
        if kwargs:
            raise TypeError("extractall() got an unexpected keyword argument 'filter'")
        return real_extractall(self, path, members, numeric_owner = numeric_owner)

    monkeypatch.setattr(tarfile.TarFile, "extractall", _old_extractall)

    archive = tmp_path / "symlink.tar"
    _write_tar(
        archive,
        [
            (_tar_member("runtime", kind = tarfile.SYMTYPE, linkname = ".."), None),
            (_tar_member("runtime/planted.txt"), b"no"),
        ],
    )
    destination = tmp_path / "deep" / "out"
    destination.mkdir(parents = True)

    with pytest.raises(arch.InstallError, match = "links outside"):
        arch._extract_archive(archive, destination)

    assert not (tmp_path / "planted.txt").exists()


def test_extraction_rejects_special_tar_members(tmp_path):
    """A fifo or device node in the install dir is not a build artifact.

    Without an explicit check the entry is either created (a fifo another local
    process can write through) or refused by ``tarfile`` with an error type the
    installer cannot report, so the job fails with "the install failed" instead
    of naming the archive as hostile.
    """
    for kind in (tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE):
        archive = tmp_path / "special.tar"
        _write_tar(archive, [(_tar_member("pipe", kind = kind), None)])
        with pytest.raises(arch.InstallError, match = "special file"):
            arch._extract_archive(archive, tmp_path / "out")


def test_extraction_rejects_duplicate_members(tmp_path):
    """Two members with one name make the result depend on archive order.

    Whatever gets extracted is decided by the last entry, so a benign-looking
    first copy cannot be what the installer inspected or reported.
    """
    archive = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        # zipfile warns while the archive is built; the archive is the point.
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("build/bin/llama-server", "first")
            bundle.writestr("build/bin/llama-server", "second")

    with pytest.raises(arch.InstallError, match = "duplicate"):
        arch._extract_archive(archive, tmp_path / "out")


def test_extraction_accepts_a_realistic_tar_gz_build(tmp_path):
    """The checks must not refuse a real release: dirs, payloads and version links.

    A fork tarball repeats directory entries and, on Unix, ships a versioned
    shared library with a symlink beside it, so both have to keep extracting.
    """
    binary_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    archive = tmp_path / "ik.tar.gz"
    members = [
        (_tar_member("build", kind = tarfile.DIRTYPE), None),
        (_tar_member("build", kind = tarfile.DIRTYPE), None),
        (_tar_member("build/bin", kind = tarfile.DIRTYPE), None),
        (_tar_member(f"build/bin/{binary_name}", kind = tarfile.REGTYPE), b"stub"),
        (_tar_member("build/lib/libllama.so.0.0.1", kind = tarfile.REGTYPE), b"lib"),
    ]
    if sys.platform != "win32":
        members.append(
            (
                _tar_member(
                    "build/lib/libllama.so",
                    kind = tarfile.SYMTYPE,
                    linkname = "libllama.so.0.0.1",
                ),
                None,
            )
        )
    _write_tar(archive, members, mode = "w:gz")

    arch._extract_archive(archive, tmp_path / "out")

    assert (tmp_path / "out" / "build" / "bin" / binary_name).read_bytes() == b"stub"
    assert arch._find_binary(tmp_path / "out").name == binary_name
    if sys.platform != "win32":
        link = tmp_path / "out" / "build" / "lib" / "libllama.so"
        assert link.is_symlink()
        assert link.read_bytes() == b"lib"


def test_release_lookup_gives_up_on_a_drip_feed(monkeypatch):
    """The API read runs on the install thread and on every status poll.

    A per-socket timeout does not bound a body that keeps trickling, so one
    stalled GitHub answer could wedge both the poll and the install lock.
    """
    monkeypatch.setattr(arch, "time", _FakeClock(step = 3.0))
    calls = {"count": 0}

    class _Api:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, _size = -1):
            calls["count"] += 1
            if calls["count"] > 200:
                raise AssertionError("the release lookup never gave up on a drip feed")
            return b"{"

    monkeypatch.setattr(arch.urllib.request, "urlopen", lambda *_args, **_kwargs: _Api())

    with pytest.raises(TimeoutError, match = "took longer than"):
        arch._read_json("https://api.github.com/repos/o/r/releases/latest", timeout = 8.0)


def test_extraction_rejects_an_archive_that_expands_too_large(tmp_path, monkeypatch):
    """A compression bomb is a disk-fill, not an install.

    The member count was capped but the expanded size was not, so one small
    entry could write until the volume filled. The cap is enforced on the bytes
    actually read, not on the size the archive claims.
    """
    monkeypatch.setattr(arch, "_MAX_EXTRACTED_BYTES", 32)
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression = zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("build/bin/llama-server", b"\0" * 4096)

    with pytest.raises(arch.InstallError, match = "expands too large"):
        arch._extract_archive(archive, tmp_path / "out")


def test_download_rejects_a_redirect_away_from_github(tmp_path):
    """urlopen follows redirects silently, including off GitHub and to plain HTTP.

    Most release assets carry no digest, so the host that finally serves the
    bytes is the only remaining guarantee about where the binary came from.
    """
    asset = _release_asset()
    for final_url in (
        "https://mirror.example/a.zip",
        "http://github.com/o/r/releases/download/t/a.zip",
        "https://github.com.evil.example/a.zip",
    ):
        with pytest.raises(arch.InstallError, match = "not GitHub"):
            arch._download_asset(
                asset,
                tmp_path / "a.zip",
                progress = lambda _value: None,
                opener = _Opener(b"body", final_url = final_url),
            )


def test_download_accepts_githubs_asset_cdn(tmp_path):
    """The asset CDN is GitHub's own, so a redirect to it must keep working."""
    for final_url in (
        "https://release-assets.githubusercontent.com/github-production-release-asset/x",
        "https://objects.githubusercontent.com/x",
        "https://codeload.github.com/o/r/tar.gz/abc",
    ):
        arch._download_asset(
            _release_asset(),
            tmp_path / "a.zip",
            progress = lambda _value: None,
            opener = _Opener(b"body", final_url = final_url),
        )


def test_download_refuses_a_build_bigger_than_the_cap(tmp_path, monkeypatch):
    """Nothing bounded the transfer, so an asset could fill the disk.

    GitHub's declared size is a hint from the same response, so the cap is
    enforced on the declared size before the first byte and on the bytes that
    actually arrive when the response declares nothing.
    """
    monkeypatch.setattr(arch, "_MAX_DOWNLOAD_BYTES", 64)
    declared = _Opener(b"")
    with pytest.raises(arch.InstallError, match = "larger than Unsloth will download"):
        arch._download_asset(
            _release_asset(size = 4096),
            tmp_path / "a.zip",
            progress = lambda _value: None,
            opener = declared,
        )
    assert declared.calls == 0

    with pytest.raises(arch.InstallError, match = "larger than Unsloth will download"):
        arch._download_asset(
            _release_asset(),
            tmp_path / "a.zip",
            progress = lambda _value: None,
            opener = _Opener(b"y" * 4096),
        )


def test_download_gives_up_on_a_drip_feed(tmp_path, monkeypatch):
    """A per-socket timeout does not bound a server that keeps trickling bytes.

    The install thread holds the install lock, so a transfer without a deadline
    wedges the whole architecture surface until the process restarts.
    """
    monkeypatch.setattr(arch, "time", _FakeClock(step = 25.0))

    with pytest.raises(arch.InstallError, match = "took longer than"):
        arch._download_asset(
            _release_asset(),
            tmp_path / "a.zip",
            progress = lambda _value: None,
            opener = _DripOpener(b""),
        )


def test_download_refuses_a_digest_it_cannot_verify(tmp_path):
    """A digest in another algorithm used to be skipped in complete silence.

    The install then claimed the bytes matched what GitHub published while
    nothing had been compared, which is worse than refusing the install.
    """
    for digest in ("md5:" + "0" * 32, "sha256:not-hex", "sha256:" + "0" * 8):
        opener = _Opener(b"body")
        with pytest.raises(arch.InstallError):
            arch._download_asset(
                _release_asset(digest = digest),
                tmp_path / "a.zip",
                progress = lambda _value: None,
                opener = opener,
            )
        assert opener.calls == 0


def test_download_verifies_a_sha512_digest(tmp_path):
    """The reported algorithm is the one compared, not always SHA-256."""
    payload = b"a real build"
    arch._download_asset(
        _release_asset(
            size = len(payload),
            digest = "sha512:" + hashlib.sha512(payload).hexdigest(),
        ),
        tmp_path / "a.zip",
        progress = lambda _value: None,
        opener = _Opener(payload),
    )
    assert (tmp_path / "a.zip").read_bytes() == payload


def test_a_marker_without_its_binary_is_a_broken_install_not_a_missing_one(monkeypatch):
    """Behaviour pinned: a deleted binary reads as ``missing_binary``, not as "installable".

    The prebuilt branch of the reason ladder used to run before the marker check, so a fork
    whose ``llama-server`` had been removed answered ``reason = None`` -- the picker showed it as
    something the user never installed, and the row's own repair text (the ``missingBinary``
    string) was unreachable. Reorder the ladder and the row says what is actually wrong.
    """
    marker, _binary = arch._installed("ik")
    assert marker is None
    binary = _install_stub("ik")
    binary.unlink()
    monkeypatch.setattr(arch, "_latest_release", lambda repo, force_refresh = False: None)

    status = arch.get_status()
    row = next(profile for profile in status["profiles"] if profile["id"] == "ik")

    assert row["reason"] == "missing_binary"
    assert row["installed"] is False
    assert row["available"] is False, "a broken install must not be offered as usable"


def test_a_prebuilt_row_without_a_marker_still_reports_why_it_cannot_install(monkeypatch):
    """Behaviour pinned: the reorder above did not swallow the not-installed reasons.

    A fork with no marker and no matching asset must still say ``no_build_for_platform``-class
    reasons rather than ``missing_binary``: the two states have different fixes (install vs
    reinstall), so telling them apart is the point of the ladder.
    """
    monkeypatch.setattr(arch, "_latest_release", lambda repo, force_refresh = False: None)

    status = arch.get_status()
    rows = {profile["id"]: profile for profile in status["profiles"]}

    for architecture in ("bee", "neurall", "ik", "dory"):
        assert rows[architecture]["reason"] != "missing_binary"
        assert rows[architecture]["installed"] is False


def test_source_build_logs_outlive_the_staging_tree(tmp_path, monkeypatch):
    """Behaviour pinned: a failure message may not point at a file that was already deleted.

    The CMake logs were written inside the staging directory, which the install removes in its
    ``finally``, so "check the CMake build log" referred to nothing by the time the user read it.
    They now live beside the install root, which survives the cleanup.
    """

    class _Process:
        def __init__(self, *args, **kwargs):
            self.returncode = 0

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(arch.shutil, "which", lambda name: "/usr/bin/cmake")
    monkeypatch.setattr(arch.subprocess, "Popen", _Process)

    source_root = tmp_path / "staging" / "source"
    source_root.mkdir(parents = True)
    arch._run_source_build(
        source_root, source_root / "build", architecture = "dory", progress = lambda _value: None
    )

    log = arch._build_log_path("dory", "build")
    assert log.is_file(), "the build log is missing"
    assert tmp_path in log.parents, "the log is not under the studio home"
    assert (tmp_path / "staging") not in log.parents, (
        "the log is inside the staging tree that an install deletes"
    )


def test_activate_refuses_while_an_install_holds_the_lock(monkeypatch):
    """A switch must not race the directory swap an install performs.

    _promote_install renames the install directory aside before the new tree and
    its marker exist, so a concurrent switch either fails as "not installed"
    for a fork that is installed, or records a tree that is about to be
    replaced. It also shares one lock with start_install for that reason.
    """
    _install_stub("ik")
    activated: list[str] = []
    monkeypatch.setattr(arch, "_set_active", lambda name: activated.append(name))
    monkeypatch.setattr(arch, "_latest_release", lambda repo, force_refresh = False: None)

    assert arch._OPERATION_LOCK.acquire(blocking = False) is True
    try:
        with pytest.raises(ValueError, match = "already running"):
            arch.activate("ik")
    finally:
        arch._OPERATION_LOCK.release()

    assert activated == []
    assert arch._OPERATION_LOCK.locked() is False


def test_source_build_runs_an_argument_list_without_a_shell(tmp_path, monkeypatch):
    """The build must never hand a manifest string to a command interpreter.

    The repository and commit come from the manifest and the paths come from a
    downloaded archive, so a shell would turn either into command execution.
    Both CMake calls are argument lists, and the pinned commit is the only thing
    the source URL names.
    """
    captured: list[dict] = []

    class _Process:
        def __init__(self, args, **kwargs):
            captured.append({"args": args, "kwargs": kwargs})
            self.returncode = 0

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(arch.shutil, "which", lambda name: "/usr/bin/cmake")
    monkeypatch.setattr(arch.subprocess, "Popen", _Process)

    source_root = tmp_path / "source"
    source_root.mkdir()
    build_dir = source_root / "build"

    arch._run_source_build(
        source_root, build_dir, architecture = "dory", progress = lambda _value: None
    )

    assert len(captured) == 2
    for call in captured:
        args = call["args"]
        assert isinstance(args, list)
        assert all(isinstance(item, str) for item in args)
        assert call["kwargs"].get("shell") is False
        assert not any(";" in item or "&&" in item or "|" in item for item in args)
    commit = arch._profile("dory").source_commit
    assert not any(commit in item for item in captured[0]["args"] + captured[1]["args"])


def test_active_architecture_directory_uses_the_installed_binary(monkeypatch):
    install_dir = arch._install_dir("ik")
    binary = install_dir / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    (install_dir / arch.ARCHITECTURE_MARKER_NAME).write_text(
        '{"architecture":"ik","binary_relpath":"build/bin/llama-server"}',
        encoding = "utf-8",
    )
    monkeypatch.setattr(arch, "_active_id", lambda: "ik")

    assert arch.active_architecture_directory() == binary.parent


def test_install_writes_an_atomic_marker_and_finds_the_binary(monkeypatch):
    profile = arch._profile("ik")
    assert profile is not None

    def _fake_download(_asset, destination, *, progress, opener = None):
        destination.parent.mkdir(parents = True, exist_ok = True)
        with zipfile.ZipFile(destination, "w") as bundle:
            bundle.writestr("build/bin/llama-server", "stub")
            bundle.writestr("build/bin/llama-server.exe", "stub")
        progress(0.5)

    monkeypatch.setattr(arch, "_download_asset", _fake_download)
    release = {"tag": "main-b5658", "assets": []}
    asset = _asset("ik_llama-main-b5658-bin-win-cpu-x86_64_v3-avx2.zip")

    marker = arch._install_from_asset(
        profile, release, asset, progress = lambda _value: None
    )

    install_dir = arch._install_dir("ik")
    binary = install_dir / marker["binary_relpath"]
    assert binary.is_file()
    assert (install_dir / arch.ARCHITECTURE_MARKER_NAME).is_file()
    assert marker["release_tag"] == "main-b5658"
    assert arch._read_marker("ik") == marker
    assert arch._installed("ik")[1] is not None


def test_a_failed_marker_write_never_leaves_a_half_install(monkeypatch):
    """A marker is what makes a directory "installed", so it is the last step.

    If anything fails after the previous tree is renamed aside, the previous
    install has to come back intact and no marker may survive for the tree that
    never finished. Otherwise the picker reports a fork whose files are gone.
    """
    profile = arch._profile("ik")

    def _fake_download(_asset, destination, *, progress, opener = None):
        destination.parent.mkdir(parents = True, exist_ok = True)
        with zipfile.ZipFile(destination, "w") as bundle:
            bundle.writestr("build/bin/llama-server", "previous build")
            bundle.writestr("build/bin/llama-server.exe", "previous build")
        progress(0.5)

    monkeypatch.setattr(arch, "_download_asset", _fake_download)
    release = {"tag": "main-b5658", "assets": []}
    asset = _asset("ik_llama-main-b5658-bin-win-cpu-x86_64_v3-avx2.zip")
    previous = arch._install_from_asset(
        profile, release, asset, progress = lambda _value: None
    )

    def _boom() -> str:
        raise RuntimeError("the marker could not be written")

    real_utcnow = arch._utcnow
    monkeypatch.setattr(arch, "_utcnow", _boom)
    release = {"tag": "main-b9999", "assets": []}
    try:
        with pytest.raises(RuntimeError):
            arch._install_from_asset(profile, release, asset, progress = lambda _value: None)
    finally:
        monkeypatch.setattr(arch, "_utcnow", real_utcnow)

    marker, binary = arch._installed("ik")
    assert binary is not None
    assert marker == previous
    assert binary.read_text(encoding = "utf-8") == "previous build"
    assert not list(arch._install_root().glob(".old-*"))
    assert not list(arch._install_root().glob(".staging-*"))


def test_source_install_builds_the_pinned_commit_and_records_it(monkeypatch):
    profile = arch._profile("dory")
    assert profile is not None
    requested = {}

    def _fake_source_download(_asset, destination, *, progress, opener = None):
        requested.update(_asset)
        destination.parent.mkdir(parents = True, exist_ok = True)
        with zipfile.ZipFile(destination, "w") as bundle:
            bundle.writestr("llama.cpp-dory-support/CMakeLists.txt", "project(dory)")
            bundle.writestr("llama.cpp-dory-support/ggml/CMakeLists.txt", "")
        progress(0.5)

    def _fake_source_build(source_root, build_dir, *, architecture, progress):
        assert architecture == "dory", "the build is handed the id its log file is named after"
        name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
        binary = build_dir / "bin" / name
        binary.parent.mkdir(parents = True)
        binary.write_text("stub", encoding = "utf-8")
        progress(0.9)

    monkeypatch.setattr(arch, "_download_asset", _fake_source_download)
    monkeypatch.setattr(arch, "_source_build_tools", lambda: (True, None))
    monkeypatch.setattr(arch, "_run_source_build", _fake_source_build)

    marker = arch._install_from_source(profile, progress = lambda _value: None)

    install_dir = arch._install_dir("dory")
    assert (install_dir / marker["binary_relpath"]).is_file()
    assert not (install_dir / "llama.cpp-dory-support").exists()
    assert marker["source_commit"] == "1c3cdd23787750814aa21a070dc4d6dc451af871"
    assert marker["source_build"] is True
    assert marker["release_tag"] == "source-1c3cdd23"
    assert arch._read_marker("dory") == marker
    assert requested["url"] == (
        "https://github.com/brianchmielNV/llama.cpp/archive/"
        "1c3cdd23787750814aa21a070dc4d6dc451af871.tar.gz"
    )
