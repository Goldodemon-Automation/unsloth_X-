# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The NPU provider probe cache: it must save work without ever changing the answer.

Every test counts probe calls against a host with no NPU runtime at all, so the numbers here say
what the cache did rather than what happens to be installed on the machine running them.
"""

from __future__ import annotations

import threading
import time

import pytest

from utils.hardware import npu_providers as np


@pytest.fixture(autouse = True)
def _fresh_cache():
    """No test may inherit rows another test cached."""
    np.invalidate()
    yield
    np.invalidate()


def _hermetic(monkeypatch) -> None:
    """A host with no NPU runtime: every probe answers from nothing and none of them raises."""
    monkeypatch.setattr(np, "_distribution", lambda package: (False, None))
    monkeypatch.setattr(np, "_module_present", lambda module: False)
    monkeypatch.setattr(np, "_spec_origin", lambda module: None)
    monkeypatch.setattr(np.shutil, "which", lambda name: None)
    monkeypatch.setattr(np.sys, "platform", "linux")


def _counting_probe(monkeypatch, name: str) -> list[int]:
    """Replace one probe with a counter that still answers exactly as the real one would."""
    calls: list[int] = []
    real = np._PROBES[name]

    def counted():
        calls.append(1)
        return real()

    monkeypatch.setitem(np._PROBES, name, counted)
    return calls


def test_two_calls_inside_the_ttl_probe_once(monkeypatch):
    _hermetic(monkeypatch)
    calls = _counting_probe(monkeypatch, "openvino")

    first = np.npu_runtime_providers()
    second = np.npu_runtime_providers()

    assert len(calls) == 1
    assert first == second
    assert tuple(row["name"] for row in second) == np.PROVIDERS


def test_invalidate_forces_a_probe(monkeypatch):
    _hermetic(monkeypatch)
    calls = _counting_probe(monkeypatch, "lemonade")

    np.npu_runtime_providers()
    np.invalidate()
    np.npu_runtime_providers()

    assert len(calls) == 2


def test_a_raising_probe_is_never_cached(monkeypatch):
    _hermetic(monkeypatch)
    calls: list[int] = []

    def boom():
        calls.append(1)
        raise RuntimeError("broken install")

    monkeypatch.setitem(np._PROBES, "vitis_ai", boom)

    for _ in range(3):
        rows = {row["name"]: row for row in np.npu_runtime_providers()}
        # Only the failing provider loses its result; the rest of the matrix still answers.
        assert len(rows) == len(np.PROVIDERS)
        assert rows["vitis_ai"]["installed"] is False
        assert "probe failed" in rows["vitis_ai"]["detail"]
    assert len(calls) == 3


def test_cached_rows_equal_a_fresh_probes_rows(monkeypatch):
    _hermetic(monkeypatch)
    monkeypatch.setattr(
        np,
        "_distribution",
        lambda package: (True, "2025.1.0") if package == "openvino" else (False, None),
    )
    calls = _counting_probe(monkeypatch, "openvino")

    warm = np.npu_runtime_providers()
    cached = np.npu_runtime_providers()
    assert len(calls) == 1  # the second call really did come from the cache
    np.invalidate()
    fresh = np.npu_runtime_providers()
    assert len(calls) == 2

    assert cached == fresh == warm
    assert cached[0]["installed"] is True and cached[0]["version"] == "2025.1.0"


def test_a_callers_edit_cannot_reach_the_cache(monkeypatch):
    _hermetic(monkeypatch)
    calls = _counting_probe(monkeypatch, "openvino")

    first = np.npu_runtime_providers()
    first[0]["installed"] = True
    first.clear()
    second = np.npu_runtime_providers()

    assert len(calls) == 1
    assert second[0]["installed"] is False


def test_a_changed_host_is_not_answered_from_the_cache(monkeypatch):
    _hermetic(monkeypatch)
    calls = _counting_probe(monkeypatch, "fastflowlm")

    def rows() -> dict[str, dict]:
        return {row["name"]: row for row in np.npu_runtime_providers()}

    assert rows()["fastflowlm"]["installed"] is False
    # A binary appearing on PATH changes what the probe would say, so the cache cannot win.
    monkeypatch.setattr(np.shutil, "which", lambda name: f"/usr/bin/{name}")
    found = rows()["fastflowlm"]

    assert found["installed"] is True
    assert found["detail"] == "/usr/bin/flm"
    assert len(calls) == 2


def test_the_ttl_running_out_probes_again(monkeypatch):
    _hermetic(monkeypatch)
    monkeypatch.setattr(np, "_CACHE_TTL_SECONDS", 0.0)
    calls = _counting_probe(monkeypatch, "openvino")

    np.npu_runtime_providers()
    np.npu_runtime_providers()

    assert len(calls) == 2


def test_concurrent_callers_share_one_probe(monkeypatch):
    _hermetic(monkeypatch)
    calls: list[int] = []
    real = np._PROBES["openvino"]

    def slow():
        calls.append(1)
        time.sleep(0.02)  # long enough for the other threads to arrive at the lock
        return real()

    monkeypatch.setitem(np._PROBES, "openvino", slow)

    results: list[list[dict]] = []
    threads = [
        threading.Thread(target = lambda: results.append(np.npu_runtime_providers()))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 30)

    assert len(results) == 8
    assert len(calls) == 1
    assert all(result == results[0] for result in results)
