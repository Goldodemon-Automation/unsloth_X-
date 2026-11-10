# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NPU runtime provider matrix: every row present, and nothing imported to find out.

Distribution metadata, module specs and PATH lookups are stubbed, so these tests never depend on
what happens to be installed on the machine running them.
"""

from __future__ import annotations

from utils.hardware import npu_providers as np

_TASK_TOKENS = {"inference", "tokenize", "eval", "qat_calibrate"}


def _distributions(table: dict[str, tuple[bool, str | None]]):
    return lambda package: table.get(package, (False, None))


def _rows() -> dict[str, dict]:
    return {row["name"]: row for row in np.npu_runtime_providers()}


def _row(name: str) -> dict:
    return _rows()[name]


def _nothing_installed(monkeypatch) -> None:
    """The host has no NPU runtime at all: every probe must still answer."""
    monkeypatch.setattr(np, "_distribution", _distributions({}))
    monkeypatch.setattr(np, "_module_present", lambda module: False)
    monkeypatch.setattr(np, "_spec_origin", lambda module: None)
    monkeypatch.setattr(np.shutil, "which", lambda name: None)
    monkeypatch.setattr(np.sys, "platform", "linux")


def test_every_provider_row_is_always_returned(monkeypatch):
    _nothing_installed(monkeypatch)
    rows = np.npu_runtime_providers()

    assert tuple(row["name"] for row in rows) == np.PROVIDERS
    assert np.PROVIDERS == (
        "openvino",
        "onnxruntime_qnn",
        "vitis_ai",
        "fastflowlm",
        "lemonade",
        "windows_ml",
    )
    for row in rows:
        assert set(row) == {"name", "installed", "version", "detail", "vendors", "supports"}
        assert row["installed"] is False
        assert isinstance(row["vendors"], tuple) and row["vendors"]
        assert set(row["supports"]) <= _TASK_TOKENS


def test_openvino_reports_its_metadata_version(monkeypatch):
    _nothing_installed(monkeypatch)
    monkeypatch.setattr(np, "_distribution", _distributions({"openvino": (True, "2025.1.0")}))

    row = _row("openvino")
    assert row["installed"] is True
    assert row["version"] == "2025.1.0"
    assert row["vendors"] == ("intel",)
    assert row["supports"] == ("inference", "tokenize", "eval", "qat_calibrate")


def test_qnn_needs_the_package_and_the_htp_backend(monkeypatch, tmp_path):
    _nothing_installed(monkeypatch)
    monkeypatch.setattr(np, "_distribution", _distributions({"onnxruntime": (True, "1.22.0")}))

    missing = _row("onnxruntime_qnn")
    assert missing["installed"] is False
    assert "QnnHtp" in missing["detail"]
    assert missing["supports"] == ("inference", "eval")

    (tmp_path / "QnnHtp.dll").write_text("", encoding = "utf-8")
    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path))
    found = _row("onnxruntime_qnn")
    assert found["installed"] is True
    assert found["version"] == "1.22.0"
    assert "QnnHtp" in found["detail"]
    assert found["vendors"] == ("qualcomm",)


def test_qnn_is_absent_without_onnxruntime_even_with_the_sdk(monkeypatch, tmp_path):
    _nothing_installed(monkeypatch)
    (tmp_path / "libQnnHtp.so").write_text("", encoding = "utf-8")
    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path))

    assert _row("onnxruntime_qnn")["installed"] is False


def test_vitis_ai_accepts_the_qat_package_on_its_own(monkeypatch):
    _nothing_installed(monkeypatch)
    monkeypatch.setattr(
        np,
        "_distribution",
        _distributions({"vitis_ai_quantization_pytorch": (True, "1.0.0")}),
    )

    row = _row("vitis_ai")
    assert row["installed"] is True
    assert row["version"] == "1.0.0"
    assert row["vendors"] == ("amd",)
    assert row["supports"] == ("inference", "eval", "qat_calibrate")


def test_vitis_ai_prefers_the_runtime_version(monkeypatch):
    _nothing_installed(monkeypatch)
    monkeypatch.setattr(
        np,
        "_distribution",
        _distributions(
            {"vitis_ai_runtime": (True, "2.0.0"), "vitis_ai_quantization_pytorch": (True, "1.0.0")}
        ),
    )

    row = _row("vitis_ai")
    assert row["installed"] is True and row["version"] == "2.0.0"


def test_path_binaries_are_detected_with_which(monkeypatch):
    _nothing_installed(monkeypatch)
    monkeypatch.setattr(
        np.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "flm" else None
    )

    rows = _rows()
    assert rows["fastflowlm"]["installed"] is True
    assert rows["fastflowlm"]["detail"] == "/usr/bin/flm"
    assert rows["fastflowlm"]["vendors"] == ("amd",)
    assert rows["lemonade"]["installed"] is False
    assert rows["lemonade"]["vendors"] == ("amd", "intel", "qualcomm")


def test_windows_ml_only_exists_on_windows(monkeypatch):
    _nothing_installed(monkeypatch)
    monkeypatch.setattr(
        np,
        "_distribution",
        _distributions({"winrt-Windows.AI.MachineLearning": (True, "1.0.0")}),
    )

    row = _row("windows_ml")
    assert row["installed"] is False
    assert row["supports"] == ("inference",)


def test_windows_ml_resolves_the_winrt_package_on_windows(monkeypatch):
    _nothing_installed(monkeypatch)
    monkeypatch.setattr(np.sys, "platform", "win32")
    monkeypatch.setattr(
        np,
        "_distribution",
        _distributions({"winrt-Windows.AI.MachineLearning": (True, "1.0.0")}),
    )

    row = _row("windows_ml")
    assert row["installed"] is True
    assert row["version"] == "1.0.0"
    assert row["vendors"] == ("microsoft", "qualcomm", "intel", "amd")
    assert row["supports"] == ("inference",)


def test_a_raising_probe_reports_only_its_own_row(monkeypatch):
    _nothing_installed(monkeypatch)

    def boom():
        raise RuntimeError("broken install")

    monkeypatch.setitem(np._PROBES, "openvino", boom)

    rows = _rows()
    assert len(rows) == len(np.PROVIDERS)
    assert rows["openvino"]["installed"] is False
    assert "probe failed" in rows["openvino"]["detail"]
    # The declared capability is a provider fact, not a probe result, so it survives the failure.
    assert rows["openvino"]["supports"] == ("inference", "tokenize", "eval", "qat_calibrate")


def _provider(name: str, installed: bool, vendors: tuple[str, ...], supports: tuple[str, ...]):
    return {
        "name": name,
        "installed": installed,
        "version": None,
        "detail": None,
        "vendors": vendors,
        "supports": supports,
    }


_FAKE = [
    _provider("openvino", True, ("intel",), ("inference", "eval")),
    _provider("vitis_ai", False, ("amd",), ("inference", "qat_calibrate")),
    _provider("lemonade", True, ("amd", "intel"), ("inference",)),
]


def test_provider_for_vendor_skips_what_is_not_installed():
    assert np.provider_for_vendor("amd", _FAKE)["name"] == "lemonade"
    assert np.provider_for_vendor("intel", _FAKE)["name"] == "openvino"
    assert np.provider_for_vendor("qualcomm", _FAKE) is None
    assert np.provider_for_vendor("apple", _FAKE) is None


def test_providers_supporting_returns_only_installed_ones():
    assert [p["name"] for p in np.providers_supporting("inference", _FAKE)] == [
        "openvino",
        "lemonade",
    ]
    # vitis_ai claims qat_calibrate but is not installed, so it cannot be offered.
    assert np.providers_supporting("qat_calibrate", _FAKE) == []
    assert np.providers_supporting("tokenize", _FAKE) == []


def test_the_default_provider_list_is_used_when_none_is_passed(monkeypatch):
    _nothing_installed(monkeypatch)
    assert np.provider_for_vendor("intel") is None
    assert np.providers_supporting("inference") == []
