# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The remembered compute target: validation, forgiveness and the one setting key.

Reads are total and writes are strict, which is the pair the Train page needs: a corrupt row must
not break the page, and a typo must not become "auto" and surprise the next run. The normalisation
is pinned because the same value arrives from a `<select>`, from a URL and from a stored row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils import compute_target_settings as settings  # noqa: E402


@pytest.fixture()
def store(monkeypatch):
    """A stand-in for storage.studio_db, injected the way the module imports it."""
    import types

    module = types.ModuleType("storage.studio_db")
    module.rows = {}
    module.get_app_setting = lambda key, default = None: module.rows.get(key, default)

    def upsert(values):
        module.rows.update(values)

    module.upsert_app_settings = upsert
    monkeypatch.setitem(sys.modules, "storage.studio_db", module)
    return module


def test_an_unset_row_reads_as_auto(store):
    assert settings.get_selected_compute_target() == "auto"


def test_a_stored_target_is_returned(store):
    store.rows[settings.COMPUTE_TARGET_SETTING_KEY] = "igpu"

    assert settings.get_selected_compute_target() == "igpu"


def test_a_corrupt_row_reads_as_auto_rather_than_raising(store):
    for corrupt in ("vulkan", 7, None, ["cpu"], ""):
        store.rows[settings.COMPUTE_TARGET_SETTING_KEY] = corrupt

        assert settings.get_selected_compute_target() == "auto"


def test_a_database_that_cannot_answer_reads_as_auto(monkeypatch):
    import types

    module = types.ModuleType("storage.studio_db")

    def explode(*_args, **_kwargs):
        raise RuntimeError("database is locked")

    module.get_app_setting = explode
    monkeypatch.setitem(sys.modules, "storage.studio_db", module)

    # A settings row must never be able to break a polled route.
    assert settings.get_selected_compute_target() == "auto"


def test_writing_normalises_and_persists(store):
    assert settings.set_selected_compute_target("  IGPU ") == "igpu"
    assert store.rows[settings.COMPUTE_TARGET_SETTING_KEY] == "igpu"


def test_writing_refuses_an_unknown_target_and_names_the_alternatives(store):
    with pytest.raises(ValueError) as caught:
        settings.set_selected_compute_target("tpu")

    message = str(caught.value)
    assert "tpu" in message
    for accepted in settings.VALID_COMPUTE_TARGETS:
        assert accepted in message
    # Nothing was written on the way out.
    assert store.rows == {}


def test_writing_refuses_a_non_string(store):
    for value in (None, 3, b"cpu"):
        with pytest.raises(ValueError):
            settings.set_selected_compute_target(value)


def test_auto_is_the_only_non_registry_value(store):
    assert settings.DEFAULT_COMPUTE_TARGET == "auto"
    assert settings.VALID_COMPUTE_TARGETS[0] == "auto"
    assert set(settings.VALID_COMPUTE_TARGETS[1:]) == {"cpu", "igpu", "dgpu", "npu"}
