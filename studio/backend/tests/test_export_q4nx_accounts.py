# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account converts to Q4NX only from GGUFs inside its own workspace."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from models import ConvertQ4NXRequest
from routes import export as export_routes
from utils.account_context import OWNER, AccountContext, arun_as
from utils.paths import workspace_root

ALICE = AccountContext("alice-q4nx-account", "alice")
BOB = AccountContext("bob-q4nx-account", "bob")


def _convert(
    account,
    gguf_path = None,
    **fields,
):
    request = ConvertQ4NXRequest(
        save_directory = "q4nx",
        gguf_path = str(gguf_path) if gguf_path else None,
        base_model = fields.pop("base_model", "unsloth/Qwen3-0.6B"),
        hf_token = "own-token",
        **fields,
    )
    return asyncio.run(
        arun_as(account, export_routes.convert_q4nx(request, "subject", allow_ambient = False))
    )


@pytest.fixture
def authorized(monkeypatch):
    checked = []
    monkeypatch.setattr(
        export_routes.account_access,
        "authorize_download",
        lambda repo, repo_type, token: checked.append(repo),
    )
    return checked


@pytest.fixture
def converted(monkeypatch, authorized):
    calls = []
    real_to_thread = asyncio.to_thread

    async def to_thread(fn, *args, **kwargs):
        if fn is export_routes.account_access.authorize_download:
            return await real_to_thread(fn, *args, **kwargs)
        calls.append(fn)
        raise RuntimeError("stop before converting")

    monkeypatch.setattr(export_routes.asyncio, "to_thread", to_thread)
    return calls


def test_hub_repos_are_authorized_for_the_caller_before_converting(converted, authorized):
    with pytest.raises(HTTPException):
        _convert(ALICE, repo_id = "org/private-GGUF", filename = "m.Q4_1.gguf")
    assert authorized == ["org/private-GGUF", "unsloth/Qwen3-0.6B"]


def test_a_refused_hub_repo_stops_before_converting(monkeypatch, converted):
    def refuse(repo, repo_type, token):
        raise HTTPException(status_code = 404, detail = "Repository not found")

    monkeypatch.setattr(export_routes.account_access, "authorize_download", refuse)
    with pytest.raises(HTTPException) as exc:
        _convert(ALICE, repo_id = "org/private-GGUF", filename = "m.Q4_1.gguf")
    assert exc.value.status_code == 404
    assert converted == []


def test_a_companion_symlink_out_of_the_workspace_is_refused(tmp_path):
    from core.export import q4nx
    from utils.account_context import run_as

    base = run_as(ALICE, workspace_root) / "models" / "base"
    base.mkdir(parents = True, exist_ok = True)
    secret = tmp_path / "other_account_template.jinja"
    secret.write_text("private")
    link = base / "chat_template.jinja"
    link.unlink(missing_ok = True)
    link.symlink_to(secret)

    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, q4nx._fetch_base_file, str(base), "chat_template.jinja", None)
    assert exc.value.status_code == 403
    assert run_as(OWNER, q4nx._fetch_base_file, str(base), "chat_template.jinja", None) == link


def test_managed_account_cannot_convert_another_accounts_gguf(converted):
    from utils.account_context import run_as

    foreign = run_as(BOB, workspace_root) / "exports" / "private.Q4_1.gguf"
    with pytest.raises(HTTPException) as exc:
        _convert(ALICE, foreign)
    assert exc.value.status_code == 403
    assert converted == []


def test_managed_account_and_owner_reach_the_converter_for_allowed_paths(converted):
    from utils.account_context import run_as

    own = run_as(ALICE, workspace_root) / "exports" / "mine.Q4_1.gguf"
    for account, path in ((ALICE, own), (OWNER, "/anywhere/model.Q4_1.gguf")):
        with pytest.raises(HTTPException) as exc:
            _convert(account, path)
        assert exc.value.status_code == 400 and "stop before converting" in exc.value.detail
    assert len(converted) == 2


def test_a_failed_conversion_reports_no_host_path(tmp_path, monkeypatch):
    """The converter's failure names the absolute source and output paths; a tunnel caller owns
    neither. The reason survives the redaction so the caller can still act on it."""
    from core.export import q4nx

    gguf = tmp_path / "model.Q4_1.gguf"
    gguf.write_bytes(b"GGUF")
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setattr(q4nx, "require_converter_deps", lambda: None)

    def boom(source, base_model, out_dir, token = None):
        raise RuntimeError(f"{source} is not a readable GGUF, cannot write {out_dir}")

    monkeypatch.setattr(q4nx, "convert_existing_gguf", boom)

    request = ConvertQ4NXRequest(
        save_directory = str(tmp_path / "out"), gguf_path = str(gguf), base_model = str(base)
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(export_routes.convert_q4nx(request, "subject", allow_ambient = False))
    assert exc.value.status_code == 400
    assert "Q4NX conversion failed" in exc.value.detail
    assert "is not a readable GGUF" in exc.value.detail
    assert str(tmp_path) not in exc.value.detail
    # The two request paths survive as their own last component, so the reason still reads.
    assert "model.Q4_1.gguf" in exc.value.detail and "out" in exc.value.detail
