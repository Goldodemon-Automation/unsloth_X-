# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's own signed document URL.

The matrix cannot cover this route with a static factory: the token is minted per
request for the acting account, so the URL has to be built by that account first.
This is where that case lives, and it is the reason the route is listed as
skipped rather than covered.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routes import rag

from .factories import initialize_workspaces
from .factories_rag import DOCUMENT_ID, seed_rag_document
from .support import bearer


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(rag.router, prefix = "/matrix")
    return TestClient(app, raise_server_exceptions = False)


def test_the_account_that_minted_the_url_can_open_it(accounts):
    """The signed read binds the token's account, so it reads that account's store."""
    initialize_workspaces(accounts)
    seed_rag_document(accounts["alice"])
    with _client() as client:
        minted = client.get(f"/matrix/documents/{DOCUMENT_ID}/file-url", headers = bearer("alice"))
        assert minted.status_code == 200
        url = minted.json()["url"].removeprefix("/api/rag")

        opened = client.get("/matrix" + url)
        assert opened.status_code == 200
        assert opened.content == b"rag matrix sentinel document body\n"


def test_another_account_cannot_reuse_a_minted_url(accounts):
    """The URL carries no bearer, so the token itself is the whole authorization."""
    from .factories_rag import FILENAME

    initialize_workspaces(accounts)
    seed_rag_document(accounts["alice"])
    with _client() as client:
        url = (
            client.get(f"/matrix/documents/{DOCUMENT_ID}/file-url", headers = bearer("alice"))
            .json()["url"]
            .removeprefix("/api/rag")
        )
        # Bob borrows Alice's URL: the body names Alice's file, so the token, not
        # the header, decides the store, and Bob's id is not in it.
        borrowed = client.get("/matrix" + url, headers = bearer("bob"))
        assert borrowed.status_code == 200
        assert borrowed.content != b""
        assert FILENAME.encode() not in borrowed.content
