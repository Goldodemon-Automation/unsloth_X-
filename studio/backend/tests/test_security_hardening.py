# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The three hardening fixes a security review asked for, pinned so they cannot quietly regress.

Each of these was a real gap: a CORS policy that reflected any origin while allowing credentials, a
preview page that permitted inline script on the origin that holds the session tokens, and a desktop
webview CSP without `base-uri`. The tests are source-level where the effect is configuration the app
builds at import time, and behavioural where it is cheap to be.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_cors_never_allows_credentials_with_a_wildcard_origin():
    """Starlette reflects the request Origin when both are set, which turns a wildcard policy into
    "any site the user has open may read credentialed responses from this backend"."""
    main_source = (_BACKEND / "main.py").read_text(encoding = "utf-8")

    assert "allow_credentials = False" in main_source
    # The wildcard policy is deliberate and still there; it is the pairing that was wrong.
    assert 'return ["*"]' in (_BACKEND / "utils" / "host_policy.py").read_text(encoding = "utf-8")


def test_the_preview_page_uses_a_nonce_instead_of_unsafe_inline():
    from routes import preview

    nonce = "test-nonce-value"
    page = preview._PREVIEW_PAGE_HTML.replace("__CSP_NONCE__", nonce)
    policy = preview._PREVIEW_PAGE_CSP.format(nonce = nonce)

    assert f'nonce="{nonce}"' in page
    # The single inline block is stamped; nothing is left permitted by a blanket rule.
    assert "__CSP_NONCE__" not in page
    script_src = policy.split("script-src", 1)[1].split(";", 1)[0]
    assert f"'nonce-{nonce}'" in script_src
    assert "unsafe-inline" not in script_src
    assert "base-uri" in policy and "object-src" in policy


def test_the_desktop_csp_does_not_fall_back_for_base_uri():
    """`base-uri` does not inherit from `default-src`, so a webview without it can be re-pointed with
    a single injected `<base>` tag -- and this webview exposes the whole Tauri IPC surface."""
    config = json.loads((_BACKEND.parent / "src-tauri" / "tauri.conf.json").read_text(encoding = "utf-8"))
    csp = config["app"]["security"]["csp"]

    assert "base-uri 'none'" in csp
    assert "object-src 'none'" in csp
