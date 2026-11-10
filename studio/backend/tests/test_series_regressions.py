# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pins for the behaviours this series added, so a later deletion cannot be silent.

Every test here answers one question: "if someone deleted the behaviour, would a test fail?"
Each block says which behaviour it holds in place, in a comment on the test itself.

Two kinds of pin are used, and the choice is deliberate:

* Behavioural pins drive the real code (``apply_compute_target_environment``, the NPU planner, the
  preview page renderer, the CORS middleware the app actually installs) and assert what it does.
* Source-level pins read the file with ``ast`` and assert the *shape* of the code, for the places
  whose behaviour cannot be exercised without a GPU, a subprocess or a live server: the training
  worker (it imports torch at module scope), the auth routes' limiter calls, and the route
  wiring.

Neither kind imports torch, starts a server or touches the network.
"""

from __future__ import annotations

import ast
import asyncio
import re
import sys
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


# --- helpers ---------------------------------------------------------------------------------

def _read(relative: str) -> str:
    return (_BACKEND / relative).read_text(encoding = "utf-8")


def _tree(relative: str) -> ast.Module:
    return ast.parse(_read(relative))


def _functions(node: ast.AST) -> dict:
    """Every function/method definition under ``node``, by name."""
    found = {}
    for sub in ast.walk(node):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(sub.name, []).append(sub)
    return found


def _called_names(node: ast.AST) -> set:
    """Every name a call inside ``node`` refers to (attribute tails included)."""
    names = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Attribute):
            names.add(func.attr)
        elif isinstance(func, ast.Name):
            names.add(func.id)
    return names


def _script_src(csp: str) -> str:
    """The ``script-src`` directive of a CSP header, without the other directives."""
    for directive in csp.split(";"):
        directive = directive.strip()
        if directive.startswith("script-src"):
            return directive
    return ""


def _constants(node: ast.AST) -> set:
    """Every string constant appearing in ``node``."""
    return {
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    }


# --- 1. CORS: a wildcard origin must never be paired with credentials -----------------------

def test_cors_never_pairs_a_wildcard_origin_with_credentials():
    """Behaviour pinned: the installed CORS middleware has allow_credentials False.

    Starlette answers "allow any origin + credentials" by *reflecting* the request's Origin and
    adding Access-Control-Allow-Credentials: true, so any page the user has open could then read
    credentialed cross-origin responses from the backend. Delete the ``allow_credentials = False``
    (or the middleware block) and this test fails.
    """
    import main

    entries = []
    for middleware in main.app.user_middleware:
        cls = getattr(middleware, "cls", None)
        if cls is not None and "CORS" in getattr(cls, "__name__", ""):
            entries.append(middleware)
    assert entries, "the CORS middleware is no longer installed on the app"

    for middleware in entries:
        kwargs = getattr(middleware, "kwargs", {})
        assert kwargs.get("allow_credentials") is False, (
            "the CORS middleware was given allow_credentials=True: with a wildcard origin "
            "that reflects the request Origin and lets any page read credentialed responses"
        )
        # The wildcard itself is expected (host_policy returns ["*"] outside the desktop
        # lockdown); it is the credentials flag that must stay off.
        if kwargs.get("allow_origins") == ["*"]:
            assert kwargs.get("allow_credentials") is False


def test_cors_source_pins_allow_credentials_false():
    """Source-level pin for the same behaviour, independent of how the app is wired.

    If the middleware call is rewritten (or the flag flipped) the literal ``False`` disappears
    from the call and this fails.
    """
    tree = _tree("main.py")
    cors_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_middleware"):
            continue
        keywords = {kw.arg for kw in node.keywords}
        if "allow_credentials" in keywords or "allow_origins" in keywords:
            cors_calls.append(node)
    assert cors_calls, "main.py no longer adds a CORS middleware with an origins/credentials pair"

    for call in cors_calls:
        for keyword in call.keywords:
            if keyword.arg != "allow_credentials":
                continue
            value = keyword.value
            assert isinstance(value, ast.Constant), (
                "allow_credentials is no longer a literal; the wildcard/credentials pairing "
                "must be decided in the source, not at runtime"
            )
            assert value.value is False


def test_default_cors_origin_is_still_a_wildcard():
    """Behaviour pinned: outside the desktop lockdown the allowlist is ["*"].

    The lockdown (api-only + not secure) narrows it, but the default web/Tauri bind does not.
    Narrowing the default by accident would break the webview and the tunnel, so the value is
    pinned here: change the default and this test fails.
    """
    from utils.host_policy import cors_origins_for_mode

    assert cors_origins_for_mode(api_only = False, secure = False) == ["*"]
    assert cors_origins_for_mode(api_only = False, secure = True) == ["*"]


# --- 2. The preview page's CSP uses a nonce, not 'unsafe-inline' -----------------------------

def test_preview_page_csp_script_src_has_no_unsafe_inline():
    """Behaviour pinned: the preview page's script-src is a nonce, never 'unsafe-inline'.

    The page shares an origin with the Studio UI, so a blanket inline-script allowance there would
    let any injected markup run with the session tokens the UI keeps. Re-add 'unsafe-inline' to
    script-src and this test fails.
    """
    from routes import preview

    directive = _script_src(preview._PREVIEW_PAGE_CSP)
    assert directive, "the preview page CSP no longer carries a script-src directive"
    assert "unsafe-inline" not in directive
    assert "'nonce-" in directive


def test_preview_page_stamps_a_nonce_into_the_script_tag():
    """Behaviour pinned: every response carries a fresh nonce, in the CSP and in the page.

    The template placeholder is replaced per response, so the served page has a concrete nonce in
    the script tag and the CSP header names the same one. Stop stamping the nonce (or leave the
    placeholder in the body) and this test fails.
    """
    from routes import preview

    original = preview._resolve_or_4xx
    preview._resolve_or_4xx = lambda *args, **kwargs: None
    try:
        response = preview._preview_page("run-name", None)
    finally:
        preview._resolve_or_4xx = original

    csp = response.headers["content-security-policy"]
    directive = _script_src(csp)
    assert "unsafe-inline" not in directive

    match = re.search(r"'nonce-([^']+)'", directive)
    assert match, "the served CSP header does not name a nonce"
    nonce = match.group(1)
    assert nonce

    body = response.body.decode("utf-8")
    assert "__CSP_NONCE__" not in body
    assert f'nonce="{nonce}"' in body


def test_preview_page_asset_carries_the_nonce_placeholder():
    """Behaviour pinned: the page template asks for the nonce in a nonce= attribute.

    The placeholder in the template is what the per-response nonce is stamped into. Remove the
    nonce attribute from the asset and this test fails.
    """
    template = _read("assets/preview_page.html")
    assert re.search(r"""nonce\s*=\s*["']__CSP_NONCE__["']""", template), (
        "the preview page template no longer asks for a nonce: the inline script would run "
        "under a CSP that does not name it"
    )


# --- 3. The credential limiter's key table is bounded, and the auth routes report outcomes ---

def test_rate_limit_key_table_is_bounded():
    """Behaviour pinned: the limiter's key table cannot grow without bound.

    An unbounded dict keyed by client address is a memory-exhaustion vector: a flood of distinct
    (or forged) source addresses would mint one entry per source. Make _MAX_KEYS unbounded (or
    drop the eviction) and this test fails.
    """
    from auth.rate_limit import AttemptLimiter, _MAX_KEYS

    assert isinstance(_MAX_KEYS, int)
    assert 0 < _MAX_KEYS < 1_000_000, "the tracked-key bound must be a finite, sane constant"

    limiter = AttemptLimiter(max_attempts = 3, window_s = 60.0, lockout_s = 60.0)
    for index in range(_MAX_KEYS + 50):
        limiter.record_failure(f"198.51.100.{index % 255}:{index}")

    entries = getattr(limiter, "_entries")
    assert len(entries) <= _MAX_KEYS


def test_auth_routes_record_failures_and_successes():
    """Behaviour pinned: /refresh and /change-password report every outcome to the limiter.

    A rejected credential must cost the caller a budget (record_failure) and an accepted one must
    clear it (record_success); without both, the lockout never engages and a guesser is free.
    Delete either call from either handler and this test fails.
    """
    tree = _tree("routes/auth.py")
    handlers = _functions(tree)

    for name in ("refresh", "change_password"):
        definitions = handlers.get(name)
        assert definitions, f"routes/auth.py no longer defines the {name} handler"

        found = False
        for definition in definitions:
            calls = _called_names(definition)
            if {"record_failure", "record_success"} <= calls:
                found = True
                break
        assert found, (
            f"the {name} handler no longer records both a failure and a success on the "
            "token_attempts limiter"
        )


def test_auth_route_paths_are_the_ones_being_pinned():
    """Behaviour pinned: the pinned handlers are the ones the routes actually publish.

    A rename of the path (or of the function) must not silently move the limiter calls onto a
    route nobody calls; the decorator is checked so the pin follows the published path.
    """
    tree = _tree("routes/auth.py")
    handlers = _functions(tree)

    expected = {"/refresh": "refresh", "/change-password": "change_password"}
    published = {}
    for name, definitions in handlers.items():
        for definition in definitions:
            for decorator in definition.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                if not (isinstance(func, ast.Attribute) and func.attr == "post"):
                    continue
                if decorator.args and isinstance(decorator.args[0], ast.Constant):
                    published[decorator.args[0].value] = name

    for path, name in expected.items():
        assert published.get(path) == name, (
            f"the {path} POST route is no longer handled by {name}(); the limiter pin would "
            "follow the wrong code"
        )


# --- 4. The worker refuses a 4-bit load on an integrated target ------------------------------

def test_compute_target_env_maps_an_igpu_without_a_device_pin():
    """Behaviour pinned: an "igpu" target sets the shared-memory flags and no device pin.

    The worker relies on this: it is the worker's own guard that refuses the 4-bit load, and the
    device pin is deliberately absent so a masked child reports the device it used instead of
    raising. Change the mapping and this test fails.
    """
    from core.training.compute_target_env import apply_compute_target_environment, target_kind

    assert target_kind("igpu") == "igpu"

    environ = {}
    applied = apply_compute_target_environment(
        {"compute_target": "igpu", "compute_target_vendor": "intel"},
        environ = environ,
    )

    assert environ.get("UNSLOTH_IGPU") == "1"
    assert "UNSLOTH_TRAINING_DEVICE" not in environ
    assert "__error__" not in applied


def test_worker_clears_load_in_4bit_for_an_igpu_target():
    """Source-level pin: the worker's run_training_process still refuses 4-bit on an iGPU.

    The worker imports torch at module scope, so it cannot be imported here; the block is read
    from the source instead. It must be an ``if`` that tests for the "igpu" target and assigns
    False to ``config["load_in_4bit"]``. Delete the block and this test fails.
    """
    tree = _tree("core/training/worker.py")
    handlers = _functions(tree)
    definitions = handlers.get("run_training_process")
    assert definitions, "core/training/worker.py no longer defines run_training_process"

    guarded = False
    for definition in definitions:
        for node in ast.walk(definition):
            if not isinstance(node, ast.If):
                continue
            constants = _constants(node.test)
            if "igpu" not in constants:
                continue
            for statement in ast.walk(node):
                if not isinstance(statement, ast.Assign):
                    continue
                for target in statement.targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    key = target.slice
                    if not isinstance(key, ast.Constant) or key.value != "load_in_4bit":
                        continue
                    if isinstance(statement.value, ast.Constant) and statement.value.value is False:
                        guarded = True
    assert guarded, (
        "run_training_process no longer clears load_in_4bit for an igpu target: a config from "
        "the CLI, a resumed run or an older build would take the 4-bit load an integrated "
        "target cannot afford"
    )


# --- 5. The compute route's probe cache is locked, and a POST invalidates it -----------------

def test_compute_probe_cache_uses_a_threading_lock():
    """Behaviour pinned: the probe cache is guarded by a threading.Lock.

    The payload is built on a worker thread, so an asyncio lock could not be taken here; a plain
    threading lock is what makes the cache single-flight. Swap it for something else and this
    test fails.
    """
    from routes import compute

    lock = getattr(compute, "_probe_lock")
    assert isinstance(lock, type(threading.Lock()))
    # Not an asyncio lock: the cache is read on a worker thread, where an asyncio lock cannot
    # be taken at all.
    assert not isinstance(lock, asyncio.Lock)


def test_compute_probe_cache_invalidation_clears_the_entry():
    """Behaviour pinned: _invalidate_probe_cache empties the cached probe.

    Without this the Train page would keep showing the pre-change answer for a whole minute.
    Empty the invalidator's body and this test fails.
    """
    from routes import compute

    compute._probe_cache.update(at = 123.0, probe = object(), npu = {"mode": "stub"})
    compute._invalidate_probe_cache()

    assert compute._probe_cache["at"] == 0.0
    assert compute._probe_cache["probe"] is None
    assert compute._probe_cache["npu"] is None


def test_compute_post_handler_invalidates_the_probe_cache(monkeypatch):
    """Behaviour pinned: choosing a target drops the cached probe.

    The POST handler must call _invalidate_probe_cache, otherwise a deliberate change is not
    reflected until the TTL expires. The handler is called directly (no server): the storage
    write and the probe are stubbed, and the cache is asserted afterwards.
    """
    from core.training.device_targets import TrainingTarget
    from routes import compute

    target = TrainingTarget(
        id = "igpu",
        device = "xpu:0",
        kind = "igpu",
        name = "integrated GPU",
        vendor = "intel",
        memory_bytes = None,
        unified_memory = True,
        supported = True,
    )

    monkeypatch.setattr(
        compute.compute_target_settings, "set_selected_compute_target", lambda choice: choice
    )
    monkeypatch.setattr(compute, "resolve_training_target", lambda requested, probe = None: target)
    monkeypatch.setattr(compute, "training_target_payload", lambda probe = None: {"targets": []})

    compute._probe_cache.update(at = 123.0, probe = object(), npu = {"mode": "stub"})

    asyncio.run(compute.choose_compute_target(compute.TargetChoice(target = "igpu"), "tester"))

    assert compute._probe_cache["probe"] is None
    assert compute._probe_cache["at"] == 0.0


def test_compute_post_handler_calls_the_invalidator_in_its_own_body():
    """Source-level pin for the same behaviour: the POST handler calls the invalidator.

    The behavioural test above can be satisfied by a call from anywhere else in the request
    path; this one insists the call is in choose_compute_target's own body. Move the call out
    and this test fails.
    """
    tree = _tree("routes/compute.py")
    handlers = _functions(tree)
    definitions = handlers.get("choose_compute_target")
    assert definitions, "routes/compute.py no longer defines choose_compute_target"

    assert any(
        "_invalidate_probe_cache" in _called_names(definition) for definition in definitions
    ), "choose_compute_target no longer calls _invalidate_probe_cache"


# --- 6. The NPU refusal token list covers every token the planner can emit -------------------

def test_refusal_tokens_cover_every_token_the_planner_can_emit():
    """Behaviour pinned: REFUSAL_TOKENS lists every token a plan's refused_reason can carry.

    The list is published so a UI can translate the reason; an incomplete list is how a reason
    reaches the user as snake_case. The planner is driven over every refusal path it has, and
    each reason it produced must be in the published list. Drop a token and this test fails.
    """
    from core.training.npu_assist import REFUSAL_TOKENS, plan_npu_assist

    assert isinstance(REFUSAL_TOKENS, (tuple, list, set, frozenset))
    published = set(REFUSAL_TOKENS)

    capabilities = [
        # An NPU that cannot be measured at all.
        {"mode": "unavailable", "reasons": [], "tasks": [], "providers": []},
        # A reason the contract does not know: the planner must map it onto a published token.
        {"mode": "unavailable", "reasons": ["something_new"], "tasks": [], "providers": []},
        # A working NPU that cannot take this run's task.
        {
            "mode": "assist",
            "backward_pass": False,
            "reasons": [],
            "tasks": ["tokenize"],
            "providers": [{"name": "stub", "installed": True, "supports": ["tokenize"]}],
        },
        # A working NPU with no installed runtime at all.
        {"mode": "assist", "backward_pass": False, "reasons": [], "tasks": [], "providers": []},
    ]
    # Each refusal the capability probe can report.
    capabilities.extend(
        {"mode": "unavailable", "reasons": [reason], "tasks": [], "providers": []}
        for reason in (
            "disabled",
            "absent",
            "no_runtime",
            "no_supported_task",
            "unsupported_family",
        )
    )

    emitted = set()
    for capability in capabilities:
        for requested in ("off", "assist", "full", "auto", None, "nonsense"):
            for primary_kind in ("dgpu", "npu", "cpu"):
                plan = plan_npu_assist(
                    capability,
                    requested = requested,
                    primary_kind = primary_kind,
                )
                if plan.refused_reason is not None:
                    emitted.add(plan.refused_reason)

    assert emitted, "the planner refused nothing; the pin would be vacuous"
    missing = sorted(emitted - published)
    assert not missing, f"the planner can emit refusal tokens that are not published: {missing}"


def test_refusal_tokens_cover_the_planner_and_the_probe():
    """Behaviour pinned: the published list names the planner's own reasons and the probe's.

    The planner's reasons ("no_backward_pass", "npu_is_primary", "no_supported_task") and every
    reason the capability probe can emit must be in REFUSAL_TOKENS. Remove one and this test
    fails.
    """
    from core.training.npu_assist import REFUSAL_TOKENS

    published = set(REFUSAL_TOKENS)

    planner_reasons = {"no_backward_pass", "npu_is_primary", "no_supported_task"}
    assert planner_reasons <= published, (
        "the planner's own refusal reasons are missing from REFUSAL_TOKENS: "
        f"{sorted(planner_reasons - published)}"
    )

    probe_source = _read("utils/hardware/npu.py")
    probe_reasons = set(
        re.findall(r'reasons"\]\s*=\s*\[([^\]]*)\]', probe_source)
    )
    emitted = set()
    for group in probe_reasons:
        for token in re.findall(r'"([^"]+)"', group):
            emitted.add(token)
    assert emitted, "the capability probe's reason tokens could not be read from the source"
    missing = sorted(emitted - published)
    assert not missing, (
        f"the capability probe can report reasons that are not published: {missing}"
    )


if __name__ == "__main__":  # pragma: no cover - manual run helper
    raise SystemExit(pytest.main([__file__, "-q"]))
