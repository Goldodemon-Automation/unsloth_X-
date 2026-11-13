# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP contract for the llama.cpp architecture picker.

GET  /api/llama/architectures          -> which forks this host has, can install, may switch to
POST /api/llama/architectures/install  -> download + extract one, optionally activate it
POST /api/llama/architectures/activate -> record one as the runtime

Installing and switching replace the runtime for the whole installation, so those two are
owner-only; reading the picker is not. The architecture helpers are stubbed wherever the test is
about the route, so nothing here downloads, extracts, spawns an install thread, or reads GitHub.
The few tests that deliberately run the real helper only use the refusals that return before any
I/O: an unknown id, and the single-writer lock.
"""

from __future__ import annotations

import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import get_current_subject
from routes import llama as llama_routes
from utils import keyless_api_access
from utils import llama_cpp_architectures as arch
from utils.account_context import AccountContext, OWNER, bind_account, reset_account

PREFIX = "/api/llama"
STATUS = f"{PREFIX}/architectures"
INSTALL = f"{PREFIX}/architectures/install"
ACTIVATE = f"{PREFIX}/architectures/activate"

ALICE = AccountContext("a" * 32, "alice")

# The fields the picker is documented to carry. A frontend row binds every one of them, so a
# missing field blanks a column instead of erroring, and a missing top-level ``active`` labels
# the wrong fork as the running one.
DOCUMENTED_STATUS_FIELDS = frozenset(
    {"active", "runtime_locked", "runtime_lock_reason", "profiles", "job"}
)
DOCUMENTED_PROFILE_FIELDS = frozenset(
    {"id", "label", "available", "installable", "selectable", "installed", "active", "reason"}
)


@pytest.fixture(autouse = True)
def _idle_installer(monkeypatch):
    """Start every test with no job recorded and the single-writer lock free.

    The architecture module keeps one process-wide job and one operation lock, so a test that
    reads them must not inherit another test's install. The two env overrides are cleared
    because either one makes the module refuse to install or activate at all.
    """
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    arch._reset_for_tests()
    yield
    arch._reset_for_tests()


def _app_for(account: AccountContext, loop_threads: list | None = None) -> FastAPI:
    """The route surface as main.py mounts it, under one signed-in account.

    The subject dependency binds the account context for the rest of the request, exactly as
    the real auth dependency does, so the real ``policy.require_owner`` decides who may write.
    """
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject

    @app.middleware("http")
    async def _record_loop_thread(request, call_next):
        # Middleware runs on the event loop's thread, which is the thread a blocking call made
        # straight from a route handler would stall.
        if loop_threads is not None:
            loop_threads.append(threading.current_thread())
        return await call_next(request)

    app.include_router(llama_routes.router, prefix = PREFIX)
    return app


def _client(account: AccountContext, loop_threads: list | None = None) -> TestClient:
    return TestClient(_app_for(account, loop_threads))


@pytest.fixture()
def anonymous_client(monkeypatch) -> TestClient:
    """A caller with no credential at all.

    Keyless API access is stubbed off instead of read from settings, so the real credential
    gate answers without touching the settings store or the network.
    """
    monkeypatch.setattr(keyless_api_access, "keyless_request_allowed", lambda request: False)
    app = FastAPI()
    app.include_router(llama_routes.router, prefix = PREFIX)
    return TestClient(app)


def _job(**overrides) -> dict:
    job = {
        "state": "idle",
        "architecture": None,
        "message": "",
        "error": None,
        "progress": None,
        "from_tag": None,
        "to_tag": None,
        "activated": None,
        "started_at": None,
        "finished_at": None,
    }
    job.update(overrides)
    return job


def _profile(**overrides) -> dict:
    """One picker row, complete: the response model declares every field here."""
    profile = {
        "id": "neurall",
        "label": "Neuralls llama.cpp (neurall/llama.cpp)",
        "repo": "neurall/llama.cpp",
        "source_build": False,
        "source_commit": None,
        "available": True,
        "installable": True,
        "selectable": False,
        "installed": False,
        "active": False,
        "reason": None,
        "asset": "llama-b6000-bin-win-vulkan-x64.zip",
        "release_tag": "b6000",
        "latest_tag": "b6000",
        "installed_tag": None,
        "installed_at_utc": None,
        "download_size_bytes": 1234,
        "update_available": False,
        "binary_path": None,
        "install_dir": "C:/studio/architectures/neurall",
        "description": "The Neuralls llama.cpp build.",
    }
    profile.update(overrides)
    return profile


def _status_payload(**overrides) -> dict:
    payload = {
        "active": "normal",
        "runtime_locked": False,
        "runtime_lock_reason": None,
        "release_lookup_failed": False,
        "profiles": [_profile()],
        "job": _job(),
    }
    payload.update(overrides)
    return payload


def _started_action(architecture: str = "neurall") -> dict:
    return {
        "started": True,
        "reason": None,
        "message": None,
        "job": _job(
            state = "running", architecture = architecture, started_at = "2026-01-01T00:00:00Z"
        ),
    }


def _running_action() -> dict:
    """What the module answers when a job is already in flight: no start, the running job."""
    return {
        "started": False,
        "reason": "already_running",
        "message": "Another llama.cpp install or switch is already running.",
        "job": _job(
            state = "running",
            architecture = "neurall",
            message = "Downloading llama-b6000-bin-win-vulkan-x64.zip...",
            progress = 0.35,
            to_tag = "b6000",
            started_at = "2026-01-01T00:00:00Z",
        ),
    }


def _stubs(monkeypatch, *, status = None, install_action = None) -> dict:
    """Stand in for the three helpers the architecture routes call, recording every call.

    ``status`` is the picker payload; the activate route answers with it too, because it
    re-reads the status after recording the switch. ``install_action`` is the installer's
    answer (a started job by default). Nothing here downloads, extracts, or resolves a release,
    and every call records the thread it ran on.
    """
    status_payload = _status_payload() if status is None else status
    recorded: dict = {"status": [], "install": [], "activate": [], "calls": []}

    def note(kind: str, value):
        recorded["calls"].append((kind, threading.current_thread()))
        return value

    def fake_get_status(*, force_refresh = False):
        recorded["status"].append(force_refresh)
        return note("status", status_payload)

    def fake_start_install(architecture, *, activate = True):
        recorded["install"].append({"architecture": architecture, "activate": activate})
        action = install_action if install_action is not None else _started_action(architecture)
        return note("install", action)

    def fake_activate(architecture):
        recorded["activate"].append(architecture)
        return note("activate", status_payload)

    monkeypatch.setattr(llama_routes, "get_architecture_status", fake_get_status)
    monkeypatch.setattr(llama_routes, "start_architecture_install", fake_start_install)
    monkeypatch.setattr(llama_routes, "activate_architecture", fake_activate)
    return recorded


# --- 1. authentication and authorisation -------------------------------------------------


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("get", STATUS, None),
        ("post", INSTALL, {"architecture": "neurall"}),
        ("post", ACTIVATE, {"architecture": "neurall"}),
    ],
)
def test_no_credential_is_refused_before_any_work(
    monkeypatch, anonymous_client, method, path, body
):
    """No credential means no architecture surface at all: 401 on all three routes.

    These routes read the host's install tree and can download and execute a third-party
    binary, so an unauthenticated caller reaching a handler would be a remote install
    primitive. The refusal also has to land before the handler: a 200 with an empty payload
    would still have paid the release lookup.
    """
    recorded = _stubs(monkeypatch)
    response = (
        getattr(anonymous_client, method)(path, json = body)
        if body is not None
        else getattr(anonymous_client, method)(path)
    )

    assert response.status_code == 401
    assert recorded["status"] == []
    assert recorded["install"] == []
    assert recorded["activate"] == []


def test_a_managed_account_may_read_the_picker_but_not_replace_the_runtime(monkeypatch):
    """Reading is allowed for any signed-in account; installing and switching are owner-only.

    The runtime is shared by every account on the installation, so a managed account that
    could install or activate would pick the executable everyone else runs. Breaking this
    hands an unprivileged account the host's process image.
    """
    recorded = _stubs(monkeypatch)
    client = _client(ALICE)

    status_response = client.get(STATUS)
    install_response = client.post(INSTALL, json = {"architecture": "neurall"})
    activate_response = client.post(ACTIVATE, json = {"architecture": "neurall"})

    assert status_response.status_code == 200
    assert len(recorded["status"]) == 1
    # 403, not 401: the account authenticated, it just may not change an installation-wide runtime.
    assert install_response.status_code == 403
    assert activate_response.status_code == 403
    # Refused before the work: no job started, nothing recorded active.
    assert recorded["install"] == []
    assert recorded["activate"] == []


def test_the_owner_may_install_and_switch(monkeypatch):
    """The owner gets both mutations through, with the request forwarded as given.

    This is the positive half of the gate: if the owner were refused too the feature would be
    unusable, and a route that dropped ``activate`` would install a fork it never switches to.
    """
    recorded = _stubs(monkeypatch)
    client = _client(OWNER)

    install_response = client.post(INSTALL, json = {"architecture": "neurall", "activate": False})
    activate_response = client.post(ACTIVATE, json = {"architecture": "neurall"})

    assert install_response.status_code == 200
    assert install_response.json()["started"] is True
    assert recorded["install"] == [{"architecture": "neurall", "activate": False}]
    assert activate_response.status_code == 200
    assert recorded["activate"] == ["neurall"]


def test_the_owner_gate_is_attached_after_the_auth_dependency():
    """Both mutations carry policy.require_owner, solved after the subject dependency.

    The behaviour above only holds while the gate is registered, and the ordering matters too:
    require_owner reads the account context the auth dependency binds. Swapping or dropping
    either leaves an installation-wide write open, or permanently refused.
    """
    mutating = [
        route
        for route in llama_routes.router.routes
        if route.path in ("/architectures/install", "/architectures/activate")
    ]
    assert len(mutating) == 2
    for route in mutating:
        solved = [
            dependency.call
            for dependency in route.dependant.dependencies
            if dependency.call is not None
        ]
        assert policy.require_owner in solved
        assert solved.index(get_current_subject) < solved.index(policy.require_owner)

    status_route = next(
        route for route in llama_routes.router.routes if route.path == "/architectures"
    )
    # The picker is deliberately readable by managed accounts, so it must not grow the gate.
    assert policy.require_owner not in [
        dependency.call for dependency in status_route.dependant.dependencies
    ]


# --- 2. request validation ---------------------------------------------------------------


@pytest.mark.parametrize("path", [INSTALL, ACTIVATE])
@pytest.mark.parametrize("architecture", ["", "x" * 65], ids = ["empty", "over-long"])
def test_an_empty_or_over_long_id_is_a_422(monkeypatch, path, architecture):
    """An empty or over-length id is refused by the request model with a 422.

    Current behaviour: 422 (``string_too_short`` / ``string_too_long``). These are the two
    shapes that could otherwise fall through to a default, so the check is that nothing was
    handed to the helper: an empty id read as "the default fork" would install and activate a
    runtime the user never asked for.
    """
    recorded = _stubs(monkeypatch)
    response = _client(OWNER).post(path, json = {"architecture": architecture})

    assert response.status_code == 422
    assert recorded["install"] == []
    assert recorded["activate"] == []


@pytest.mark.parametrize("path", [INSTALL, ACTIVATE])
def test_a_missing_id_is_a_422(monkeypatch, path):
    """A body with no ``architecture`` key is a 422, never a default fork.

    The UI posts the row it rendered, so a body without an id is a bug on the caller's side;
    treating it as "the bundled runtime" would silently install something else. Current
    behaviour: 422 ``missing``.
    """
    recorded = _stubs(monkeypatch)
    response = _client(OWNER).post(path, json = {})

    assert response.status_code == 422
    assert recorded["install"] == []
    assert recorded["activate"] == []


def test_activate_refuses_an_unknown_id_with_a_409():
    """An unknown id on activate answers 409, not a 500 and not a silent switch.

    Current behaviour: the route turns the module's ``ValueError`` into a 409 carrying its
    message. Uncaught this is a 500 for ordinary bad input; swallowed, the UI reports a switch
    that never happened.
    """
    response = _client(OWNER).post(ACTIVATE, json = {"architecture": "not_a_real_architecture"})

    assert response.status_code == 409
    assert "not_a_real_architecture" in response.json()["detail"]
    # Nothing was recorded as running.
    assert arch.get_job()["state"] == "idle"


def test_install_of_an_unknown_id_is_a_404_and_starts_nothing():
    """Behaviour pinned: an install of an unknown id is refused with a 404 that names the ids.

    It used to answer 200 with ``started=false, reason="unknown_architecture"``, which a
    generated client or a naive ``response.ok`` check reads as "the install began". Every other
    bad-input path on this surface is already a 4xx (422 for the id's shape, 409 for activate of
    an unknown id), so the route checks the manifest before calling ``start_install``. Drop that
    check and a caller can no longer tell a refused request from a running install.
    """
    response = _client(OWNER).post(INSTALL, json = {"architecture": "not_a_real_architecture"})
    detail = response.json()["detail"]

    assert response.status_code == 404
    assert "not_a_real_architecture" in detail
    # The message is a remedy, not just a refusal: it lists what could have been asked for.
    assert "dory" in detail and "ik" in detail
    # Nothing was started, so no job was recorded either.
    assert arch.get_job()["state"] == "idle"


def test_install_of_a_known_but_unbuildable_id_still_answers_200_with_started_false():
    """Behaviour pinned: a known id that cannot be built here keeps the 200 + reason contract.

    ``x`` (LLama.zig) is in the manifest but has no build source, so the id check must let it
    through and ``start_install`` must answer honestly. Narrowing the route's check to
    "installable here" would turn a capability answer (the row the picker shows) into an error.
    """
    response = _client(OWNER).post(INSTALL, json = {"architecture": "x"})
    body = response.json()

    assert response.status_code == 200
    assert body["started"] is False
    assert body["reason"], "a refusal without a reason cannot be shown to the user"
    assert body["job"]["state"] == "idle"


# --- 3. response shape -------------------------------------------------------------------


def test_the_status_payload_matches_the_documented_model(monkeypatch):
    """GET /architectures returns the documented fields, with the values passed through.

    The payload is the picker's whole contract: every documented profile field drives a row
    (``available``/``installable``/``selectable`` decide which control is live, ``reason``
    explains why one is not). A renamed or dropped field blanks a column, and a value that is
    not passed through -- ``active`` not echoing the recorded setting, say -- labels the wrong
    fork as the one running.
    """
    _stubs(monkeypatch, status = _status_payload())
    response = _client(OWNER).get(STATUS)

    assert response.status_code == 200
    body = response.json()
    # Exactly the declared model: the response is the model, nothing invented or dropped.
    assert set(body) == set(llama_routes.LlamaArchitecturesStatus.model_fields)
    assert DOCUMENTED_STATUS_FIELDS <= set(body)
    assert set(body["profiles"][0]) == set(llama_routes.LlamaArchitectureInfo.model_fields)
    assert DOCUMENTED_PROFILE_FIELDS <= set(body["profiles"][0])

    assert body["active"] == "normal"
    assert body["runtime_locked"] is False
    assert body["runtime_lock_reason"] is None
    profile = body["profiles"][0]
    assert profile["id"] == "neurall"
    assert profile["label"] == "Neuralls llama.cpp (neurall/llama.cpp)"
    assert profile["installable"] is True
    assert profile["installed"] is False
    assert profile["reason"] is None
    assert body["job"]["state"] == "idle"


def test_a_locked_runtime_is_reported_as_locked_and_unselectable(monkeypatch):
    """An env-pinned llama.cpp path comes back with ``runtime_locked`` and its reason.

    Without the flag the picker offers switches the runtime resolver will ignore, and the user
    is told they changed something that did not change. The reason is carried too, so the UI
    can name the variable instead of only dimming the row.
    """
    _stubs(
        monkeypatch,
        status = _status_payload(
            runtime_locked = True,
            runtime_lock_reason = "UNSLOTH_LLAMA_CPP_PATH",
            profiles = [
                _profile(
                    available = False,
                    installable = False,
                    selectable = False,
                    reason = "environment_override",
                )
            ],
        ),
    )
    response = _client(OWNER).get(STATUS)

    assert response.status_code == 200
    body = response.json()
    assert body["runtime_locked"] is True
    assert body["runtime_lock_reason"] == "UNSLOTH_LLAMA_CPP_PATH"
    assert body["profiles"][0]["selectable"] is False
    assert body["profiles"][0]["reason"] == "environment_override"


def test_the_status_route_forwards_force_refresh(monkeypatch):
    """``?force_refresh=true`` reaches the status helper; a bare call asks for the cached answer.

    The explicit re-check is the only way a user can clear a stale "no release" row, so a route
    that dropped the flag leaves a broken picker with no recourse, and one that defaulted it to
    true pays a GitHub round trip on every poll.
    """
    recorded = _stubs(monkeypatch)
    client = _client(OWNER)

    assert client.get(STATUS).status_code == 200
    assert client.get(f"{STATUS}?force_refresh=true").status_code == 200

    assert recorded["status"] == [False, True]


def test_activate_answers_with_the_new_picker_status(monkeypatch):
    """Activating answers with the full picker payload, not just an acknowledgement.

    The panel re-renders straight from this response, so a bare ``{"ok": true}`` would leave it
    showing the previous fork as active until the next poll -- and the payload it does return
    must be the post-switch one, which is why the route re-reads the status after recording.
    """
    recorded = _stubs(monkeypatch, status = _status_payload(active = "neurall"))
    response = _client(OWNER).post(ACTIVATE, json = {"architecture": "neurall"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == set(llama_routes.LlamaArchitecturesStatus.model_fields)
    assert body["active"] == "neurall"
    # Recorded first, then re-read: the answer is the state after the switch, not before.
    assert recorded["activate"] == ["neurall"]
    assert recorded["status"] == [False]


# --- 4. the job contract -----------------------------------------------------------------


def test_a_second_install_while_one_runs_does_not_claim_a_new_job(monkeypatch):
    """A second install during a running job reports the refusal, with the running job.

    The route is a pass-through: when the module answers ``started=False,
    reason=already_running`` the response must keep that answer and echo the job already in
    flight (same progress, same start time) instead of claiming a start or resetting the job to
    idle. Breaking it makes the UI show a second fresh progress bar for work that never began.
    """
    recorded = _stubs(monkeypatch, install_action = _running_action())
    client = _client(OWNER)

    first = client.post(INSTALL, json = {"architecture": "neurall"})
    second = client.post(INSTALL, json = {"architecture": "neurall"})

    assert first.status_code == 200 and second.status_code == 200
    for response in (first, second):
        body = response.json()
        assert body["started"] is False
        assert body["reason"] == "already_running"
        # The job reported is the one already running, unchanged.
        assert body["job"] == _running_action()["job"]
        assert body["job"]["progress"] == 0.35

    assert first.json() == second.json()
    # The route does not cache the refusal: the guard lives in the module, so a retry is safe.
    assert recorded["install"] == [
        {"architecture": "neurall", "activate": True},
        {"architecture": "neurall", "activate": True},
    ]


def test_the_single_writer_lock_refuses_a_second_install_through_the_route():
    """With the module's operation lock held, the route answers started=False, already_running.

    This is the real guard behind the pin above: ``start_install`` takes the lock without
    blocking and refuses rather than queueing a second writer against the same install tree. It
    returns before any download or thread start, which is why driving the real helper here is
    safe. Without it, two clicks would race two extractions into one directory.
    """
    held = arch._OPERATION_LOCK.acquire(blocking = False)
    assert held is True
    try:
        response = _client(OWNER).post(INSTALL, json = {"architecture": "neurall"})
    finally:
        arch._OPERATION_LOCK.release()

    assert response.status_code == 200
    body = response.json()
    assert body["started"] is False
    assert body["reason"] == "already_running"
    # Nothing was recorded: the refused request did not touch the job at all.
    assert body["job"]["state"] == "idle"
    assert arch.get_job()["state"] == "idle"


def test_installing_the_bundled_runtime_does_not_start_a_job():
    """``normal`` is the bundled runtime, so installing it is refused with reason ``bundled``.

    The picker lists ``normal`` as a row and the UI may post it. Treating it as installable
    would extract a fork over the bundled runtime, the one install the app must never replace
    out from under itself.
    """
    response = _client(OWNER).post(INSTALL, json = {"architecture": "normal"})

    assert response.status_code == 200
    body = response.json()
    assert body["started"] is False
    assert body["reason"] == "bundled"
    assert arch.get_job()["state"] == "idle"


# --- 5. the event loop -------------------------------------------------------------------


def test_all_three_routes_hand_their_work_to_a_worker_thread(monkeypatch):
    """The status read, the install start and the activation all run off the event loop.

    Each ends up in code that touches the disk and can block for seconds -- a release lookup, a
    download, a settings write -- and on the loop thread that stalls every other request in the
    process, including the SSE streams the app serves. The stubs record the thread they run on
    and the middleware records the thread the loop serves the request on, so a direct ``await``
    of one of them (a dropped ``asyncio.to_thread``) shows up as the same thread twice.
    """
    loop_threads: list[threading.Thread] = []
    recorded = _stubs(monkeypatch, install_action = _running_action())

    client = _client(OWNER, loop_threads)
    assert client.get(STATUS).status_code == 200
    assert client.post(INSTALL, json = {"architecture": "neurall"}).status_code == 200
    assert client.post(ACTIVATE, json = {"architecture": "neurall"}).status_code == 200

    kinds = [kind for kind, _thread in recorded["calls"]]
    assert {"status", "install", "activate"} <= set(kinds)
    assert len(loop_threads) == 3

    loop_idents = {thread.ident for thread in loop_threads}
    worker_idents = {thread.ident for _kind, thread in recorded["calls"]}
    assert loop_idents.isdisjoint(worker_idents)