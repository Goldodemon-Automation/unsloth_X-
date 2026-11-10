# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which device the NPU card names, on a host where more than one device could answer.

Every sentence `/api/compute/npu/assist` writes is built around one fact: the device that owns the
backward pass. `/api/compute/targets` shows the device the user picked. Those used to be two different
answers, because the assist handler resolved `auto` for itself -- hardcoded, with its own fresh probe --
instead of reading the stored choice like the payload beside it does. On a hybrid host whose picker read
CPU, the card said "The backward pass runs on the discrete GPU" directly underneath it. Both routes now
go through `_resolve_primary()`, and what is pinned here is that the card cannot name a device the
picker is not showing.

The registry and the planner run for real, because the bug lived in the seam between the two routes and
stubbing either one would have hidden it. Only the host is invented: an Intel iGPU, an NVIDIA card that
`auto` wins with, and an NPU that can assist but can never train -- a machine no test runner is required
to have. The route's probe cache is process-wide, so it is emptied before and after every test here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth.authentication import get_current_subject  # noqa: E402
from core.training.device_targets import TargetProbe  # noqa: E402

GB = 1024**3

# Pinned as text because they are user-facing, and the finding *is* this sentence: the card led with the
# wrong device in it.
BACKWARD_ON_CPU = "The backward pass runs on the CPU."
BACKWARD_ON_IGPU = "The backward pass runs on the integrated GPU."
BACKWARD_ON_DGPU = "The backward pass runs on the discrete GPU."
CONTINUES_ON_CPU = "The NPU takes tokenization while the training step continues on the CPU."

INTEGRATED_GPU = {
    "index": 0,
    "name": "Intel(R) Arc(TM) Graphics",
    "vendor": "intel",
    "integrated": True,
    "memory_bytes": 512 * 1024**2,
    "arch": None,
    "unified": True,
}
DISCRETE_GPU = {
    "index": 1,
    "name": "NVIDIA GeForce RTX 4090",
    "vendor": "nvidia",
    "integrated": False,
    "memory_bytes": 24 * GB,
    "arch": "sm_89",
    "unified": False,
}
# Present, enabled and supported as an assist device -- which is exactly why it is listed and still not
# trainable: `list_training_targets()` marks every NPU row unsupported with `no_backward_pass`.
NPU = {
    "present": True,
    "enabled": True,
    "supported": True,
    "reason": None,
    "primary": {"name": "Intel(R) AI Boost", "vendor": "intel"},
}

# The capability of a host whose NPU can take work off a run: assist mode, tasks a runtime claims, and
# no backward pass. The card only has something to say about a device it cannot train on.
CAPABILITY = {
    "mode": "assist",
    "backward_pass": False,
    "device": "npu:0",
    "tasks": ["tokenize", "eval"],
    "providers": [{"name": "Lemonade", "installed": True, "supports": ["tokenize", "eval"]}],
    "reasons": [],
    "notes": "The NPU can assist a run that trains on another device.",
}

# The five keys `utils.hardware.xpu.xpu_torch_status` publishes. This host has an Intel GPU row, so the
# route would read the wheel for real; the answer is stubbed because the question is not under test.
XPU_RUNTIME = {
    "available": True,
    "build": "xpu",
    "device_count": 1,
    "driver": "31.0.101.5333",
    "note": "PyTorch here can train on this Intel GPU.",
}


def _row(row: dict, **overrides) -> dict:
    merged = dict(row)
    merged.update(overrides)
    return merged


def _probe(**overrides) -> TargetProbe:
    """A torch host with both kinds of GPU and an NPU: the machine the finding needs.

    `torch_available` True is what makes the iGPU and dGPU rows supported, so a stored "igpu" or "dgpu"
    resolves instead of falling back -- the fallback path has its own test below.
    """
    base = dict(
        cpu_name = "Intel(R) Core(TM) Ultra 7 155H",
        cpu_cores = 16,
        system_memory_bytes = 64 * GB,
        system_memory_available_bytes = 40 * GB,
        torch_available = True,
        torch_device = "cuda",
        gpus = [_row(INTEGRATED_GPU), _row(DISCRETE_GPU)],
        npu = dict(NPU),
        bnb_available = True,
        triton_available = False,
        flash_attn_available = False,
    )
    base.update(overrides)
    return TargetProbe(**base)


@pytest.fixture()
def host(monkeypatch):
    """The route, on a hand-built hybrid host whose stored compute target a test can set.

    The store is read per request -- that is the whole fix -- so changing `host.target` between two GETs
    is the only thing a test has to do to move the picker. `runtime_calls` counts the torch read the
    probe cache is responsible for, so a test can see the hardware answer stayed cached.
    """
    from routes import compute
    from utils import compute_target_settings

    state = SimpleNamespace(probe = _probe(), target = "auto", runtime_calls = 0)

    def _runtime():
        state.runtime_calls += 1
        return dict(XPU_RUNTIME)

    monkeypatch.setattr(compute, "_probe", lambda: state.probe)
    monkeypatch.setattr(compute, "_npu_capability", lambda: dict(CAPABILITY))
    monkeypatch.setattr(compute, "_xpu_torch_status", _runtime)
    monkeypatch.setattr(
        compute_target_settings, "get_selected_compute_target", lambda: state.target
    )
    compute._invalidate_probe_cache()
    yield state
    compute._invalidate_probe_cache()


@pytest.fixture()
def client(host):
    from routes import compute

    app = FastAPI()
    app.include_router(compute.router, prefix = "/api/compute")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    return TestClient(app)


def _assist(client, **params) -> dict:
    response = client.get("/api/compute/npu/assist", params = params)
    assert response.status_code == 200
    return response.json()


def test_a_stored_cpu_makes_every_sentence_name_the_cpu(client, host):
    """The finding, exactly: a stored "cpu" on a host that also has a discrete card.

    The plan is asserted enabled first, because a card that had nothing to offload would name the CPU
    for the wrong reason -- a refusal, not a resolution.
    """
    host.target = "cpu"

    body = _assist(client)

    assert body["enabled"] is True
    assert body["mode"] == "assist"
    assert body["tasks"] == ["tokenize"]
    assert body["providers"] == ["Lemonade"]
    assert body["resolved"] == "cpu"
    assert body["notes"][0] == BACKWARD_ON_CPU
    # The label reaches the sentence about the offloaded work too, not just the lead line.
    assert CONTINUES_ON_CPU in body["notes"]
    assert BACKWARD_ON_DGPU not in body["notes"]


def test_a_stored_integrated_gpu_names_the_integrated_gpu(client, host):
    host.target = "igpu"

    body = _assist(client)

    assert body["resolved"] == "igpu"
    assert body["notes"][0] == BACKWARD_ON_IGPU
    assert BACKWARD_ON_DGPU not in body["notes"]
    assert BACKWARD_ON_CPU not in body["notes"]


def test_an_auto_request_on_this_host_names_the_discrete_card(client, host):
    """The premise, pinned: `auto` resolves the dGPU, so the two tests above cannot be accidents.

    This was also the answer the card gave for every store before the fix, which is why a stored CPU was
    invisible to it. Pinning it is what keeps the two CPU/iGPU tests above from passing by accident on a
    host whose discrete card a broken fixture had quietly dropped.
    """
    host.target = "auto"

    body = _assist(client)

    assert body["resolved"] == "dgpu"
    assert body["notes"][0] == BACKWARD_ON_DGPU


@pytest.mark.parametrize(
    ("stored", "resolved", "lead_note"),
    [
        ("auto", "dgpu", BACKWARD_ON_DGPU),
        ("cpu", "cpu", BACKWARD_ON_CPU),
        ("igpu", "igpu", BACKWARD_ON_IGPU),
        ("dgpu", "dgpu", BACKWARD_ON_DGPU),
        # `npu` is the interesting row: the device is present, listed, and can never train, so both
        # routes must fall back to the same winner rather than one refusing and the other naming it.
        ("npu", "dgpu", BACKWARD_ON_DGPU),
    ],
)
def test_the_card_and_the_picker_agree_on_the_device(client, host, stored, resolved, lead_note):
    """Two routes, one store, one device -- and the sentence has to carry the same one.

    `resolved` is compared as well as the prose, so a UI that reads the field cannot drift from a card
    that reads the notes, and a planner that reworded a sentence would not silently break this.
    """
    host.target = stored

    targets = client.get("/api/compute/targets").json()
    assist = _assist(client)

    assert targets["requested"] == stored
    assert targets["resolved"] == resolved
    assert assist["resolved"] == targets["resolved"]
    assert assist["notes"][0] == lead_note
    # The capability block is the same measured answer, not a second NPU detection pass.
    assert assist["capability"] == targets["npu"]


def test_a_stored_target_this_host_cannot_serve_falls_back_the_same_way_in_both_routes(client, host):
    """An unavailable choice must not split the two answers: same fallback, same reason, same card.

    A host whose only GPU is integrated cannot serve a remembered "dgpu" -- the eGPU went back in the
    bag. `/targets` already handled that by falling back; the card resolved `auto` and therefore named
    the same device by accident rather than by construction, which is the disagreement this route fixes.
    """
    host.probe = _probe(gpus = [_row(INTEGRATED_GPU)])
    host.target = "dgpu"

    targets = client.get("/api/compute/targets").json()
    assist = _assist(client)

    assert [target["id"] for target in targets["targets"]] == ["cpu", "igpu", "npu"]
    assert targets["requested"] == "dgpu"
    assert targets["resolved"] == "igpu"
    assert "dgpu" in targets["fallback_reason"]
    assert assist["resolved"] == targets["resolved"]
    assert assist["notes"][0] == BACKWARD_ON_IGPU


def test_the_store_is_read_per_poll_not_from_the_probe_cache(client, host):
    """Hardware is cached for a minute; a session choice is not allowed to be.

    Both routes poll on the same page, so a stored change that only reached the card after the TTL
    expired would be the same disagreement wearing a different symptom -- and a fresh probe per poll is
    what the cache exists to prevent.
    """
    host.target = "auto"
    assert client.get("/api/compute/targets").json()["resolved"] == "dgpu"
    reads_after_first_probe = host.runtime_calls

    host.target = "cpu"

    assert _assist(client)["resolved"] == "cpu"
    assert client.get("/api/compute/targets").json()["resolved"] == "cpu"
    # Same cache window: the store moved, the probe did not re-run.
    assert host.runtime_calls == reads_after_first_probe


def test_a_full_request_still_refuses_with_no_backward_pass(client, host):
    """The honesty rule is untouched by the device fix: an explicit full request is refused by name.

    And the refusal still leads with the stored device, because where the run *does* train is the fact
    a user needs in order to read the refusal.
    """
    host.target = "cpu"

    body = _assist(client, requested = "full")

    assert body["enabled"] is False
    assert body["mode"] == "off"
    assert body["refused_reason"] == "no_backward_pass"
    assert body["tasks"] == []
    assert body["resolved"] == "cpu"
    assert body["notes"][0] == BACKWARD_ON_CPU


def test_a_probe_that_raises_still_answers_with_a_plan(client, host, monkeypatch):
    """A host that cannot be probed costs the card its device name, not the card: None, and the CPU.

    `/targets` fails its GET in this case because it has a page to render. The preview has nothing to
    fail about, so it keeps the behaviour the handler had before it learned to share the probe.
    """
    from routes import compute

    def _boom():
        raise RuntimeError("the driver hung during detection")

    monkeypatch.setattr(compute, "_probe", _boom)
    compute._invalidate_probe_cache()

    body = _assist(client)

    assert body["resolved"] is None
    assert body["notes"][0] == BACKWARD_ON_CPU
    assert body["capability"] == CAPABILITY
