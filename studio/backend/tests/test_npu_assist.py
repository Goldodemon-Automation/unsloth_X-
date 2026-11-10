# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the NPU is allowed to do for a training run, and the queue that does it.

The planner has one job worth guarding: never claim more than an NPU can do. So the tests below
pin the refusal tokens, the "full" mode staying unreachable while no vendor ships a backward
pass, and the notes staying plain English a UI can print unedited. The queue half is about the
training thread never waiting: bounded submissions, drops counted, a raising runner absorbed.
"""

from __future__ import annotations

import threading

import pytest

from core.training.npu_assist import (
    AssistQueue,
    NpuAssistPlan,
    plan_npu_assist,
)

_ALL_TASKS = ("tokenize", "eval", "qat_calibrate", "inference")


def _provider(name = "openvino", *, installed = True, supports = _ALL_TASKS):
    return {
        "name": name,
        "installed": installed,
        "version": "2026.1.0" if installed else None,
        "detail": "",
        "vendors": ("8086",),
        "supports": tuple(supports),
    }


def _capability(
    *,
    mode = "assist",
    backward_pass = False,
    tasks = _ALL_TASKS,
    providers = None,
    reasons = (),
):
    return {
        "mode": mode,
        "backward_pass": backward_pass,
        "device": {"name": "Intel Meteor Lake NPU", "vendor": "8086"},
        "tasks": list(tasks),
        "providers": (
            [_provider(), _provider("qnn", installed = False)] if providers is None else providers
        ),
        "reasons": list(reasons),
        "notes": "",
    }


# --- planner: auto -----------------------------------------------------------------------


def test_auto_enables_assist_on_a_run_that_trains_on_the_dgpu():
    plan = plan_npu_assist(_capability(), primary_kind = "dgpu")

    assert isinstance(plan, NpuAssistPlan)
    assert plan.enabled is True
    assert plan.mode == "assist"
    assert plan.tasks == ("tokenize",)
    assert plan.providers == ("openvino",)
    assert plan.refused_reason is None
    assert "the discrete GPU" in plan.notes[0]


def test_auto_refuses_when_the_npu_is_the_primary_device():
    plan = plan_npu_assist(_capability(), primary_kind = "npu")

    assert plan.enabled is False
    assert plan.mode == "off"
    assert plan.refused_reason == "npu_is_primary"
    assert plan.tasks == ()
    assert plan.providers == ()


def test_auto_leaves_the_npu_alone_when_the_capability_is_unavailable():
    plan = plan_npu_assist(_capability(mode = "unavailable", tasks = (), reasons = ["absent"]))

    assert plan.enabled is False
    assert plan.mode == "off"
    # Nothing was asked for, so there is nothing to refuse.
    assert plan.refused_reason is None


# --- planner: off ------------------------------------------------------------------------


def test_off_is_a_choice_and_not_a_refusal():
    plan = plan_npu_assist(_capability(), requested = "off")

    assert plan.enabled is False
    assert plan.mode == "off"
    assert plan.refused_reason is None
    assert plan.tasks == ()
    assert plan.providers == ()
    assert "off" in " ".join(plan.notes).lower()


# --- planner: assist ---------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["disabled", "absent", "no_runtime", "no_supported_task"])
def test_assist_refusal_carries_the_capability_reason(reason):
    plan = plan_npu_assist(
        _capability(mode = "unavailable", tasks = (), reasons = [reason]),
        requested = "assist",
    )

    assert plan.enabled is False
    assert plan.mode == "off"
    assert plan.refused_reason == reason


def test_assist_refusal_reports_the_first_reason_the_capability_gave():
    plan = plan_npu_assist(
        _capability(mode = "unavailable", tasks = (), reasons = ["present_but_odd", "absent"]),
        requested = "assist",
    )

    assert plan.refused_reason == "absent"


def test_assist_without_a_measurable_capability_refuses_as_not_measured(monkeypatch):
    npu = pytest.importorskip("utils.hardware.npu")

    def boom():
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(npu, "npu_training_capability", boom)
    plan = plan_npu_assist(requested = "assist")

    assert plan.enabled is False
    assert plan.refused_reason == "not_measured"
    assert "could not be measured" in " ".join(plan.notes)


# --- planner: full -----------------------------------------------------------------------


def test_full_is_refused_today_and_names_assist_as_the_alternative():
    plan = plan_npu_assist(_capability(), requested = "full")

    assert plan.enabled is False
    assert plan.mode == "off"
    assert plan.refused_reason == "no_backward_pass"
    notes = " ".join(plan.notes).lower()
    assert "no npu vendor exposes a backward pass" in notes
    assert "assist mode" in notes


def test_full_is_enabled_when_a_runtime_claims_a_backward_pass():
    capability = _capability(mode = "full", backward_pass = True)
    plan = plan_npu_assist(capability, requested = "full", primary_kind = "npu")

    assert plan.enabled is True
    assert plan.mode == "full"
    assert plan.refused_reason is None
    assert plan.tasks == ("tokenize",)
    assert plan.providers == ("openvino",)
    assert "the NPU" in plan.notes[0]


# --- planner: task selection -------------------------------------------------------------


def test_task_selection_follows_requested_order_and_drops_inference():
    plan = plan_npu_assist(_capability(), eval_enabled = True, qat = True)

    assert plan.tasks == ("tokenize", "eval", "qat_calibrate")
    assert "inference" not in plan.tasks


def test_tokenize_offload_off_leaves_the_other_tasks_in_order():
    plan = plan_npu_assist(
        _capability(),
        eval_enabled = True,
        tokenize_offload = False,
    )

    assert plan.tasks == ("eval",)


def test_a_task_no_installed_provider_serves_drops_out():
    providers = [
        _provider("openvino", supports = ("tokenize", "eval")),
        _provider("qnn", installed = False, supports = ("qat_calibrate",)),
    ]
    plan = plan_npu_assist(
        _capability(providers = providers),
        eval_enabled = True,
        qat = True,
    )

    assert plan.tasks == ("tokenize", "eval")
    assert plan.providers == ("openvino",)


def test_providers_are_listed_in_provider_order_without_repeats():
    providers = [
        _provider("openvino", supports = ("tokenize", "eval")),
        _provider("windows_ml", supports = ("tokenize", "eval")),
        _provider("lemonade", installed = False, supports = ("eval",)),
    ]
    plan = plan_npu_assist(_capability(providers = providers), eval_enabled = True)

    assert plan.tasks == ("tokenize", "eval")
    assert plan.providers == ("openvino", "windows_ml")


def test_an_empty_task_set_refuses_with_no_supported_task():
    plan = plan_npu_assist(_capability(tasks = ("inference",)), requested = "assist")

    assert plan.enabled is False
    assert plan.mode == "off"
    assert plan.refused_reason == "no_supported_task"
    assert plan.tasks == ()


# --- planner: notes ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "primary_kind,label",
    [
        ("cpu", "the CPU"),
        ("igpu", "the integrated GPU"),
        ("dgpu", "the discrete GPU"),
        ("npu", "the NPU"),
    ],
)
def test_every_plan_names_the_device_that_runs_the_backward_pass(primary_kind, label):
    plan = plan_npu_assist(_capability(), primary_kind = primary_kind)

    assert label in plan.notes[0]
    assert "backward pass" in plan.notes[0]


def test_assist_notes_say_the_work_is_off_the_critical_path():
    plan = plan_npu_assist(_capability())

    assert "off the critical path of a training step" in " ".join(plan.notes)


@pytest.mark.parametrize(
    "plan",
    [
        plan_npu_assist(_capability(), primary_kind = "dgpu"),
        plan_npu_assist(_capability(), primary_kind = "npu"),
        plan_npu_assist(_capability(), requested = "off"),
        plan_npu_assist(_capability(), requested = "assist"),
        plan_npu_assist(_capability(), requested = "full"),
        plan_npu_assist(_capability(mode = "unavailable", tasks = (), reasons = ["absent"])),
        plan_npu_assist(_capability(), eval_enabled = True, qat = True),
        plan_npu_assist(_capability(mode = "full", backward_pass = True), requested = "full"),
    ],
)
def test_notes_stay_plain_english(plan):
    assert plan.notes
    for note in plan.notes:
        assert note.strip() == note
        assert note.endswith(".")
        assert "_" not in note
        # A leading lowercase fragment is a fragment, not a sentence.
        assert note[0].isupper()


# --- queue -------------------------------------------------------------------------------


def test_queue_buffers_before_start_and_runs_in_order():
    seen = []

    def runner(task):
        seen.append(task.sequence)
        return task.payload.get("row")

    queue = AssistQueue(runner = runner)
    assert queue.submit("tokenize", {"row": 1}) is True
    assert queue.submit("eval", {"row": 2}) is True
    assert queue.submit("qat_calibrate", {"row": 3}) is True
    assert queue.stats()["pending"] == 3

    queue.start()
    assert queue.close() == 0

    assert seen == [1, 2, 3]
    assert [result for _, result in queue.results()] == [1, 2, 3]
    stats = queue.stats()
    assert stats["completed"] == 3
    assert stats["submitted"] == 3
    assert stats["pending"] == 0
    assert stats["failed"] == 0
    assert stats["dropped"] == 0
    assert stats["last_error"] is None


def test_a_full_queue_drops_the_submission_instead_of_blocking():
    queue = AssistQueue(runner = lambda task: None, max_pending = 2)

    assert queue.submit("tokenize") is True
    assert queue.submit("eval") is True
    assert queue.submit("eval") is False

    stats = queue.stats()
    assert stats["dropped"] == 1
    assert stats["submitted"] == 2
    assert stats["pending"] == 2
    assert queue.close(drain = False) == 2


def test_a_raising_runner_is_recorded_and_the_thread_keeps_going():
    def runner(task):
        if task.payload.get("explode"):
            raise ValueError("boom")
        return "ok"

    queue = AssistQueue(runner = runner)
    queue.submit("eval", {"explode": True})
    queue.submit("eval", {})
    queue.start()
    assert queue.close() == 0

    stats = queue.stats()
    assert stats["failed"] == 1
    assert stats["completed"] == 1
    assert "boom" in stats["last_error"]
    assert "ValueError" in stats["last_error"]
    assert [result for _, result in queue.results()] == ["ok"]


def test_close_without_drain_discards_the_backlog_and_counts_it():
    release = threading.Event()
    started = threading.Event()

    def runner(task):
        started.set()
        release.wait(5.0)
        return task.sequence

    queue = AssistQueue(runner = runner, drain_timeout_s = 0.05)
    queue.submit("eval")
    queue.start()
    assert started.wait(5.0)
    queue.submit("eval")
    queue.submit("eval")

    # The worker is inside the first task, so the two behind it never run.
    assert queue.close(drain = False) == 2
    assert queue.stats()["pending"] == 0
    release.set()


def test_close_without_start_reports_the_buffer_it_threw_away():
    queue = AssistQueue(runner = lambda task: None)
    queue.submit("tokenize")
    queue.submit("eval")

    assert queue.close(drain = False) == 2


def test_close_is_safe_twice_and_submit_after_close_is_refused():
    queue = AssistQueue(runner = lambda task: None)
    queue.start()

    assert queue.close() == 0
    assert queue.close() == 0
    assert queue.submit("tokenize") is False
    assert queue.stats()["submitted"] == 0


def test_results_stay_bounded_by_max_pending():
    release = threading.Event()
    entered = threading.Event()

    def runner(task):
        if task.sequence == 1:
            entered.set()
            release.wait(5.0)
        return task.sequence

    queue = AssistQueue(runner = runner, max_pending = 2)
    queue.submit("eval")
    queue.start()
    assert entered.wait(5.0)
    queue.submit("eval")
    queue.submit("eval")
    release.set()

    assert queue.close() == 0
    assert queue.stats()["completed"] == 3
    assert [result for _, result in queue.results()] == [2, 3]
