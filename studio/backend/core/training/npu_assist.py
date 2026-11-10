# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NPU assist for training: what an NPU can really do for a run, and a queue that does it elsewhere.

No NPU vendor exposes a backward pass today, so the honest maximum is a run that keeps training
on the primary compute device and hands the NPU the work it can finish on its own: tokenization,
checkpoint evaluation and quantization calibration. The planner below writes that arrangement out
in plain sentences instead of labelling the run "NPU training", and its "full" mode stays
unreachable until a runtime actually claims a backward pass -- so the UI can offer the mode and
explain the refusal rather than quietly downgrading an explicit request.

The queue is deliberately boring: bounded, single consumer and lossy. A training step that waits
on a queue is a slower training step, so a full queue drops the submission and the run keeps the
step time it had without the NPU.

Nothing here imports torch or touches hardware at import time: the capability probe is behind a
guarded import inside the planner, so a checkout without the NPU stack still imports this module.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

# Execution order of the tasks an NPU may take over; "inference" is deliberately absent because a
# training run has nothing to infer, so a capability that lists it must never get it offloaded.
_TASK_ORDER = ("tokenize", "eval", "qat_calibrate")
_TASK_LABELS = {
    "tokenize": "tokenization",
    "eval": "checkpoint evaluation",
    "qat_calibrate": "quantization calibration",
}
_PRIMARY_LABELS = {
    "cpu": "the CPU",
    "igpu": "the integrated GPU",
    "dgpu": "the discrete GPU",
    "npu": "the NPU",
}
_DEFAULT_PRIMARY_LABEL = "the primary compute device"
_NOT_MEASURED = "not_measured"
# Tokens the capability probe may refuse with, in the order the UI prefers to explain them.
_REFUSAL_REASONS = ("disabled", "absent", "no_runtime", "no_supported_task")
# Reasons a probe can report that are not part of the refusal contract but must still be visible.
_EXTRA_REFUSAL_REASONS = ("unsupported_family",)
# Every token `NpuAssistPlan.refused_reason` can carry, probe reasons plus the planner's own. Published
# because a UI has to translate them, and an incomplete list is how one reaches a user as snake_case.
REFUSAL_TOKENS = _REFUSAL_REASONS + _EXTRA_REFUSAL_REASONS + ("no_backward_pass", "npu_is_primary", _NOT_MEASURED)
_DEGRADED_CAPABILITY = {"mode": "unavailable", "reasons": [_NOT_MEASURED]}


class NpuAssistError(Exception):
    """Raised when an assist plan or queue cannot be built as asked."""


@dataclass(frozen = True)
class NpuAssistPlan:
    """What one training run will hand to the NPU, and the sentences that explain it."""

    enabled: bool
    mode: str  # "off" | "assist" | "full"
    tasks: tuple[str, ...]
    providers: tuple[str, ...]
    refused_reason: Optional[str]
    notes: tuple[str, ...]


@dataclass(frozen = True)
class AssistTask:
    """One unit of NPU work, opaque to the queue beyond its kind and turn."""

    kind: str  # "tokenize" | "eval" | "qat_calibrate"
    payload: dict
    sequence: int


def _measured_capability() -> dict:
    """The capability probe, or the "not measured" stand-in when the NPU stack is not installed.

    Both imports are optional on purpose: this module is the thing that must keep importing on a
    checkout where the hardware layer has not landed, and an unmeasurable NPU is not an error --
    it is a run that trains on the primary device, which is the default anyway.
    """
    try:
        from utils.hardware.npu import npu_training_capability
    except ImportError:
        return dict(_DEGRADED_CAPABILITY)
    try:
        measured = npu_training_capability()
    except Exception:  # noqa: BLE001 -- a flaky probe must not break a training run
        return dict(_DEGRADED_CAPABILITY)
    if not isinstance(measured, dict):
        return dict(_DEGRADED_CAPABILITY)
    return dict(measured)


def _runtime_providers() -> list:
    """Every NPU runtime row the provider module knows about, or nothing when it is absent."""
    try:
        from utils.hardware.npu_providers import npu_runtime_providers
    except ImportError:
        return []
    try:
        rows = npu_runtime_providers()
    except Exception:  # noqa: BLE001 -- provider probing is best effort
        return []
    return list(rows) if isinstance(rows, (list, tuple)) else []


def _capability(capability: Optional[dict]) -> dict:
    if capability is not None:
        return dict(capability)
    return _measured_capability()


def _normalise_requested(requested: Optional[str]) -> str:
    """Anything unrecognised falls back to "auto": a typo must not pin a run to a device."""
    if requested is None:
        return "auto"
    text = str(requested).strip().lower()
    return text if text in ("off", "assist", "full") else "auto"


def _primary_label(primary_kind: str) -> str:
    return _PRIMARY_LABELS.get(str(primary_kind or "").strip().lower(), _DEFAULT_PRIMARY_LABEL)


def _english(tasks: tuple) -> str:
    labels = [_TASK_LABELS.get(task, str(task).replace("_", " ")) for task in tasks]
    if len(labels) <= 1:
        return labels[0] if labels else "no tasks"
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _refusal_reason(capability: dict) -> str:
    """The reason the capability probe gave, mapped onto the tokens the contract publishes."""
    for reason in capability.get("reasons") or ():
        if reason in _REFUSAL_REASONS or reason in _EXTRA_REFUSAL_REASONS:
            return reason
    return _NOT_MEASURED


def _refusal_note(reason: str) -> str:
    """One UI-ready sentence per reason; capability notes are not reused because they carry
    environment variable names and provider ids."""
    if reason == "disabled":
        return "NPU support is switched off, so the run stays on the primary compute device."
    if reason == "absent":
        return (
            "No NPU was detected on this machine, so the run stays on the primary compute device."
        )
    if reason == "no_runtime":
        return "The NPU is present, but no installed NPU runtime can drive it, so it stays idle."
    if reason == "no_supported_task":
        return (
            "No NPU runtime here can run a task this run is able to hand off, so nothing is "
            "offloaded."
        )
    if reason == "unsupported_family":
        return "This NPU generation is not one any installed runtime supports, so it stays idle."
    return "NPU capability could not be measured, so the run assumes the NPU cannot help."


def _installed_providers(capability: dict) -> list:
    providers = capability.get("providers")
    if providers is None:
        providers = _runtime_providers()
    return [provider for provider in providers or () if provider.get("installed")]


def _select_tasks(
    capability: dict,
    *,
    tokenize_offload: bool,
    eval_enabled: bool,
    qat: bool,
) -> tuple:
    """The tasks this run will really hand over, and the installed engines that serve them.

    A capability can list a task that no installed runtime claims; offering it would put a task
    in the plan that nothing can run, so the provider rows have the last word here.
    """
    listed = set(capability.get("tasks") or ())
    installed = _installed_providers(capability)
    wanted_flags = {"tokenize": tokenize_offload, "eval": eval_enabled, "qat_calibrate": qat}
    wanted = [kind for kind in _TASK_ORDER if wanted_flags[kind] and kind in listed]
    tasks = tuple(
        kind for kind in wanted if any(kind in (row.get("supports") or ()) for row in installed)
    )
    serving = [
        row.get("name")
        for row in installed
        if row.get("name") and any(kind in (row.get("supports") or ()) for kind in tasks)
    ]
    return tasks, tuple(dict.fromkeys(serving))


def _plan(
    enabled: bool,
    mode: str,
    tasks: tuple,
    providers: tuple,
    refused_reason: Optional[str],
    notes: list,
) -> NpuAssistPlan:
    return NpuAssistPlan(
        enabled = enabled,
        mode = mode,
        tasks = tuple(tasks),
        providers = tuple(providers),
        refused_reason = refused_reason,
        notes = tuple(notes),
    )


def plan_npu_assist(
    capability: Optional[dict] = None,
    *,
    requested: Optional[str] = None,
    primary_kind: str = "dgpu",
    eval_enabled: bool = False,
    tokenize_offload: bool = True,
    qat: bool = False,
) -> NpuAssistPlan:
    """Decide what the NPU does for this run, and say so in words a UI can show unedited."""
    measured = _capability(capability)
    request = _normalise_requested(requested)
    label = _primary_label(primary_kind)
    # Every plan names the device that owns the backward pass, including the refusals: it is the
    # one fact the user needs in order to read the rest of the sentence.
    notes = [f"The backward pass runs on {label}."]

    if request == "off":
        notes.append("NPU assist is off for this run, so nothing is handed to the NPU.")
        return _plan(False, "off", (), (), None, notes)

    if request == "full":
        if not measured.get("backward_pass"):
            # The honesty rule: the mode exists so the UI can offer it and explain the refusal.
            notes.append(
                "No NPU vendor exposes a backward pass, so the NPU cannot run the training step "
                "itself; the assist mode runs the tasks the NPU can do instead, off the critical "
                "path of a training step."
            )
            return _plan(False, "off", (), (), "no_backward_pass", notes)
        tasks, providers = _select_tasks(
            measured,
            tokenize_offload = tokenize_offload,
            eval_enabled = eval_enabled,
            qat = qat,
        )
        notes.append(f"The NPU runs the whole training step, backward pass included, on {label}.")
        if tasks:
            notes.append(f"It also handles {_english(tasks)} without leaving the device.")
        return _plan(True, "full", tasks, providers, None, notes)

    if primary_kind == "npu":
        # A device cannot both train and assist the run that is training on it. "auto" is
        # expected to refuse here, and an explicit request gets the same reason, so the UI can
        # explain the disabled toggle instead of showing an unexplained switch.
        notes.append(
            "The NPU is the primary training device here, so it cannot also assist this run."
        )
        return _plan(False, "off", (), (), "npu_is_primary", notes)

    if measured.get("mode") not in ("assist", "full"):
        reason = _refusal_reason(measured)
        notes.append(_refusal_note(reason))
        # "auto" asked for nothing in particular, so an unavailable NPU is not a refusal.
        return _plan(False, "off", (), (), reason if request == "assist" else None, notes)

    tasks, providers = _select_tasks(
        measured,
        tokenize_offload = tokenize_offload,
        eval_enabled = eval_enabled,
        qat = qat,
    )
    if not tasks:
        notes.append(
            "No installed NPU runtime can run a task this run is able to hand off, so nothing is "
            "offloaded and the run keeps its usual step time."
        )
        return _plan(False, "off", (), (), "no_supported_task", notes)

    notes.append(f"The NPU takes {_english(tasks)} while the training step continues on {label}.")
    notes.append(
        "That work runs off the critical path of a training step, so the step never waits for it."
    )
    return _plan(True, "assist", tasks, providers, None, notes)


class AssistQueue:
    """Bounded, single-consumer queue that runs assist tasks on one worker thread."""

    def __init__(
        self,
        *,
        runner: Callable[[AssistTask], Any],
        max_pending: int = 32,
        drain_timeout_s: float = 5.0,
    ):
        self._runner = runner
        # A queue that accepted more than this would hold tokenized batches alive for no gain.
        self._max_pending = max(1, int(max_pending))
        self._drain_timeout_s = float(drain_timeout_s)
        self._condition = threading.Condition()
        self._shutdown = threading.Event()
        self._finished = threading.Event()
        self._pending: deque = deque()
        self._results: deque = deque(maxlen = self._max_pending)
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._closed = False
        self._sequence = 0
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._failed = 0
        self._last_error: Optional[str] = None

    def submit(self, kind: str, payload: Optional[dict] = None) -> bool:
        """Queue one task, or refuse it. Never blocks: a full queue is a dropped task, not a
        stalled training step -- the step time matters more than the offload does."""
        with self._condition:
            if self._closed:
                return False
            if len(self._pending) >= self._max_pending:
                self._dropped += 1
                return False
            self._sequence += 1
            task = AssistTask(kind = kind, payload = dict(payload or {}), sequence = self._sequence)
            self._pending.append(task)
            self._submitted += 1
            self._condition.notify()
            return True

    def start(self) -> None:
        """Start the worker. Task submissions are buffered before this, so a trainer can open the
        queue before the model finishes loading."""
        with self._condition:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(target = self._run, name = "npu-assist", daemon = True)
        self._thread.start()

    def close(self, *, drain: bool = True) -> int:
        """Stop the worker and return how many tasks never ran. Safe to call more than once."""
        with self._condition:
            if self._closed:
                return 0
            self._closed = True
            self._shutdown.set()
            thread = self._thread
            never_ran = 0
            if thread is None or not drain:
                # No worker to drain into, or the caller wants the backlog gone: either way the
                # buffered work is discarded here and reported to the caller.
                never_ran = len(self._pending)
                self._pending.clear()
            self._condition.notify_all()
        if thread is not None:
            self._finished.wait(self._drain_timeout_s)
        with self._condition:
            # A runner that outlived the drain timeout still owns the task it is inside; the rest
            # are discarded so the number we return is the number that will not run.
            never_ran += len(self._pending)
            self._pending.clear()
        return never_ran

    def stats(self) -> dict:
        with self._condition:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "dropped": self._dropped,
                "failed": self._failed,
                "pending": len(self._pending),
                "last_error": self._last_error,
            }

    def results(self) -> list:
        with self._condition:
            return list(self._results)

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._shutdown.is_set():
                        self._condition.wait()
                    if not self._pending:
                        return
                    task = self._pending.popleft()
                try:
                    result = self._runner(task)
                except Exception as exc:  # noqa: BLE001 -- one bad task must not stop the queue
                    with self._condition:
                        self._failed += 1
                        self._last_error = f"{type(exc).__name__}: {exc}"
                        self._condition.notify_all()
                    continue
                with self._condition:
                    self._completed += 1
                    self._results.append((task, result))
                    self._condition.notify_all()
        finally:
            self._finished.set()
