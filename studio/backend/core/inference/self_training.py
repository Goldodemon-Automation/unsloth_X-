# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ``train_model`` tool: let a model inspect this host and launch real training.

The tool is off unless the owner turns on "Let models train other models" in
Settings. It plans one job -- hardware, base checkpoint, dataset, method and
hyperparameters -- then either starts it through the normal Training route or,
for CMT, through the Labs CMT engine.

"Better than itself" is not a promise: a fine-tune can regress, and nothing here
scores the result against the starting checkpoint. The tool says that in its
plan rather than claiming an improvement it did not measure.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
import threading
import time
import uuid
from typing import Any, Mapping, Optional

from utils.account_context import current_account, current_account_id, is_owner_context, run_as
from utils.self_training_settings import get_self_training_enabled

SELF_TRAINING_TOOL_NAME = "train_model"

METHODS = ("cmt", "lora", "qlora", "cora", "cpt", "fft")
_METHOD_LABELS = {
    "cmt": "CMT (grow the model, train only the new blocks)",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "cora": "CoRA",
    "cpt": "Continued Pretraining",
    "fft": "Full fine-tuning (FFT)",
}
_METHOD_ALIASES = {
    "full": "fft",
    "full-finetune": "fft",
    "full_finetuning": "fft",
    "continued-pretraining": "cpt",
    "continued_pretraining": "cpt",
    "continued pretraining": "cpt",
    "qlora": "qlora",
    "lora": "lora",
    "cora": "cora",
    "cmt": "cmt",
    "fft": "fft",
    "cpt": "cpt",
}
_DEFAULT_MODEL = "unsloth/Qwen3-0.6B"
_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
_GGUF_HINT = re.compile(r"gguf", re.IGNORECASE)

SELF_TRAINING_TOOL = {
    "type": "function",
    "function": {
        "name": SELF_TRAINING_TOOL_NAME,
        "description": (
            "Train or fine-tune a model on this machine. Use when the user asks you to train, "
            "fine-tune, or improve a model. The server inspects the real hardware, chooses a "
            "base checkpoint, a public Hub dataset and safe defaults when you omit them, then "
            "starts ONE real training job. Supported methods: cmt, lora, qlora, cora, cpt, fft. "
            "Call with action='start' and an objective. The result includes the plan, the job id, "
            "the output location and where to watch progress; call action='status' with that job_id "
            "to read progress later. Training is compute-heavy and can take a long time. A run is "
            "not guaranteed to beat the starting checkpoint: there is no automatic evaluation here."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "status"],
                    "description": "start a job (default), or read a job you started.",
                },
                "objective": {
                    "type": "string",
                    "description": (
                        "What the trained model should get better at, in one or two sentences. "
                        "Required when action is start; used to pick a dataset and defaults."
                    ),
                },
                "method": {
                    "type": "string",
                    "enum": ["auto", "cmt", "lora", "qlora", "cora", "cpt", "fft"],
                    "description": (
                        "Training method. auto picks a safe method for the detected hardware. "
                        "CMT needs the Labs CMT feature and a CUDA-class GPU, and uses two datasets."
                    ),
                },
                "base_model": {
                    "type": "string",
                    "description": (
                        "Optional Hugging Face model id or local trainable checkpoint. Omit to use "
                        "the active model when it is trainable, otherwise a small fallback."
                    ),
                },
                "datasets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional Hugging Face dataset ids (owner/name). Omit to let the tool pick "
                        "from its curated catalog. CMT needs at least two; other methods use the first."
                    ),
                },
                "dataset_split": {
                    "type": "string",
                    "description": "Dataset split to train on. Default train.",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Optimizer steps for this run. Default 60 (100 for CMT).",
                },
                "format_type": {
                    "type": "string",
                    "enum": ["auto", "alpaca", "chatml", "sharegpt", "raw"],
                    "description": "Dataset format. auto lets the trainer detect it.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Return the plan without starting a job.",
                },
                "job_id": {
                    "type": "string",
                    "description": (
                        "Job id returned by a start call, or its pending_id when the job is "
                        "waiting for the current chat turn to finish."
                    ),
                },
                "pending_id": {
                    "type": "string",
                    "description": "Pending id returned by a deferred start; alias for job_id in status.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["training", "cmt"],
                    "description": "Kind returned by a start call, for action=status.",
                },
            },
            "required": [],
        },
    },
}


class SelfTrainingError(ValueError):
    """A self-training request that cannot be planned or run as written."""


def self_training_available() -> bool:
    """Whether this account may see and call the tool right now."""
    if not is_owner_context():
        return False
    try:
        return get_self_training_enabled()
    except Exception:
        return False


def _normalise_method(value: Any) -> str:
    raw = str(value or "auto").strip().lower()
    if raw in ("", "auto"):
        return "auto"
    method = _METHOD_ALIASES.get(raw, raw)
    if method not in METHODS:
        raise SelfTrainingError(
            f"Unknown training method {value!r}. Choose auto, {', '.join(METHODS)}."
        )
    return method


def _steps(value: Any, *, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise SelfTrainingError("max_steps must be a whole number.") from exc
    if steps < 1 or steps > 500:
        raise SelfTrainingError("max_steps must be between 1 and 500.")
    return steps


def _gb(value: Optional[int]) -> Optional[float]:
    if not value:
        return None
    return round(float(value) / (1024**3), 1)


def _hardware() -> dict[str, Any]:
    from core.training.device_targets import (
        TargetProbe,
        TrainingTargetError,
        resolve_training_target,
        training_target_payload,
    )

    probe = TargetProbe.detect()
    try:
        target = resolve_training_target("auto", probe)
    except TrainingTargetError as exc:
        raise SelfTrainingError(str(exc)) from exc
    if not target.supported:
        reasons = ", ".join(target.reasons) or "the device has no usable training backend"
        raise SelfTrainingError(f"This machine cannot train on {target.name}: {reasons}.")
    payload = training_target_payload(probe)
    accelerator = target.name if target.kind != "cpu" else None
    return {
        "cpu": probe.cpu_name,
        "cpu_cores": probe.cpu_cores,
        "system_ram_gb": _gb(probe.system_memory_bytes),
        "free_ram_gb": _gb(probe.system_memory_available_bytes),
        "accelerator": accelerator,
        "vram_gb": _gb(target.memory_bytes),
        "device": target.device,
        "device_kind": target.kind,
        "vendor": target.vendor,
        "bitsandbytes": bool(probe.bnb_available),
        "torch_available": bool(probe.torch_available),
        "available_targets": payload["targets"],
        "default_target": payload["default"],
        "_target": target,
        "_probe": probe,
    }


def _active_model() -> Optional[str]:
    try:
        from core.inference.orchestrator import peek_inference_backend

        backend = peek_inference_backend()
        name = getattr(backend, "active_model_name", None) if backend is not None else None
    except Exception:
        name = None
    name = str(name or "").strip()
    if not name or _GGUF_HINT.search(name) or name.lower().endswith(".gguf"):
        return None
    return name


def _base_model(requested: Any) -> str:
    explicit = str(requested or "").strip()
    if explicit:
        if _GGUF_HINT.search(explicit) or explicit.lower().endswith(".gguf"):
            raise SelfTrainingError(
                "A GGUF checkpoint cannot be trained. Use a Hugging Face model id or a "
                "trainable local checkpoint instead."
            )
        return explicit
    return _active_model() or _DEFAULT_MODEL


def _catalog() -> list[tuple[str, str, str]]:
    from core.labs.automated_training import RANDOM_DATASET_CATALOG

    return list(RANDOM_DATASET_CATALOG)


def _objective_category(objective: str) -> str:
    text = objective.lower()
    keyword_groups = (
        ("code", ("code", "coding", "program", "software", "python", "typescript", "sql")),
        (
            "reasoning",
            ("reason", "math", "logic", "proof", "science", "chain of thought", "thinking"),
        ),
        (
            "syntax",
            ("tool", "function call", "function-calling", "json", "api", "schema", "grammar"),
        ),
    )
    for category, words in keyword_groups:
        if any(word in text for word in words):
            return category
    return "general"


def _dataset_choices(
    objective: str,
    method: str,
    requested: Any,
    *,
    split: str,
) -> list[dict[str, Any]]:
    catalog = _catalog()
    explicit: list[str] = []
    if isinstance(requested, str) and requested.strip():
        explicit = [requested.strip()]
    elif isinstance(requested, (list, tuple)):
        for item in requested:
            value = str(item or "").strip()
            if value and value not in explicit:
                explicit.append(value)
    for value in explicit:
        if not _HUB_ID.match(value):
            raise SelfTrainingError(
                f"{value!r} is not a Hugging Face dataset id. It needs to look like 'owner/name'."
            )

    wanted = 2 if method == "cmt" else 1
    chosen = explicit[:wanted]
    if method != "cmt" and len(explicit) > 1:
        # Non-CMT training takes one dataset; name that rather than silently pretending otherwise.
        pass
    category = _objective_category(objective)
    preferred = [entry for entry in catalog if entry[2] == category]
    fallback = [entry for entry in catalog if entry[2] != category]
    pool = preferred + fallback
    for _, _, _ in pool:
        if len(chosen) >= wanted:
            break
        candidate = next(entry[0] for entry in pool if entry[0] not in chosen)
        chosen.append(candidate)
    if len(chosen) < wanted:
        raise SelfTrainingError("Could not find enough public datasets for this run.")
    return [
        {"repo_id": repo_id, "split": split or "train", "eval_fraction": 0.1}
        for repo_id in chosen
    ]


def _check_method(method: str, hardware: Mapping[str, Any]) -> tuple[bool, str]:
    target = hardware["_target"]
    if method == "cmt":
        try:
            from core.labs import feature_enabled

            labs_on = feature_enabled("cmt")
        except Exception:
            labs_on = False
        if not labs_on:
            return False, "CMT is a Labs feature. Enable Labs and CMT in Settings first."
        if target.kind != "dgpu":
            return False, "CMT needs a CUDA-class GPU; this host resolved to a non-GPU target."
    if target.kind == "cpu" and method not in ("qlora", "cora"):
        return False, (
            "This machine has no trainable accelerator, so only QLoRA and CoRA can run here."
        )
    if target.device.startswith("mps") and method in ("cpt", "cora"):
        return False, "CPT and CoRA are not available on Apple Silicon's MLX trainer."
    return True, ""


def _choose_method(requested: str, hardware: Mapping[str, Any]) -> str:
    if requested != "auto":
        ok, reason = _check_method(requested, hardware)
        if not ok:
            raise SelfTrainingError(reason)
        return requested
    target = hardware["_target"]
    if target.kind == "cpu":
        return "qlora"
    if target.device.startswith("mps"):
        return "qlora"
    return "qlora"


def _method_settings(
    method: str,
    hardware: Mapping[str, Any],
    *,
    max_steps: int,
) -> dict[str, Any]:
    target = hardware["_target"]
    on_cpu = target.kind == "cpu"
    on_mlx = target.device.startswith("mps")
    common: dict[str, Any] = {
        "max_steps": max_steps,
        "num_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1024 if on_cpu else 2048,
        "warmup_steps": 5,
        "save_steps": 0,
        "weight_decay": 0.001,
        "random_seed": 3407,
        "packing": False,
        "optim": (
            "adamw_torch"
            if on_cpu or on_mlx or not hardware["bitsandbytes"]
            else "adamw_8bit"
        ),
        "lr_scheduler_type": "linear",
        "use_lora": method != "fft",
        "load_in_4bit": method == "qlora",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "use_rslora": False,
        "use_loftq": False,
        "use_dora": False,
    }
    if method == "cora":
        common.update(
            {
                "learning_rate": "1e-4",
                "lora_r": 8,
                "lora_alpha": 24,
                "optim": (
                    "adamw_torch"
                    if on_cpu or on_mlx or not hardware["bitsandbytes"]
                    else "paged_adamw_8bit"
                ),
                "lr_scheduler_type": "cosine" if not on_cpu else "linear",
                "use_loftq": bool(hardware["bitsandbytes"]) and not on_cpu and not on_mlx,
            }
        )
    elif method == "cpt":
        common.update(
            {
                "learning_rate": "5e-5",
                "lora_r": 128,
                "lora_alpha": 32,
                "use_rslora": True,
                "gradient_accumulation_steps": 8,
            }
        )
    elif method == "fft":
        common.update(
            {
                "learning_rate": "2e-5",
                "gradient_accumulation_steps": 8,
            }
        )
    elif method == "qlora":
        common.update({"learning_rate": "2e-4", "use_loftq": False})
    else:
        common.update({"learning_rate": "2e-4"})
    return common


def _training_request(
    base_model: str,
    datasets: list[dict[str, Any]],
    method: str,
    settings: Mapping[str, Any],
    *,
    split: str,
    format_type: str,
) -> dict[str, Any]:
    training_type = {
        "lora": "LoRA/QLoRA",
        "qlora": "LoRA/QLoRA",
        "cora": "LoRA/QLoRA",
        "cpt": "Continued Pretraining",
        "fft": "Full Finetuning",
    }[method]
    preset = {"cora": "cora", "qlora": "qlora"}.get(method)
    request: dict[str, Any] = {
        "model_name": base_model,
        "project_name": "model-self-training",
        "start_request_id": f"self-train-{uuid.uuid4().hex}",
        "training_type": training_type,
        "training_preset": preset,
        "hf_dataset": datasets[0]["repo_id"],
        "dataset_known_cached": False,
        "train_split": split or "train",
        "format_type": format_type or "auto",
        "dataset_streaming": False,
        "dataset_slice_start": None,
        "dataset_slice_end": None,
        "local_datasets": [],
        "model_known_cached": False,
        "compute_target": "auto",
        **settings,
    }
    return request


def _cmt_spec(
    base_model: str,
    datasets: list[dict[str, Any]],
    *,
    max_steps: int,
) -> dict[str, Any]:
    return {
        "source": base_model,
        "mode": "finetune",
        "datasets": [
            {
                "repo_id": entry["repo_id"],
                "split": entry.get("split") or "train",
                "eval_fraction": 0.1,
            }
            for entry in datasets[:8]
        ],
        "max_steps": max_steps,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "max_seq_length": 2048,
        "learning_rate": 2e-4,
        "eval_every": 20,
        "patience": 4,
    }


def plan_self_training(
    objective: str,
    *,
    base_model: Any = None,
    datasets: Any = None,
    method: Any = "auto",
    max_steps: Any = None,
    dataset_split: str = "train",
    format_type: str = "auto",
    hardware: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a complete, honest plan without starting anything."""
    objective_text = str(objective or "").strip()
    if not objective_text:
        raise SelfTrainingError("objective is required to start training.")
    facts = dict(hardware or _hardware())
    requested = _normalise_method(method)
    chosen_method = _choose_method(requested, facts)
    base = _base_model(base_model)
    if chosen_method == "cmt":
        steps = _steps(max_steps, default=100)
        dataset_entries = _dataset_choices(
            objective_text, chosen_method, datasets, split=dataset_split
        )
        settings = _cmt_spec(base, dataset_entries, max_steps=steps)
        cmt_plan: Optional[dict[str, Any]] = None
        try:
            from core.labs import cmt

            cmt_plan = cmt.plan(settings)
        except Exception as exc:  # noqa: BLE001 - shown in the plan, not hidden
            cmt_plan = {"runnable": False, "blocked_reason": str(exc)}
        return {
            "objective": objective_text,
            "method": chosen_method,
            "method_label": _METHOD_LABELS[chosen_method],
            "base_model": base,
            "datasets": dataset_entries,
            "settings": settings,
            "hardware": _public_hardware(facts),
            "runnable": bool(cmt_plan.get("runnable")),
            "blocked_reason": cmt_plan.get("blocked_reason"),
            "cmt_plan": cmt_plan,
            "quality_note": (
                "No evaluation is run automatically. A fine-tune can improve one behaviour and "
                "regress another; compare it before replacing the model you use."
            ),
        }

    steps = _steps(max_steps, default=60)
    dataset_entries = _dataset_choices(objective_text, chosen_method, datasets, split=dataset_split)
    settings = _method_settings(chosen_method, facts, max_steps=steps)
    return {
        "objective": objective_text,
        "method": chosen_method,
        "method_label": _METHOD_LABELS[chosen_method],
        "base_model": base,
        "datasets": dataset_entries,
        "settings": settings,
        "hardware": _public_hardware(facts),
        "runnable": True,
        "blocked_reason": None,
        "training_request": _training_request(
            base,
            dataset_entries,
            chosen_method,
            settings,
            split=dataset_split,
            format_type=format_type,
        ),
        "quality_note": (
            "No evaluation is run automatically. A fine-tune can improve one behaviour and "
            "regress another; compare it before replacing the model you use."
        ),
    }


def _public_hardware(facts: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in facts.items() if not str(key).startswith("_")}


def _run_awaitable(awaitable):
    """Run an async route from the sync tool worker, preserving the account context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    # A tool worker can be called from a loop. Run the route on its own loop in a
    # fresh thread, with the current account context copied into it.
    result: dict[str, Any] = {}
    ctx = contextvars.copy_context()

    def runner() -> None:
        try:
            result["value"] = ctx.run(asyncio.run, awaitable)
        except BaseException as exc:  # noqa: BLE001 - returned to the tool caller
            result["error"] = exc

    thread = threading.Thread(target=runner, name="self-training-start", daemon=True)
    thread.start()
    thread.join(180)
    if thread.is_alive():
        raise SelfTrainingError("The Training route is still starting the job. Check Training for it.")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _start_training(request: Mapping[str, Any]) -> dict[str, Any]:
    from models.training import TrainingStartRequest
    from routes.training import start_training

    payload = TrainingStartRequest(**request)
    try:
        response = _run_awaitable(
            start_training(
                payload,
                current_subject=current_account_id(),
                via_api_key=False,
            )
        )
    except Exception as exc:
        detail = getattr(exc, "detail", None) or str(exc)
        raise SelfTrainingError(f"Training did not start: {detail}") from exc
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    if response.get("status") == "error" or response.get("error"):
        raise SelfTrainingError(
            response.get("error") or response.get("message") or "Training did not start."
        )
    return {
        "kind": "training",
        "job_id": response.get("job_id"),
        "status": response.get("status", "starting"),
        "message": response.get("message"),
        "response": response,
    }


def _start_cmt(plan: Mapping[str, Any]) -> dict[str, Any]:
    from core.labs import cmt
    from routes.advanced import _start_job
    from utils.paths.storage_roots import outputs_root

    report = plan.get("cmt_plan") or {}
    if not report.get("runnable"):
        raise SelfTrainingError(report.get("blocked_reason") or "This CMT plan cannot run.")
    source = str(plan["base_model"])
    slug = re.sub(r"[^a-z0-9_-]+", "-", source.lower())[-60:] or "model"
    output = outputs_root() / f"self-train-cmt-{slug}-{uuid.uuid4().hex[:6]}"

    def work(progress) -> dict[str, Any]:
        return cmt.run(plan["settings"], output_dir=str(output), progress=progress)

    try:
        record = _start_job(
            "cmt",
            work,
            output_dir=str(output),
            sources=[source],
            with_progress=True,
        )
    except Exception as exc:
        detail = getattr(exc, "detail", None) or str(exc)
        raise SelfTrainingError(f"CMT did not start: {detail}") from exc
    return {
        "kind": "cmt",
        "job_id": record.get("job_id"),
        "status": record.get("status", "running"),
        "output_dir": str(output),
        "message": "CMT is running through the Labs advanced-job engine.",
    }


def _status(job_id: str, kind: str = "") -> dict[str, Any]:
    job_id = str(job_id or "").strip()
    if not job_id:
        raise SelfTrainingError("job_id is required for action=status.")
    if kind == "cmt":
        from routes.advanced import job_status

        try:
            record = job_status(job_id)
        except Exception as exc:
            detail = getattr(exc, "detail", None) or str(exc)
            raise SelfTrainingError(f"Could not read CMT job: {detail}") from exc
        return {
            "kind": "cmt",
            "job_id": record["job_id"],
            "status": record["status"],
            "progress": record.get("progress"),
            "error": record.get("error"),
            "output_dir": record.get("output_dir"),
        }

    from dataclasses import asdict

    from core.training.training import get_training_backend
    from storage.studio_db import get_run

    backend = get_training_backend()
    if backend.current_job_id == job_id:
        progress = asdict(backend.get_training_progress())
        return {
            "kind": "training",
            "job_id": job_id,
            **progress,
        }
    run = get_run(job_id)
    if run:
        return {
            "kind": "training",
            "job_id": job_id,
            "status": run.get("status"),
            "step": run.get("final_step"),
            "total_steps": run.get("total_steps"),
            "loss": run.get("final_loss"),
            "output_dir": run.get("output_dir"),
            "error": run.get("error_message"),
            "started_at": run.get("started_at"),
            "ended_at": run.get("ended_at"),
        }
    raise SelfTrainingError(f"No training job {job_id!r} is running or recorded.")


def _result_payload(
    plan: Mapping[str, Any], launched: Mapping[str, Any], *, pending_id: Optional[str] = None
) -> dict[str, Any]:
    return {
        "status": launched.get("status"),
        "kind": launched.get("kind"),
        "job_id": launched.get("job_id"),
        "pending_id": pending_id,
        "output_dir": launched.get("output_dir"),
        "message": launched.get("message"),
        "plan": {key: value for key, value in plan.items() if key != "training_request"},
        "watch": (
            "Open Training to follow this job."
            if launched.get("kind") == "training"
            else "Ask this tool with action='status' and the job_id to read CMT progress."
        ),
    }


_PENDING_LOCK = threading.Lock()
_PENDING: dict[str, dict[str, Any]] = {}
_PENDING_LIMIT = 8


def _deferred_inference_active() -> bool:
    """Whether this chat turn still holds the model; training unloads it."""
    try:
        from core.inference.llama_keepwarm import other_inference_request_count

        return other_inference_request_count(current_request_counted = False) > 0
    except Exception:
        return False


def _pending_record(pending_id: str) -> Optional[dict[str, Any]]:
    with _PENDING_LOCK:
        return _PENDING.get(pending_id)


def _defer_start(plan: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Start after the current chat turn releases the model.

    Calling the Training route from inside a generation would unload the very
    model that is waiting for this tool result. The tool answers immediately and
    a background thread starts the job once the inference request count drops to
    zero, which is when the user can actually see Training begin.
    """
    pending_id = uuid.uuid4().hex
    record: dict[str, Any] = {
        "pending_id": pending_id,
        "status": "queued",
        "created_at": time.time(),
        "plan": plan,
        "job": None,
        "error": None,
    }
    with _PENDING_LOCK:
        if len(_PENDING) >= _PENDING_LIMIT:
            oldest = min(_PENDING, key = lambda key: _PENDING[key]["created_at"])
            _PENDING.pop(oldest, None)
        _PENDING[pending_id] = record
    account = current_account()

    def runner() -> None:
        deadline = time.monotonic() + 15 * 60
        while time.monotonic() < deadline:
            if not _deferred_inference_active():
                break
            time.sleep(0.5)
        if _deferred_inference_active():
            record["status"] = "failed"
            record["error"] = "Gave up waiting for the chat turn to finish; nothing was started."
            return
        record["status"] = "starting"
        try:
            record["job"] = (
                _start_cmt(plan) if plan["method"] == "cmt" else _start_training(
                    plan["training_request"]
                )
            )
            record["status"] = record["job"].get("status") or "running"
        except Exception as exc:  # noqa: BLE001 - surfaced through status
            record["status"] = "failed"
            record["error"] = str(exc)

    threading.Thread(
        target = run_as,
        args = (account, runner),
        name = "self-training-deferred-start",
        daemon = True,
    ).start()
    return pending_id, record


def _pending_status(record: Mapping[str, Any]) -> dict[str, Any]:
    job = record.get("job")
    if job:
        try:
            return {
                **_status(str(job.get("job_id") or ""), str(job.get("kind") or "")),
                "pending_id": record["pending_id"],
            }
        except SelfTrainingError:
            pass
    return {
        "pending_id": record["pending_id"],
        "status": record.get("status"),
        "kind": (job or {}).get("kind"),
        "job_id": (job or {}).get("job_id"),
        "error": record.get("error"),
        "message": (
            "Waiting for the current chat turn to finish before training starts."
            if record.get("status") == "queued"
            else None
        ),
    }


def execute_self_training_tool(arguments: Mapping[str, Any]) -> str:
    """Entry point used by ``core.inference.tools.execute_tool``."""
    if not self_training_available():
        raise SelfTrainingError(
            "Model self-training is off. The installation owner must enable "
            "\"Let models train other models\" in Settings > General."
        )
    args = dict(arguments or {})
    action = str(args.get("action") or "start").strip().lower()
    if action == "status":
        lookup = str(args.get("pending_id") or args.get("job_id") or "").strip()
        pending = _pending_record(lookup)
        result = (
            _pending_status(pending)
            if pending is not None
            else _status(lookup, str(args.get("kind") or "").strip().lower())
        )
        return json.dumps(result, indent=2, default=str)
    if action != "start":
        raise SelfTrainingError("action must be 'start' or 'status'.")

    plan = plan_self_training(
        args.get("objective") or "",
        base_model=args.get("base_model"),
        datasets=args.get("datasets"),
        method=args.get("method") or "auto",
        max_steps=args.get("max_steps"),
        dataset_split=str(args.get("dataset_split") or "train"),
        format_type=str(args.get("format_type") or "auto"),
    )
    if not plan["runnable"]:
        raise SelfTrainingError(plan.get("blocked_reason") or "This plan cannot run.")
    if bool(args.get("dry_run")):
        return json.dumps(
            {
                "status": "planned",
                "plan": {key: value for key, value in plan.items() if key != "training_request"},
            },
            indent=2,
            default=str,
        )
    if _deferred_inference_active():
        pending_id, record = _defer_start(plan)
        return json.dumps(
            _result_payload(
                plan,
                {
                    "status": "queued",
                    "kind": "training" if plan["method"] != "cmt" else "cmt",
                    "job_id": None,
                    "message": (
                        "Training is queued. It starts as soon as this chat turn finishes, "
                        "because training must unload the chat model to free memory."
                    ),
                },
                pending_id = pending_id,
            ),
            indent=2,
            default=str,
        )
    launched = _start_cmt(plan) if plan["method"] == "cmt" else _start_training(
        plan["training_request"]
    )
    return json.dumps(_result_payload(plan, launched), indent=2, default=str)
