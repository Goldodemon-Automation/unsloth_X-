# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-gated tools for Unsloth's Labs and Training surfaces.

These tools do not invent a second permission system: they are advertised and
executed only while the owner's "Let models train other models" switch is on.
Every action calls the same route function the UI calls, so Labs feature gates,
training preflights, job leases and account scoping all still apply.

Arguments stay as JSON objects because the Labs panels already define rich
specs. The tool descriptions name the shapes; the route's own Pydantic model
reports the exact missing or invalid field.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Mapping, Optional

from fastapi import HTTPException

from .self_training import (
    SELF_TRAINING_TOOL_NAME,
    _defer_start,
    _deferred_inference_active,
    _pending_record,
    _pending_status,
    _run_awaitable,
    _start_training,
    self_training_available,
)

MAX_RESULT_CHARS = 60_000

LABS_ACTIONS = (
    "state",
    "artifacts",
    "artifact_detail",
    "artifact_diff",
    "artifact_manifest",
    "lineage",
    "pipeline_presets",
    "repair_preset_plan",
    "pipeline_list",
    "pipeline_plan",
    "pipeline_save",
    "pipeline_delete",
    "benchmark_list",
    "benchmark_run",
    "benchmark_pin",
    "benchmark_unpin",
    "benchmark_delete",
    "cmt_plan",
    "cmt_run",
    "auto_scale_plan",
    "auto_scale_run",
    "surgery_plan",
    "surgery_run",
    "arch_swap_plan",
    "arch_swap_run",
    "housekeeping_scan",
    "housekeeping_delete",
    "housekeeping_rules",
    "housekeeping_rules_update",
    "housekeeping_rules_preview",
    "housekeeping_rules_apply",
    "automated_presets",
    "automated_plan",
    "automated_run",
    "job_status",
    "job_list",
)

TRAINING_ACTIONS = (
    "status",
    "metrics",
    "hardware",
    "offload",
    "start",
    "stop",
    "reset",
    "runs",
    "run_detail",
    "run_update",
    "run_delete",
    "resume_check",
    "diffusion_status",
    "diffusion_runs",
    "diffusion_run",
    "stop_diffusion",
    "dataset_preflight",
    "disk_preflight",
    "start_request",
)

MERGE_ACTIONS = ("validate", "plan", "run", "eval", "adapters", "moe", "status")

AUTOMATED_ACTIONS = ("presets", "plan", "run", "status")


LABS_TOOL = {
    "type": "function",
    "function": {
        "name": "labs",
        "description": (
            "Read and operate Unsloth Labs: artifacts, lineage, pipelines, benchmarks, CMT, "
            "auto-scale, model surgery, architecture swap, cleanup, automated training, and "
            "advanced job status. Only the installation owner can call this while "
            "\"Let models train other models\" is on in Settings. Plan actions start nothing; "
            "run actions write checkpoints or delete data and may need approval. CMT needs Labs "
            "CMT enabled and a CUDA-class GPU. Pass the same spec object the Labs panel sends."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(LABS_ACTIONS)},
                "spec": {
                    "type": "object",
                    "description": (
                        "Panel spec. CMT: {source, datasets:[{repo_id|path, split, eval_fraction}], "
                        "mode:'cpt'|'finetune', ...}. auto-scale: {model, datasets:[...], method, ...}. "
                        "surgery: {model, drop_layers, ...}. arch-swap: {source, target, ...}. "
                        "pipeline: {source, name, stages:[{kind, ...}]}. automated: {source, datasets:[...], "
                        "methods:['cmt'|'cpt'|'fft'], rounds, runs_per_round, ...}."
                    ),
                },
                "path": {"type": "string", "description": "Artifact path for detail or manifest."},
                "source": {"type": "string", "description": "Source checkpoint for repair_preset_plan."},
                "base": {"type": "string", "description": "Base model for repair_preset_plan."},
                "tolerance": {"type": "number"},
                "steps": {"type": "integer"},
                "sequence_length": {"type": "integer"},
                "learning_rate": {"type": "number"},
                "left": {"type": "string", "description": "Left artifact path for a diff."},
                "right": {"type": "string", "description": "Right artifact path for a diff."},
                "artifact": {"type": "string", "description": "Artifact path for a benchmark."},
                "baseline": {"type": "string", "description": "Baseline model for a benchmark."},
                "suites": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer"},
                "batch_size": {"type": "integer"},
                "timeout_seconds": {"type": "integer"},
                "output_dir": {"type": "string"},
                "pipeline_id": {"type": "string"},
                "record_id": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
                "confirm": {"type": "boolean", "description": "Required for housekeeping_delete."},
                "rules": {"type": "object", "description": "Cleanup retention rules."},
                "job_id": {"type": "string"},
            },
            "required": ["action"],
        },
    },
}


AUTOMATED_TRAINING_TOOL = {
    "type": "function",
    "function": {
        "name": "automated_training",
        "description": (
            "Run Unsloth Labs automated training: plan rounds over datasets and methods, then "
            "train, benchmark, keep the best checkpoint, and continue-train it. Requires the "
            "owner's model-access switch and Labs automated-training enabled. Use action=plan "
            "first; action=run launches a real long-running job. Use action=status with its "
            "job_id to follow it. The spec is the Labs automated panel spec: {source, datasets, "
            "methods:['cmt'|'cpt'|'fft'], rounds, runs_per_round, continue_runs, steps_per_run, "
            "preset, benchmark:{suites,limit,batch_size}}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(AUTOMATED_ACTIONS)},
                "spec": {"type": "object"},
                "output_dir": {"type": "string"},
                "job_id": {"type": "string"},
            },
            "required": ["action"],
        },
    },
}


MERGE_TOOL = {
    "type": "function",
    "function": {
        "name": "merge_models",
        "description": (
            "Merge 2-6 full checkpoints (linear, slerp, TIES, DARE), merge LoRA adapters, or "
            "build a MoE merge. This is the Merge page's backend and requires the owner's "
            "model-access switch. Use plan/validate first; run writes a checkpoint and may "
            "need approval. Adapter merging is synchronous; full merges return a job id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(MERGE_ACTIONS)},
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-6 model ids or local checkpoint folders.",
                },
                "model_a": {"type": "string"},
                "model_b": {"type": "string"},
                "weights": {"type": "array", "items": {"type": "number"}},
                "output_dir": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["linear", "slerp", "ties", "dare"],
                },
                "weight": {"type": "number"},
                "density": {"type": "number"},
                "seed": {"type": "integer"},
                "max_shard_gb": {"type": "number"},
                "allow_remote_code": {"type": "boolean"},
                "allow_non_safetensors": {"type": "boolean"},
                "allow_tokenizer_conflict": {"type": "boolean"},
                "top_k": {"type": "integer"},
                "router_hidden": {"type": "integer"},
                "temperature": {"type": "number"},
                "layer_start": {"type": "integer"},
                "layer_end": {"type": "integer"},
                "source": {"type": "string", "description": "Merged checkpoint for eval."},
                "sample_tensors": {"type": "integer"},
                "adapters": {"type": "array", "items": {"type": "string"}},
                "rank": {"type": "integer"},
                "job_id": {"type": "string"},
            },
            "required": ["action"],
        },
    },
}


TRAINING_TOOL = {
    "type": "function",
    "function": {
        "name": "training",
        "description": (
            "Read and control Unsloth Training: current status and metrics, hardware, offload, "
            "stop/reset, run history and detail, resume checks, diffusion training status/history, "
            "dataset and disk preflights, and explicit LLM training starts. Requires the owner's "
            "model-access switch. Prefer train_model for an automatic plan; action=start takes a "
            "full TrainingStartRequest config. Training can unload the chat model, so a start "
            "during a reply is queued until that reply finishes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(TRAINING_ACTIONS)},
                "config": {
                    "type": "object",
                    "description": (
                        "TrainingStartRequest for action=start: model_name, training_type, "
                        "format_type, hf_dataset or local_datasets, plus optional hyperparameters. "
                        "Do not include tokens."
                    ),
                },
                "kind": {"type": "string", "enum": ["llm", "diffusion"]},
                "run_id": {"type": "string"},
                "display_name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
                "delete_artifacts": {"type": "boolean"},
                "confirm": {"type": "boolean", "description": "Required for run_delete."},
                "job_id": {"type": "string"},
                "pending_id": {"type": "string"},
                "start_request_id": {"type": "string"},
                "expected_job_id": {"type": "string"},
                "save": {"type": "boolean"},
                "path": {"type": "string"},
                "model": {"type": "string"},
                "method": {"type": "string"},
                "load_in_4bit": {"type": "boolean"},
                "max_seq_length": {"type": "integer"},
                "batch_size": {"type": "integer"},
                "gradient_accumulation": {"type": "integer"},
                "lora_rank": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "save_steps": {"type": "integer"},
                "model_known_cached": {"type": "boolean"},
                "max_rows": {"type": "integer"},
                "long_row_chars": {"type": "integer"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
                "query": {"type": "string"},
                "tag": {"type": "string"},
            },
            "required": ["action"],
        },
    },
}


STUDIO_ACCESS_TOOLS = (LABS_TOOL, AUTOMATED_TRAINING_TOOL, MERGE_TOOL, TRAINING_TOOL)
STUDIO_ACCESS_TOOL_NAMES = frozenset(
    {SELF_TRAINING_TOOL_NAME}
    | {tool["function"]["name"] for tool in STUDIO_ACCESS_TOOLS}
)


class StudioAccessError(ValueError):
    """A Studio operation could not run as written."""


def _require_access() -> None:
    if not self_training_available():
        raise StudioAccessError(
            "Studio access is off. The installation owner must enable "
            "\"Let models train other models\" in Settings > General."
        )


def _detail(error: HTTPException) -> str:
    detail = error.detail
    if isinstance(detail, str):
        return detail
    return json.dumps(detail, default = str)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode = "json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _call_route(action: str, target, *args, **kwargs) -> Any:
    try:
        result = target(*args, **kwargs)
        if inspect.isawaitable(result):
            result = _run_awaitable(result)
    except HTTPException as error:
        raise StudioAccessError(f"{action} failed: {_detail(error)}") from error
    except StudioAccessError:
        raise
    except Exception as error:  # noqa: BLE001 - route and Pydantic failures become readable
        raise StudioAccessError(f"{action} failed: {error}") from error
    return _jsonable(result)


def _result(value: Any) -> str:
    text = json.dumps(_jsonable(value), indent = 2, default = str)
    if len(text) > MAX_RESULT_CHARS:
        return text[:MAX_RESULT_CHARS] + "\n... result truncated; narrow the request for the rest."
    return text


def _action(arguments: Mapping[str, Any], actions: tuple[str, ...]) -> str:
    action = str(arguments.get("action") or "").strip().lower()
    if action not in actions:
        raise StudioAccessError(f"action must be one of: {', '.join(actions)}.")
    return action


def _scrub_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _scrub_secrets(item)
            for key, item in value.items()
            if key not in {"hf_token", "wandb_token", "s3_config"}
        }
    if isinstance(value, list):
        return [_scrub_secrets(item) for item in value]
    return value


def _spec(arguments: Mapping[str, Any]) -> dict[str, Any]:
    spec = arguments.get("spec")
    if not isinstance(spec, Mapping):
        raise StudioAccessError("spec must be an object.")
    return _scrub_secrets(dict(spec))


def _strings(arguments: Mapping[str, Any], key: str, *, default: Optional[list[str]] = None) -> list[str]:
    value = arguments.get(key)
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise StudioAccessError(f"{key} must be a list of strings.")


def _int(arguments: Mapping[str, Any], key: str, default: Optional[int] = None) -> Optional[int]:
    value = arguments.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise StudioAccessError(f"{key} must be an integer.") from error


def _float(arguments: Mapping[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    value = arguments.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise StudioAccessError(f"{key} must be a number.") from error


def _present(arguments: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: arguments[key] for key in keys if key in arguments and arguments[key] is not None}


def execute_labs_tool(arguments: Mapping[str, Any]) -> str:
    _require_access()
    args = dict(arguments or {})
    action = _action(args, LABS_ACTIONS)
    from routes import labs as routes

    if action == "state":
        return _result(
            _call_route(
                "labs state", routes.read_labs
            )
        )
    if action == "artifacts":
        return _result(
            _call_route(
                "labs artifacts",
                routes.list_artifacts,
                limit = _int(args, "limit", 50) or 50,
            )
        )
    if action == "artifact_detail":
        path = str(args.get("path") or args.get("artifact") or "").strip()
        return _result(_call_route("labs artifact detail", routes.artifact_detail, path))
    if action == "artifact_diff":
        left = str(args.get("left") or "").strip()
        right = str(args.get("right") or "").strip()
        return _result(_call_route("labs artifact diff", routes.artifact_diff, left, right))
    if action == "artifact_manifest":
        path = str(args.get("path") or args.get("artifact") or "").strip()
        return _result(_call_route("labs artifact manifest", routes.artifact_manifest, path))
    if action == "lineage":
        return _result(_call_route("labs lineage", routes.run_lineage))
    if action == "pipeline_presets":
        return _result(_call_route("labs pipeline presets", routes.pipeline_presets))
    if action == "repair_preset_plan":
        request = routes.RepairPresetRequest(
            source = str(args.get("source") or "").strip(),
            base = str(args.get("base") or "").strip(),
            suites = _strings(args, "suites", default = ["mmlu"]),
            tolerance = _float(args, "tolerance", 0.02) or 0.02,
            steps = _int(args, "steps", 120) or 120,
            batch_size = _int(args, "batch_size", 1) or 1,
            sequence_length = _int(args, "sequence_length", 2048) or 2048,
            learning_rate = _float(args, "learning_rate", 5e-5) or 5e-5,
        )
        return _result(_call_route("labs repair preset plan", routes.plan_preset, request))
    if action == "pipeline_list":
        return _result(_call_route("labs pipeline list", routes.list_pipelines))
    if action == "pipeline_plan":
        return _result(
            _call_route(
                "labs pipeline plan",
                routes.plan_pipeline,
                routes.PipelinePlanRequest(spec = _spec(args)),
            )
        )
    if action == "pipeline_save":
        return _result(
            _call_route(
                "labs pipeline save",
                routes.save_pipeline,
                routes.PipelineSaveRequest(spec = _spec(args)),
            )
        )
    if action == "pipeline_delete":
        pipeline_id = str(args.get("pipeline_id") or args.get("record_id") or "").strip()
        return _result(
            _call_route("labs pipeline delete", routes.delete_pipeline, pipeline_id)
        )
    if action == "benchmark_list":
        return _result(
            _call_route(
                "labs benchmark list",
                routes.list_benchmarks,
                limit = _int(args, "limit", 50) or 50,
                artifact = args.get("artifact") or None,
            )
        )
    if action == "benchmark_run":
        request = routes.BenchmarkRunRequest(
            artifact = str(args.get("artifact") or "").strip(),
            baseline = str(args.get("baseline") or "").strip(),
            suites = _strings(args, "suites"),
            limit = _int(args, "limit"),
            batch_size = _int(args, "batch_size", 8) or 8,
            timeout_seconds = _int(args, "timeout_seconds", 3600) or 3600,
        )
        return _result(_call_route("labs benchmark run", routes.run_benchmark, request))
    if action == "benchmark_pin":
        record_id = str(args.get("record_id") or "").strip()
        return _result(_call_route("labs benchmark pin", routes.pin_benchmark, record_id))
    if action == "benchmark_unpin":
        record_id = str(args.get("record_id") or "").strip()
        return _result(_call_route("labs benchmark unpin", routes.unpin_benchmark, record_id))
    if action == "benchmark_delete":
        record_id = str(args.get("record_id") or "").strip()
        return _result(_call_route("labs benchmark delete", routes.delete_benchmark, record_id))
    if action == "cmt_plan":
        return _result(
            _call_route("labs CMT plan", routes.plan_cmt, routes.CmtPlanRequest(spec = _spec(args)))
        )
    if action == "cmt_run":
        return _result(
            _call_route(
                "labs CMT run",
                routes.run_cmt,
                routes.CmtRunRequest(spec = _spec(args), output_dir = args.get("output_dir")),
            )
        )
    if action == "auto_scale_plan":
        return _result(
            _call_route(
                "labs auto-scale plan",
                routes.plan_auto_scale,
                routes.AutoScaleRequest(spec = _spec(args)),
            )
        )
    if action == "auto_scale_run":
        return _result(
            _call_route(
                "labs auto-scale run",
                routes.run_auto_scale,
                routes.AutoScaleRequest(spec = _spec(args), output_dir = args.get("output_dir")),
            )
        )
    if action == "surgery_plan":
        return _result(
            _call_route(
                "labs surgery plan",
                routes.plan_surgery,
                routes.SurgeryRequest(spec = _spec(args)),
            )
        )
    if action == "surgery_run":
        return _result(
            _call_route(
                "labs surgery run",
                routes.run_surgery,
                routes.SurgeryRequest(spec = _spec(args), output_dir = args.get("output_dir")),
            )
        )
    if action == "arch_swap_plan":
        return _result(
            _call_route(
                "labs arch-swap plan",
                routes.plan_arch_swap,
                routes.ArchSwapRequest(spec = _spec(args)),
            )
        )
    if action == "arch_swap_run":
        return _result(
            _call_route(
                "labs arch-swap run",
                routes.run_arch_swap,
                routes.ArchSwapRequest(spec = _spec(args), output_dir = args.get("output_dir")),
            )
        )
    if action == "housekeeping_scan":
        return _result(_call_route("labs housekeeping scan", routes.scan_housekeeping))
    if action == "housekeeping_delete":
        request = routes.HousekeepingDeleteRequest(
            paths = _strings(args, "paths"),
            confirm = bool(args.get("confirm")),
        )
        return _result(_call_route("labs housekeeping delete", routes.delete_housekeeping, request))
    if action == "housekeeping_rules":
        return _result(_call_route("labs housekeeping rules", routes.read_janitor_rules))
    if action.startswith("housekeeping_rules_"):
        rules = args.get("rules")
        if not isinstance(rules, Mapping):
            raise StudioAccessError("rules must be an object.")
        request = routes.JanitorRulesUpdate.model_validate(dict(rules))
        function = {
            "housekeeping_rules_update": routes.update_janitor_rules,
            "housekeeping_rules_preview": routes.preview_janitor_rules,
            "housekeeping_rules_apply": routes.apply_janitor_rules,
        }[action]
        return _result(_call_route(f"labs {action}", function, request))
    if action in ("automated_presets", "automated_plan", "automated_run"):
        return _result(_execute_automated_action(routes, action, args))
    if action == "job_status":
        from routes import advanced

        job_id = str(args.get("job_id") or "").strip()
        return _result(_call_route("labs job status", advanced.job_status, job_id))
    if action == "job_list":
        from routes import advanced

        return _result(_call_route("labs job list", advanced.list_jobs))
    raise StudioAccessError(f"Unsupported Labs action {action!r}.")


def _execute_automated_action(routes: Any, action: str, args: Mapping[str, Any]) -> Any:
    if action == "automated_presets":
        return _call_route("automated presets", routes.automated_presets)
    if action == "automated_plan":
        return _call_route(
            "automated plan",
            routes.plan_automated_training,
            routes.AutomatedTrainingPlanRequest(spec = _spec(args)),
        )
    return _call_route(
        "automated run",
        routes.run_automated_training,
        routes.AutomatedTrainingRunRequest(
            spec = _spec(args), output_dir = args.get("output_dir")
        ),
    )


def execute_automated_training_tool(arguments: Mapping[str, Any]) -> str:
    _require_access()
    args = dict(arguments or {})
    action = _action(args, AUTOMATED_ACTIONS)
    if action == "status":
        from routes import advanced

        job_id = str(args.get("job_id") or "").strip()
        return _result(_call_route("automated status", advanced.job_status, job_id))
    from routes import labs as routes

    return _result(
        _execute_automated_action(routes, f"automated_{action}", args)
    )


def _merge_payload(args: Mapping[str, Any]) -> dict[str, Any]:
    payload = _present(
        args,
        (
            "model_a",
            "model_b",
            "models",
            "weights",
            "output_dir",
            "method",
            "weight",
            "density",
            "seed",
            "max_shard_gb",
            "allow_remote_code",
            "allow_non_safetensors",
            "allow_tokenizer_conflict",
            "top_k",
            "router_hidden",
            "temperature",
            "layer_start",
            "layer_end",
        ),
    )
    if "models" in args:
        payload["models"] = _strings(args, "models")
    return payload


def execute_merge_tool(arguments: Mapping[str, Any]) -> str:
    _require_access()
    args = dict(arguments or {})
    action = _action(args, MERGE_ACTIONS)
    from routes import advanced

    if action == "status":
        job_id = str(args.get("job_id") or "").strip()
        return _result(_call_route("merge status", advanced.job_status, job_id))
    if action == "eval":
        request = advanced.MergeEvalRequest(
            source = str(args.get("source") or "").strip(),
            sample_tensors = _int(args, "sample_tensors", 24) or 24,
        )
        return _result(_call_route("merge eval", advanced.eval_merged_model, request))
    if action == "adapters":
        request = advanced.AdapterMergeRequest(
            adapters = _strings(args, "adapters"),
            output_dir = str(args.get("output_dir") or "").strip(),
            method = str(args.get("method") or "linear"),
            weights = args.get("weights"),
            density = _float(args, "density", 0.5) or 0.5,
            seed = _int(args, "seed", 3407) or 3407,
            rank = _int(args, "rank"),
        )
        return _result(_call_route("merge adapters", advanced.merge_lora_adapters, request))
    payload = _merge_payload(args)
    if action == "validate":
        return _result(
            _call_route(
                "merge validate",
                advanced.validate_merge,
                advanced.MergeValidateRequest(**payload),
            )
        )
    if action == "plan":
        return _result(
            _call_route(
                "merge plan",
                advanced.plan_merge,
                advanced.MergePlanRequest(**payload),
            )
        )
    if action == "moe":
        if not payload.get("models"):
            pair = [payload.get("model_a"), payload.get("model_b")]
            if all(pair):
                payload["models"] = pair
        return _result(
            _call_route(
                "merge moe",
                advanced.run_moe,
                advanced.MoeRunRequest(**payload),
            )
        )
    return _result(
        _call_route(
            "merge run",
            advanced.run_merge,
            advanced.MergeRunRequest(**payload),
        )
    )


def execute_training_tool(arguments: Mapping[str, Any]) -> str:
    _require_access()
    args = dict(arguments or {})
    action = _action(args, TRAINING_ACTIONS)
    from utils.account_context import current_account_id
    from routes import training as routes

    subject = current_account_id()
    if action == "status":
        pending_id = str(args.get("pending_id") or "").strip()
        if pending_id:
            pending = _pending_record(pending_id)
            if pending is None:
                raise StudioAccessError(f"Unknown pending training id {pending_id!r}.")
            return _result(_pending_status(pending))
        return _result(
            _call_route(
                "training status",
                routes.get_training_status,
                current_subject = subject,
                via_api_key = False,
            )
        )
    if action == "metrics":
        return _result(
            _call_route(
                "training metrics",
                routes.get_training_metrics,
                expected_job_id = args.get("job_id") or None,
                current_subject = subject,
            )
        )
    if action == "hardware":
        return _result(
            _call_route(
                "training hardware",
                routes.get_hardware_utilization,
                current_subject = subject,
            )
        )
    if action == "offload":
        return _result(
            _call_route(
                "training offload",
                routes.get_offload_state,
                current_subject = subject,
            )
        )
    if action == "start":
        from models.training import TrainingStartRequest

        if str(args.get("kind") or "llm").lower() != "llm":
            raise StudioAccessError(
                "action=start supports LLM training only. Use diffusion_status or the diffusion "
                "run actions for image/video training."
            )
        config = args.get("config")
        if not isinstance(config, Mapping):
            raise StudioAccessError("config must be a TrainingStartRequest object.")
        config = dict(config)
        for secret in ("hf_token", "wandb_token", "s3_config"):
            config.pop(secret, None)
        try:
            validated = TrainingStartRequest(**config)
        except Exception as error:  # noqa: BLE001 - pydantic's message names the bad field
            raise StudioAccessError(f"Invalid training config: {error}") from error
        clean_config = validated.model_dump(mode = "json")
        for secret in ("hf_token", "wandb_token", "s3_config"):
            clean_config.pop(secret, None)
        if _deferred_inference_active():
            pending_id, _record = _defer_start(
                {"method": "custom", "training_request": clean_config}
            )
            return _result(
                {
                    "status": "queued",
                    "kind": "training",
                    "pending_id": pending_id,
                    "job_id": None,
                    "message": (
                        "Training is queued and starts when this chat turn finishes, because "
                        "training has to unload the chat model."
                    ),
                }
            )
        return _result(_call_route("training start", _start_training, clean_config))
    if action == "stop":
        expected = str(args.get("expected_job_id") or args.get("job_id") or "").strip()
        if not expected:
            from core.training.training import get_training_backend

            expected = str(get_training_backend().current_job_id or "")
        if not expected:
            raise StudioAccessError("No training job is running to stop.")
        request = routes.TrainingStopRequest(
            save = args.get("save") is not False,
            expected_job_id = expected,
        )
        return _result(
            _call_route("training stop", routes.stop_training, request, current_subject = subject)
        )
    if action == "reset":
        request = None
        expected = str(args.get("expected_job_id") or "").strip()
        if expected:
            request = routes.TrainingResetRequest(expected_job_id = expected)
        return _result(
            _call_route("training reset", routes.reset_training, request, current_subject = subject)
        )
    if action in ("runs", "run_detail", "run_update", "run_delete", "resume_check"):
        from routes import training_history as history

        if action == "runs":
            return _result(
                _call_route(
                    "training runs",
                    history.list_training_runs,
                    limit = _int(args, "limit", 50) or 50,
                    offset = _int(args, "offset", 0) or 0,
                    q = args.get("query") or None,
                    tag = args.get("tag") or None,
                    current_subject = subject,
                    no_credential = False,
                    via_api_key = False,
                )
            )
        run_id = str(args.get("run_id") or "").strip()
        if action == "run_update":
            from models.training import TrainingRunUpdateRequest

            fields = _present(args, ("display_name", "tags", "notes"))
            if not fields:
                raise StudioAccessError("run_update needs display_name, tags or notes.")
            payload = TrainingRunUpdateRequest(**fields)
            return _result(
                _call_route(
                    "training run update",
                    history.update_training_run,
                    run_id,
                    payload,
                    current_subject = subject,
                    no_credential = False,
                    via_api_key = False,
                )
            )
        if action == "run_delete":
            if args.get("confirm") is not True:
                raise StudioAccessError("run_delete requires confirm=true.")
            return _result(
                _call_route(
                    "training run delete",
                    history.delete_training_run,
                    run_id,
                    delete_artifacts = bool(args.get("delete_artifacts")),
                    current_subject = subject,
                )
            )
        function = (
            history.get_training_run_detail
            if action == "run_detail"
            else history.check_training_run_resume
        )
        return _result(
            _call_route(
                f"training {action}",
                function,
                run_id,
                current_subject = subject,
                via_api_key = False,
                **({"no_credential": False} if action == "run_detail" else {}),
            )
        )
    if action == "diffusion_status":
        return _result(
            _call_route(
                "diffusion training status",
                routes.diffusion_training_status,
                current_subject = subject,
            )
        )
    if action == "diffusion_runs":
        return _result(
            _call_route(
                "diffusion training runs",
                routes.list_diffusion_training_runs,
                limit = _int(args, "limit", 20) or 20,
                current_subject = subject,
            )
        )
    if action == "diffusion_run":
        job_id = str(args.get("run_id") or args.get("job_id") or "").strip()
        return _result(
            _call_route(
                "diffusion training run",
                routes.get_diffusion_training_run,
                job_id,
                current_subject = subject,
            )
        )
    if action == "stop_diffusion":
        request = None
        if "save" in args:
            request = routes.DiffusionTrainingStopRequest(save = args.get("save") is not False)
        return _result(
            _call_route(
                "diffusion training stop",
                routes.stop_diffusion_training,
                request,
                current_subject = subject,
            )
        )
    if action == "dataset_preflight":
        request = routes.DatasetPreflightRequest(
            path = str(args.get("path") or "").strip(),
            max_rows = _int(args, "max_rows", 200_000) or 200_000,
            long_row_chars = _int(args, "long_row_chars", 32_768) or 32_768,
        )
        return _result(
            _call_route("training dataset preflight", routes.dataset_preflight, request, current_subject = subject)
        )
    if action == "disk_preflight":
        payload = _present(
            args,
            (
                "model",
                "method",
                "load_in_4bit",
                "max_seq_length",
                "batch_size",
                "gradient_accumulation",
                "lora_rank",
                "max_steps",
                "save_steps",
                "model_known_cached",
            ),
        )
        request = routes.TrainingDiskPreflightRequest(**payload)
        return _result(
            _call_route(
                "training disk preflight",
                routes.training_disk_preflight,
                request,
                current_subject = subject,
                via_api_key = False,
            )
        )
    if action == "start_request":
        request_id = str(args.get("start_request_id") or args.get("job_id") or "").strip()
        return _result(
            _call_route(
                "training start request",
                routes.get_training_start_request,
                start_request_id = request_id,
                current_subject = subject,
            )
        )
    raise StudioAccessError(f"Unsupported Training action {action!r}.")


TOOL_EXECUTORS = {
    "labs": execute_labs_tool,
    "automated_training": execute_automated_training_tool,
    "merge_models": execute_merge_tool,
    "training": execute_training_tool,
}


def execute_studio_access_tool(name: str, arguments: Mapping[str, Any]) -> str:
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        raise StudioAccessError(f"Unknown Studio tool {name!r}.")
    return executor(arguments)
