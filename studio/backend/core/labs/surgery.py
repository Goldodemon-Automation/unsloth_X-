# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Model surgery: prune a dense checkpoint, then upcycle its FFN into experts.

Two operations, one job, because the second is only worth running on the first:

* **Prune** drops whole transformer blocks, whole KV groups (with the query
  heads that share them) and FFN channels. Every choice is made from the stored
  weights, never from a name: a head is kept when its query *and* output column
  carry magnitude, a channel when its gate, up and down slices all do.
* **MoE-ify** copies the (pruned) dense FFN into ``experts`` routed experts and
  adds a zero-initialised router, which is the upcycling recipe every MoE paper
  starts from. With ``top_k = 1`` and a zero gate the converted model computes
  exactly what the dense model computed -- softmax over equal logits gives every
  expert ``1/1`` and expert 0 *is* the dense FFN -- so the checkpoint starts
  numerically unchanged and only training teaches the router to specialise.

The pruning is deliberately conservative where the architecture leaves no room:

* Heads are removed one **KV group** at a time. Qwen/Llama group several query
  heads onto one K/V head and ``repeat_kv`` divides the two, so dropping half a
  group would leave a checkpoint no engine can load. Asking for 2 heads when a
  group holds 4 therefore removes 4, and the plan says so instead of silently
  disagreeing with the UI.
* Dropping middle blocks renumbers the survivors. The output's layer 0 is the
  source's first surviving layer, which is what the config it writes describes.

Everything reads and writes safetensors headers/tensors through
``core.advanced.tensor_io``, so bf16 survives and nothing here needs a GPU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from core.advanced import merge, policy as policy_module, run_manifest, tensor_io
from core.labs import cmt

MIN_FFN_KEEP = 0.25
MAX_EXPERTS = 16
DEFAULT_SHARD_BYTES = 2 * 1024 * 1024 * 1024

#: Dense families whose FFN has a routed-expert twin that llama.cpp, vLLM and
#: transformers can all load. Anything else is refused by name rather than
#: written into a checkpoint no engine would open.
MOE_TARGETS: dict[str, str] = {
    "llama": "mixtral",
    "mistral": "mixtral",
    "qwen2": "qwen2_moe",
}

_ARCHITECTURES = {
    "mixtral": "MixtralForCausalLM",
    "qwen2_moe": "Qwen2MoeForCausalLM",
}

_LAYER_RE = re.compile(r"^(?P<head>.+?\.layers\.)(?P<index>\d+)(?P<tail>\..+)$")


# ---------------------------------------------------------------------------
# Reading a checkpoint
# ---------------------------------------------------------------------------


def _weight_files(directory: Path) -> list[Path]:
    """Every safetensors shard of a local checkpoint."""
    return merge._weight_files(_checkpoint_dir(directory))


def _checkpoint_dir(directory: Path) -> Path:
    """The folder that actually holds config.json and the shards.

    A Hugging Face cache copy nests the revision under ``snapshots/<hash>``, and
    ``merge._snapshot_dirs`` is the one place that unwrapping is written down.
    """
    candidates = merge._snapshot_dirs(directory)
    return candidates[0] if candidates else directory


def _close_all(opened: Sequence[Any]) -> None:
    """Release every reader :func:`_open` handed out; never raises."""
    for handle in opened:
        closer = getattr(handle, "__exit__", None)
        if callable(closer):
            try:
                closer(None, None, None)
            except Exception:  # noqa: BLE001 - closing a reader must not fail a plan
                pass


def _open(directory: Path) -> tuple[dict[str, Any], dict[str, tuple[int, ...]], list[Any]]:
    """(key -> handle, key -> shape, open handles) for one checkpoint.

    Only headers are touched here: ``safe_open`` and ``get_slice`` read the
    offset table, so planning a 70B checkpoint costs a few kilobytes of reads.
    """
    handles: dict[str, Any] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    opened: list[Any] = []
    try:
        for path in _weight_files(directory):
            handle = tensor_io.open_file(path)
            opened.append(handle)
            for key in handle.keys():
                handles[key] = handle
                shapes[key] = tensor_io.shape_of(handle, key)
        if not shapes:
            raise ValueError(f"No safetensors weights found in {str(directory)!r}.")
    except BaseException:
        # The caller never receives *opened* when this raises, so the shards
        # already read must be released here or they leak for the process life.
        _close_all(opened)
        raise
    return handles, shapes, opened


def _read(handles: Mapping[str, Any], key: str):
    return tensor_io.load_tensor(handles[key], key)


def _layer_indices(keys: Sequence[str]) -> list[int]:
    found: set[int] = set()
    for key in keys:
        match = _LAYER_RE.match(key)
        if match:
            found.add(int(match.group("index")))
    return sorted(found)


def _rolename(key: str) -> str:
    """The part after a layer index, so one rule covers every layer."""
    match = _LAYER_RE.match(key)
    return match.group("tail").lstrip(".") if match else key


def _is_llama_layout(shapes: Mapping[str, tuple[int, ...]]) -> bool:
    return any(_rolename(key) == "self_attn.q_proj.weight" for key in shapes)


# ---------------------------------------------------------------------------
# Tensor helpers (torch or numpy, whichever the checkpoint was opened with)
# ---------------------------------------------------------------------------


def _select(value, axis: int, indices: Sequence[int]):
    torch = tensor_io.torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        index = torch.tensor(list(indices), dtype = torch.long, device = value.device)
        return value.index_select(axis, index).contiguous()
    import numpy as np  # noqa: PLC0415

    return np.take(value, np.asarray(list(indices), dtype = np.int64), axis = axis)


def _copy(value):
    """A tensor that owns its storage, so two experts never share one file range."""
    torch = tensor_io.torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    return value.copy()


def _zeros(shape: Sequence[int], like):
    torch = tensor_io.torch_module()
    if torch is not None and isinstance(like, torch.Tensor):
        return torch.zeros(tuple(int(dim) for dim in shape), dtype = like.dtype)
    import numpy as np  # noqa: PLC0415

    return np.zeros(tuple(int(dim) for dim in shape), dtype = like.dtype)


def _axis_norms(value, axis: int) -> list[float]:
    """L2 norm of every slice along *axis*, over the other axes."""
    import numpy as np  # noqa: PLC0415

    array = tensor_io.to_float32(value)
    axes = tuple(index for index in range(array.ndim) if index != axis)
    return [float(norm) for norm in np.sqrt((array * array).sum(axis = axes))]


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


def _int(spec: Mapping[str, Any], key: str, default: int, *, minimum: int = 0, maximum: int = 10**9) -> int:
    value = spec.get(key)
    if value in (None, ""):
        return default
    try:
        parsed = int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a whole number.") from error
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    return parsed


def _ratio(spec: Mapping[str, Any], key: str, default: float) -> float:
    value = spec.get(key)
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be a number.") from error
    if parsed <= 0 or parsed > 1:
        raise ValueError(f"{key} must be greater than 0 and at most 1.")
    return parsed


def _source(spec: Mapping[str, Any]) -> Path:
    raw = str(spec.get("model") or "").strip()
    if not raw:
        raise ValueError("A checkpoint is required: a local folder with safetensors weights.")
    path = Path(raw).expanduser()
    if not path.is_dir():
        if merge.hf_repo_id(raw) is not None:
            raise ValueError(
                f"{raw!r} is a Hugging Face id. Surgery edits weights on disk, so download "
                "the checkpoint first and point at the local folder."
            )
        raise ValueError(f"No checkpoint folder at {raw!r}.")
    return path


def _dims(config: Mapping[str, Any], shapes: Mapping[str, tuple[int, ...]], layers: Sequence[int]) -> dict[str, Any]:
    """The dimensions the plan needs, read from config.json with shape fallbacks."""
    hidden = merge._first_int(config, merge._HIDDEN_KEYS)
    heads = merge._first_int(config, merge._HEAD_KEYS)
    kv_heads = merge._first_int(config, ("num_key_value_heads", "num_kv_heads")) or heads
    ffn = merge._first_int(config, merge._FFN_KEYS)
    head_dim = merge._first_int(config, ("head_dim",))

    layer = layers[0] if layers else None
    if layer is not None:
        q = shapes.get(f"model.layers.{layer}.self_attn.q_proj.weight")
        if q is not None and heads:
            head_dim = head_dim or max(1, int(q[0]) // int(heads))
            hidden = hidden or int(q[1])
        k = shapes.get(f"model.layers.{layer}.self_attn.k_proj.weight")
        if k is not None and kv_heads is None and head_dim:
            kv_heads = max(1, int(k[0]) // int(head_dim))
        gate = shapes.get(f"model.layers.{layer}.mlp.gate_proj.weight")
        if gate is not None:
            ffn = ffn or int(gate[0])
            hidden = hidden or int(gate[1])
    return {
        "hidden_size": hidden,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "ffn_width": ffn,
    }


def _layer_order(count: int, drop: int, strategy: str) -> tuple[list[int], list[int]]:
    """(kept indices in order, dropped indices) for one layer-pruning request."""
    if drop <= 0:
        return list(range(count)), []
    keep = max(1, count - drop)
    dropped = count - keep
    if dropped <= 0:
        return list(range(count)), []
    if strategy == "uniform":
        # Spread the loss evenly instead of guillotining the tail: drop the block
        # nearest (k + 0.5) * count / dropped for each k, never the first one.
        chosen: list[int] = []
        for step in range(dropped):
            candidate = int(round((step + 0.5) * count / dropped))
            candidate = min(max(1, candidate), count - 1)
            for _step in range(count):
                if candidate not in chosen:
                    break
                candidate = min(count - 1, candidate + 1)
            if candidate in chosen:
                candidate = next(index for index in range(1, count) if index not in chosen)
            chosen.append(candidate)
        return [index for index in range(count) if index not in chosen], sorted(chosen)
    return list(range(keep)), list(range(keep, count))


def _moe_target(architecture: str) -> Optional[str]:
    return MOE_TARGETS.get(str(architecture or "").lower())


# ---------------------------------------------------------------------------
# Planning (headers only, no tensor is read into memory)
# ---------------------------------------------------------------------------


def _projected_params(
    shapes: Mapping[str, tuple[int, ...]],
    handles: Mapping[str, Any],
    *,
    kept_layers: Sequence[int],
    heads_after: int,
    kv_after: int,
    ffn_after: int,
    head_dim: int,
    experts: int,
) -> tuple[int, int]:
    """(params, bytes) the surgery would write, from the header alone."""
    layer_map = {old: new for new, old in enumerate(kept_layers)}
    total_params = 0
    total_bytes = 0
    for key, shape in shapes.items():
        match = _LAYER_RE.match(key)
        target_roles: list[tuple[int, ...]] = []
        if match is None:
            target_roles.append(tuple(shape))
        else:
            index = int(match.group("index"))
            if index not in layer_map:
                continue
            role = match.group("tail").lstrip(".")
            target_roles.extend(_role_shapes(role, shape, heads_after, kv_after, ffn_after, head_dim, experts))
        if not target_roles:
            continue
        itemsize = tensor_io.dtype_itemsize(handles[key], key)
        for target in target_roles:
            count = 1
            for dim in target:
                count *= int(dim)
            total_params += count
            total_bytes += count * itemsize
    return total_params, total_bytes


def _role_shapes(
    role: str,
    shape: Sequence[int],
    heads: int,
    kv_heads: int,
    ffn: int,
    head_dim: int,
    experts: int,
) -> list[tuple[int, ...]]:
    """The shapes one source tensor becomes, per role.

    Shared by the plan (which only has headers) and by :func:`run`, so a plan
    cannot promise a shape the writer does not produce.
    """
    rows, cols = int(shape[0]), int(shape[-1])
    if role in ("self_attn.q_proj.weight", "self_attn.o_proj.weight"):
        width = heads * head_dim
        return [(width, cols) if role.endswith("q_proj.weight") else (rows, width)]
    if role in ("self_attn.k_proj.weight", "self_attn.v_proj.weight"):
        return [(kv_heads * head_dim, cols)]
    if role in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
        # gate/up are [ffn_width, hidden]; the FFN width is the axis a prune keeps.
        return [(ffn, cols)] * max(1, experts)
    if role == "mlp.down_proj.weight":
        # down is [hidden, ffn_width], so its FFN axis is the trailing one.
        return [(rows, ffn)] * max(1, experts)
    return [tuple(int(dim) for dim in shape)]


def _moe_shape_roles(role: str) -> bool:
    return role in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def _router_key(index: int, target: str, hidden: int) -> str:
    if target == "mixtral":
        return f"model.layers.{index}.block_sparse_moe.gate.weight"
    return f"model.layers.{index}.mlp.gate.weight"


def _expert_key(index: int, expert: int, role: str, target: str) -> str:
    if target == "mixtral":
        leaf = {"gate": "w1", "up": "w3", "down": "w2"}[role]
        return f"model.layers.{index}.block_sparse_moe.experts.{expert}.{leaf}.weight"
    leaf = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}[role]
    return f"model.layers.{index}.mlp.experts.{expert}.{leaf}.weight"


def plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Size the surgery from config.json and the safetensors headers.

    Nothing is loaded, nothing is written: the caller can price a prune before
    deciding to run it.
    """
    directory = _source(spec)
    label, config = merge.read_model_config(str(directory))
    handles, shapes, opened = _open(directory)
    try:
        if not _is_llama_layout(shapes):
            raise ValueError(
                f"{label!r} does not have the layer layout this surgery understands "
                "(model.layers.N.self_attn.q_proj.weight). Llama, Mistral and Qwen2 "
                "checkpoints have it."
            )
        indices = _layer_indices(shapes)
        if not indices:
            raise ValueError(f"{label!r} has no transformer blocks to edit.")
        dims = _dims(config, shapes, indices)
        architecture = str(config.get("model_type") or "").lower()
        layer_count = len(indices)

        drop_layers = _int(spec, "drop_layers", 0, maximum = max(0, layer_count - 1))
        strategy = str(spec.get("layer_strategy") or "last").strip().lower()
        if strategy not in ("last", "uniform"):
            raise ValueError("layer_strategy must be 'last' or 'uniform'.")
        kept_layers, dropped_layers = _layer_order(layer_count, drop_layers, strategy)

        heads = int(dims["num_heads"] or 0)
        kv_heads = int(dims["num_kv_heads"] or heads)
        head_dim = int(dims["head_dim"] or 0)
        ffn_width = int(dims["ffn_width"] or 0)
        if not heads or not head_dim or not ffn_width:
            raise ValueError(
                f"{label!r} does not record its head count, head width or FFN width, so the "
                "plan cannot say what a prune would remove."
            )
        if kv_heads <= 0 or heads % kv_heads:
            raise ValueError(
                f"{label!r} has {heads} query heads over {kv_heads} KV heads, which does not "
                "divide into groups. Pruning it by head would make a checkpoint no engine can load."
            )
        group = heads // kv_heads
        requested_heads = _int(spec, "drop_heads", 0, maximum = max(0, heads - group))
        # Whole groups only, rounded up: half a group is a checkpoint no engine loads.
        groups_dropped = (
            min(kv_heads - 1, -(-requested_heads // group)) if requested_heads else 0
        )
        heads_after = heads - groups_dropped * group
        kv_after = kv_heads - groups_dropped

        ffn_keep = _ratio(spec, "ffn_keep", 1.0)
        if ffn_keep < 1.0 and ffn_keep < MIN_FFN_KEEP:
            raise ValueError(
                f"ffn_keep must be at least {MIN_FFN_KEEP:.0%}; below that the MLP has no "
                "capacity left to train back."
            )
        ffn_after = max(1, int(round(ffn_width * ffn_keep))) if ffn_keep < 1.0 else ffn_width

        experts = _int(spec, "experts", 0, maximum = MAX_EXPERTS)
        top_k = _int(spec, "top_k", 1, minimum = 1, maximum = max(1, experts or 1))
        target = _moe_target(architecture) if experts else None

        warnings: list[str] = []
        blocked_reason: Optional[str] = None
        if experts and target is None:
            blocked_reason = (
                f"{label!r} is {architecture or 'an unnamed architecture'}. MoE-ification "
                "needs a routed-expert twin that llama.cpp and vLLM both load: "
                + ", ".join(f"{name} → {twin}" for name, twin in sorted(MOE_TARGETS.items()))
                + "."
            )
        if experts and top_k > experts:
            raise ValueError("top_k cannot be larger than the expert count.")
        if experts == 1:
            blocked_reason = blocked_reason or "One expert is a dense model with a router on top. Ask for 2 or more."

        base_params, base_bytes = _projected_params(
            shapes,
            handles,
            kept_layers = indices,
            heads_after = heads,
            kv_after = kv_heads,
            ffn_after = ffn_width,
            head_dim = head_dim,
            experts = 1,
        )
        after_params, after_bytes = _projected_params(
            shapes,
            handles,
            kept_layers = kept_layers,
            heads_after = heads_after,
            kv_after = kv_after,
            ffn_after = ffn_after,
            head_dim = head_dim,
            experts = experts or 1,
        )
    finally:
        _close_all(opened)

    if dropped_layers:
        warnings.append(
            f"Dropping {len(dropped_layers)} block(s) ({strategy}) leaves {len(kept_layers)}. "
            "Survivors are renumbered from 0, and a middle block's removal costs more quality "
            "than the count suggests: train the result before trusting it."
        )
    if groups_dropped:
        warnings.append(
            f"{group} query head(s) share each KV head, so asking for {requested_heads} removed "
            f"{groups_dropped * group} query heads and {groups_dropped} KV head(s): "
            f"{heads} → {heads_after} query, {kv_heads} → {kv_after} KV."
        )
    if ffn_after < ffn_width:
        warnings.append(
            f"FFN width {ffn_width} → {ffn_after}. Channels are chosen by gate·up·down magnitude, "
            "which is a weight-only proxy: fine-tuning recovers most, not all, of the loss."
        )
    if experts:
        warnings.extend(
            [
                f"Upcycling writes {experts} copies of every FFN tensor, so the FFN share of the "
                f"checkpoint is written {experts} times and the result is "
                f"{cmt._format_params(after_params)} parameters "
                f"({report_ratio(after_params, base_params)} of the source).",
                "The router starts at zero, so with top_k=1 the converted model computes exactly "
                "what the dense model computed. Until it is trained every expert is a copy and "
                "the router is a no-op.",
            ]
        )

    return {
        "model": label,
        "architecture": architecture or "unknown",
        "moe_architecture": target,
        "runnable": blocked_reason is None,
        "blocked_reason": blocked_reason,
        "base": {
            "layers": layer_count,
            "layer_indices": indices,
            "hidden_size": int(dims["hidden_size"] or 0),
            "num_heads": heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "ffn_width": ffn_width,
            "params": base_params,
            "params_label": cmt._format_params(base_params),
            "bytes": base_bytes,
            "bytes_label": _format_bytes(base_bytes),
        },
        "prune": {
            "layers": {
                "drop": len(dropped_layers),
                "keep": len(kept_layers),
                "strategy": strategy,
                "dropped_indices": dropped_layers,
            },
            "heads": {
                "requested": requested_heads,
                "removed": groups_dropped * group,
                "keep": heads_after,
                "kv_keep": kv_after,
                "group_size": group,
            },
            "ffn": {
                "keep_ratio": round(ffn_after / ffn_width, 4) if ffn_width else 1.0,
                "keep": ffn_after,
                "drop": max(0, ffn_width - ffn_after),
            },
        },
        "moe": (
            {
                "experts": experts,
                "top_k": top_k,
                "architecture": target,
                "router": "zeros",
                "ffn_width": ffn_after,
            }
            if experts
            else None
        ),
        "projection": {
            "params": after_params,
            "params_label": cmt._format_params(after_params),
            "bytes": after_bytes,
            "bytes_label": _format_bytes(after_bytes),
            "ratio": round(after_params / base_params, 4) if base_params else 1.0,
        },
        "warnings": warnings,
        "notes": [
            "Heads and channels are chosen from the stored weights, not from activations: a "
            "data-free prune, which is why the result is a starting point for training.",
            "The tokenizer is copied unchanged; the vocabulary is never touched.",
        ],
    }


def report_ratio(after: int, before: int) -> str:
    """``1.42x`` for a plan that grew, ``0.71x`` for one that shrank."""
    if not before:
        return "1.00x"
    return f"{after / before:.2f}x"


def _format_bytes(value: float) -> str:
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PiB"


# ---------------------------------------------------------------------------
# Running the surgery
# ---------------------------------------------------------------------------


def _head_selection(
    q,
    o,
    *,
    heads: int,
    group: int,
    groups_dropped: int,
    head_dim: int,
) -> tuple[list[int], list[int]]:
    """(rows of q_proj/o_proj to keep, rows of k_proj/v_proj to keep).

    A head's importance is the magnitude of its query rows *and* of its output
    columns: a head that reads strongly but writes nothing is as droppable as the
    reverse. Rows, not head numbers, because the caller slices storage.
    """
    q_norm = _axis_norms(q, 0)
    o_norm = _axis_norms(o, 1)
    kv_heads = max(1, heads // group)
    score: list[float] = []
    for head in range(heads):
        start = head * head_dim
        stop = start + head_dim
        score.append(
            sum(q_norm[start:stop]) + sum(o_norm[start:stop])
        )
    group_scores = [sum(score[start : start + group]) for start in range(0, heads, group)]
    order = sorted(range(kv_heads), key = lambda index: group_scores[index])
    dropped = set(order[:groups_dropped])
    keep_heads = [head for head in range(heads) if head // group not in dropped]
    keep_kv = [index for index in range(kv_heads) if index not in dropped]
    keep_q_rows = [
        row for head in keep_heads for row in range(head * head_dim, (head + 1) * head_dim)
    ]
    keep_kv_rows = [
        row for index in keep_kv for row in range(index * head_dim, (index + 1) * head_dim)
    ]
    return keep_q_rows, keep_kv_rows


def _channel_selection(gate, up, down, *, width: int, keep: int) -> list[int]:
    """``keep`` kept FFN channels, chosen by the product of the three slices' norms."""
    gate_norm = _axis_norms(gate, 0)
    up_norm = _axis_norms(up, 0)
    down_norm = _axis_norms(down, 1)
    score = [
        gate_norm[index] * up_norm[index] * down_norm[index] for index in range(width)
    ]
    order = sorted(range(width), key = lambda index: score[index], reverse = True)
    return sorted(order[: max(1, keep)])


def run(
    spec: Mapping[str, Any],
    *,
    output_dir: str,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Prune and (optionally) upcycle one checkpoint into *output_dir*."""
    def stage(name: str, message: str, **extra: Any) -> None:
        if progress is not None:
            try:
                progress({"stage": name, "message": message, **extra})
            except Exception:  # noqa: BLE001 - progress must never fail the job
                pass

    report = plan(spec)
    if not report["runnable"]:
        raise ValueError(report.get("blocked_reason") or "This surgery plan cannot run.")

    directory = _checkpoint_dir(_source(spec))
    config = json.loads((directory / "config.json").read_text(encoding = "utf-8"))
    out = Path(output_dir).expanduser()
    out.mkdir(parents = True, exist_ok = True)
    checkpoint = out / "checkpoint"
    checkpoint.mkdir(parents = True, exist_ok = True)

    base = report["base"]
    prune = report["prune"]
    moe = report["moe"]
    dropped = set(prune["layers"]["dropped_indices"])
    kept_layers = [index for index in base["layer_indices"] if index not in dropped]
    layer_map = {old: new for new, old in enumerate(kept_layers)}
    group = prune["heads"]["group_size"]
    groups_dropped = (
        (base["num_heads"] - prune["heads"]["keep"]) // group if group else 0
    )
    ffn_after = prune["ffn"]["keep"]
    target = moe["architecture"] if moe else None
    experts = int(moe["experts"]) if moe else 0

    handles, shapes, opened = _open(directory)
    shard: dict[str, Any] = {}
    shard_bytes = 0
    shard_index = 1
    weight_map: dict[str, str] = {}
    budget = DEFAULT_SHARD_BYTES

    def flush() -> None:
        nonlocal shard, shard_bytes, shard_index
        if not shard:
            return
        name = f"model-{shard_index:05d}.safetensors"
        policy_module.atomic_save_tensors(shard, checkpoint / name)
        for key in shard:
            weight_map[key] = name
        shard = {}
        shard_bytes = 0
        shard_index += 1

    def emit(key: str, value) -> None:
        nonlocal shard, shard_bytes
        owned = _copy(value)
        size = tensor_io.nbytes(owned)
        if shard and shard_bytes + size > budget:
            flush()
        shard[key] = owned
        shard_bytes += size

    selections: dict[int, tuple[list[int], list[int]]] = {}
    channels: dict[int, list[int]] = {}

    def layer_selection(old: int) -> tuple[list[int], list[int]]:
        if old not in selections:
            q = _read(handles, f"model.layers.{old}.self_attn.q_proj.weight")
            o = _read(handles, f"model.layers.{old}.self_attn.o_proj.weight")
            selections[old] = _head_selection(
                q,
                o,
                heads = base["num_heads"],
                group = group,
                groups_dropped = groups_dropped,
                head_dim = base["head_dim"],
            )
            del q, o
        return selections[old]

    def layer_channels(old: int) -> list[int]:
        if old not in channels:
            gate = _read(handles, f"model.layers.{old}.mlp.gate_proj.weight")
            up = _read(handles, f"model.layers.{old}.mlp.up_proj.weight")
            down = _read(handles, f"model.layers.{old}.mlp.down_proj.weight")
            channels[old] = _channel_selection(
                gate, up, down, width = base["ffn_width"], keep = ffn_after
            )
            del gate, up, down
        return channels[old]

    def trim_ffn(role_value, old: int, role: str):
        """Select the kept FFN channels: rows of gate/up, columns of down."""
        if ffn_after == base["ffn_width"]:
            return role_value
        axis = 1 if role == "mlp.down_proj.weight" else 0
        return _select(role_value, axis, layer_channels(old))

    keys = sorted(shapes)
    total = max(1, len(keys))
    written = 0
    try:
        stage("surgery", f"Editing {report['model']}...", done = 0, total = total)
        for position, key in enumerate(keys, start = 1):
            match = _LAYER_RE.match(key)
            if match is None:
                emit(key, _read(handles, key))
                written += 1
                continue
            old = int(match.group("index"))
            if old not in layer_map:
                continue
            new = layer_map[old]
            role = match.group("tail").lstrip(".")
            value = _read(handles, key)
            renamed = f"{match.group('head')}{new}{match.group('tail')}"

            if role == "self_attn.q_proj.weight" and groups_dropped:
                keep_q, _ = layer_selection(old)
                emit(renamed, _select(value, 0, keep_q))
            elif role == "self_attn.o_proj.weight" and groups_dropped:
                keep_q, _ = layer_selection(old)
                emit(renamed, _select(value, 1, keep_q))
            elif role in ("self_attn.k_proj.weight", "self_attn.v_proj.weight") and groups_dropped:
                _, keep_kv = layer_selection(old)
                emit(renamed, _select(value, 0, keep_kv))
            elif _moe_shape_roles(role) and experts:
                trimmed = trim_ffn(value, old, role)
                leaf = {
                    "mlp.gate_proj.weight": "gate",
                    "mlp.up_proj.weight": "up",
                    "mlp.down_proj.weight": "down",
                }[role]
                for expert in range(experts):
                    emit(_expert_key(new, expert, leaf, target), trimmed)
                if leaf == "down":
                    # Zero logits + top_k=1 means expert 0 alone reproduces the dense
                    # FFN, so the upcycled checkpoint starts numerically unchanged.
                    emit(
                        _router_key(new, target, base["hidden_size"]),
                        _zeros((experts, base["hidden_size"]), value),
                    )
                del trimmed
            elif role in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
                emit(renamed, trim_ffn(value, old, role))
            else:
                emit(renamed, value)
            written += 1
            if position % 25 == 0 or position == total:
                stage(
                    "surgery",
                    f"{position}/{total} tensors",
                    done = position,
                    total = total,
                    bytes_written = shard_bytes,
                )
        flush()
    finally:
        _close_all(opened)

    stage("finalizing", "Writing index, config and tokenizer...")
    if shard_index == 2 and weight_map:
        # One shard is written under the plain name every loader looks for first.
        single = checkpoint / "model-00001.safetensors"
        if single.is_file():
            single.replace(checkpoint / "model.safetensors")
            weight_map = {key: "model.safetensors" for key in weight_map}
    index = {
        "metadata": {"total_size": _index_bytes(checkpoint, weight_map)},
        "weight_map": weight_map,
    }
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(index, indent = 2), encoding = "utf-8"
    )
    total_bytes = int(index["metadata"]["total_size"])

    updated = _updated_config(config, report)
    (checkpoint / "config.json").write_text(
        json.dumps(updated, indent = 2), encoding = "utf-8"
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        candidate = directory / name
        if candidate.is_file():
            try:
                (checkpoint / name).write_bytes(candidate.read_bytes())
            except OSError:
                pass

    stage_value = "moe-ification" if experts else "prune"
    manifest = run_manifest.build_manifest(
        stage_value,
        config = {
            "drop_layers": prune["layers"]["drop"],
            "layer_strategy": prune["layers"]["strategy"],
            "drop_heads": prune["heads"]["requested"],
            "heads_removed": prune["heads"]["removed"],
            "ffn_keep": prune["ffn"]["keep_ratio"],
            "experts": experts,
            "top_k": moe["top_k"] if moe else None,
            "moe_architecture": target,
        },
        inputs = {"source": run_manifest.checkpoint_digest(str(directory))},
        artifacts = {
            "output_dir": str(out),
            "checkpoint": str(checkpoint),
            "params": report["projection"]["params"],
            "weight_bytes": total_bytes,
            "written_tensors": written,
        },
    )
    manifest_path = run_manifest.write_manifest(out, manifest)

    stage("completed", "The edited checkpoint is ready to train.", done = 1, total = 1)
    return {
        "output_dir": str(out),
        "checkpoint": str(checkpoint),
        "manifest_path": str(manifest_path),
        "params": report["projection"]["params"],
        "params_label": report["projection"]["params_label"],
        "weight_bytes": total_bytes,
        "written_tensors": written,
        "architecture": target or report["architecture"],
        "warnings": report["warnings"],
        "plan": report,
    }


def _index_bytes(checkpoint: Path, weight_map: Mapping[str, str]) -> int:
    """Total weight bytes, summed from the shards that were just written."""
    total = 0
    for name in set(weight_map.values()):
        path = checkpoint / name
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _updated_config(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """The source config with the layers, heads, width and architecture it now has."""
    updated = json.loads(json.dumps(config))
    prune = report["prune"]
    updated["num_hidden_layers"] = int(prune["layers"]["keep"])
    updated["num_attention_heads"] = int(prune["heads"]["keep"])
    updated["num_key_value_heads"] = int(prune["heads"]["kv_keep"])
    updated["intermediate_size"] = int(prune["ffn"]["keep"])
    moe = report.get("moe")
    if moe:
        target = str(moe["architecture"])
        updated["model_type"] = target
        updated["architectures"] = [_ARCHITECTURES.get(target, "MixtralForCausalLM")]
        updated["num_experts_per_tok"] = int(moe["top_k"])
        updated["router_aux_loss_coef"] = 0.001
        if target == "mixtral":
            updated["num_local_experts"] = int(moe["experts"])
        else:
            updated["num_experts"] = int(moe["experts"])
            updated["moe_intermediate_size"] = int(prune["ffn"]["keep"])
            updated["shared_expert_intermediate_size"] = 0
            updated["decoder_sparse_step"] = 1
            updated.pop("num_local_experts", None)
    return updated
