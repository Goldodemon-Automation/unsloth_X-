# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Arch swap: move weights from one architecture into another by role, not by name.

A hand-written conversion script matches tensor names and dies on the first
shape the two checkpoints disagree about. This module goes through a **role**
instead: every name is identified as ``attn.q`` or ``mlp.gate`` or ``norm.final``
for its family, the source's roles are read once per block, and the target's
names are written from those roles. Fusions the target wants are built and
fusions the source had are split, so ``c_attn`` (GPT-2, one fused Conv1D) and
``q_proj/k_proj/v_proj`` (Llama, three Linears) convert in either direction.

Two things it will not pretend:

* A cross-architecture copy is an **initialization**, not a conversion. MiniGPT's
  ``c_fc → GELU`` is not Llama's ``gate × up → SiLU``; wiring one into the other
  loads and runs, and then needs a fine-tune. The plan says which role changed
  meaning.
* Shapes are fitted by **keeping the leading slice and zero-padding the tail**.
  For a vocabulary that is the only safe rule -- centring rows would renumber
  every embedding -- and for FFN channels it is a plain, documented choice.

Layer counts need not agree either: surplus target blocks repeat the last mapped
block so the checkpoint still loads, and the plan says so instead of leaving keys
missing that a loader would fill with random noise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from core.advanced import merge, policy as policy_module, run_manifest, tensor_io
from core.labs import surgery

DEFAULT_SHARD_BYTES = 2 * 1024 * 1024 * 1024
MAX_MAPPING_ROWS = 60


@dataclass(frozen = True)
class Rule:
    """One name pattern and the role it carries."""

    pattern: str
    role: str
    layer_group: Optional[int] = None
    transpose: bool = False
    fused: Optional[str] = None


#: Name → role per family. ``layer_group`` names the capture group holding the
#: block index, so renumbering is the same operation for every family.
FAMILIES: dict[str, tuple[Rule, ...]] = {
    "llama": (
        Rule(r"^model\.embed_tokens\.(weight)$", "embed"),
        Rule(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$", "q", 1),
        Rule(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$", "k", 1),
        Rule(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$", "v", 1),
        Rule(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$", "o", 1),
        Rule(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$", "gate", 1),
        Rule(r"^model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$", "up", 1),
        Rule(r"^model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$", "down", 1),
        Rule(r"^model\.layers\.(\d+)\.input_layernorm\.(weight|bias)$", "norm_attn", 1),
        Rule(
            r"^model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$",
            "norm_ffn",
            1,
        ),
        Rule(r"^model\.norm\.(weight|bias)$", "norm_final"),
        Rule(r"^lm_head\.(weight|bias)$", "lm_head"),
    ),
    "qwen2": (
        Rule(r"^model\.embed_tokens\.(weight)$", "embed"),
        Rule(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$", "q", 1),
        Rule(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$", "k", 1),
        Rule(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$", "v", 1),
        Rule(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$", "o", 1),
        Rule(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$", "gate", 1),
        Rule(r"^model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$", "up", 1),
        Rule(r"^model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$", "down", 1),
        Rule(r"^model\.layers\.(\d+)\.input_layernorm\.(weight|bias)$", "norm_attn", 1),
        Rule(
            r"^model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$",
            "norm_ffn",
            1,
        ),
        Rule(r"^model\.norm\.(weight|bias)$", "norm_final"),
        Rule(r"^lm_head\.(weight|bias)$", "lm_head"),
    ),
    "gpt2": (
        Rule(r"^transformer\.wte\.(weight)$", "embed"),
        Rule(r"^transformer\.wpe\.(weight)$", "pos"),
        # GPT-2 stores every linear as Conv1D: [in, out], hence the transpose.
        Rule(r"^transformer\.h\.(\d+)\.attn\.c_attn\.(weight|bias)$", "qkv", 1, True, "gpt2"),
        Rule(r"^transformer\.h\.(\d+)\.attn\.c_proj\.(weight|bias)$", "o", 1, True),
        Rule(r"^transformer\.h\.(\d+)\.mlp\.c_fc\.(weight|bias)$", "fc", 1, True),
        Rule(r"^transformer\.h\.(\d+)\.mlp\.c_proj\.(weight|bias)$", "fc_out", 1, True),
        Rule(r"^transformer\.h\.(\d+)\.ln_1\.(weight|bias)$", "norm_attn", 1),
        Rule(r"^transformer\.h\.(\d+)\.ln_2\.(weight|bias)$", "norm_ffn", 1),
        Rule(r"^transformer\.ln_f\.(weight|bias)$", "norm_final"),
        Rule(r"^lm_head\.(weight)$", "lm_head"),
    ),
    "neox": (
        Rule(r"^gpt_neox\.embed_in\.(weight)$", "embed"),
        Rule(
            r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.(weight|bias)$",
            "qkv",
            1,
            False,
            "neox",
        ),
        Rule(r"^gpt_neox\.layers\.(\d+)\.attention\.dense\.(weight|bias)$", "o", 1),
        Rule(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_h_to_4h\.(weight|bias)$", "fc", 1),
        Rule(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_4h_to_h\.(weight|bias)$", "fc_out", 1),
        Rule(
            r"^gpt_neox\.layers\.(\d+)\.input_layernorm\.(weight|bias)$",
            "norm_attn",
            1,
        ),
        Rule(
            r"^gpt_neox\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$",
            "norm_ffn",
            1,
        ),
        Rule(r"^gpt_neox\.final_layer_norm\.(weight|bias)$", "norm_final"),
        Rule(r"^embed_out\.(weight)$", "lm_head"),
    ),
}

#: ``model_type`` values that share a family's layout.
ALIASES = {
    "mistral": "llama",
    "qwen3": "qwen2",
    "phi3": "llama",
    "gpt2": "gpt2",
    "gpt_neox": "neox",
    "pythia": "neox",
}

_ROLE_LABELS = {
    "embed": "Token embedding",
    "pos": "Position embedding",
    "q": "Query projection",
    "k": "Key projection",
    "v": "Value projection",
    "qkv": "Fused QKV projection",
    "o": "Attention output projection",
    "gate": "FFN gate",
    "up": "FFN up",
    "fc": "FFN in",
    "down": "FFN down",
    "fc_out": "FFN out",
    "norm_attn": "Attention norm",
    "norm_ffn": "FFN norm",
    "norm_final": "Final norm",
    "lm_head": "Output head",
}


def family_of(config: Mapping[str, Any]) -> Optional[str]:
    """The layout family a config's ``model_type`` uses, if it is known."""
    model_type = str(config.get("model_type") or "").lower()
    if model_type in FAMILIES:
        return model_type
    return ALIASES.get(model_type)


def identify(family: str, name: str) -> Optional[tuple[Optional[int], str, str, Rule]]:
    """(layer, role, kind, rule) for one tensor name, or None."""
    rules = FAMILIES.get(family)
    if rules is None:
        return None
    for rule in rules:
        match = re.match(rule.pattern, name)
        if match is None:
            continue
        layer = int(match.group(rule.layer_group)) if rule.layer_group else None
        kind = match.group(rule.layer_group + 1) if rule.layer_group else match.group(1)
        return layer, rule.role, kind, rule
    return None


def _inventory(family: str, shapes: Mapping[str, tuple[int, ...]]) -> dict[tuple[Optional[int], str, str], tuple[str, Rule]]:
    """(layer, role, kind) → (tensor name, rule), for one checkpoint."""
    found: dict[tuple[Optional[int], str, str], tuple[str, Rule]] = {}
    for name in shapes:
        identified = identify(family, name)
        if identified is None:
            continue
        layer, role, kind, rule = identified
        found[(layer, role, kind)] = (name, rule)
    return found


def _source(spec: Mapping[str, Any], key: str, label: str) -> Path:
    raw = str(spec.get(key) or "").strip()
    if not raw:
        raise ValueError(f"A {label} checkpoint is required.")
    path = Path(raw).expanduser()
    if not path.is_dir():
        if merge.hf_repo_id(raw) is not None:
            raise ValueError(
                f"{raw!r} is a Hugging Face id. An arch swap reads and writes local weights, "
                "so download the checkpoint first."
            )
        raise ValueError(f"No checkpoint folder at {raw!r}.")
    return surgery._checkpoint_dir(path)


def _layer_count(inventory: Mapping[tuple[Optional[int], str, str], Any]) -> int:
    layers = {layer for layer, _role, _kind in inventory if layer is not None}
    return max(layers) + 1 if layers else 0


def _roles(inventory: Mapping[tuple[Optional[int], str, str], Any], layer: int) -> set[str]:
    return {role for (index, role, _kind) in inventory if index == layer}


def plan(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Work out what each target tensor is filled from. Writes nothing."""
    source = _source(spec, "source", "source")
    target = _source(spec, "target", "target")
    source_label, source_config = merge.read_model_config(str(source))
    target_label, target_config = merge.read_model_config(str(target))
    source_family = family_of(source_config)
    target_family = family_of(target_config)
    blocked: Optional[str] = None
    if source_family is None:
        raise ValueError(
            f"{source_label!r} is {source_config.get('model_type')!r}, which has no name table "
            "here. Known layouts: " + ", ".join(sorted(FAMILIES)) + "."
        )
    if target_family is None:
        raise ValueError(
            f"{target_label!r} is {target_config.get('model_type')!r}, which has no name table "
            "here. Known layouts: " + ", ".join(sorted(FAMILIES)) + "."
        )

    source_handles, source_shapes, source_opened = surgery._open(source)
    target_handles, target_shapes, target_opened = surgery._open(target)
    try:
        source_inventory = _inventory(source_family, source_shapes)
        target_inventory = _inventory(target_family, target_shapes)
        if not source_inventory:
            raise ValueError(f"None of {source_label!r}'s tensors match the {source_family} layout.")
        if not target_inventory:
            raise ValueError(f"None of {target_label!r}'s tensors match the {target_family} layout.")

        source_layers = _layer_count(source_inventory)
        target_layers = _layer_count(target_inventory)
        policy = str(spec.get("extra_layers") or "repeat").strip().lower()
        if policy not in ("repeat", "zeros"):
            raise ValueError("extra_layers must be 'repeat' or 'zeros'.")

        mapping: list[dict[str, Any]] = []
        counts = {"copied": 0, "fitted": 0, "built": 0, "unmapped": 0, "source_only": 0}
        role_summary: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        seen_roles: set[tuple[Optional[int], str]] = set()

        for (layer, role, kind), (name, rule) in sorted(
            target_inventory.items(), key = lambda item: item[1][0]
        ):
            source_layer = (
                min(layer, source_layers - 1) if layer is not None and source_layers else layer
            )
            providers = _providers(source_inventory, source_layer, role, kind)
            if not providers:
                counts["unmapped"] += 1
                action = "unmapped"
                note = "The source has no tensor for this role; the target keeps its own value."
            else:
                action, note = _action_for(role, providers, rule)
                counts["built" if action in ("split", "fuse", "transpose") else "copied"] += 1
            summary = role_summary.setdefault(
                role, {"role": role, "label": _ROLE_LABELS.get(role, role), "tensors": 0, "action": action}
            )
            summary["tensors"] += 1
            if len(mapping) < MAX_MAPPING_ROWS:
                mapping.append(
                    {
                        "target": name,
                        "role": role,
                        "layer": layer,
                        "from": sorted(provider[0] for provider in providers),
                        "action": action,
                        "note": note,
                    }
                )
            seen_roles.add((layer, role))

        for (layer, role, kind), (name, _rule) in sorted(
            source_inventory.items(), key = lambda item: item[1][0]
        ):
            target_layer = min(layer, target_layers - 1) if layer is not None else layer
            if (target_layer, role) not in seen_roles:
                counts["source_only"] += 1

        if counts["unmapped"]:
            warnings.append(
                f"{counts['unmapped']} target tensor(s) have no source role (a position embedding, "
                "for instance, which Llama does not have). Those keep zero, which is the neutral "
                "value for an addition and a no-op for a norm."
            )
        if target_layers > source_layers:
            warnings.append(
                f"The target has {target_layers} blocks and the source {source_layers}. The extra "
                f"{target_layers - source_layers} block(s) "
                + (
                    "repeat the last mapped block, so the checkpoint loads but carries no new "
                    "information: training has to separate them."
                    if policy == "repeat"
                    else "are zero-filled, so those blocks start as a no-op and training has to fill them."
                )
            )
        if target_layers < source_layers:
            warnings.append(
                f"The target has {target_layers} blocks and the source {source_layers}; "
                f"{source_layers - target_layers} source block(s) are not used."
            )

        source_vocab = _dim(source_shapes, source_inventory, "embed", 0)
        target_vocab = _dim(target_shapes, target_inventory, "embed", 0)
        if source_vocab and target_vocab and source_vocab != target_vocab:
            warnings.append(
                f"Vocabulary {source_vocab} → {target_vocab}: rows past {min(source_vocab, target_vocab)} "
                "are zeroed (a larger target) or dropped (a smaller one). A target with more tokens "
                "needs an embedding warm-up before the model is usable."
            )
        source_head_dim = _head_dim(source_config)
        target_head_dim = _head_dim(target_config)
        if source_head_dim and target_head_dim and source_head_dim != target_head_dim:
            warnings.append(
                f"Head width {source_head_dim} → {target_head_dim}. Attention rows are cropped or "
                "zero-padded, so the copied heads are not the source's heads: expect a long fine-tune."
            )
        if (source_family, target_family) in (
            ("llama", "gpt2"),
            ("gpt2", "llama"),
            ("qwen2", "gpt2"),
            ("gpt2", "qwen2"),
        ):
            warnings.append(
                "SwiGLU (gate × up) and GPT-2's single GELU projection are not the same function. "
                "The mapping builds one from the other and the result needs training; it is an "
                "initialization, not a conversion."
            )
        if source_family == target_family:
            warnings.append(
                "Both checkpoints have the same layout. This is a shape fit (crop/pad), which is "
                "what a size change needs -- a plain copy is a file operation, not a swap."
            )
    finally:
        for handle in (*source_opened, *target_opened):
            closer = getattr(handle, "__exit__", None)
            if callable(closer):
                try:
                    closer(None, None, None)
                except Exception:  # noqa: BLE001
                    pass

    return {
        "source": source_label,
        "target": target_label,
        "source_family": source_family,
        "target_family": target_family,
        "source_layers": source_layers,
        "target_layers": target_layers,
        "extra_layers": policy,
        "runnable": blocked is None,
        "blocked_reason": blocked,
        "mapping": mapping,
        "mapping_truncated": len(mapping) >= MAX_MAPPING_ROWS,
        "roles": list(role_summary.values()),
        "counts": counts,
        "warnings": warnings,
        "notes": [
            "Shapes are fitted by keeping the leading slice and zero-padding the tail.",
            "The target's config.json is written, because it is the config that describes the "
            "shapes this swap produces.",
        ],
    }


def _providers(
    inventory: Mapping[tuple[Optional[int], str, str], tuple[str, Rule]],
    layer: Optional[int],
    role: str,
    kind: str,
) -> list[tuple[str, Rule, str]]:
    """The source tensors that can fill one target tensor, as (name, rule, role)."""
    wanted = [role] if role != "qkv" else ["qkv", "q", "k", "v"]
    if role in ("q", "k", "v", "o"):
        wanted.append("qkv")
    if role in ("gate", "up"):
        wanted.append("fc")
    if role == "fc":
        wanted.extend(["gate", "up"])
    if role == "down":
        wanted.append("fc_out")
    if role == "fc_out":
        wanted.append("down")
    # A tied model has no output head of its own, and its embedding *is* the head.
    if role in ("lm_head", "embed"):
        wanted.append("embed" if role == "lm_head" else "lm_head")
    found: list[tuple[str, Rule, str]] = []
    for candidate in wanted:
        entry = inventory.get((layer, candidate, kind))
        if entry is not None and entry[0] not in {name for name, _rule, _role in found}:
            found.append((entry[0], entry[1], candidate))
    return found


def _action_for(role: str, providers: Sequence[tuple[str, Rule, str]], target_rule: Rule) -> tuple[str, str]:
    """The action label and its explanation for one target tensor."""
    source_roles = [provider[2] for provider in providers]
    if role == "qkv" and source_roles[:1] == ["qkv"]:
        if target_rule.fused != providers[0][1].fused:
            return "transpose", "Fused QKV is re-ordered from the source's layout into the target's."
        return "copy", "Fused QKV already has the target's layout."
    if role == "qkv":
        return "fuse", "Fused from the source's separate q/k/v projections."
    if source_roles[:1] == ["qkv"]:
        return "split", "Split out of the source's fused QKV projection."
    if role in ("fc",) and set(source_roles) & {"gate", "up"}:
        return "fuse", "Built from the source's gate and up projections (GELU vs SwiGLU: see warnings)."
    if role in ("gate", "up") and source_roles == ["fc"]:
        return "split", "Split out of the source's single FFN input projection."
    if role in ("down",) and source_roles == ["fc_out"]:
        return "transpose", "Taken from the source's FFN output projection."
    if role == "fc_out" and source_roles == ["down"]:
        return "transpose", "Taken from the source's FFN down projection (GELU vs SwiGLU: see warnings)."
    if target_rule.transpose != providers[0][1].transpose:
        return "transpose", "Stored transposed in the source (Conv1D [in, out])."
    return "copy", "Same role on both sides."


def _dim(
    shapes: Mapping[str, tuple[int, ...]],
    inventory: Mapping[tuple[Optional[int], str, str], tuple[str, Rule]],
    role: str,
    axis: int,
) -> Optional[int]:
    entry = inventory.get((None, role, "weight"))
    if entry is None:
        return None
    shape = shapes.get(entry[0])
    if shape is None or len(shape) <= axis:
        return None
    return int(shape[axis])


def _head_dim(config: Mapping[str, Any]) -> Optional[int]:
    explicit = merge._first_int(config, ("head_dim",))
    if explicit:
        return explicit
    hidden = merge._first_int(config, merge._HIDDEN_KEYS)
    heads = merge._first_int(config, merge._HEAD_KEYS)
    if hidden and heads:
        return max(1, hidden // heads)
    return None


# ---------------------------------------------------------------------------
# Running the swap
# ---------------------------------------------------------------------------


def _fit(value, shape: Sequence[int]):
    """Crop the tail or zero-pad it, so ``value`` has exactly ``shape``."""
    torch = tensor_io.torch_module()
    current = tuple(int(dim) for dim in getattr(value, "shape", ()))
    target = tuple(int(dim) for dim in shape)
    if current == target:
        return value
    if len(current) != len(target):
        # A rank change is not a fit: a 1-D norm cannot be padded into a matrix.
        return surgery._zeros(target, value)
    keep = tuple(min(a, b) for a, b in zip(current, target))
    window = tuple(slice(0, size) for size in keep)
    if torch is not None and isinstance(value, torch.Tensor):
        result = torch.zeros(target, dtype = value.dtype)
        result[window] = value[window]
        return result
    import numpy as np  # noqa: PLC0415

    result = np.zeros(target, dtype = value.dtype)
    result[window] = value[window]
    return result


def _cat(values: Sequence[Any], axis: int):
    torch = tensor_io.torch_module()
    if torch is not None and all(isinstance(value, torch.Tensor) for value in values):
        return torch.cat(list(values), dim = axis)
    import numpy as np  # noqa: PLC0415

    return np.concatenate(list(values), axis = axis)


def _transpose(value):
    """[in, out] ↔ [out, in] for whichever framework the tensor came from.

    ``torch.transpose`` swaps the two dims it is given, but numpy's ``transpose``
    takes an axis *permutation*, so ``transpose(0, 1)`` there is the identity and
    would silently leave every Conv1D weight unturned.
    """
    if getattr(value, "ndim", 2) != 2:
        return value
    torch = tensor_io.torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.transpose(0, 1)
    import numpy as np  # noqa: PLC0415

    return np.swapaxes(value, 0, 1)


def _split_qkv(value, *, pattern: str, heads: int, head_dim: int):
    """(q, k, v) out of a fused projection stored in *pattern* layout.

    ``gpt2`` concatenates the three projections, ``neox`` interleaves them head by
    head. Both are read as rows of ``[3 * heads * head_dim, hidden]``.
    """
    if pattern == "neox" and heads and head_dim:
        import numpy as np  # noqa: PLC0415

        array = tensor_io.to_float32(value)
        rows = heads * 3 * head_dim
        if array.shape[0] < rows:
            raise ValueError("The fused QKV tensor is smaller than its head count claims.")
        blocks = array[:rows].reshape(heads, 3, head_dim, array.shape[1])
        parts = [blocks[:, index, :, :].reshape(heads * head_dim, array.shape[1]) for index in range(3)]
        return tuple(_cast_back(part, value) for part in parts)
    third = value.shape[0] // 3
    return tuple(value[start : start + third] for start in (0, third, 2 * third))


def _cast_back(array, reference):
    torch = tensor_io.torch_module()
    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.from_numpy(array).to(reference.dtype)
    import numpy as np  # noqa: PLC0415

    return np.asarray(array).astype(reference.dtype, copy = False)


def run(
    spec: Mapping[str, Any],
    *,
    output_dir: str,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Map the source's roles onto the target's names and write the result."""
    def stage(name: str, message: str, **extra: Any) -> None:
        if progress is not None:
            try:
                progress({"stage": name, "message": message, **extra})
            except Exception:  # noqa: BLE001 - progress must never fail a job
                pass

    report = plan(spec)
    if not report["runnable"]:
        raise ValueError(report.get("blocked_reason") or "This swap cannot run.")

    source = _source(spec, "source", "source")
    target = _source(spec, "target", "target")
    out = Path(output_dir).expanduser()
    out.mkdir(parents = True, exist_ok = True)
    checkpoint = out / "checkpoint"
    checkpoint.mkdir(parents = True, exist_ok = True)

    _, source_config = merge.read_model_config(str(source))
    _, target_config = merge.read_model_config(str(target))
    source_family = str(report["source_family"])
    target_family = str(report["target_family"])
    policy = str(report["extra_layers"])

    source_handles, source_shapes, source_opened = surgery._open(source)
    target_handles, target_shapes, target_opened = surgery._open(target)
    source_inventory = _inventory(source_family, source_shapes)
    target_inventory = _inventory(target_family, target_shapes)
    source_layers = _layer_count(source_inventory)
    target_layers = _layer_count(target_inventory)
    source_heads = merge._first_int(source_config, merge._HEAD_KEYS) or 0
    source_head_dim = _head_dim(source_config) or 0

    shard: dict[str, Any] = {}
    shard_bytes = 0
    shard_index = 1
    weight_map: dict[str, str] = {}
    budget = DEFAULT_SHARD_BYTES
    zeroed = 0

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
        owned = surgery._copy(value)
        size = tensor_io.nbytes(owned)
        if shard and shard_bytes + size > budget:
            flush()
        shard[key] = owned
        shard_bytes += size

    def source_layer_for(target_layer: Optional[int]) -> Optional[int]:
        if target_layer is None:
            return None
        return min(target_layer, max(0, source_layers - 1))

    def values(layer: Optional[int]) -> dict[tuple[str, str], Any]:
        """role → value for one source block, fusions already split."""
        slot = source_layer_for(layer)
        found: dict[tuple[str, str], Any] = {}
        for (index, role, kind), (name, rule) in source_inventory.items():
            if index != slot:
                continue
            value = tensor_io.load_tensor(source_handles[name], name)
            if rule.transpose:
                value = _transpose(value)
            if role == "qkv" and kind == "weight":
                q, k, v = _split_qkv(
                    value,
                    pattern = rule.fused or "gpt2",
                    heads = source_heads,
                    head_dim = source_head_dim,
                )
                for split_role, part in (("q", q), ("k", k), ("v", v)):
                    found[(split_role, kind)] = part
                found[(role, kind)] = value
                continue
            found[(role, kind)] = value
        return found

    def for_target(role: str, kind: str, values_map: Mapping[tuple[str, str], Any]):
        """The value a target role is built from, or None."""
        if role == "qkv":
            parts = [values_map.get((part, kind)) for part in ("q", "k", "v")]
            if any(part is None for part in parts):
                direct = values_map.get(("qkv", kind))
                return direct
            return _cat(parts, 0)
        if role == "fc" and ("fc", kind) in values_map:
            return values_map[("fc", kind)]
        if role in ("gate", "up") and (role, kind) in values_map:
            return values_map[(role, kind)]
        if role == "fc":
            gate = values_map.get(("gate", kind))
            up = values_map.get(("up", kind))
            if gate is not None and up is not None:
                return _cat([gate, up], 0)
        if role == "down" and ("down", kind) in values_map:
            return values_map[("down", kind)]
        if role == "fc_out":
            direct = values_map.get(("fc_out", kind))
            if direct is not None:
                return direct
            down = values_map.get(("down", kind))
            if down is not None:
                return _transpose(down)
        if role == "down" and ("fc_out", kind) in values_map:
            return _transpose(values_map[("fc_out", kind)])
        if role in ("gate", "up") and ("fc", kind) in values_map:
            fused = values_map[("fc", kind)]
            half = fused.shape[0] // 2
            return fused[:half] if role == "gate" else fused[half : half * 2]
        if role == "lm_head":
            direct = values_map.get(("lm_head", kind))
            return direct if direct is not None else values_map.get(("embed", kind))
        if role == "embed":
            direct = values_map.get(("embed", kind))
            return direct if direct is not None else values_map.get(("lm_head", kind))
        return values_map.get((role, kind))

    keys = sorted(target_shapes)
    total = max(1, len(keys))
    try:
        stage("mapping", f"Swapping {report['source']} into {report['target']}...", done = 0, total = total)
        cache_layer: Optional[int] = -1
        cache: dict[tuple[str, str], Any] = {}
        for position, key in enumerate(keys, start = 1):
            identified = identify(target_family, key)
            if identified is None:
                # A target tensor with no role here keeps its own value: dropping it
                # would leave a key the target's config still declares.
                emit(key, tensor_io.load_tensor(target_handles[key], key))
                continue
            layer, role, kind, rule = identified
            slot = source_layer_for(layer)
            if slot != cache_layer:
                cache = values(slot)
                cache_layer = slot
            past_source = (
                policy == "zeros"
                and layer is not None
                and source_layers > 0
                and layer >= source_layers
            )
            value = None if past_source else for_target(role, kind, cache)
            if value is None:
                reference = tensor_io.load_tensor(target_handles[key], key)
                value = surgery._zeros(target_shapes[key], reference)
                zeroed += 1
            if rule.transpose:
                value = _transpose(value)
            emit(key, _fit(value, target_shapes[key]))
            if position % 25 == 0 or position == total:
                stage("mapping", f"{position}/{total} tensors", done = position, total = total)
        flush()
    finally:
        for handle in (*source_opened, *target_opened):
            closer = getattr(handle, "__exit__", None)
            if callable(closer):
                try:
                    closer(None, None, None)
                except Exception:  # noqa: BLE001
                    pass

    stage("finalizing", "Writing index, config and tokenizer...")
    if shard_index == 2 and weight_map:
        single = checkpoint / "model-00001.safetensors"
        if single.is_file():
            single.replace(checkpoint / "model.safetensors")
            weight_map = {key: "model.safetensors" for key in weight_map}
    index = {
        "metadata": {"total_size": surgery._index_bytes(checkpoint, weight_map)},
        "weight_map": weight_map,
    }
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(index, indent = 2), encoding = "utf-8"
    )
    (checkpoint / "config.json").write_text(
        json.dumps(target_config, indent = 2), encoding = "utf-8"
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        candidate = target / name
        if candidate.is_file():
            try:
                (checkpoint / name).write_bytes(candidate.read_bytes())
            except OSError:
                pass

    manifest = run_manifest.build_manifest(
        "arch-swap",
        config = {
            "source_family": source_family,
            "target_family": target_family,
            "source_layers": source_layers,
            "target_layers": target_layers,
            "extra_layers": policy,
            "zeroed_tensors": zeroed,
        },
        inputs = {
            "source": run_manifest.checkpoint_digest(str(source)),
            "target": run_manifest.checkpoint_digest(str(target)),
        },
        artifacts = {
            "output_dir": str(out),
            "checkpoint": str(checkpoint),
            "tensors": len(weight_map),
            "weight_bytes": int(index["metadata"]["total_size"]),
        },
    )
    manifest_path = run_manifest.write_manifest(out, manifest)

    stage("completed", "The mapped checkpoint is ready to train.", done = 1, total = 1)
    return {
        "output_dir": str(out),
        "checkpoint": str(checkpoint),
        "manifest_path": str(manifest_path),
        "source_family": source_family,
        "target_family": target_family,
        "tensors": len(weight_map),
        "zeroed_tensors": zeroed,
        "weight_bytes": int(index["metadata"]["total_size"]),
        "warnings": report["warnings"],
        "plan": report,
    }
