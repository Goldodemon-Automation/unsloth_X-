# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Arch swap: roles, not names, and a plan that says what changed meaning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.labs import arch_swap

HIDDEN = 32
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
FFN = 64
VOCAB = 40
CTX = 8


def _llama(root: Path, *, layers: int = 2, vocab: int = VOCAB, kv_heads: int = HEADS) -> Path:
    from safetensors.numpy import save_file

    rng = np.random.default_rng(1)
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.normal(size = (vocab, HIDDEN)).astype(np.float32),
        "model.norm.weight": np.ones(HIDDEN, dtype = np.float32),
        "lm_head.weight": rng.normal(size = (vocab, HIDDEN)).astype(np.float32),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(HIDDEN, dtype = np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(HIDDEN, dtype = np.float32)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = rng.normal(size = (HIDDEN, HIDDEN)).astype(np.float32)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = rng.normal(
            size = (kv_heads * HEAD_DIM, HIDDEN)
        ).astype(np.float32)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = rng.normal(
            size = (kv_heads * HEAD_DIM, HIDDEN)
        ).astype(np.float32)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = rng.normal(size = (HIDDEN, HIDDEN)).astype(np.float32)
        tensors[f"{prefix}.mlp.gate_proj.weight"] = rng.normal(size = (FFN, HIDDEN)).astype(np.float32)
        tensors[f"{prefix}.mlp.up_proj.weight"] = rng.normal(size = (FFN, HIDDEN)).astype(np.float32)
        tensors[f"{prefix}.mlp.down_proj.weight"] = rng.normal(size = (HIDDEN, FFN)).astype(np.float32)
    root.mkdir(parents = True, exist_ok = True)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": HIDDEN,
                "num_hidden_layers": layers,
                "num_attention_heads": HEADS,
                "num_key_value_heads": kv_heads,
                "intermediate_size": FFN,
                "vocab_size": vocab,
            }
        ),
        encoding = "utf-8",
    )
    return root


def _gpt2(root: Path, *, layers: int = 2, vocab: int = VOCAB) -> Path:
    """A GPT-2 checkpoint whose fused QKV rows are 1, 2 and 3, so a split is visible."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(2)
    fused = np.zeros((HIDDEN, 3 * HIDDEN), dtype = np.float32)
    fused[:, :HIDDEN] = 1.0
    fused[:, HIDDEN : 2 * HIDDEN] = 2.0
    fused[:, 2 * HIDDEN :] = 3.0
    tensors: dict[str, np.ndarray] = {
        "transformer.wte.weight": rng.normal(size = (vocab, HIDDEN)).astype(np.float32),
        "transformer.wpe.weight": rng.normal(size = (CTX, HIDDEN)).astype(np.float32),
        "transformer.ln_f.weight": np.ones(HIDDEN, dtype = np.float32),
    }
    for layer in range(layers):
        prefix = f"transformer.h.{layer}"
        tensors[f"{prefix}.ln_1.weight"] = np.ones(HIDDEN, dtype = np.float32)
        tensors[f"{prefix}.ln_2.weight"] = np.ones(HIDDEN, dtype = np.float32)
        tensors[f"{prefix}.attn.c_attn.weight"] = fused.copy()
        tensors[f"{prefix}.attn.c_proj.weight"] = rng.normal(size = (HIDDEN, HIDDEN)).astype(np.float32)
        tensors[f"{prefix}.mlp.c_fc.weight"] = rng.normal(size = (HIDDEN, 4 * HIDDEN)).astype(np.float32)
        tensors[f"{prefix}.mlp.c_proj.weight"] = rng.normal(size = (HIDDEN, 4 * HIDDEN)).astype(np.float32)
    root.mkdir(parents = True, exist_ok = True)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "n_embd": HIDDEN,
                "n_head": HEADS,
                "n_layer": layers,
                "n_positions": CTX,
                "n_inner": 4 * HIDDEN,
                "vocab_size": vocab,
            }
        ),
        encoding = "utf-8",
    )
    return root


def _load(path: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    return load_file(str(path / "model.safetensors"))


def test_the_plan_names_the_roles_and_the_actions(tmp_path):
    report = arch_swap.plan(
        {"source": str(_llama(tmp_path / "llama")), "target": str(_gpt2(tmp_path / "gpt2"))}
    )

    assert report["runnable"] is True
    assert report["source_family"] == "llama"
    assert report["target_family"] == "gpt2"
    actions = {row["role"]: row["action"] for row in report["mapping"]}
    assert actions["qkv"] == "fuse"
    assert actions["fc"] == "fuse"
    assert any("SwiGLU" in warning for warning in report["warnings"])


def test_a_fused_source_is_split_into_separate_projections(tmp_path):
    result = arch_swap.run(
        {"source": str(_gpt2(tmp_path / "gpt2")), "target": str(_llama(tmp_path / "llama"))},
        output_dir = str(tmp_path / "out"),
    )
    tensors = _load(Path(result["checkpoint"]))

    np.testing.assert_allclose(tensors["model.layers.0.self_attn.q_proj.weight"], 1.0)
    np.testing.assert_allclose(tensors["model.layers.0.self_attn.k_proj.weight"], 2.0)
    np.testing.assert_allclose(tensors["model.layers.0.self_attn.v_proj.weight"], 3.0)
    assert tensors["model.layers.0.self_attn.q_proj.weight"].shape == (HIDDEN, HIDDEN)


def test_the_target_config_is_the_one_written(tmp_path):
    result = arch_swap.run(
        {"source": str(_gpt2(tmp_path / "gpt2")), "target": str(_llama(tmp_path / "llama"))},
        output_dir = str(tmp_path / "out"),
    )
    config = json.loads((Path(result["checkpoint"]) / "config.json").read_text(encoding = "utf-8"))

    assert config["model_type"] == "llama"
    assert config["num_hidden_layers"] == 2
    assert (Path(result["output_dir"]) / "run_manifest.json").is_file()


def test_a_position_embedding_with_no_source_role_is_zeroed(tmp_path):
    result = arch_swap.run(
        {"source": str(_llama(tmp_path / "llama")), "target": str(_gpt2(tmp_path / "gpt2"))},
        output_dir = str(tmp_path / "out"),
    )
    tensors = _load(Path(result["checkpoint"]))

    assert tensors["transformer.wpe.weight"].shape == (CTX, HIDDEN)
    assert not tensors["transformer.wpe.weight"].any()
    assert result["zeroed_tensors"] >= 1


def test_a_tied_source_head_fills_the_target_head(tmp_path):
    """GPT-2 has no lm_head; its embedding is the head, so the target gets one."""
    result = arch_swap.run(
        {"source": str(_gpt2(tmp_path / "gpt2")), "target": str(_llama(tmp_path / "llama"))},
        output_dir = str(tmp_path / "out"),
    )
    tensors = _load(Path(result["checkpoint"]))
    source = _load(tmp_path / "gpt2")

    np.testing.assert_array_equal(tensors["lm_head.weight"], source["transformer.wte.weight"])


def test_a_larger_target_vocabulary_is_zero_padded(tmp_path):
    result = arch_swap.run(
        {
            "source": str(_llama(tmp_path / "small", vocab = 20)),
            "target": str(_llama(tmp_path / "big", vocab = 40)),
        },
        output_dir = str(tmp_path / "out"),
    )
    tensors = _load(Path(result["checkpoint"]))

    assert tensors["model.embed_tokens.weight"].shape == (40, HIDDEN)
    assert not tensors["model.embed_tokens.weight"][20:].any()
    assert any("Vocabulary 20 → 40" in warning for warning in result["warnings"])


def test_extra_target_blocks_repeat_the_last_mapped_block(tmp_path):
    result = arch_swap.run(
        {
            "source": str(_llama(tmp_path / "small", layers = 1)),
            "target": str(_llama(tmp_path / "big", layers = 3)),
        },
        output_dir = str(tmp_path / "out"),
    )
    tensors = _load(Path(result["checkpoint"]))

    np.testing.assert_array_equal(
        tensors["model.layers.2.self_attn.q_proj.weight"],
        tensors["model.layers.0.self_attn.q_proj.weight"],
    )
    assert any("repeat the last mapped block" in warning for warning in result["warnings"])


def test_an_unknown_layout_is_refused_by_name(tmp_path):
    source = _llama(tmp_path / "llama")
    config = json.loads((source / "config.json").read_text(encoding = "utf-8"))
    config["model_type"] = "mamba"
    (source / "config.json").write_text(json.dumps(config), encoding = "utf-8")

    with pytest.raises(ValueError, match = "no name table"):
        arch_swap.plan({"source": str(source), "target": str(_llama(tmp_path / "other"))})


def test_a_hub_id_is_refused_because_a_swap_writes_local_weights(tmp_path):
    with pytest.raises(ValueError, match = "download the checkpoint first"):
        arch_swap.plan({"source": "unsloth/Llama-3.2-1B", "target": str(_llama(tmp_path / "t"))})
