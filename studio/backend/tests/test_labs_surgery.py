# SPDX-License-Identifier: NRC-3
# Copyright 2026-present the X8 authors. All rights reserved. See /LICENSE-NRC-3.md

"""Model surgery: what a prune removes, and what an upcycle writes.

The fixtures are 2-layer, 32-hidden Llama checkpoints written with numpy, so the
tests read the same headers and tensors a real checkpoint does without needing a
GPU, a tokenizer or ``transformers``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.labs import surgery

HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS
FFN = 64
LAYERS = 2
VOCAB = 50


def _config(model_type: str = "llama") -> dict:
    return {
        "model_type": model_type,
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "intermediate_size": FFN,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }


def _checkpoint(root: Path, model_type: str = "llama", seed: int = 0) -> Path:
    """A tiny Llama checkpoint with deliberately uneven FFN channel magnitudes."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": rng.normal(size = (VOCAB, HIDDEN)).astype(np.float32),
        "model.norm.weight": np.ones(HIDDEN, dtype = np.float32),
        "lm_head.weight": rng.normal(size = (VOCAB, HIDDEN)).astype(np.float32),
    }
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(HIDDEN, dtype = np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(HIDDEN, dtype = np.float32)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = rng.normal(size = (HIDDEN, HIDDEN)).astype(np.float32)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = rng.normal(size = (KV_HEADS * HEAD_DIM, HIDDEN)).astype(np.float32)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = rng.normal(size = (KV_HEADS * HEAD_DIM, HIDDEN)).astype(np.float32)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = rng.normal(size = (HIDDEN, HIDDEN)).astype(np.float32)
        # Channel c has magnitude 1e-4 * (c + 1), so the top channels are known.
        scale = (np.arange(FFN, dtype = np.float32) + 1.0) * 1e-4
        tensors[f"{prefix}.mlp.gate_proj.weight"] = (
            rng.normal(size = (FFN, HIDDEN)).astype(np.float32) * scale[:, None]
        )
        tensors[f"{prefix}.mlp.up_proj.weight"] = (
            rng.normal(size = (FFN, HIDDEN)).astype(np.float32) * scale[:, None]
        )
        tensors[f"{prefix}.mlp.down_proj.weight"] = (
            rng.normal(size = (HIDDEN, FFN)).astype(np.float32) * scale[None, :]
        )
    root.mkdir(parents = True, exist_ok = True)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps(_config(model_type)), encoding = "utf-8")
    return root


def _load(path: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    return load_file(str(path / "model.safetensors"))


def test_the_plan_prices_a_prune_from_headers_only(tmp_path):
    report = surgery.plan({"model": str(_checkpoint(tmp_path / "base")), "drop_layers": 1, "ffn_keep": 0.5})

    assert report["runnable"] is True
    assert report["base"]["layers"] == LAYERS
    assert report["prune"]["layers"]["keep"] == 1
    assert report["prune"]["layers"]["dropped_indices"] == [1]
    assert report["prune"]["ffn"]["keep"] == FFN // 2
    assert report["projection"]["params"] < report["base"]["params"]
    assert report["moe"] is None


def test_a_head_request_is_rounded_up_to_whole_kv_groups(tmp_path):
    report = surgery.plan({"model": str(_checkpoint(tmp_path / "base")), "drop_heads": 1})

    # Two query heads share each KV head, so asking for one removes both.
    assert report["prune"]["heads"]["requested"] == 1
    assert report["prune"]["heads"]["removed"] == 2
    assert report["prune"]["heads"]["keep"] == HEADS - 2
    assert report["prune"]["heads"]["kv_keep"] == KV_HEADS - 1
    assert any("share each KV head" in warning for warning in report["warnings"])


def test_moe_ification_is_refused_for_an_architecture_with_no_twin(tmp_path):
    report = surgery.plan(
        {"model": str(_checkpoint(tmp_path / "base", model_type = "gpt2")), "experts": 4}
    )

    assert report["runnable"] is False
    assert "llama → mixtral" in report["blocked_reason"]


def test_dropping_a_block_renumbers_the_survivors(tmp_path):
    source = _checkpoint(tmp_path / "base")
    result = surgery.run({"model": str(source), "drop_layers": 1}, output_dir = str(tmp_path / "out"))
    out = Path(result["checkpoint"])

    tensors = _load(out)
    assert "model.layers.0.self_attn.q_proj.weight" in tensors
    assert not any(key.startswith("model.layers.1.") for key in tensors)
    config = json.loads((out / "config.json").read_text(encoding = "utf-8"))
    assert config["num_hidden_layers"] == 1
    assert json.loads((out / "model.safetensors.index.json").read_text(encoding = "utf-8"))["weight_map"]
    assert (Path(result["output_dir"]) / "run_manifest.json").is_file()


def test_dropping_a_kv_group_shrinks_query_and_output_together(tmp_path):
    source = _checkpoint(tmp_path / "base")
    result = surgery.run({"model": str(source), "drop_heads": 2}, output_dir = str(tmp_path / "out"))
    tensors = _load(Path(result["checkpoint"]))

    assert tensors["model.layers.0.self_attn.q_proj.weight"].shape == (HIDDEN - 2 * HEAD_DIM, HIDDEN)
    assert tensors["model.layers.0.self_attn.k_proj.weight"].shape == (HEAD_DIM, HIDDEN)
    assert tensors["model.layers.0.self_attn.o_proj.weight"].shape == (HIDDEN, HIDDEN - 2 * HEAD_DIM)
    config = json.loads((Path(result["checkpoint"]) / "config.json").read_text(encoding = "utf-8"))
    assert config["num_attention_heads"] == HEADS - 2
    assert config["num_key_value_heads"] == KV_HEADS - 1


def test_ffn_pruning_keeps_the_heaviest_channels(tmp_path):
    source = _checkpoint(tmp_path / "base")
    original = _load(source)["model.layers.0.mlp.gate_proj.weight"]
    result = surgery.run({"model": str(source), "ffn_keep": 0.5}, output_dir = str(tmp_path / "out"))
    pruned = _load(Path(result["checkpoint"]))["model.layers.0.mlp.gate_proj.weight"]

    assert pruned.shape == (FFN // 2, HIDDEN)
    # The fixture's channel c has magnitude ~(c + 1), so the kept half is the top half.
    for row, channel in enumerate(sorted(range(FFN), key = lambda c: c)[FFN // 2 :]):
        np.testing.assert_allclose(pruned[row], original[channel], rtol = 1e-6)
    down = _load(Path(result["checkpoint"]))["model.layers.0.mlp.down_proj.weight"]
    assert down.shape == (HIDDEN, FFN // 2)


def test_upcycling_writes_experts_and_a_zero_router(tmp_path):
    source = _checkpoint(tmp_path / "base")
    original = _load(source)["model.layers.0.mlp.gate_proj.weight"]
    result = surgery.run(
        {"model": str(source), "experts": 3, "top_k": 1},
        output_dir = str(tmp_path / "out"),
    )
    tensors = _load(Path(result["checkpoint"]))

    for expert in range(3):
        np.testing.assert_array_equal(
            tensors[f"model.layers.0.block_sparse_moe.experts.{expert}.w1.weight"], original
        )
    router = tensors["model.layers.0.block_sparse_moe.gate.weight"]
    assert router.shape == (3, HIDDEN)
    assert not router.any()
    assert "model.layers.0.mlp.gate_proj.weight" not in tensors

    config = json.loads((Path(result["checkpoint"]) / "config.json").read_text(encoding = "utf-8"))
    assert config["model_type"] == "mixtral"
    assert config["num_local_experts"] == 3
    assert config["num_experts_per_tok"] == 1
    assert config["architectures"] == ["MixtralForCausalLM"]


def test_upcycling_a_qwen2_checkpoint_uses_the_qwen2_moe_layout(tmp_path):
    source = _checkpoint(tmp_path / "base", model_type = "qwen2")
    result = surgery.run(
        {"model": str(source), "experts": 2}, output_dir = str(tmp_path / "out")
    )
    tensors = _load(Path(result["checkpoint"]))

    assert "model.layers.0.mlp.experts.1.gate_proj.weight" in tensors
    assert "model.layers.0.mlp.gate.weight" in tensors
    config = json.loads((Path(result["checkpoint"]) / "config.json").read_text(encoding = "utf-8"))
    assert config["model_type"] == "qwen2_moe"
    assert config["num_experts"] == 2
    assert config["moe_intermediate_size"] == FFN


def test_upcycling_grows_the_projection_because_experts_are_copied(tmp_path):
    source = _checkpoint(tmp_path / "base")
    report = surgery.plan({"model": str(source), "experts": 4})

    assert report["runnable"] is True
    assert report["projection"]["params"] > report["base"]["params"]
    assert report["projection"]["ratio"] > 1.0
    assert any("written 4 times" in warning for warning in report["warnings"])


def test_a_hub_id_is_fetched_because_the_picker_hands_one_over(tmp_path, monkeypatch):
    """Behaviour pinned: a Hub id is downloaded, not refused.

    The Train wizard's checkpoint picker hands these engines the id the user selected, so refusing
    an id would ask them to find the cache folder by hand. The download is stubbed here; what
    matters is that the engine resolves an id to a local snapshot, and that it asks the Hub for
    the files a weight reader needs rather than for the whole repository.
    """
    local = _checkpoint(tmp_path / "downloaded")
    seen: dict = {}

    def fake_download(repo_id, **kwargs):
        seen["repo"] = repo_id
        seen["allow_patterns"] = kwargs.get("allow_patterns")
        return str(local)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    report = surgery.plan({"model": "unsloth/Llama-3.2-1B", "experts": 4})

    assert seen["repo"] == "unsloth/Llama-3.2-1B", "the Hub id never reached snapshot_download"
    assert seen["allow_patterns"], "the download is unbounded: it would fetch GGUF and optimiser state"
    assert report["runnable"] is True


def test_a_hub_id_that_cannot_be_fetched_reports_the_hub_reason(monkeypatch):
    import huggingface_hub

    def failing_download(repo_id, **kwargs):
        raise RuntimeError("401 Client Error: gated repo")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", failing_download)
    with pytest.raises(ValueError, match = "gated repo"):
        surgery.plan({"model": "unsloth/Llama-3.2-1B"})


def test_one_expert_is_refused(tmp_path):
    report = surgery.plan({"model": str(_checkpoint(tmp_path / "base")), "experts": 1})
    assert report["runnable"] is False
    assert "dense model with a router" in report["blocked_reason"]


def test_a_shard_that_fails_to_open_closes_the_shards_read_before_it(tmp_path, monkeypatch):
    # _open only hands its readers back on success, so a later shard failing used
    # to leave the earlier ones open for the life of the process.
    source = _checkpoint(tmp_path / "base")
    broken = source / "model-00002.safetensors"
    broken.write_bytes(b"this is not a safetensors file")
    monkeypatch.setattr(
        surgery, "_weight_files", lambda directory: [source / "model.safetensors", broken]
    )

    real_open = surgery.tensor_io.open_file
    closed: list[bool] = []

    class _Spy:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            closed.append(True)
            return self._handle.__exit__(*exc_info)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    monkeypatch.setattr(surgery.tensor_io, "open_file", lambda path: _Spy(real_open(path)))

    with pytest.raises(Exception):
        surgery.plan({"model": str(source)})

    assert closed == [True]
