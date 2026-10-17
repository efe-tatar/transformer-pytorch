import itertools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ModelConfig, MODEL_PRESETS
from network.causal_lm import CausalLM

VARIANTS = list(itertools.product(["rms", "layer"], ["swiglu", "gelu"], ["rope", "learned"]))


def make_model(norm, mlp, pos, n_kv_heads=2):
    torch.manual_seed(0)
    cfg = ModelConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=n_kv_heads, vocab_size=50,
                      max_seq_len=16, norm=norm, mlp=mlp, pos=pos)
    return CausalLM(cfg).eval()


@pytest.mark.parametrize("norm,mlp,pos", VARIANTS)
def test_variant_forward_shape(norm, mlp, pos):
    model = make_model(norm, mlp, pos)
    logits = model(torch.randint(0, 50, (2, 7)))
    assert logits.shape == (2, 7, 50)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("norm,mlp,pos", VARIANTS)
def test_variant_kv_cache_matches_full_forward(norm, mlp, pos):
    model = make_model(norm, mlp, pos)
    tokens = torch.randint(0, 50, (1, 7))
    full = model(tokens)

    logits, cache, n = model(tokens[:, :3], use_cache=True)
    outs = [logits]
    for t in range(3, 7):
        logits, cache, n = model(tokens[:, t:t + 1], use_cache=True, kv_cache_list=cache, cached_seq_len=n)
        outs.append(logits)
    assert torch.allclose(full, torch.cat(outs, dim=1), atol=1e-4)


def test_causality_learned_positions():
    model = make_model("rms", "swiglu", "learned")
    tokens = torch.randint(0, 50, (1, 8))
    perturbed = tokens.clone()
    perturbed[:, 5:] = (perturbed[:, 5:] + 1) % 50
    assert torch.allclose(model(tokens)[:, :5], model(perturbed)[:, :5], atol=1e-5)


def test_rope_tables_shared_and_not_in_state_dict():
    model = make_model("rms", "swiglu", "rope")
    keys = model.state_dict().keys()
    assert not any("rope" in k or k.endswith("mask") for k in keys)
    assert not hasattr(model.blocks[0], "angle_cos_cache")


def test_kv_cache_follows_activation_dtype():
    model = make_model("rms", "swiglu", "rope").to(torch.bfloat16)
    _, cache, _ = model(torch.randint(0, 50, (1, 4)), use_cache=True)
    assert cache[0][0].dtype == torch.bfloat16


@pytest.mark.parametrize("name,low,high", [
    ("debug_cpu", 1e6, 4e6),
    ("small_ablation", 30e6, 50e6),
    ("gpt2_small", 90e6, 130e6),
])
def test_preset_param_counts(name, low, high):
    with torch.device("meta"):
        model = CausalLM(MODEL_PRESETS[name])
    n = sum(p.numel() for p in model.parameters())
    assert low < n < high, f"{name}: {n / 1e6:.1f}M"
