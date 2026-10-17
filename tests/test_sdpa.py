"""Spec for step 4 (yours): the SDPA attention path must match the manual reference."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ModelConfig
from network.causal_lm import CausalLM


def make_pair(n_kv_heads):
    cfg = dict(dim=64, n_layers=2, n_heads=8, n_kv_heads=n_kv_heads, vocab_size=100, max_seq_len=32)
    torch.manual_seed(0)
    manual = CausalLM(ModelConfig(**cfg, attn_impl="manual")).eval()
    sdpa = CausalLM(ModelConfig(**cfg, attn_impl="sdpa")).eval()
    sdpa.load_state_dict(manual.state_dict())
    return manual, sdpa


@pytest.mark.parametrize("n_kv_heads", [8, 4, 1], ids=["mha", "gqa", "mqa"])
def test_sdpa_matches_manual_full_forward(n_kv_heads):
    manual, sdpa = make_pair(n_kv_heads)
    tokens = torch.randint(0, 100, (3, 20))
    assert torch.allclose(manual(tokens), sdpa(tokens), atol=1e-5)


@pytest.mark.parametrize("n_kv_heads", [8, 4, 1], ids=["mha", "gqa", "mqa"])
def test_sdpa_matches_manual_with_kv_cache(n_kv_heads):
    # prefill 5 tokens (seq_len > 1 *and* past_len > 0 is the tricky mask case
    # on the second chunk), then a 3-token chunk, then single-token decoding
    manual, sdpa = make_pair(n_kv_heads)
    tokens = torch.randint(0, 100, (2, 12))
    ref = manual(tokens)

    outs, cache, n = [], None, 0
    for start, end in [(0, 5), (5, 8), (8, 9), (9, 10), (10, 12)]:
        logits, cache, n = sdpa(tokens[:, start:end], use_cache=True, kv_cache_list=cache, cached_seq_len=n)
        outs.append(logits)
    assert torch.allclose(ref, torch.cat(outs, dim=1), atol=1e-5)


def test_sdpa_backward_works():
    _, sdpa = make_pair(4)
    sdpa.train()
    tokens = torch.randint(0, 100, (2, 16))
    sdpa(tokens).sum().backward()
    assert sdpa.blocks[0].W_QKV.weight.grad is not None
