"""Spec for step 3 (yours): weight init inside CausalLM._init_weights."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ModelConfig
from network.causal_lm import CausalLM

N_LAYERS = 6


def make_model():
    torch.manual_seed(0)
    return CausalLM(ModelConfig(dim=128, n_layers=N_LAYERS, n_heads=4, n_kv_heads=2,
                                vocab_size=1000, max_seq_len=64))


def test_embedding_std_is_0_02():
    std = make_model().token_embedding.weight.std().item()
    assert abs(std - 0.02) < 0.002, std


def test_plain_linears_std_is_0_02():
    block = make_model().blocks[0]
    for linear in (block.W_QKV, block.swiglu_linear_1, block.swiglu_linear_2):
        assert abs(linear.weight.std().item() - 0.02) < 0.002


def test_residual_projections_are_scaled():
    target = 0.02 / math.sqrt(2 * N_LAYERS)
    for block in make_model().blocks:
        for linear in block.output_projections():
            assert abs(linear.weight.std().item() - target) < 0.1 * target


def test_norms_untouched():
    block = make_model().blocks[0]
    assert torch.all(block.layer_norm_1.weight == 1)


def test_initial_loss_is_about_ln_vocab():
    # a well-initialised LM starts out ~uniform over the vocabulary
    model = make_model().eval()
    tokens = torch.randint(0, 1000, (4, 64))
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 1000), tokens[:, 1:].reshape(-1))
    assert abs(loss.item() - math.log(1000)) < 0.1, loss.item()
