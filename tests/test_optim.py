"""Spec for step 7 (yours): AdamW parameter groups."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ModelConfig
from network.causal_lm import CausalLM
from training.optim import configure_optimizer


@pytest.fixture(params=["rope", "learned"])
def model(request):
    return CausalLM(ModelConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=50,
                                max_seq_len=16, norm="layer", pos=request.param))


def groups(model):
    opt = configure_optimizer(model, weight_decay=0.1, lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    decay = [g for g in opt.param_groups if g["weight_decay"] > 0]
    no_decay = [g for g in opt.param_groups if g["weight_decay"] == 0]
    assert len(decay) == 1 and len(no_decay) == 1
    assert decay[0]["weight_decay"] == 0.1
    return {id(p) for p in decay[0]["params"]}, {id(p) for p in no_decay[0]["params"]}, opt


def test_every_parameter_exactly_once(model):
    decay, no_decay, opt = groups(model)
    all_ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert set(all_ids) == {id(p) for p in model.parameters()}


def test_linear_weights_decay(model):
    decay, _, _ = groups(model)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module is not model.lm_head:
            assert id(module.weight) in decay


def test_norms_and_embeddings_do_not_decay(model):
    _, no_decay, _ = groups(model)
    assert id(model.token_embedding.weight) in no_decay
    if model.cfg.pos == "learned":
        assert id(model.position_embedding.weight) in no_decay
    norm = model.blocks[0].layer_norm_1
    assert id(norm.weight) in no_decay and id(norm.bias) in no_decay


def test_betas_and_lr(model):
    opt = configure_optimizer(model, weight_decay=0.1, lr=3e-4, betas=(0.9, 0.95))
    assert all(g["lr"] == 3e-4 and g["betas"] == (0.9, 0.95) for g in opt.param_groups)
