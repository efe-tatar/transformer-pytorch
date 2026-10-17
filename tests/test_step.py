"""Spec for step 7 (yours): gradient accumulation must equal one big batch."""
import copy
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ModelConfig
from network.causal_lm import CausalLM
from training.step import train_step

CPU = torch.device("cpu")


def setup():
    torch.manual_seed(0)
    model = CausalLM(ModelConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=50, max_seq_len=16))
    tokens = torch.randint(0, 50, (8, 17))
    # contiguous like the batches training/loader.py produces
    return model, tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def full_batch_reference(model, x, y, lr, clip=1e9):
    model = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss = F.cross_entropy(model(x).reshape(-1, 50), y.reshape(-1))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return model, loss.item(), norm.item()


def micro_batches(x, y, n):
    pairs = list(zip(x.chunk(n), y.chunk(n)))
    return lambda: pairs.pop(0)


def test_accumulation_matches_full_batch():
    model, x, y = setup()
    ref_model, ref_loss, ref_norm = full_batch_reference(model, x, y, lr=0.5)

    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    loss, norm = train_step(model, micro_batches(x, y, 4), opt, accum_steps=4, device=CPU, grad_clip=1e9)

    assert torch.is_tensor(loss) and not loss.requires_grad
    assert abs(loss.item() - ref_loss) < 1e-5
    assert abs(float(norm) - ref_norm) < 1e-4
    for p, q in zip(model.parameters(), ref_model.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_gradients_cleared_after_step():
    model, x, y = setup()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    train_step(model, micro_batches(x, y, 2), opt, accum_steps=2, device=CPU)
    assert all(p.grad is None for p in model.parameters())


def test_clipping_applied_and_pre_clip_norm_returned():
    model, x, y = setup()
    ref_model, _, ref_norm = full_batch_reference(model, x, y, lr=0.5, clip=1e-2)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    _, norm = train_step(model, micro_batches(x, y, 2), opt, accum_steps=2, device=CPU, grad_clip=1e-2)
    assert abs(float(norm) - ref_norm) < 1e-4 and float(norm) > 1e-2
    for p, q in zip(model.parameters(), ref_model.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_bf16_autocast_runs_and_keeps_fp32_weights():
    model, x, y = setup()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, _ = train_step(model, micro_batches(x, y, 2), opt, accum_steps=2, device=CPU, amp_dtype=torch.bfloat16)
    assert torch.isfinite(loss)
    assert all(p.dtype == torch.float32 for p in model.parameters())
