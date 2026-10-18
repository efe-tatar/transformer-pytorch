import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.hellaswag import render, score_endings

EXAMPLE = {"ctx": "a b", "endings": ["c", "d d", "e", "f f f"], "label": 1}
encode = lambda text: [ord(ch) % 50 for ch in text]  # toy tokenizer: 1 char = 1 token


def test_render_masks_only_the_endings():
    tokens, mask, label = render(EXAMPLE, encode, bos_id=0)
    assert tokens.shape == mask.shape == (4, 1 + 3 + 6)  # bos + "a b" + longest " f f f"
    ctx_len = 1 + 3
    for i, ending in enumerate(EXAMPLE["endings"]):
        assert mask[i, :ctx_len].sum() == 0
        assert mask[i].sum() == len(" " + ending)
    assert label == 1


def test_score_uses_the_logit_one_position_before_each_token():
    tokens, mask, _ = render(EXAMPLE, encode, bos_id=0)
    vocab = 50
    logits = torch.zeros(4, tokens.shape[1], vocab)
    # make row 1's ending tokens certain, predicted from the previous position
    for t in range(1, tokens.shape[1]):
        if mask[1, t]:
            logits[1, t - 1, tokens[1, t]] = 50.0
    total, mean = score_endings(logits, tokens, mask)
    assert total.argmax().item() == 1 and mean.argmax().item() == 1
    assert abs(mean[1].item()) < 1e-6
    # uniform rows: log(1/50) per token
    assert torch.allclose(mean[[0, 2, 3]], torch.full((3,), -torch.log(torch.tensor(50.0)).item()), atol=1e-5)


def test_mean_normalises_by_ending_length():
    tokens, mask, _ = render(EXAMPLE, encode, bos_id=0)
    logits = torch.zeros(4, tokens.shape[1], 50)  # uniform: every token costs log(50)
    total, mean = score_endings(logits, tokens, mask)
    # longer endings have lower totals but the same per-token mean
    assert total[3] < total[0]
    assert torch.allclose(mean, mean[0].expand(4))
