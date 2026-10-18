"""The Hugging Face export must reproduce our model and tokenizer exactly."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

transformers = pytest.importorskip("transformers")

from config import ModelConfig
from data.prepare_fineweb import SPECIAL_TOKENS
from network.causal_lm import CausalLM
from tokenizer.bpe import BPETokenizer
from tools.export_hf import SAMPLE_TEXTS, check, to_hf_config, to_hf_state_dict, to_hf_tokenizer
from tools.load_hf import load_hf


@pytest.mark.parametrize("n_kv_heads", [4, 2, 1], ids=["mha", "gqa", "mqa"])
def test_export_matches_our_model_and_tokenizer(tmp_path, n_kv_heads):
    bpe = BPETokenizer(400, special_tokens=SPECIAL_TOKENS)
    bpe.train_from_iterator([" ".join(SAMPLE_TEXTS) * 20, "the quick brown fox jumps over the lazy dog " * 50])

    torch.manual_seed(0)
    model = CausalLM(ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=n_kv_heads,
                                 vocab_size=bpe.vocab_size, max_seq_len=96, rope_base=50000.0)).eval()
    # non-trivial norm weights, so a swapped norm would be caught
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 1:
                p.uniform_(0.5, 1.5)

    hf_model = transformers.LlamaForCausalLM(to_hf_config(model.cfg, bpe.special_tokens))
    hf_model.load_state_dict(to_hf_state_dict(model), strict=False)
    hf_model.tie_weights()
    hf_model.save_pretrained(tmp_path)
    to_hf_tokenizer(bpe).save_pretrained(tmp_path)

    reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    tok = transformers.AutoTokenizer.from_pretrained(tmp_path)
    assert check(model, bpe, reloaded, tok) < 1e-4

    # and back: the Hugging Face files load into our own classes bit for bit
    back_model, back_tok = load_hf(str(tmp_path))
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, back_model.state_dict()[name]), name
    assert back_model.cfg.rope_base == model.cfg.rope_base
    assert (back_tok.vocab, back_tok.merges, back_tok.special_tokens) == (bpe.vocab, bpe.merges, bpe.special_tokens)
