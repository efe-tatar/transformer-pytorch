import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from config import ModelConfig
from network.causal_lm import CausalLM


class TestCausalLM(unittest.TestCase):

    def _make_model(self, embedding_dimension=32, nb_trans_blocks=2, vocabulary_size=50,
                     max_seq_len=16, nb_heads=4, nb_kv_heads=2, seed=0):
        torch.manual_seed(seed)
        model = CausalLM(ModelConfig(dim=embedding_dimension, n_layers=nb_trans_blocks, vocab_size=vocabulary_size,
                                     max_seq_len=max_seq_len, n_heads=nb_heads, n_kv_heads=nb_kv_heads))
        model.eval()
        return model

    def test_forward_output_shape(self):
        model = self._make_model(vocabulary_size=50)
        tokens = torch.randint(0, 50, (2, 6))
        logits = model(tokens)
        self.assertEqual(logits.shape, (2, 6, 50))

    def test_weight_tying(self):
        model = self._make_model()
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_logits_no_nan_or_inf(self):
        model = self._make_model()
        tokens = torch.randint(0, 50, (1, 8))
        logits = model(tokens)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_use_cache_incremental_generation_matches_full_forward(self):
        model = self._make_model(vocabulary_size=50, max_seq_len=16)
        torch.manual_seed(42)
        tokens = torch.randint(0, 50, (1, 6))

        logits_full = model(tokens)

        kv_cache_list = None
        cached_seq_len = 0
        outs = []
        for t in range(tokens.shape[1]):
            tok_t = tokens[:, t:t + 1]
            logits_t, kv_cache_list, cached_seq_len = model(
                tok_t, use_cache=True, kv_cache_list=kv_cache_list, cached_seq_len=cached_seq_len
            )
            outs.append(logits_t)
        logits_incremental = torch.cat(outs, dim=1)

        self.assertTrue(torch.allclose(logits_full, logits_incremental, atol=1e-3))

    def test_use_cache_prefill_then_decode_matches_full_forward(self):
        model = self._make_model(vocabulary_size=50, max_seq_len=16)
        torch.manual_seed(7)
        tokens = torch.randint(0, 50, (1, 7))

        logits_full = model(tokens)

        prefill = tokens[:, :3]
        logits_prefill, kv_cache_list, cached_seq_len = model(
            prefill, use_cache=True, kv_cache_list=None, cached_seq_len=0
        )
        outs = [logits_prefill]
        for t in range(3, tokens.shape[1]):
            tok_t = tokens[:, t:t + 1]
            logits_t, kv_cache_list, cached_seq_len = model(
                tok_t, use_cache=True, kv_cache_list=kv_cache_list, cached_seq_len=cached_seq_len
            )
            outs.append(logits_t)
        logits_mixed = torch.cat(outs, dim=1)

        self.assertTrue(torch.allclose(logits_full, logits_mixed, atol=1e-3))

    def test_cached_seq_len_advances_correctly(self):
        model = self._make_model(vocabulary_size=50, max_seq_len=16)
        tokens = torch.randint(0, 50, (1, 4))
        _, kv_cache_list, cached_seq_len = model(tokens, use_cache=True)
        self.assertEqual(cached_seq_len, 4)
        self.assertEqual(len(kv_cache_list), model.blocks.__len__())


if __name__ == "__main__":
    unittest.main()
