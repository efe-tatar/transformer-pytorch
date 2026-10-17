import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from config import ModelConfig
from network.transformer import TransformerBlock, build_rope_cache


class TestTransformer(unittest.TestCase):

    def _make_block(self, embedding_dimension=32, nb_heads=4, nb_kv_heads=2, max_seq_len=16, seed=0):
        torch.manual_seed(seed)
        cfg = ModelConfig(dim=embedding_dimension, n_layers=1, n_heads=nb_heads,
                          n_kv_heads=nb_kv_heads, max_seq_len=max_seq_len)
        block = TransformerBlock(cfg)
        block.eval()
        self.rope = build_rope_cache(cfg.head_dim, max_seq_len)
        return block

    def test_forward_output_shape(self):
        block = self._make_block()
        x = torch.randn(2, 5, 32)
        out = block(x, self.rope)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_head_dim_divisibility_assert(self):
        with self.assertRaises(AssertionError):
            ModelConfig(dim=30, n_heads=4, n_kv_heads=2, max_seq_len=16)

    def test_kv_heads_divisibility_assert(self):
        with self.assertRaises(AssertionError):
            ModelConfig(dim=32, n_heads=4, n_kv_heads=3, max_seq_len=16)

    def test_mha_equivalent_to_gqa_when_kv_heads_equal_heads(self):
        # nb_kv_heads == nb_heads degenerates GQA into standard MHA.
        block = self._make_block(nb_heads=4, nb_kv_heads=4)
        x = torch.randn(1, 4, 32)
        out = block(x, self.rope)
        self.assertEqual(out.shape, (1, 4, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_causal_masking_future_tokens_do_not_affect_past(self):
        block = self._make_block(max_seq_len=10)
        torch.manual_seed(1)
        x = torch.randn(1, 6, 32)

        out_full = block(x, self.rope)

        x_perturbed = x.clone()
        x_perturbed[:, 4:, :] = torch.randn(1, 2, 32)  # change only future positions
        out_perturbed = block(x_perturbed, self.rope)

        # positions 0..3 must be identical since they cannot see positions 4,5
        self.assertTrue(torch.allclose(out_full[:, :4, :], out_perturbed[:, :4, :], atol=1e-5))
        # position 4/5 (which changed) should generally differ
        self.assertFalse(torch.allclose(out_full[:, 4:, :], out_perturbed[:, 4:, :], atol=1e-6))

    def test_kv_cache_matches_full_forward(self):
        embedding_dimension, nb_heads, nb_kv_heads, max_seq_len = 32, 4, 2, 16
        block = self._make_block(embedding_dimension, nb_heads, nb_kv_heads, max_seq_len)
        torch.manual_seed(2)
        x = torch.randn(1, 5, embedding_dimension)

        out_full = block(x, self.rope)

        # incremental (token-by-token) forward using kv cache
        kv_cache = None
        cached_seq_len = 0
        outs = []
        for t in range(x.shape[1]):
            x_t = x[:, t:t + 1, :]
            out_t, kv_cache, cached_seq_len = block(
                x_t, self.rope, use_cache=True, kv_cache=kv_cache, cached_seq_len=cached_seq_len
            )
            outs.append(out_t)
        out_incremental = torch.cat(outs, dim=1)

        self.assertTrue(torch.allclose(out_full, out_incremental, atol=1e-4))

    def test_prefill_then_decode_matches_full_forward(self):
        embedding_dimension, nb_heads, nb_kv_heads, max_seq_len = 32, 4, 2, 16
        block = self._make_block(embedding_dimension, nb_heads, nb_kv_heads, max_seq_len)
        torch.manual_seed(3)
        x = torch.randn(1, 6, embedding_dimension)

        out_full = block(x, self.rope)

        # prefill first 4 tokens in one go, then decode the rest one at a time
        prefill = x[:, :4, :]
        out_prefill, kv_cache, cached_seq_len = block(
            prefill, self.rope, use_cache=True, kv_cache=None, cached_seq_len=0
        )
        outs = [out_prefill]
        for t in range(4, x.shape[1]):
            x_t = x[:, t:t + 1, :]
            out_t, kv_cache, cached_seq_len = block(
                x_t, self.rope, use_cache=True, kv_cache=kv_cache, cached_seq_len=cached_seq_len
            )
            outs.append(out_t)
        out_mixed = torch.cat(outs, dim=1)

        self.assertTrue(torch.allclose(out_full, out_mixed, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
