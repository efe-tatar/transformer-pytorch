import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.bpe import BPETokenizer


class TestBPETokenizer(unittest.TestCase):

    def _train_on_text(self, text, max_vocab_size):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(text)
            path = f.name
        try:
            tok = BPETokenizer(max_vocab_size)
            tok.train([path])
            return tok
        finally:
            os.remove(path)

    def test_base_vocab_is_256_bytes(self):
        tok = self._train_on_text("hello world", max_vocab_size=256)
        self.assertEqual(len(tok.vocab), 256)
        self.assertEqual(len(tok.merges), 0)
        for i in range(256):
            self.assertEqual(tok.vocab[i], bytes([i]))

    def test_vocab_grows_with_merges(self):
        text = "the quick brown fox jumps over the lazy dog " * 50
        tok = self._train_on_text(text, max_vocab_size=300)
        self.assertEqual(len(tok.vocab), 300)
        self.assertEqual(len(tok.merges), 300 - 256)

    def test_encode_decode_roundtrip_ascii(self):
        text = "the quick brown fox jumps over the lazy dog. " * 20
        tok = self._train_on_text(text, max_vocab_size=310)
        sample = "the quick fox jumps over the dog"
        tokens = tok.encode(sample)
        decoded = tok.decode(tokens)
        self.assertEqual(decoded, sample)

    def test_encode_decode_roundtrip_unicode(self):
        text = "café naïve résumé " * 30
        tok = self._train_on_text(text, max_vocab_size=280)
        sample = "café naïve"
        tokens = tok.encode(sample)
        decoded = tok.decode(tokens)
        self.assertEqual(decoded, sample)

    def test_encode_on_unseen_text_still_roundtrips(self):
        train_text = "the quick brown fox jumps over the lazy dog " * 30
        tok = self._train_on_text(train_text, max_vocab_size=300)
        unseen = "zzz qqq xyz123 !!! completely novel input"
        tokens = tok.encode(unseen)
        decoded = tok.decode(tokens)
        self.assertEqual(decoded, unseen)

    def test_frequent_pair_merged_before_rare_pair(self):
        # "ab" appears far more often than "xy" so it should be merged first,
        # meaning it gets a lower (earlier-assigned) token id.
        text = ("ab " * 100) + ("xy " * 2)
        tok = self._train_on_text(text, max_vocab_size=258)
        ab_id = tok.merges.get((ord("a"), ord("b")))
        xy_id = tok.merges.get((ord("x"), ord("y")))
        self.assertIsNotNone(ab_id)
        if xy_id is not None:
            self.assertLess(ab_id, xy_id)

    def test_encode_empty_string(self):
        tok = self._train_on_text("hello world " * 10, max_vocab_size=260)
        self.assertEqual(tok.encode(""), [])

    def test_single_character_word(self):
        tok = self._train_on_text("a b c a b c " * 10, max_vocab_size=260)
        tokens = tok.encode("a")
        self.assertEqual(tok.decode(tokens), "a")

    def test_encoded_tokens_are_valid_vocab_ids(self):
        text = "the quick brown fox jumps over the lazy dog " * 20
        tok = self._train_on_text(text, max_vocab_size=300)
        tokens = tok.encode("the quick brown fox")
        for t in tokens:
            self.assertIn(t, tok.vocab)


if __name__ == "__main__":
    unittest.main()
