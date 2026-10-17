import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.bpe import BPETokenizer


class TestBPEPersistence(unittest.TestCase):

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

    def test_dump_load_roundtrip_preserves_vocab_and_merges(self):
        text = "the quick brown fox jumps over the lazy dog " * 30
        tok = self._train_on_text(text, max_vocab_size=300)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dump_path = f.name
        try:
            tok.dump(dump_path)

            reloaded = BPETokenizer(max_vocab_size=0)  # overwritten by load
            reloaded.load(dump_path)

            self.assertEqual(tok.max_vocab_size, reloaded.max_vocab_size)
            self.assertEqual(tok.vocab, reloaded.vocab)
            self.assertEqual(tok.merges, reloaded.merges)
        finally:
            os.remove(dump_path)

    def test_reloaded_tokenizer_encodes_decodes_identically(self):
        text = "the quick brown fox jumps over the lazy dog " * 30
        tok = self._train_on_text(text, max_vocab_size=300)
        sample = "the quick fox jumps over the dog"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dump_path = f.name
        try:
            tok.dump(dump_path)

            reloaded = BPETokenizer(max_vocab_size=0)
            reloaded.load(dump_path)

            self.assertEqual(tok.encode(sample), reloaded.encode(sample))
            self.assertEqual(reloaded.decode(reloaded.encode(sample)), sample)
        finally:
            os.remove(dump_path)

    def test_dump_file_is_valid_json(self):
        tok = self._train_on_text("hello world " * 10, max_vocab_size=260)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dump_path = f.name
        try:
            tok.dump(dump_path)
            import json
            with open(dump_path, "r", encoding="utf-8") as fd:
                data = json.load(fd)  # raises if malformed
            self.assertIn("vocab", data)
            self.assertIn("merges", data)
        finally:
            os.remove(dump_path)

    def test_reloaded_vocab_keys_are_ints_and_values_are_bytes(self):
        tok = self._train_on_text("hello world " * 10, max_vocab_size=260)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dump_path = f.name
        try:
            tok.dump(dump_path)

            reloaded = BPETokenizer(max_vocab_size=0)
            reloaded.load(dump_path)

            for k, v in reloaded.vocab.items():
                self.assertIsInstance(k, int)
                self.assertIsInstance(v, bytes)
        finally:
            os.remove(dump_path)

    def test_reloaded_merges_keys_are_int_tuples(self):
        text = "the quick brown fox jumps over the lazy dog " * 30
        tok = self._train_on_text(text, max_vocab_size=300)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dump_path = f.name
        try:
            tok.dump(dump_path)

            reloaded = BPETokenizer(max_vocab_size=0)
            reloaded.load(dump_path)

            self.assertGreater(len(reloaded.merges), 0)
            for k, v in reloaded.merges.items():
                self.assertIsInstance(k, tuple)
                self.assertEqual(len(k), 2)
                self.assertIsInstance(k[0], int)
                self.assertIsInstance(k[1], int)
                self.assertIsInstance(v, int)
        finally:
            os.remove(dump_path)


if __name__ == "__main__":
    unittest.main()
