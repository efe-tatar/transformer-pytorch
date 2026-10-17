import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.bpe import BPETokenizer

EOT = "<|endoftext|>"
SPECIALS = (EOT, "<|user|>", "<|assistant|>")
TEXT = "the quick brown fox jumps over the lazy dog. " * 40


@pytest.fixture(scope="module")
def tok():
    t = BPETokenizer(300, special_tokens=SPECIALS)
    t.train_from_iterator([TEXT, EOT + TEXT])
    return t


def test_special_ids_follow_merges_and_respect_budget(tok):
    assert tok.vocab_size == 300
    assert len(tok.vocab) == 300 - len(SPECIALS)
    assert sorted(tok.special_tokens.values()) == list(range(len(tok.vocab), 300))


def test_special_token_never_merged(tok):
    # "<|" must not have been learned from the EOT occurrence in the corpus
    assert not any(b"<|" in v for v in tok.vocab.values())


def test_encode_allowed_special(tok):
    ids = tok.encode("the fox" + EOT + "the dog")
    assert ids.count(tok.special_tokens[EOT]) == 1
    assert tok.decode(ids) == "the fox" + EOT + "the dog"


def test_encode_disallowed_special_is_plain_text(tok):
    ids = tok.encode("a " + EOT, allowed_special="none")
    assert tok.special_tokens[EOT] not in ids
    assert tok.decode(ids) == "a " + EOT


def test_encode_subset_of_specials(tok):
    ids = tok.encode(EOT + "<|user|>", allowed_special={"<|user|>"})
    assert tok.special_tokens["<|user|>"] in ids
    assert tok.special_tokens[EOT] not in ids


def test_decode_invalid_utf8_does_not_raise(tok):
    e_acute = "é".encode("utf-8")  # 2 bytes, decode only the first one
    assert tok.decode([e_acute[0]]) == "�"


def test_dump_load_keeps_specials(tok):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tok.json")
        tok.dump(path)
        reloaded = BPETokenizer(0)
        reloaded.load(path)
    assert reloaded.special_tokens == tok.special_tokens
    sample = "x" + EOT + "the lazy dog"
    assert reloaded.encode(sample) == tok.encode(sample)


def test_train_from_files_matches_iterator():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "corpus.txt")
        with open(path, "w", encoding="utf-8") as fd:
            fd.write(TEXT)
        from_file = BPETokenizer(290)
        from_file.train([path])
    from_iter = BPETokenizer(290)
    from_iter.train_from_iterator([TEXT])
    assert from_file.merges == from_iter.merges
