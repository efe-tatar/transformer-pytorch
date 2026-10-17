"""Spec for step 5 (yours): memoize word -> ids in BPETokenizer.encode_ordinary."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.bpe import BPETokenizer

TEXT = "the quick brown fox jumps over the lazy dog. " * 40


@pytest.fixture
def tok():
    t = BPETokenizer(300)
    t.train_from_iterator([TEXT])
    return t


def count_word_encodings(tok, monkeypatch):
    calls = []
    original = tok._encode_word
    monkeypatch.setattr(tok, "_encode_word", lambda word: calls.append(word) or original(word))
    return calls


def test_each_unique_word_encoded_once(tok, monkeypatch):
    calls = count_word_encodings(tok, monkeypatch)
    text = "the cat the cat the cat sat"
    ids = tok.encode(text)
    assert sorted(calls) == sorted(set(tok.regex.findall(text)))
    assert tok.decode(ids) == text


def test_cache_persists_across_calls(tok, monkeypatch):
    calls = count_word_encodings(tok, monkeypatch)
    first = tok.encode("over the lazy dog")
    n = len(calls)
    second = tok.encode("over the lazy dog")
    assert len(calls) == n and first == second


def test_cache_does_not_change_output(tok):
    words = tok.regex.findall(TEXT + " unseen wörds 123")
    reference = [i for w in words for i in tok._encode_word(w)]
    assert tok.encode_ordinary(TEXT + " unseen wörds 123") == reference


def test_cached_ids_cannot_be_corrupted_by_caller(tok):
    # if you cache lists, make sure a caller mutating the result can't poison the cache
    ids = tok.encode("dog")
    ids.append(999)
    assert tok.encode("dog") == ids[:-1]


def test_cache_cleared_on_retrain(tok):
    tok.encode("the lazy dog")
    tok.train_from_iterator(["completely different corpus " * 30])
    assert tok.encode_ordinary("the lazy dog") == [i for w in tok.regex.findall("the lazy dog") for i in tok._encode_word(w)]
