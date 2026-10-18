import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["SERVE_NO_AUTOLOAD"] = "1"

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from config import ModelConfig
from data.prepare_fineweb import SPECIAL_TOKENS
from network.causal_lm import CausalLM
from serve.app import IncrementalDecoder, create_app
from tokenizer.bpe import BPETokenizer


@pytest.fixture(scope="module")
def setup():
    tok = BPETokenizer(300, special_tokens=SPECIAL_TOKENS)
    tok.train_from_iterator(["the quick brown fox jumps over the lazy dog. café naïve " * 40])
    torch.manual_seed(0)
    model = CausalLM(ModelConfig(dim=32, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=tok.vocab_size,
                                 max_seq_len=64)).eval()
    return model, tok, TestClient(create_app(model, tok))


def test_health(setup):
    _, _, client = setup
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["context"] == 64


def test_generate_is_greedy_and_matches_model_generate(setup):
    model, tok, client = setup
    body = client.post("/generate", json={"prompt": "the quick", "max_new_tokens": 10, "temperature": 0}).json()
    ids = torch.tensor([[tok.special_tokens["<|endoftext|>"]] + tok.encode("the quick")])
    expected = model.generate(ids, 10, temperature=0)[0, ids.shape[1]:].tolist()
    eot = tok.special_tokens["<|endoftext|>"]
    expected = expected[:expected.index(eot)] if eot in expected else expected
    assert body["text"] == tok.decode(expected) and body["tokens"] == len(expected)


def test_stream_concatenates_to_the_same_text(setup):
    _, _, client = setup
    req = {"prompt": "the lazy", "max_new_tokens": 12, "temperature": 0}
    full = client.post("/generate", json=req).json()["text"]
    with client.stream("POST", "/generate/stream", json=req) as response:
        lines = [l for l in response.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert "".join(json.loads(l[6:])["text"] for l in lines[:-1]) == full


def test_seed_makes_sampling_reproducible(setup):
    _, _, client = setup
    req = {"prompt": "the", "max_new_tokens": 15, "temperature": 1.0, "seed": 123}
    assert client.post("/generate", json=req).json() ["text"] == client.post("/generate", json=req).json()["text"]


def test_long_prompt_is_truncated_not_rejected(setup):
    _, _, client = setup
    r = client.post("/generate", json={"prompt": "the quick brown fox " * 100, "max_new_tokens": 8})
    assert r.status_code == 200


def test_invalid_requests_rejected(setup):
    _, _, client = setup
    assert client.post("/generate", json={"prompt": "x", "max_new_tokens": 0}).status_code == 422
    assert client.post("/generate", json={"prompt": "x", "temperature": 5}).status_code == 422


def test_incremental_decoder_holds_back_split_characters(setup):
    _, tok, _ = setup
    decoder = IncrementalDecoder(tok)
    e_acute = list("é".encode("utf-8"))  # two byte tokens
    assert decoder.push(e_acute[0]) == ""
    assert decoder.push(e_acute[1]) == "é"
