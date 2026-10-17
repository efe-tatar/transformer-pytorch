import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.prepare_fineweb import write_shards
from training.loader import ShardLoader


def docs(n_docs, doc_len):
    # doc i is filled with values i*doc_len ... so the stream is 0, 1, 2, ...
    for i in range(n_docs):
        yield np.arange(i * doc_len, (i + 1) * doc_len, dtype=np.uint16)


def read(path):
    return np.load(path)


def test_write_shards_layout(tmp_path):
    counts = write_shards(docs(10, 37), tmp_path, val_tokens=50, shard_size=100, max_tokens=370)
    assert counts == {"val": 50, "train": 320}
    files = sorted(os.listdir(tmp_path))
    assert files == ["train_000000.npy", "train_000001.npy", "train_000002.npy", "train_000003.npy", "val_000000.npy"]
    stream = np.concatenate([read(tmp_path / f) for f in ["val_000000.npy"] + files[:4]])
    assert np.array_equal(stream, np.arange(370))


def test_write_shards_stops_at_max_tokens(tmp_path):
    counts = write_shards(docs(100, 50), tmp_path, val_tokens=20, shard_size=1000, max_tokens=120)
    assert counts == {"val": 20, "train": 100}


@pytest.fixture
def shards(tmp_path):
    write_shards(docs(1, 1000), tmp_path, val_tokens=100, shard_size=300, max_tokens=1000)
    return tmp_path  # val: 0..99, train: 100..399, 400..699, 700..999


def test_batches_are_contiguous_and_shifted(shards):
    loader = ShardLoader(shards, "train", batch_size=2, seq_len=8)
    x, y = loader.next_batch()
    assert x.dtype == torch.long and x.shape == (2, 8)
    assert x.flatten().tolist() == list(range(100, 116))
    assert torch.equal(y, x + 1)
    x2, _ = loader.next_batch()
    assert x2[0, 0].item() == 116


def test_ranks_read_disjoint_windows(shards):
    seen = []
    for rank in range(3):
        loader = ShardLoader(shards, "train", 2, 8, rank=rank, world_size=3)
        for _ in range(4):
            seen.extend(loader.next_batch()[0].flatten().tolist())
    assert len(seen) == len(set(seen))


def test_moves_to_next_shard_and_wraps(shards):
    loader = ShardLoader(shards, "train", 2, 8)
    firsts = [loader.next_batch()[0][0, 0].item() for _ in range(60)]
    assert 400 in firsts and 700 in firsts      # visited shard 1 and 2
    assert firsts.count(100) >= 2               # wrapped back to shard 0


def test_state_dict_resumes_exactly(shards):
    a = ShardLoader(shards, "train", 2, 8)
    for _ in range(23):
        a.next_batch()
    state = a.state_dict()
    expected = [a.next_batch()[0] for _ in range(10)]

    b = ShardLoader(shards, "train", 2, 8)
    b.load_state_dict(state)
    assert all(torch.equal(e, b.next_batch()[0]) for e in expected)


def test_too_small_shard_is_rejected(shards):
    with pytest.raises(AssertionError):
        ShardLoader(shards, "val", batch_size=8, seq_len=16)
