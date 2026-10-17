"""End to end: train.py interrupted at step 5 and resumed must reproduce an
uninterrupted run exactly. Skipped until the step 3/7 functions are implemented."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from data.prepare_fineweb import SPECIAL_TOKENS, write_shards
from tokenizer.bpe import BPETokenizer


def stubs_implemented():
    from training.schedule import get_lr
    from training.optim import configure_optimizer  # noqa: F401
    from training.step import train_step  # noqa: F401
    try:
        get_lr(0, 1, 2, 1.0, 0.1)
    except NotImplementedError:
        return False
    src = open(os.path.join(ROOT, "training", "optim.py")).read() + open(os.path.join(ROOT, "training", "step.py")).read()
    return "NotImplementedError" not in src


pytestmark = pytest.mark.skipif(not stubs_implemented(), reason="waiting on step 7 (get_lr / configure_optimizer / train_step)")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    tok = BPETokenizer(300, special_tokens=SPECIAL_TOKENS)
    tok.train_from_iterator(["the quick brown fox jumps over the lazy dog. " * 50])
    tok.dump(str(d / "tokenizer.json"))
    rng = np.random.default_rng(0)
    write_shards([rng.integers(0, 296, 40_000).astype(np.uint16)], d, val_tokens=10_000, shard_size=15_000, max_tokens=40_000)
    return d


def run(data_dir, out_dir, *extra):
    cmd = [sys.executable, "train.py", "--preset", "debug_cpu", f"--data_dir={data_dir}", f"--out_dir={out_dir}",
           "--max_steps=10", "--warmup_steps=3", "--log_interval=1", "--eval_interval=4",
           "--checkpoint_interval=100", "--sample_tokens=8", "--dim=32", "--max_seq_len=32", "--micro_batch_size=4",
           "--total_batch_tokens=256", *extra]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def metrics(out_dir):
    rows = [json.loads(line) for line in open(out_dir / "metrics.jsonl")]
    return {(r["step"], k): v for r in rows for k, v in r.items() if k in ("train/loss", "val/loss")}


def test_interrupted_run_resumes_exactly(data_dir, tmp_path):
    run(data_dir, tmp_path / "a")
    out = run(data_dir, tmp_path / "b", "--stop_step=5")
    assert "stopped at step 5" in out
    out = run(data_dir, tmp_path / "b", "--resume")
    assert "resumed" in out and "at step 5" in out

    a, b = metrics(tmp_path / "a"), metrics(tmp_path / "b")
    assert sorted(a) == sorted(b)
    assert a == b


def test_resume_rejects_different_architecture(data_dir, tmp_path):
    run(data_dir, tmp_path / "c", "--stop_step=2")
    cmd = [sys.executable, "train.py", "--preset", "debug_cpu", f"--data_dir={data_dir}", f"--out_dir={tmp_path / 'c'}",
           "--dim=32", "--max_seq_len=32", "--micro_batch_size=4", "--total_batch_tokens=256", "--n_layers=3", "--resume"]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "different model config" in result.stdout + result.stderr
