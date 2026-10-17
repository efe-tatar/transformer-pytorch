"""
Train the from-scratch CausalLM on the Tiny Shakespeare dataset.

Usage:
    python examples/train_shakespeare.py
    python examples/train_shakespeare.py --steps 2000 --device cuda
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console default codec can't print U+FFFD etc.

import torch

from config import ModelConfig
from network.causal_lm import CausalLM
from tokenizer.bpe import BPETokenizer

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare_tokenizer.json")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# ---- hyperparameters ----
# Sized to be tractable on CPU (a fwd+bwd pass at these dims is ~0.3s here vs.
# ~9s at a "real" 256-dim/4-block/seq-256 config). Bump these up if you have a GPU.
VOCAB_SIZE = 4000
BLOCK_SIZE = 128          # max_seq_len
EMBEDDING_DIM = 128
NB_TRANS_BLOCKS = 2
NB_HEADS = 4
NB_KV_HEADS = 2

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

DEFAULT_STEPS = 3000
EVAL_INTERVAL = 200
EVAL_ITERS = 10
LOG_INTERVAL = 20


def get_tokenizer():
    tokenizer = BPETokenizer(VOCAB_SIZE)
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading cached tokenizer from {TOKENIZER_PATH}")
        tokenizer.load(TOKENIZER_PATH)
    else:
        print("Training tokenizer on Tiny Shakespeare...")
        tokenizer.train([DATA_PATH])
        tokenizer.dump(TOKENIZER_PATH)
        print(f"Saved tokenizer to {TOKENIZER_PATH}")
    print(f"Vocab size: {len(tokenizer.vocab)}, merges: {len(tokenizer.merges)}")
    return tokenizer


def load_data(tokenizer, device):
    with open(DATA_PATH, "r", encoding="utf-8") as fd:
        text = fd.read()

    print("Encoding corpus...")
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    print(f"Corpus: {len(text)} chars -> {len(ids)} tokens")

    split = int(0.9 * len(data))
    train_data = data[:split].to(device)
    val_data = data[split:].to(device)
    return train_data, val_data


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), device=device)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(data, block_size, batch_size, device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def init_weights(module):
    # network/ uses PyTorch's default init, which for nn.Embedding is N(0, 1) --
    # far too large once tied into lm_head as the output projection (leads to
    # huge initial logits/loss). Re-init here from the training script instead
    # of touching network/.
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def safe_decode(tokenizer, tokens):
    # tokenizer.decode() does a strict utf-8 decode; an undertrained model can
    # sample byte sequences that aren't valid utf-8 mid-stream. Decode
    # leniently here instead of touching tokenizer/.
    raw = b"".join(tokenizer.vocab[t] for t in tokens)
    return raw.decode("utf-8", errors="replace")


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, device, block_size):
    model.eval()
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -block_size:]
        logits = model(tokens_cond)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    model.train()
    return safe_decode(tokenizer, tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download Tiny Shakespeare first, e.g.:\n"
            f"curl -sSL -o {DATA_PATH} "
            f"https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    tokenizer = get_tokenizer()
    train_data, val_data = load_data(tokenizer, device)

    model = CausalLM(ModelConfig(
        dim=EMBEDDING_DIM,
        n_layers=NB_TRANS_BLOCKS,
        vocab_size=len(tokenizer.vocab),
        max_seq_len=BLOCK_SIZE,
        n_heads=NB_HEADS,
        n_kv_heads=NB_KV_HEADS,
        dropout=0.1,
    )).to(device)
    model.apply(init_weights)

    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {nb_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model.train()
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | {elapsed:.1f}s")

        if step % EVAL_INTERVAL == 0 or step == args.steps - 1:
            losses = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, device)
            print(f"  eval @ step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

            ckpt_path = os.path.join(CHECKPOINT_DIR, "shakespeare_latest.pt")
            torch.save({"model": model.state_dict(), "step": step}, ckpt_path)

    print("\nSample generation:")
    print(generate(model, tokenizer, prompt="ROMEO:", max_new_tokens=200, device=device, block_size=BLOCK_SIZE))


if __name__ == "__main__":
    main()
