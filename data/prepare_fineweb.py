"""
Download FineWeb-Edu, train the BPE tokenizer on a sample, then encode the
corpus into uint16 token shards that training/loader.py memory-maps.

Usage:
    # tiny local debug set (~5M tokens)
    python data/prepare_fineweb.py --max-tokens 5_000_000 --tokenizer-chars 20_000_000
    # real run
    python data/prepare_fineweb.py --max-tokens 3_000_000_000

Output (in --out-dir):
    tokenizer.json        the trained BPETokenizer
    val_000000.npy        first --val-tokens tokens
    train_000000.npy ...  --shard-size tokens each
    meta.json             counts + settings
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from tokenizer.bpe import BPETokenizer

EOT = "<|endoftext|>"
# chat tokens are reserved now so the vocabulary doesn't change for Phase 3 (SFT)
SPECIAL_TOKENS = (EOT, "<|system|>", "<|user|>", "<|assistant|>")


def stream_documents(dataset, subset):
	from datasets import load_dataset
	ds = load_dataset(dataset, name=subset, split="train", streaming=True)
	for row in ds:
		yield row["text"]


def take_chars(docs, nb_chars):
	total = 0
	for doc in docs:
		yield doc
		total += len(doc)
		if total >= nb_chars:
			return


def train_tokenizer(docs, vocab_size, path):
	tokenizer = BPETokenizer(vocab_size, special_tokens=SPECIAL_TOKENS)
	t0 = time.time()
	tokenizer.train_from_iterator(docs)
	tokenizer.dump(path)
	print(f"tokenizer: {tokenizer.vocab_size} tokens, trained in {time.time() - t0:.0f}s -> {path}")
	return tokenizer


# ---- worker side (one tokenizer per process) ----
_worker_tokenizer = None

def _init_worker(tokenizer_path):
	global _worker_tokenizer
	_worker_tokenizer = BPETokenizer(0)
	_worker_tokenizer.load(tokenizer_path)

def _encode_doc(text):
	# every document starts with EOT, so the model learns where documents begin.
	# allowed_special="none": a literal "<|endoftext|>" inside web text is just text.
	ids = [_worker_tokenizer.special_tokens[EOT]]
	ids.extend(_worker_tokenizer.encode(text, allowed_special="none"))
	return np.array(ids, dtype=np.uint16)


def write_shards(token_arrays, out_dir, val_tokens, shard_size, max_tokens):
	"""Pack a stream of uint16 arrays into val_000000.npy then train_XXXXXX.npy.
	Documents are allowed to straddle shard boundaries. Returns token counts."""
	os.makedirs(out_dir, exist_ok=True)
	counts = {"val": 0, "train": 0}
	split, index = "val", 0
	capacity = min(val_tokens, max_tokens)
	buffer = np.empty(capacity, dtype=np.uint16)
	filled = 0
	written = 0

	def flush(n):
		np.save(os.path.join(out_dir, f"{split}_{index:06d}.npy"), buffer[:n])
		counts[split] += n

	for arr in token_arrays:
		while len(arr) > 0 and written < max_tokens:
			take = min(len(arr), capacity - filled, max_tokens - written)
			buffer[filled:filled + take] = arr[:take]
			filled += take
			written += take
			arr = arr[take:]
			if filled == capacity:
				flush(filled)
				filled = 0
				if split == "val":
					split, index = "train", 0
				else:
					index += 1
				capacity = min(shard_size, max_tokens - written)
				if capacity == 0:
					break
				buffer = np.empty(capacity, dtype=np.uint16)
				print(f"  {written / 1e6:,.1f}M tokens written")
		if written >= max_tokens:
			break

	if filled > 0:
		flush(filled)
	return counts


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "fineweb_edu"))
	parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
	parser.add_argument("--subset", default="sample-10BT")
	parser.add_argument("--vocab-size", type=int, default=32768)
	parser.add_argument("--tokenizer-chars", type=lambda s: int(float(s)), default=200_000_000)
	parser.add_argument("--max-tokens", type=lambda s: int(float(s)), required=True)
	parser.add_argument("--val-tokens", type=lambda s: int(float(s)), default=10_000_000)
	parser.add_argument("--shard-size", type=lambda s: int(float(s)), default=100_000_000)
	parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
	args = parser.parse_args()

	assert args.vocab_size <= 2 ** 16, "token ids are stored as uint16"
	# keep at least ~half the data for training on small debug sets
	args.val_tokens = min(args.val_tokens, args.max_tokens // 2)

	os.makedirs(args.out_dir, exist_ok=True)
	tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")
	if os.path.exists(tokenizer_path):
		print(f"reusing tokenizer at {tokenizer_path}")
	else:
		print(f"training tokenizer on the first {args.tokenizer_chars / 1e6:.0f}M characters...")
		train_tokenizer(take_chars(stream_documents(args.dataset, args.subset), args.tokenizer_chars),
						args.vocab_size, tokenizer_path)

	print(f"encoding up to {args.max_tokens / 1e6:,.0f}M tokens with {args.workers} workers...")
	t0 = time.time()
	with mp.Pool(args.workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
		arrays = pool.imap(_encode_doc, stream_documents(args.dataset, args.subset), chunksize=32)
		counts = write_shards(arrays, args.out_dir, args.val_tokens, args.shard_size, args.max_tokens)
		pool.terminate()  # stop workers still encoding past max_tokens
	elapsed = time.time() - t0
	total = counts["val"] + counts["train"]
	print(f"done: {total / 1e6:,.1f}M tokens in {elapsed:.0f}s ({total / elapsed / 1e3:,.0f}k tokens/s)")

	with open(os.path.join(args.out_dir, "meta.json"), "w") as fd:
		json.dump({**vars(args), "val_tokens_written": counts["val"], "train_tokens_written": counts["train"]}, fd, indent=2)


if __name__ == "__main__":
	main()
