"""
Compare the trained BPE tokenizer's compression with GPT-2's and GPT-4's
(via tiktoken) on held-out text. Prints a markdown table.

Usage:
    python tokenizer/evaluate.py --tokenizer data/fineweb_edu_3b/tokenizer.json
"""
import argparse
import glob
import itertools
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import tiktoken

from tokenizer.bpe import BPETokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(ROOT, "data", "eval_texts")

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHERLOCK_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
# far from the start of the stream: the tokenizer only saw the first ~300M characters,
# which all come from the first parquet files
HELD_OUT_PARQUET = "sample/10BT/013_00000.parquet"


def download(url, name):
	os.makedirs(CACHE_DIR, exist_ok=True)
	path = os.path.join(CACHE_DIR, name)
	if not os.path.exists(path):
		urllib.request.urlretrieve(url, path)
	with open(path, encoding="utf-8") as fd:
		return fd.read()


def held_out_fineweb(nb_docs):
	path = os.path.join(CACHE_DIR, "fineweb_edu_held_out.txt")
	if not os.path.exists(path):
		from datasets import load_dataset
		ds = load_dataset("HuggingFaceFW/fineweb-edu", data_files=HELD_OUT_PARQUET, split="train", streaming=True)
		os.makedirs(CACHE_DIR, exist_ok=True)
		with open(path, "w", encoding="utf-8") as fd:
			fd.write("\n\n".join(row["text"] for row in itertools.islice(ds, nb_docs)))
	with open(path, encoding="utf-8") as fd:
		return fd.read()


def gutenberg_body(text):
	# drop the Project Gutenberg licence header/footer, keep the book
	start = text.find("*** START OF")
	end = text.find("*** END OF")
	if start != -1 and end != -1:
		text = text[text.index("\n", start) + 1:end]
	return text


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tokenizer", default="data/fineweb_edu_3b/tokenizer.json")
	parser.add_argument("--fineweb-docs", type=int, default=400)
	args = parser.parse_args()

	ours = BPETokenizer(0)
	ours.load(args.tokenizer)
	baselines = {
		"GPT-2 (50k)": tiktoken.get_encoding("gpt2"),
		"GPT-4 (100k)": tiktoken.get_encoding("cl100k_base"),
	}

	python_code = "\n\n".join(
		open(path, encoding="utf-8").read()
		for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True))
		if os.sep + "data" + os.sep + "eval_texts" not in path
	)
	texts = {
		"FineWeb-Edu (held out)": held_out_fineweb(args.fineweb_docs),
		"Sherlock Holmes (Gutenberg)": gutenberg_body(download(SHERLOCK_URL, "sherlock_holmes.txt")),
		"Shakespeare": download(SHAKESPEARE_URL, "tinyshakespeare.txt"),
		"Python (this repo)": python_code,
	}

	header = ["Text", "Characters", f"Ours ({ours.vocab_size // 1000}k)", *baselines]
	print("| " + " | ".join(header) + " |")
	print("|" + "---|" * len(header))
	for name, text in texts.items():
		ids = ours.encode(text, allowed_special="none")
		assert ours.decode(ids) == text, f"round trip failed on {name}"
		row = [name, f"{len(text):,}", f"{len(text) / len(ids):.2f}"]
		row += [f"{len(text) / len(enc.encode(text, disallowed_special=())):.2f}" for enc in baselines.values()]
		print("| " + " | ".join(row) + " |")
	print("\ncharacters per token, higher = fewer tokens for the same text")


if __name__ == "__main__":
	main()
