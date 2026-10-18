"""
HellaSwag (validation split, 10,042 questions): pick the right ending out of 4.

Each ending is scored by the log-probability the model gives to its tokens, given
the context. Two standard ways to pick the winner:
  acc       highest total log-probability (biased towards short endings)
  acc_norm  highest average log-probability per token   <- the number usually reported

Usage:
    python evaluation/hellaswag.py --checkpoint checkpoints/base_100m/best.pt --tokenizer data/fineweb_edu_3b/tokenizer.json
    python evaluation/hellaswag.py --hf gpt2          # GPT-2 124M baseline, same questions, same scoring
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F

def load_examples():
	# the original GitHub file (rowanz/hellaswag) is gone; the same validation set is on the Hub
	from datasets import load_dataset
	ds = load_dataset("Rowan/hellaswag", split="validation")
	return [{"ctx": row["ctx"], "endings": row["endings"], "label": int(row["label"])} for row in ds]


def render(example, encode, bos_id):
	"""-> tokens (4, T), mask (4, T) marking the ending tokens, label.
	Rows are right-padded with 0s; the mask makes the padding irrelevant."""
	ctx = [bos_id] + encode(example["ctx"])
	rows, masks = [], []
	for ending in example["endings"]:
		end = encode(" " + ending)
		rows.append(ctx + end)
		masks.append([0] * len(ctx) + [1] * len(end))
	width = max(len(r) for r in rows)
	tokens = torch.zeros(4, width, dtype=torch.long)
	mask = torch.zeros(4, width, dtype=torch.long)
	for i, (row, m) in enumerate(zip(rows, masks)):
		tokens[i, :len(row)] = torch.tensor(row)
		mask[i, :len(m)] = torch.tensor(m)
	return tokens, mask, example["label"]


def score_endings(logits, tokens, mask):
	"""logits (4, T, V) -> (total log-prob, mean log-prob per token) of each ending, shape (4,)."""
	# the logits at position t predict token t+1
	log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
	target_log_probs = log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
	ending_mask = mask[:, 1:].float()
	total = (target_log_probs * ending_mask).sum(dim=1)
	return total, total / ending_mask.sum(dim=1)


def load_ours(checkpoint, tokenizer_path, device):
	from config import ModelConfig
	from network.causal_lm import CausalLM
	from tokenizer.bpe import BPETokenizer

	ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
	model = CausalLM(ModelConfig(**ckpt["model_cfg"]))
	model.load_state_dict(ckpt["model"])
	tokenizer = BPETokenizer(0)
	tokenizer.load(tokenizer_path)
	encode = lambda text: tokenizer.encode(text, allowed_special="none")
	name = f"ours ({os.path.basename(os.path.dirname(os.path.abspath(checkpoint)))}, step {ckpt['step']})"
	return model.to(device).eval(), lambda x: model(x), encode, tokenizer.special_tokens["<|endoftext|>"], name


def load_hf(model_name, device):
	import tiktoken
	from transformers import AutoModelForCausalLM

	assert model_name.startswith("gpt2"), "only GPT-2 checkpoints use the tiktoken gpt2 encoding"
	model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
	enc = tiktoken.get_encoding("gpt2")
	encode = lambda text: enc.encode(text, disallowed_special=())
	return model, lambda x: model(x).logits, encode, enc.eot_token, f"{model_name} (Hugging Face)"


@torch.no_grad()
def evaluate(forward, encode, bos_id, device, limit=None, log_every=1000):
	examples = load_examples()[:limit]
	amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
	correct = correct_norm = 0
	t0 = time.time()
	for i, example in enumerate(examples, 1):
		tokens, mask, label = render(example, encode, bos_id)
		tokens, mask = tokens.to(device), mask.to(device)
		with amp:
			logits = forward(tokens)
		total, mean = score_endings(logits, tokens, mask)
		correct += int(total.argmax().item() == label)
		correct_norm += int(mean.argmax().item() == label)
		if i % log_every == 0 or i == len(examples):
			print(f"  {i:5d}/{len(examples)} | acc {correct / i:.4f} | acc_norm {correct_norm / i:.4f} | {time.time() - t0:.0f}s")
	return {"examples": len(examples), "acc": correct / len(examples), "acc_norm": correct_norm / len(examples)}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", help="one of our checkpoints (best.pt / latest.pt)")
	parser.add_argument("--tokenizer", default="data/fineweb_edu_3b/tokenizer.json")
	parser.add_argument("--hf", help="a Hugging Face GPT-2 model instead, e.g. gpt2")
	parser.add_argument("--limit", type=int, default=None, help="only the first N questions (quick checks)")
	parser.add_argument("--out", default=None, help="write the result as JSON here")
	args = parser.parse_args()
	assert bool(args.checkpoint) != bool(args.hf), "pass exactly one of --checkpoint / --hf"

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if args.checkpoint:
		_, forward, encode, bos_id, name = load_ours(args.checkpoint, args.tokenizer, device)
	else:
		_, forward, encode, bos_id, name = load_hf(args.hf, device)

	print(f"HellaSwag val | {name} | {device}")
	result = {"model": name, **evaluate(forward, encode, bos_id, device, args.limit)}
	print(f"{name}: acc {result['acc']:.4f} | acc_norm {result['acc_norm']:.4f} ({result['examples']} questions)")
	if args.out:
		os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
		with open(args.out, "w") as fd:
			json.dump(result, fd, indent=2)


if __name__ == "__main__":
	main()
