"""
Inference benchmarks for a trained checkpoint:
  1. greedy generation speed with vs without the KV cache (and a check that both give the same tokens)
  2. KV-cache memory: this model's GQA layout vs the same model with full multi-head attention
  3. optional: the same weights run by Hugging Face transformers (tools/export_hf.py output)

Usage:
    python tools/benchmark_inference.py --checkpoint checkpoints/base_100m/best.pt
    python tools/benchmark_inference.py --hf exported/base_100m          # also time HF's LlamaForCausalLM
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from config import ModelConfig
from network.causal_lm import CausalLM


@torch.inference_mode()
def generate_without_cache(model, ids, max_new_tokens):
	# recompute the whole sequence for every new token: O(n^2) work overall
	for _ in range(max_new_tokens):
		next_token = model(ids)[:, -1, :].argmax(dim=-1, keepdim=True)
		ids = torch.cat([ids, next_token], dim=1)
	return ids


@torch.inference_mode()
def generate_with_cache(model, ids, max_new_tokens):
	return model.generate(ids, max_new_tokens, temperature=0)


def timed(fn, device, repeats):
	fn()  # warm-up
	times = []
	for _ in range(repeats):
		if device.type == "cuda":
			torch.cuda.synchronize()
		t0 = time.perf_counter()
		out = fn()
		if device.type == "cuda":
			torch.cuda.synchronize()
		times.append(time.perf_counter() - t0)
	return out, min(times)


def kv_cache_bytes(cfg, n_kv_heads, seq_len, bytes_per_value):
	# K and V, for every layer: (n_kv_heads, seq_len, head_dim) each, per sequence
	return 2 * cfg.n_layers * n_kv_heads * seq_len * cfg.head_dim * bytes_per_value


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", default="checkpoints/base_100m/best.pt")
	parser.add_argument("--hf", default=None, help="exported Hugging Face folder, to time transformers too")
	parser.add_argument("--prompt-len", type=int, default=16)
	parser.add_argument("--new-tokens", type=int, nargs="+", default=[128, 512])
	parser.add_argument("--repeats", type=int, default=2)
	parser.add_argument("--out", default=None, help="write results as JSON")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
	cfg = ModelConfig(**ckpt["model_cfg"])
	model = CausalLM(cfg)
	model.load_state_dict(ckpt["model"])
	model.to(device).eval()
	hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else f"CPU, {torch.get_num_threads()} threads"
	print(f"{hardware} | fp32 | batch 1 | prompt {args.prompt_len} tokens | greedy decoding\n")

	torch.manual_seed(0)
	prompt = torch.randint(0, cfg.vocab_size, (1, args.prompt_len), device=device)
	results = {"hardware": hardware, "speed": [], "kv_cache": {}}

	print("| new tokens | without KV cache | with KV cache | speed-up |")
	print("|---|---|---|---|")
	for n in args.new_tokens:
		n = min(n, cfg.max_seq_len - args.prompt_len)
		slow_ids, slow = timed(lambda: generate_without_cache(model, prompt, n), device, args.repeats)
		fast_ids, fast = timed(lambda: generate_with_cache(model, prompt, n), device, args.repeats)
		assert torch.equal(slow_ids, fast_ids), "the KV cache changed the generated tokens"
		row = {"new_tokens": n, "no_cache_tok_s": n / slow, "cache_tok_s": n / fast, "speedup": slow / fast}
		results["speed"].append(row)
		print(f"| {n} | {row['no_cache_tok_s']:.1f} tok/s | {row['cache_tok_s']:.1f} tok/s | **{row['speedup']:.1f}x** |")
	print("(both produce identical tokens)\n")

	seq = cfg.max_seq_len
	gqa = kv_cache_bytes(cfg, cfg.n_kv_heads, seq, 2)
	mha = kv_cache_bytes(cfg, cfg.n_heads, seq, 2)
	_, cache, _ = model(prompt, use_cache=True)
	measured = sum(t.numel() * t.element_size() for kv in cache for t in kv)
	assert measured == kv_cache_bytes(cfg, cfg.n_kv_heads, seq, cache[0][0].element_size())
	results["kv_cache"] = {"seq_len": seq, "gqa_bf16_bytes": gqa, "mha_bf16_bytes": mha}
	print(f"KV cache per sequence at {seq} tokens (bf16): GQA {cfg.n_kv_heads} KV heads {gqa / 2**20:.0f} MiB "
		  f"vs MHA {cfg.n_heads} heads {mha / 2**20:.0f} MiB -> {mha / gqa:.0f}x smaller "
		  f"(measured allocation matches the formula)")

	if args.hf:
		from transformers import AutoModelForCausalLM
		hf_model = AutoModelForCausalLM.from_pretrained(args.hf).to(device).eval()
		n = min(args.new_tokens[-1], cfg.max_seq_len - args.prompt_len)
		run = lambda: hf_model.generate(prompt, max_new_tokens=n, do_sample=False, attention_mask=torch.ones_like(prompt),
										pad_token_id=0, eos_token_id=None)
		hf_ids, hf_time = timed(lambda: torch.inference_mode()(run)(), device, args.repeats)
		ours = results["speed"][-1]
		same = torch.equal(hf_ids, generate_with_cache(model, prompt, n))
		results["hf"] = {"new_tokens": n, "tok_s": n / hf_time, "identical_tokens": same}
		print(f"\nsame weights in transformers' LlamaForCausalLM, {n} tokens with its KV cache: {n / hf_time:.1f} tok/s "
			  f"(ours: {ours['cache_tok_s']:.1f} tok/s) | identical tokens: {same}")

	if args.out:
		with open(args.out, "w") as fd:
			json.dump(results, fd, indent=2)


if __name__ == "__main__":
	main()
