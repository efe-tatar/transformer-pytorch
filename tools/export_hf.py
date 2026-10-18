"""
Export a checkpoint + tokenizer to the Hugging Face format (LlamaForCausalLM + a fast BPE
tokenizer), so the model loads with plain `transformers`:

    tok = AutoTokenizer.from_pretrained("exported/base_100m")
    model = AutoModelForCausalLM.from_pretrained("exported/base_100m")

Usage:
    python tools/export_hf.py --checkpoint checkpoints/base_100m/best.pt \
        --tokenizer data/fineweb_edu_3b/tokenizer.json --out exported/base_100m

The export is checked at the end: same logits as our CausalLM, same token ids as our BPETokenizer.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from config import ModelConfig
from network.causal_lm import CausalLM
from tokenizer.bpe import BPETokenizer


# ---------------------------------------------------------------- model

def rope_interleaved_to_halves(weight, n_heads, head_dim):
	"""Our RoPE rotates neighbouring dims (0,1), (2,3), ... while HF Llama rotates dim i with
	dim i + head_dim/2. Reorder each head's output rows so that HF's layout computes exactly
	the same rotation (q.k dot products are unchanged by a shared permutation of q and k)."""
	dim_in = weight.shape[1]
	return weight.view(n_heads, head_dim // 2, 2, dim_in).transpose(1, 2).reshape(n_heads * head_dim, dim_in)


def to_hf_state_dict(model):
	cfg = model.cfg
	assert cfg.norm == "rms" and cfg.mlp == "swiglu" and cfg.pos == "rope", "only the Llama-style variant maps to LlamaForCausalLM"
	q_dim, kv_dim = cfg.dim, cfg.n_kv_heads * cfg.head_dim
	sd = {"model.embed_tokens.weight": model.token_embedding.weight,
		  "model.norm.weight": model.post_trans_norm_layer.weight}
	for i, block in enumerate(model.blocks):
		p = f"model.layers.{i}."
		q, k, v = torch.split(block.W_QKV.weight, [q_dim, kv_dim, kv_dim], dim=0)
		sd[p + "self_attn.q_proj.weight"] = rope_interleaved_to_halves(q, cfg.n_heads, cfg.head_dim)
		sd[p + "self_attn.k_proj.weight"] = rope_interleaved_to_halves(k, cfg.n_kv_heads, cfg.head_dim)
		sd[p + "self_attn.v_proj.weight"] = v
		sd[p + "self_attn.o_proj.weight"] = block.W_o.weight
		# swiglu: linear_3(linear_1(x) * silu(linear_2(x)))  ->  down(up(x) * silu(gate(x)))
		sd[p + "mlp.gate_proj.weight"] = block.swiglu_linear_2.weight
		sd[p + "mlp.up_proj.weight"] = block.swiglu_linear_1.weight
		sd[p + "mlp.down_proj.weight"] = block.swiglu_linear_3.weight
		sd[p + "input_layernorm.weight"] = block.layer_norm_1.weight
		sd[p + "post_attention_layernorm.weight"] = block.layer_norm_2.weight
	return {k: v.detach().clone().contiguous() for k, v in sd.items()}


def to_hf_config(cfg, special_tokens):
	from transformers import LlamaConfig
	eot = special_tokens["<|endoftext|>"]
	return LlamaConfig(
		vocab_size=cfg.vocab_size,
		hidden_size=cfg.dim,
		intermediate_size=2 * 4 * cfg.dim // 3,  # same formula as TransformerBlock
		num_hidden_layers=cfg.n_layers,
		num_attention_heads=cfg.n_heads,
		num_key_value_heads=cfg.n_kv_heads,
		head_dim=cfg.head_dim,
		max_position_embeddings=cfg.max_seq_len,
		# torch.nn.RMSNorm(eps=None) uses the float32 machine epsilon
		rms_norm_eps=torch.finfo(torch.float32).eps,
		rope_theta=cfg.rope_base,
		hidden_act="silu",
		attention_bias=False,
		mlp_bias=False,
		tie_word_embeddings=True,
		bos_token_id=eot,  # every training document started with <|endoftext|>
		eos_token_id=eot,
		torch_dtype="float32",
	)


# ---------------------------------------------------------------- tokenizer

def bytes_to_unicode():
	"""GPT-2's reversible byte -> printable unicode character map, used by HF byte-level BPE."""
	bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
	cs = bs[:]
	n = 0
	for b in range(256):
		if b not in bs:
			bs.append(b)
			cs.append(256 + n)
			n += 1
	return dict(zip(bs, map(chr, cs)))


def to_hf_tokenizer(bpe):
	from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
	from transformers import PreTrainedTokenizerFast

	byte_map = bytes_to_unicode()
	as_str = lambda b: "".join(byte_map[x] for x in b)
	vocab = {as_str(b): i for i, b in bpe.vocab.items()}
	# HF applies merges by rank = position in this list; ours by merged-token id. same order.
	merges = [(as_str(bpe.vocab[a]), as_str(bpe.vocab[b])) for (a, b), _ in sorted(bpe.merges.items(), key=lambda kv: kv[1])]

	tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))
	# use_regex=True applies GPT-2's pre-tokenization pattern, the same one tokenizer/bpe.py uses
	tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
	tok.decoder = decoders.ByteLevel()
	for name, idx in sorted(bpe.special_tokens.items(), key=lambda kv: kv[1]):
		assert tok.get_vocab_size() == idx, "special token ids must follow the merges"
		tok.add_special_tokens([AddedToken(name, special=True, normalized=False)])
	eot = "<|endoftext|>"
	return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=eot, eos_token=eot, unk_token=None,
								   model_max_length=None, clean_up_tokenization_spaces=False)


# ---------------------------------------------------------------- export + check

SAMPLE_TEXTS = [
	"Photosynthesis is the process by which plants turn light into chemical energy.",
	"  Whitespace,\ttabs\nand newlines   — plus unicode: café, naïve, 日本語, emoji 🙂, numbers 1234567.",
	"It's, they're, we'll, I'd: contractions!",
]


def export(checkpoint, tokenizer_path, out_dir, repo_id=None):
	from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

	ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
	model = CausalLM(ModelConfig(**ckpt["model_cfg"]))
	model.load_state_dict(ckpt["model"])
	model.eval()
	bpe = BPETokenizer(0)
	bpe.load(tokenizer_path)

	hf_model = LlamaForCausalLM(to_hf_config(model.cfg, bpe.special_tokens))
	missing, unexpected = hf_model.load_state_dict(to_hf_state_dict(model), strict=False)
	assert not unexpected, unexpected
	assert set(missing) <= {"lm_head.weight"}, missing  # tied to embed_tokens
	hf_model.tie_weights()
	hf_model.save_pretrained(out_dir)
	to_hf_tokenizer(bpe).save_pretrained(out_dir)
	if repo_id:
		card = open(os.path.join(os.path.dirname(__file__), "model_card.md"), encoding="utf-8").read()
		with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fd:
			fd.write(card.replace("REPO_ID", repo_id))
	print(f"exported to {out_dir}")

	# reload from disk exactly like a user would, then compare against our implementation
	hf_model = AutoModelForCausalLM.from_pretrained(out_dir).eval()
	hf_tok = AutoTokenizer.from_pretrained(out_dir)
	return check(model, bpe, hf_model, hf_tok)


@torch.no_grad()
def check(model, bpe, hf_model, hf_tok):
	for text in SAMPLE_TEXTS:
		ours = bpe.encode(text, allowed_special="none")
		theirs = hf_tok(text, add_special_tokens=False)["input_ids"]
		assert ours == theirs, f"tokenizer mismatch on {text!r}:\n{ours}\n{theirs}"
		assert hf_tok.decode(theirs) == text
	eot = bpe.special_tokens["<|endoftext|>"]
	assert hf_tok("a<|endoftext|>b", add_special_tokens=False)["input_ids"] == bpe.encode("a<|endoftext|>b")

	torch.manual_seed(0)
	tokens = torch.randint(0, model.cfg.vocab_size, (2, min(64, model.cfg.max_seq_len)))
	max_diff = (model(tokens) - hf_model(tokens).logits).abs().max().item()
	assert max_diff < 1e-3, f"logits differ by {max_diff}"

	prompt = torch.tensor([[eot] + bpe.encode(SAMPLE_TEXTS[0][:30])])
	n_new = min(20, model.cfg.max_seq_len - prompt.shape[1])
	ours = model.generate(prompt, n_new, temperature=0)
	theirs = hf_model.generate(prompt, max_new_tokens=n_new, do_sample=False, attention_mask=torch.ones_like(prompt),
							   pad_token_id=eot, eos_token_id=None)
	assert torch.equal(ours, theirs), f"greedy generation differs:\n{ours}\n{theirs}"
	print(f"check passed: identical token ids, max logit difference {max_diff:.2e}, identical greedy generation")
	return max_diff


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", default="checkpoints/base_100m/best.pt")
	parser.add_argument("--tokenizer", default="data/fineweb_edu_3b/tokenizer.json")
	parser.add_argument("--out", default="exported/base_100m")
	parser.add_argument("--repo-id", default=None, help="Hub repo the export is meant for; adds the model card")
	args = parser.parse_args()
	export(args.checkpoint, args.tokenizer, args.out, args.repo_id)


if __name__ == "__main__":
	main()
