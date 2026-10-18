"""
The reverse of tools/export_hf.py: load a Hugging Face export (local folder or Hub repo id)
back into our own CausalLM and BPETokenizer. Needs only safetensors + huggingface_hub, not
transformers, so the API server and the demo run on this project's own inference code.

    model, tokenizer = load_hf("Efe-Tatar/base-100m-fineweb-edu")
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from safetensors.torch import load_file

from config import ModelConfig
from network.causal_lm import CausalLM
from tokenizer.bpe import BPETokenizer
from tools.export_hf import bytes_to_unicode

FILES = ("config.json", "model.safetensors", "tokenizer.json")


def resolve(path_or_repo, token=None):
	if os.path.isdir(path_or_repo):
		return path_or_repo
	from huggingface_hub import snapshot_download
	return snapshot_download(path_or_repo, allow_patterns=list(FILES), token=token)


def rope_halves_to_interleaved(weight, n_heads, head_dim):
	# inverse of export_hf.rope_interleaved_to_halves
	dim_in = weight.shape[1]
	return weight.view(n_heads, 2, head_dim // 2, dim_in).transpose(1, 2).reshape(n_heads * head_dim, dim_in)


def config_from_hf(hf):
	rope_theta = hf.get("rope_parameters", {}).get("rope_theta", hf.get("rope_theta", 10000.0))
	return ModelConfig(dim=hf["hidden_size"], n_layers=hf["num_hidden_layers"], n_heads=hf["num_attention_heads"],
					   n_kv_heads=hf["num_key_value_heads"], vocab_size=hf["vocab_size"],
					   max_seq_len=hf["max_position_embeddings"], rope_base=rope_theta, attn_impl="sdpa")


def from_hf_state_dict(sd, cfg):
	ours = {"token_embedding.weight": sd["model.embed_tokens.weight"],
			"lm_head.weight": sd["model.embed_tokens.weight"],
			"post_trans_norm_layer.weight": sd["model.norm.weight"]}
	for i in range(cfg.n_layers):
		p, b = f"model.layers.{i}.", f"blocks.{i}."
		q = rope_halves_to_interleaved(sd[p + "self_attn.q_proj.weight"], cfg.n_heads, cfg.head_dim)
		k = rope_halves_to_interleaved(sd[p + "self_attn.k_proj.weight"], cfg.n_kv_heads, cfg.head_dim)
		ours[b + "W_QKV.weight"] = torch.cat([q, k, sd[p + "self_attn.v_proj.weight"]], dim=0)
		ours[b + "W_o.weight"] = sd[p + "self_attn.o_proj.weight"]
		ours[b + "swiglu_linear_1.weight"] = sd[p + "mlp.up_proj.weight"]
		ours[b + "swiglu_linear_2.weight"] = sd[p + "mlp.gate_proj.weight"]
		ours[b + "swiglu_linear_3.weight"] = sd[p + "mlp.down_proj.weight"]
		ours[b + "layer_norm_1.weight"] = sd[p + "input_layernorm.weight"]
		ours[b + "layer_norm_2.weight"] = sd[p + "post_attention_layernorm.weight"]
	return ours


def tokenizer_from_hf(tokenizer_json):
	unicode_to_byte = {c: b for b, c in bytes_to_unicode().items()}
	as_bytes = lambda s: bytes(unicode_to_byte[c] for c in s)
	model = tokenizer_json["model"]
	tok = BPETokenizer(0)
	tok.vocab = {i: as_bytes(s) for s, i in model["vocab"].items()}
	ids = {s: i for s, i in model["vocab"].items()}
	merges = [m if isinstance(m, list) else m.split(" ") for m in model["merges"]]
	tok.merges = {(ids[a], ids[b]): ids[a + b] for a, b in merges}
	tok.special_tokens = {t["content"]: t["id"] for t in tokenizer_json["added_tokens"]}
	tok.special_token_names = list(tok.special_tokens)
	tok.max_vocab_size = tok.vocab_size
	tok.encode_cache = {}
	return tok


def load_hf(path_or_repo, device="cpu", token=None):
	folder = resolve(path_or_repo, token)
	with open(os.path.join(folder, "config.json"), encoding="utf-8") as fd:
		cfg = config_from_hf(json.load(fd))
	model = CausalLM(cfg)
	model.load_state_dict(from_hf_state_dict(load_file(os.path.join(folder, "model.safetensors")), cfg))
	with open(os.path.join(folder, "tokenizer.json"), encoding="utf-8") as fd:
		tokenizer = tokenizer_from_hf(json.load(fd))
	return model.to(device).eval(), tokenizer
