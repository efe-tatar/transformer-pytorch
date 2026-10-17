import torch
import math


def build_norm(cfg):
	if cfg.norm == "rms":
		return torch.nn.RMSNorm(cfg.dim)
	return torch.nn.LayerNorm(cfg.dim)


def build_rope_cache(head_dim, max_seq_len, base=10000.0):
	# rope precomputations, done once per model and shared by every block
	pairs = torch.arange(0, head_dim, 2)
	# axes are paired consecutively.
	# frequency gets smaller as the pair index increases
	# having multiple rotation speeds let's us avoid having identical rotations for different tokens.
	inv_frequency = base ** (-pairs / head_dim) # (head_dim / 2, )
	positions = torch.arange(max_seq_len) # (max_seq_len,)

	# positions[:, None] -> (max_seq_len, 1)
	# inv_frequency[None, :] -> (1, head_dim/2)
	# this notation adds an additional dimension
	angles = positions[:, None] * inv_frequency[None, :] # build a matrix out of these
	# so we get (max_seq_len, head_dim / 2)

	# 2d rotation matrix: [[cos, -sin], [sin, cos]]
	return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
	# ... takes all previous dimensions
	# ::2 is every second element and 1::2 is every second element starting from 1
	# same as x[:, :, :, ::2]
	# I don't really know why we choose to pair adjacent dims.
	# Couldn't find an anwser.
	# Wouldn't splitting the dimensions into two be easier because
	# the elements would be contigious in memory ? idk
	# oh wait askip that's what llama does ?
	x_even = x[..., ::2]
	x_odd = x[..., 1::2]
	x_even_rot = x_even * cos - x_odd * sin
	x_odd_rot = x_even * sin + x_odd * cos

	# oh wait turns out this breaks autograd :(
	# ok so what breaks autograd is directly editing the Q matrix, quite expected actually hmmm
	# just create a new one
	out = torch.empty_like(x)
	out[..., ::2] = x_even_rot
	out[..., 1::2] = x_odd_rot
	return out


class TransformerBlock(torch.nn.Module):

	def __init__(self, cfg):
		super().__init__()

		self.cfg = cfg
		self.embedding_dimension = cfg.dim
		self.nb_heads = cfg.n_heads
		self.head_dim = cfg.head_dim
		self.head_dim_sqrt = math.sqrt(self.head_dim)
		self.max_seq_len = cfg.max_seq_len
		self.nb_kv_heads = cfg.n_kv_heads
		# divisibility of dim / heads / kv heads is asserted in ModelConfig

		self.layer_norm_1 = build_norm(cfg)

		# added GQA now:
		self.W_QKV = torch.nn.Linear(
			self.embedding_dimension,
			self.embedding_dimension + 2 * self.nb_kv_heads * self.head_dim,
			bias=False
		)

		# how to cache the causal mask: build it once, slice it in forward.
		# persistent=False keeps it out of the state_dict (it's not a weight)
		self.register_buffer(
			"mask",
			torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1).bool(),
			persistent=False
		)

		self.attention_weights_dropout = torch.nn.Dropout(cfg.dropout)

		self.attention_residual_dropout = torch.nn.Dropout(cfg.dropout)

		self.W_o = torch.nn.Linear(self.embedding_dimension, self.embedding_dimension, bias=False)

		self.layer_norm_2 = build_norm(cfg)

		if cfg.mlp == "swiglu":
			expansion_dim = 2 * 4 * self.embedding_dimension // 3
			self.swiglu_linear_1 = torch.nn.Linear(self.embedding_dimension, expansion_dim, bias=False)
			self.swiglu_linear_2 = torch.nn.Linear(self.embedding_dimension, expansion_dim, bias=False)
			self.swiglu_swish = torch.nn.SiLU()
			self.swiglu_linear_3 = torch.nn.Linear(expansion_dim, self.embedding_dimension, bias=False)
		else:
			self.mlp_1 = torch.nn.Linear(self.embedding_dimension, 4 * self.embedding_dimension, bias=False)
			self.mlp_2 = torch.nn.GELU()
			self.mlp_3 = torch.nn.Linear(4 * self.embedding_dimension, self.embedding_dimension, bias=False)

		self.post_mlp_dropout = torch.nn.Dropout(cfg.dropout)

	def output_projections(self):
		# the linears that write into the residual stream (used for scaled init)
		return [self.W_o, self.swiglu_linear_3 if self.cfg.mlp == "swiglu" else self.mlp_3]

	def forward(self, x, rope=None, use_cache=False, kv_cache=None, cached_seq_len=0):

		batch_size = x.shape[0]
		seq_len = x.shape[1]

		residual = x

		x = self.layer_norm_1(x)

		QKV = self.W_QKV(x)
		Q, K, V = torch.split(
			QKV,
			[self.embedding_dimension, self.nb_kv_heads * self.head_dim, self.nb_kv_heads * self.head_dim],
			-1
		)

		Q = Q.view(batch_size, seq_len, self.nb_heads, self.head_dim)
		K = K.view(batch_size, seq_len, self.nb_kv_heads, self.head_dim)
		V = V.view(batch_size, seq_len, self.nb_kv_heads, self.head_dim)

		Q = Q.transpose(1, 2) # -> batch_size, head, seqlen, head_dim
		K = K.transpose(1, 2)
		V = V.transpose(1, 2)

		if self.cfg.pos == "rope":
			assert rope is not None, "pos='rope' needs the (cos, sin) tables from build_rope_cache"
			cos_cache, sin_cache = rope
			position_ids = torch.arange(cached_seq_len, cached_seq_len + seq_len, device=Q.device)
			cos = cos_cache[position_ids][None, None, :, :] # to add dimensions for batch size and number of heads
			sin = sin_cache[position_ids][None, None, :, :]
			Q = apply_rope(Q, cos, sin)
			K = apply_rope(K, cos, sin)

		if use_cache is True:
			if kv_cache is not None:
				K_cache, V_cache = kv_cache

			else:
				# dtype=K.dtype so the cache is bf16 when running under autocast
				K_cache = torch.zeros((batch_size, self.nb_kv_heads, self.max_seq_len, self.head_dim), device=x.device, dtype=K.dtype)
				V_cache = torch.zeros((batch_size, self.nb_kv_heads, self.max_seq_len, self.head_dim), device=x.device, dtype=V.dtype)

			K_cache[:, :, cached_seq_len:cached_seq_len+seq_len, :] = K
			V_cache[:, :, cached_seq_len:cached_seq_len+seq_len, :] = V

		past_len = cached_seq_len if (kv_cache is not None and use_cache is True) else 0
		total_len = past_len + seq_len

		effective_K = K_cache[:, :, :total_len, :] if use_cache else K
		effective_V = V_cache[:, :, :total_len, :] if use_cache else V

		if self.cfg.attn_impl == "sdpa":
			attention = self._sdpa_attention(Q, effective_K, effective_V, past_len, total_len)
		else:
			attention = self._manual_attention(Q, effective_K, effective_V, past_len, total_len)

		attention = attention.transpose(1, 2)
		attention = attention.reshape(batch_size, seq_len, self.embedding_dimension)

		attention = self.W_o(attention)
		attention = self.attention_residual_dropout(attention)

		x = residual + attention

		residual_2 = x

		x = self.layer_norm_2(x)

		if self.cfg.mlp == "swiglu":
			swiglu_1 = self.swiglu_linear_1(x)
			swiglu_2 = self.swiglu_linear_2(x)
			silu = self.swiglu_swish(swiglu_2)
			element_wise_mult = swiglu_1 * silu
			x = self.swiglu_linear_3(element_wise_mult)
		else:
			x = self.mlp_1(x)
			x = self.mlp_2(x)
			x = self.mlp_3(x)

		x = self.post_mlp_dropout(x)

		x = residual_2 + x

		if use_cache:
			return x, (K_cache, V_cache), cached_seq_len + seq_len
		else:
			return x

	def _manual_attention(self, Q, K, V, past_len, total_len):
		# reference implementation. Q: (batch, nb_heads, seq_len, head_dim)
		# K, V: (batch, nb_kv_heads, total_len, head_dim)
		batch_size, _, seq_len, _ = Q.shape

		# expands rereferences the same address, it doesn't copy into memory.
		K = K[:, :, None, :, :] # K -> (batch, nb_k_heads, 1, total_len, head_dim)
		K = K.expand(-1, -1, self.nb_heads // self.nb_kv_heads, -1, -1)
		K = K.reshape(batch_size, self.nb_heads, total_len, self.head_dim)

		V = V[:, :, None, :, :]
		V = V.expand(-1, -1, self.nb_heads // self.nb_kv_heads, -1, -1)
		V = V.reshape(batch_size, self.nb_heads, total_len, self.head_dim)

		# can't do this because of third dimension: batch
		# scores = (Q @ K.T) / math.sqrt(self.embedding_dimension)
		scores = (Q @ K.transpose(-2, -1)) / self.head_dim_sqrt

		# mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
		# this allocated an unnecessarily big tensor of size (batch size, nb_head, seqlen, seqlen)
		# whereas we can have (seqlen, seqlen) and let pytorch broadcast,
		# and slice it out of the cached mask (look at init)
		mask = self.mask[past_len:past_len + seq_len, :total_len]

		scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

		attention_weights = torch.softmax(scores, dim=-1)
		attention_weights = self.attention_weights_dropout(attention_weights)
		return attention_weights @ V

	def _sdpa_attention(self, Q, K, V, past_len, total_len):
		# TODO(step 4, yours): same contract as _manual_attention, using
		# torch.nn.functional.scaled_dot_product_attention. Hints:
		#  - enable_gqa=True lets it consume the nb_kv_heads K/V directly (no expand)
		#  - no cache (past_len == 0 and seq_len == total_len): is_causal=True
		#  - with a cache: pass attn_mask instead -- careful, SDPA's boolean mask
		#    means True = "may attend", the opposite of self.mask
		#  - dropout_p=self.cfg.dropout only while self.training
		# tests/test_sdpa.py tells you when it's right.
		raise NotImplementedError("step 4: implement _sdpa_attention")
