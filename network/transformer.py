import numpy as np
import torch
import math

class Transformer(torch.nn.Module):

	def __init__(self, embedding_dimension, nb_heads, nb_kv_heads, max_seq_len):
		super().__init__()

		self.embedding_dimension = embedding_dimension
		self.nb_heads = nb_heads
		self.head_dim = embedding_dimension // nb_heads
		self.head_dim_squared = math.sqrt(self.head_dim)
		self.max_seq_len = max_seq_len
		# self.nb_query_heads = nb_heads
		self.nb_kv_heads = nb_kv_heads

		# if embedding_dimension % nb_heads != 0:
		#	print("embedding_dimension % nb_heads != 0")
		assert embedding_dimension % nb_heads == 0, "embedding_dimension % nb_heads != 0"

		assert nb_heads % nb_kv_heads == 0, "nb_heads % nb_kv_heads != 0"

		# self.layer_norm_1 = torch.nn.LayerNorm(embedding_dimension)
		self.layer_norm_1 = torch.nn.RMSNorm(embedding_dimension)

		# self.W_QKV = torch.nn.Linear(embedding_dimension, 3*embedding_dimension, bias=False)
		# added GQA now:
		self.W_QKV = torch.nn.Linear(
			embedding_dimension,
			embedding_dimension + 2 * nb_kv_heads * self.head_dim,
			bias=False
		)

		# rope precomputations
		pairs = torch.arange(0, self.head_dim, 2)
		base = 10000
		# axes are paired consecutively.
		# frequency gets smaller as the pair index increases
		# having multiple rotation speeds let's us avoid having identical rotations for different tokens.
		inv_frequency = base ** (-pairs / self.head_dim) # (head_dim / 2, )
		positions = torch.arange(max_seq_len) # (max_seq_len,)

		# positions[:, None] -> (max_seq_len, 1)
		# inv_frequency[None, :] -> (1, head_dim/2)
		# this notation adds an additional dimension
		angles = positions[:, None] * inv_frequency[None, :] # build a matrix out of these
		# so we get (max_seq_len, head_dim / 2)

		# 2d rotation matrix: [[cos, -sin], [sin, cos]]
		# compute and cache
		cos_angles = angles.cos()
		sin_angles = angles.sin()
		self.register_buffer("angle_cos_cache", cos_angles)
		self.register_buffer("angle_sin_cache", sin_angles)



		self.register_buffer(
			"mask",
			torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
		)

		self.attention_weights_dropout = torch.nn.Dropout(0.1)

		self.attention_residual_dropout = torch.nn.Dropout(0.1)

		self.W_o = torch.nn.Linear(embedding_dimension, embedding_dimension, bias=False)

		# self.layer_norm_2 = torch.nn.LayerNorm(embedding_dimension)
		self.layer_norm_2 = torch.nn.RMSNorm(embedding_dimension)

		#mlp
		# self.mlp_1 = torch.nn.Linear(embedding_dimension, 4 * embedding_dimension, bias=False)
		# self.mlp_2 = torch.nn.GELU()
		# self.mlp_3 = torch.nn.Linear(4 * embedding_dimension, embedding_dimension, bias=False)

		# swiglu
		expansion_dim = 2 * 4 * embedding_dimension // 3
		self.swiglu_linear_1 = torch.nn.Linear(embedding_dimension, expansion_dim, bias=False)
		self.swiglu_linear_2 = torch.nn.Linear(embedding_dimension, expansion_dim, bias=False)
		self.swiglu_swish = torch.nn.SiLU()
		self.swiglu_linear_3 = torch.nn.Linear(expansion_dim, embedding_dimension, bias=False)

		self.post_mlp_dropout = torch.nn.Dropout(0.1)
	
	def forward(self, x, use_cache=False, kv_cache=None, cached_seq_len=0):

		batch_size = x.shape[0]
		seq_len = x.shape[1]

		residual = x

		x = self.layer_norm_1(x)

		QKV = self.W_QKV(x)
		# Q, K, V = QKV.chunk(3, dim=-1)
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

		position_ids = torch.arange(cached_seq_len, cached_seq_len + seq_len, device=Q.device)
		cos = self.angle_cos_cache[position_ids][None, None, :, :] # to add dimensions for batch size and number of heads
		sin = self.angle_sin_cache[position_ids][None, None, :, :]

		# ... takes all previous dimensions
		# ::2 is every second element and 1::2 is every second element starting from 1
		# same as Q[:, :, :, ::2]
		# I don't really know why we choose to pair adjacent dims.
		# Couldn't find an anwser.
		# Wouldn't splitting the dimensions into two be easier because
		# the elements would be contigious in memory ? idk
		# oh wait askip that's what llama does ?
		Q_even = Q[..., ::2]
		Q_odd = Q[..., 1::2]
		Q_even_rot = Q_even * cos - Q_odd * sin
		Q_odd_rot = Q_even * sin + Q_odd * cos

		# oh wait turns out this breaks autograd :(
		# ok so what breaks autograd is directly editing the Q matrix, quite expected actually hmmm
		# just create a new one
		Q = torch.empty_like(Q)
		Q[..., ::2] = Q_even_rot
		Q[..., 1::2] = Q_odd_rot

		K_even = K[..., ::2]
		K_odd = K[..., 1::2]
		K_even_rot = K_even * cos - K_odd * sin
		K_odd_rot = K_even * sin + K_odd * cos

		K = torch.empty_like(K)
		K[..., ::2] = K_even_rot
		K[..., 1::2] = K_odd_rot

		if use_cache is True:
			if kv_cache is not None:
				K_cache, V_cache = kv_cache

			else:
				K_cache = torch.zeros((batch_size, self.nb_kv_heads, self.max_seq_len, self.head_dim), device=x.device)
				V_cache = torch.zeros((batch_size, self.nb_kv_heads, self.max_seq_len, self.head_dim), device=x.device)
			
			K_cache[:, :, cached_seq_len:cached_seq_len+seq_len, :] = K
			V_cache[:, :, cached_seq_len:cached_seq_len+seq_len, :] = V
		else:
			pass

		past_len = cached_seq_len if (kv_cache is not None and use_cache is True) else 0
		total_len = past_len + seq_len

		effective_K = K_cache[:, :, :total_len, :] if use_cache else K
		effective_V = V_cache[:, :, :total_len, :] if use_cache else V

		# K (batch, nb_k_heads, seq_len, head_dim)
		# expands rereferences the same address, it doesn't copy into memory.
		effective_K = effective_K[:, :, None, :, :] # K -> (batch, nb_k_heads, 1, seq_len, head_dim)
		effective_K = effective_K.expand(-1, -1, self.nb_heads // self.nb_kv_heads, -1, -1)
		effective_K = effective_K.reshape(batch_size, self.nb_heads, total_len, self.head_dim)

		effective_V = effective_V[:, :, None, :, :]
		effective_V = effective_V.expand(-1, -1, self.nb_heads // self.nb_kv_heads, -1, -1)
		effective_V = effective_V.reshape(batch_size, self.nb_heads, total_len, self.head_dim)

		# can't do this because of third dimension: batch
		# scores = (Q @ K.T) / math.sqrt(self.embedding_dimension)
		scores = (Q @ effective_K.transpose(-2, -1)) / self.head_dim_squared

		# mask = torch.triu(torch.ones(scores.shape), diagonal=1).bool()
		# turns out torch.ones(scores.shape) will need to be carried over to the gpu
		# mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
		# or mask = torch.triu(torch.ones(scores.shape, device=scores.device), diagonal=1).bool()
		# fixes that
		# mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
		# this thing above allocated an unnecessarily big tensor of size (batch size, nb_head, seqlen, seqlen)
		# whereas we can have (seqlen, seqlen) and let pytorch broadcast:
		# mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
		# how to chache: (look at init)
		# mask = self.mask[:seq_len, :seq_len]
		mask = self.mask[past_len:past_len + seq_len, :total_len]

		scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

		# attention = self.attention_softmax_layer(scores) @ V
		# attention = torch.softmax(scores, dim=-1) @ V
		attention_weights = torch.softmax(scores, dim=-1)
		attention_weights = self.attention_weights_dropout(attention_weights)
		attention = attention_weights @ effective_V

		attention = attention.transpose(1, 2)
		attention = attention.reshape(batch_size, seq_len, self.embedding_dimension)

		attention = self.W_o(attention)
		attention = self.attention_residual_dropout(attention)

		x = residual + attention

		residual_2 = x

		x = self.layer_norm_2(x)

		# x = self.mlp_1(x)
		# x = self.mlp_2(x)
		# x = self.mlp_3(x)
		swiglu_1 = self.swiglu_linear_1(x)
		swiglu_2 = self.swiglu_linear_2(x)
		silu = self.swiglu_swish(swiglu_2)
		element_wise_mult = swiglu_1 * silu
		swiglu = self.swiglu_linear_3(element_wise_mult)

		x = self.post_mlp_dropout(swiglu)

		x = residual_2 + x

		if use_cache:
			return x, (K_cache, V_cache), cached_seq_len + seq_len
		else:
			return x


