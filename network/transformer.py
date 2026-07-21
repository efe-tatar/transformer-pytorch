import numpy as np
import torch
import math

class Transformer(torch.nn.Module):

	def __init__(self, embedding_dimension, nb_heads, max_seq_len):
		super().__init__()

		self.embedding_dimension = embedding_dimension
		self.nb_heads = nb_heads
		self.head_dim = embedding_dimension // nb_heads
		self.max_seq_len = max_seq_len

		# if embedding_dimension % nb_heads != 0:
		#	print("embedding_dimension % nb_heads != 0")
		assert embedding_dimension % nb_heads == 0, "embedding_dimension % nb_heads != 0"

		self.layer_norm_1 = torch.nn.LayerNorm(embedding_dimension)

		self.W_QKV = torch.nn.Linear(embedding_dimension, 3*embedding_dimension, bias=False)

		self.register_buffer(
			"mask",
			torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
		)

		self.attention_weights_dropout = torch.nn.Dropout(0.1)

		self.attention_residual_dropout = torch.nn.Dropout(0.1)

		self.W_o = torch.nn.Linear(embedding_dimension, embedding_dimension, bias=False)

		self.layer_norm_2 = torch.nn.LayerNorm(embedding_dimension)

		#mlp
		self.mlp_1 = torch.nn.Linear(embedding_dimension, 4 * embedding_dimension)
		self.mlp_2 = torch.nn.GELU()
		self.mlp_3 = torch.nn.Linear(4 * embedding_dimension, embedding_dimension)

		self.post_mlp_dropout = torch.nn.Dropout(0.1)
	
	def forward(self, x, use_cache=False, kv_cache=None, cached_seq_len=0):

		batch_size = x.shape[0]
		seq_len = x.shape[1]

		residual = x

		x = self.layer_norm_1(x)

		QKV = self.W_QKV(x)
		Q, K, V = QKV.chunk(3, dim=-1)

		Q = Q.view(batch_size, seq_len, self.nb_heads, self.head_dim)
		K = K.view(batch_size, seq_len, self.nb_heads, self.head_dim)
		V = V.view(batch_size, seq_len, self.nb_heads, self.head_dim)

		Q = Q.transpose(1, 2) # -> batch_size, head, seqlen, head_dim
		K = K.transpose(1, 2)
		V = V.transpose(1, 2)

		if use_cache is True:
			if kv_cache is not None:
				K_cache, V_cache = kv_cache

			else:
				K_cache = torch.zeros((batch_size, self.nb_heads, self.max_seq_len, self.head_dim), device=x.device)
				V_cache = torch.zeros((batch_size, self.nb_heads, self.max_seq_len, self.head_dim), device=x.device)
			
			K_cache[:, :, cached_seq_len:cached_seq_len+seq_len, :] = K
			V_cache[:, :, cached_seq_len:cached_seq_len+seq_len, :] = V
		else:
			pass

		past_len = cached_seq_len if (kv_cache is not None and use_cache is True) else 0
		total_len = past_len + seq_len

		effective_K = K_cache[:, :, :total_len, :] if use_cache else K
		effective_V = V_cache[:, :, :total_len, :] if use_cache else V
		# can't do this because of third dimension: batch
		# scores = (Q @ K.T) / math.sqrt(self.embedding_dimension)
		scores = (Q @ effective_K.transpose(-2, -1)) / math.sqrt(self.head_dim)

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

		x = self.mlp_1(x)
		x = self.mlp_2(x)
		x = self.mlp_3(x)

		x = self.post_mlp_dropout(x)

		x = residual_2 + x

		if use_cache:
			return x, (K_cache, V_cache), cached_seq_len + seq_len
		else:
			return x


