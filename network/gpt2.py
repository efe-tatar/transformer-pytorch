import numpy as np
import torch
import math
from network.transformer import Transformer

class GPT_2(torch.nn.Module):

	def __init__(self, embedding_dimension, nb_trans_blocks, vocabulary_size, max_seq_len, nb_heads):
		super().__init__()

		self.embedding_dimension = embedding_dimension
		self.vocabulary_size = vocabulary_size
		self.max_seq_len = max_seq_len

		self.token_embedding = torch.nn.Embedding(vocabulary_size, embedding_dimension)
		self.position_embedding = torch.nn.Embedding(max_seq_len, embedding_dimension)

		self.embedding_dropout = torch.nn.Dropout(0.1)

		self.blocks = torch.nn.ModuleList([Transformer(embedding_dimension, nb_heads, max_seq_len) for i in range(nb_trans_blocks)])

		self.post_trans_norm_layer = torch.nn.LayerNorm(embedding_dimension)

		self.lm_head = torch.nn.Linear(embedding_dimension, vocabulary_size, bias=False)
		# this is called weight tying.
		# since we want to convert from embeddings to vocab,
		# we use the transposed weight of vocab to embeddings
		# and since (weirdly) pytorch stores the weights in transposed form,
		# we can directly link the two
		self.lm_head.weight = self.token_embedding.weight
	
	def forward(self, tokens, use_cache=False, kv_cache_list=None, cached_seq_len=0):

		# don't do this
		# token_embeddings = torch.tensor([embedding_matrix[i] for i in tokens])
		# torch.tensor is for raw data whereas embedding_matrix[i] return a pytorch tensor.
		# you can alternatively do this:
		# token_embeddings = torch.stack([embedding_matrix[i] for i in tokens])
		# the following would work if token_embedding were a tensor 
		# token_embeddings = self.token_embedding[tokens]
		# the fix:
		token_embeddings = self.token_embedding(tokens)

		# the following line doesnt take into account kv cache.
		# position_ids = torch.arange(tokens.shape[1], device=tokens.device)
		position_ids = torch.arange(cached_seq_len, cached_seq_len + tokens.shape[1], device=tokens.device)
		position_embeddings = self.position_embedding(position_ids)
		# added RoPE to transformers
		# x = token_embeddings + position_embeddings
		x = self.embedding_dropout(x)

		if use_cache and kv_cache_list is None:
			kv_cache_list = [None for j in range(len(self.blocks))]

		for i, transformer in enumerate(self.blocks):
			if use_cache:
				x, kv_cache, new_cached_seq_len = transformer(x, True, kv_cache_list[i], cached_seq_len)
				kv_cache_list[i] = kv_cache
			else:
				x = transformer(x)
		
		x = self.post_trans_norm_layer(x)

		logits = self.lm_head(x)

		if use_cache:
			return logits, kv_cache_list, new_cached_seq_len
		else:
			return logits




