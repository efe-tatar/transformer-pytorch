import torch
import math
from network.transformer import TransformerBlock, build_norm, build_rope_cache

class CausalLM(torch.nn.Module):

	def __init__(self, cfg):
		super().__init__()

		self.cfg = cfg
		self.embedding_dimension = cfg.dim
		self.vocabulary_size = cfg.vocab_size
		self.max_seq_len = cfg.max_seq_len

		self.token_embedding = torch.nn.Embedding(cfg.vocab_size, cfg.dim)

		if cfg.pos == "learned":
			self.position_embedding = torch.nn.Embedding(cfg.max_seq_len, cfg.dim)
		else:
			# computed once here and handed to every block instead of each block keeping a copy
			cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
			self.register_buffer("rope_cos", cos, persistent=False)
			self.register_buffer("rope_sin", sin, persistent=False)

		self.embedding_dropout = torch.nn.Dropout(cfg.dropout)

		self.blocks = torch.nn.ModuleList([TransformerBlock(cfg) for i in range(cfg.n_layers)])

		self.post_trans_norm_layer = build_norm(cfg)

		self.lm_head = torch.nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
		# this is called weight tying.
		# since we want to convert from embeddings to vocab,
		# we use the transposed weight of vocab to embeddings
		# and since (weirdly) pytorch stores the weights in transposed form,
		# we can directly link the two
		self.lm_head.weight = self.token_embedding.weight

		self.apply(self._init_weights)
		for block in self.blocks:
			for projection in block.output_projections():
				torch.nn.init.normal_(projection.weight, 0, 0.02/math.sqrt(2*self.cfg.n_layers))

	def _init_weights(self, module):
		# TODO(step 3, yours): pytorch's default init is N(0, 1) for nn.Embedding,
		# far too large once tied into lm_head. Port init_weights from
		# examples/train_shakespeare.py here, then additionally scale the
		# residual output projections (block.output_projections()) to
		# std = 0.02 / sqrt(2 * n_layers).
		# tests/test_init.py tells you when it's right.
		if isinstance(module, torch.nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, torch.nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, tokens, use_cache=False, kv_cache_list=None, cached_seq_len=0):

		# don't do this
		# token_embeddings = torch.tensor([embedding_matrix[i] for i in tokens])
		# torch.tensor is for raw data whereas embedding_matrix[i] return a pytorch tensor.
		# you can alternatively do this:
		# token_embeddings = torch.stack([embedding_matrix[i] for i in tokens])
		# the following would work if token_embedding were a tensor
		# token_embeddings = self.token_embedding[tokens]
		# the fix:
		x = self.token_embedding(tokens)

		rope = None
		if self.cfg.pos == "learned":
			# offset by cached_seq_len so positions stay right with the kv cache
			position_ids = torch.arange(cached_seq_len, cached_seq_len + tokens.shape[1], device=tokens.device)
			x = x + self.position_embedding(position_ids)
		else:
			rope = (self.rope_cos, self.rope_sin)

		x = self.embedding_dropout(x)

		if use_cache and kv_cache_list is None:
			kv_cache_list = [None for j in range(len(self.blocks))]

		for i, transformer in enumerate(self.blocks):
			if use_cache:
				x, kv_cache, new_cached_seq_len = transformer(x, rope, True, kv_cache_list[i], cached_seq_len)
				kv_cache_list[i] = kv_cache
			else:
				x = transformer(x, rope)

		x = self.post_trans_norm_layer(x)

		logits = self.lm_head(x)

		if use_cache:
			return logits, kv_cache_list, new_cached_seq_len
		else:
			return logits

	@torch.no_grad()
	def generate(self, ids, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
		# ids: (batch, prompt_len) LongTensor -> (batch, prompt_len + new) LongTensor
		new_tokens = list(self.stream(ids, max_new_tokens, temperature, top_k, stop_token))
		return torch.cat([ids, *new_tokens], dim=1)

	@torch.no_grad()
	def stream(self, ids, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
		# yields each new token as a (batch, 1) LongTensor as soon as it's sampled (for streaming APIs).
		# prefill the whole prompt once, then feed one token at a time through the kv cache.
		# the cache is allocated for max_seq_len positions, so generation stops there.
		was_training = self.training
		self.eval()
		try:
			max_new_tokens = min(max_new_tokens, self.max_seq_len - ids.shape[1])
			if max_new_tokens <= 0:
				return
			logits, kv_cache_list, cached_seq_len = self(ids, use_cache=True)
			for step in range(max_new_tokens):
				next_logits = logits[:, -1, :].float()
				if temperature == 0:
					next_token = next_logits.argmax(dim=-1, keepdim=True)
				else:
					next_logits = next_logits / temperature
					if top_k is not None:
						kth_largest = torch.topk(next_logits, min(top_k, next_logits.shape[-1])).values[:, -1:]
						next_logits = next_logits.masked_fill(next_logits < kth_largest, float("-inf"))
					probs = torch.softmax(next_logits, dim=-1)
					next_token = torch.multinomial(probs, num_samples=1)
				yield next_token
				if (stop_token is not None and (next_token == stop_token).all()) or step == max_new_tokens - 1:
					return
				logits, kv_cache_list, cached_seq_len = self(
					next_token, use_cache=True, kv_cache_list=kv_cache_list, cached_seq_len=cached_seq_len
				)
		finally:
			self.train(was_training)
