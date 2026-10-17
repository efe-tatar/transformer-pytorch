import torch


def configure_optimizer(model, weight_decay, lr, betas=(0.9, 0.95), device_type="cpu"):
	# weight decay pulls weights towards 0 at every step. That's a useful regulariser
	# for the big matmul weights, but harmful elsewhere:
	#  - norm weights are scales initialised at 1, decaying them fights the normalisation
	#  - embeddings are lookup rows, a rare token's row would just shrink to 0 between the
	#    few steps where it gets a gradient
	# so: decay every nn.Linear weight, nothing else.

	# dict keyed by id(): lm_head.weight IS token_embedding.weight (weight tying),
	# so it shows up twice in model.modules() but must only be counted once.
	# it's an embedding first, so it goes in the no-decay group.
	decay = {id(m.weight): m.weight for m in model.modules() if isinstance(m, torch.nn.Linear)}
	decay.pop(id(model.token_embedding.weight), None)

	# model.parameters() already de-duplicates shared tensors
	no_decay = [p for p in model.parameters() if id(p) not in decay]

	return torch.optim.AdamW(
		[
			{"params": list(decay.values()), "weight_decay": weight_decay},
			{"params": no_decay, "weight_decay": 0.0},
		],
		lr=lr,
		# beta2 = 0.95 instead of pytorch's 0.999: the second-moment estimate adapts faster,
		# which makes LLM pretraining more stable against loss spikes (GPT-3, Llama)
		betas=betas,
		# fused: one CUDA kernel updates every parameter, instead of one launch per tensor
		fused=device_type == "cuda",
	)
