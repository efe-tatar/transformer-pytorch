import torch


def configure_optimizer(model, weight_decay, lr, betas=(0.9, 0.95), device_type="cpu"):
	# TODO(step 7, yours): return a torch.optim.AdamW with two param groups
	#  - weight_decay applied to the matmul weights (every nn.Linear weight)
	#  - weight_decay = 0 for everything else: norm weights/biases, embeddings
	#    (careful: lm_head.weight IS token_embedding.weight -- weight tying --
	#    each parameter must appear in exactly one group)
	#  - fused=True when device_type == "cuda" (one kernel for the whole update)
	# tests/test_optim.py tells you when it's right.
	raise NotImplementedError("step 7: implement configure_optimizer")
	torch.optim.AdamW(
		[
			{
				"params": [m for m in model.modules() if isinstance(m, nn)],
				"weight_decay": weight_decay
			},
			{
				"params": 1,
				"weight_decay": 0
			}
		]
	)
