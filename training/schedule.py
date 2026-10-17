import math


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
	# TODO(step 7, yours): learning rate for optimizer step `step` (0-based)
	#  - steps 0 .. warmup_steps-1: linear warmup, reaching max_lr * (step + 1) / warmup_steps
	#  - warmup_steps .. max_steps: cosine decay from max_lr down to min_lr
	#  - after max_steps: stay at min_lr
	# tests/test_schedule.py tells you when it's right.
	if step > max_steps: return min_lr
	elif step < warmup_steps: return max_lr * (step + 1) / warmup_steps
	else:
		return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (step - warmup_steps) / (max_steps - warmup_steps)))
