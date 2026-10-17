import torch
import torch.nn.functional as F


def set_grad_sync(model, enabled):
	# with DDP every backward() all-reduces gradients across GPUs. During gradient
	# accumulation only the last micro-step needs to; this turns it off for the others.
	# no-op for a plain (non-DDP) model.
	if isinstance(model, torch.nn.parallel.DistributedDataParallel):
		model.require_backward_grad_sync = enabled


def train_step(model, next_batch, optimizer, accum_steps, device, amp_dtype=None, grad_clip=1.0, scaler=None):
	"""One optimizer step over `accum_steps` micro-batches.

	model:       the (maybe compiled / DDP-wrapped) CausalLM, already in train mode
	next_batch:  callable -> (x, y) cpu LongTensors of shape (B, T)
	optimizer:   learning rate already set by the caller
	device:      torch.device to move x, y to
	amp_dtype:   torch.bfloat16 / torch.float16 for autocast, None for plain fp32
	scaler:      a torch.amp.GradScaler when amp_dtype is float16, else None

	returns (loss, grad_norm): loss = mean loss over all micro-batches (a detached
	tensor, don't .item() it per micro-step: that syncs the GPU), grad_norm = the
	gradient norm before clipping.
	"""
	# TODO(step 7, yours). Per micro-step:
	#  - set_grad_sync(model, <only on the last micro-step>)
	#  - forward under torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
	#  - cross-entropy on (B*T, vocab) logits vs (B*T,) targets
	#  - why must the loss be divided by accum_steps before backward()?
	#  - fp16: backward on scaler.scale(loss); unscale_ the optimizer before clipping,
	#    then scaler.step + scaler.update instead of optimizer.step
	# then clip, step, zero_grad(set_to_none=True).
	# tests/test_step.py tells you when it's right.
	raise NotImplementedError("step 7: implement train_step")
