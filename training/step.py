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
	# a tensor on the device: adding to it doesn't force a GPU -> CPU sync every micro-step
	total_loss = torch.zeros((), device=device)

	for micro_step in range(accum_steps):
		if micro_step == accum_steps-1:
			set_grad_sync(model, True)
		else:
			set_grad_sync(model, False)

		x, y = next_batch()
		# .to() returns a new tensor, it doesn't move x in place
		x = x.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True)

		with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
			logits = model(x)

		# cross_entropy wants (N, vocab) predictions vs (N,) targets: flatten batch and time.
		# .float(): softmax in fp32 even when the logits came out of autocast in bf16
		loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
		# backward() *adds* into .grad, so accum_steps backwards of the mean loss would sum to
		# accum_steps x the big-batch gradient. dividing makes the sum equal the mean.
		loss = loss / accum_steps

		# detach: only the value is needed for logging, not the graph behind it
		total_loss += loss.detach()

		if scaler is not None:
			scaler.scale(loss).backward()
		else:
			loss.backward()

	if scaler is not None:
		# gradients are still multiplied by the fp16 loss scale: undo that before clipping
		scaler.unscale_(optimizer)
	# returns the norm *before* clipping, which is what's worth logging
	grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

	if scaler is not None:
		# skips the step if the fp16 gradients overflowed to inf/nan, and adjusts the scale
		scaler.step(optimizer)
		scaler.update()
	else:
		optimizer.step()

	# set_to_none: frees the gradient memory instead of filling it with zeros
	optimizer.zero_grad(set_to_none=True)

	return total_loss, grad_norm
