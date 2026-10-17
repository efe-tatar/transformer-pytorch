"""
Pretrain CausalLM on the token shards made by data/prepare_fineweb.py.

Usage:
    python train.py --preset debug_cpu
    python train.py --preset gpt2_small --max_steps=2000 --wandb=true
    python train.py --preset gpt2_small --resume          # continue from out_dir/latest.pt
    torchrun --nproc_per_node=2 train.py --preset gpt2_small --dtype=fp16

Any ModelConfig / TrainConfig field (see config.py) can be overridden with --key=value.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console default codec

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import TRAIN_PRESETS, load_preset
from network.causal_lm import CausalLM
from tokenizer.bpe import BPETokenizer
from training.checkpoint import load_checkpoint, save_checkpoint, set_rng_state, unwrap
from training.loader import ShardLoader
from training.optim import configure_optimizer
from training.schedule import get_lr
from training.step import train_step

# dense bf16/fp16 tensor-core peak, for the MFU estimate
PEAK_FLOPS = {"H100": 989e12, "A100": 312e12, "L40": 181e12, "4090": 165e12, "A10": 125e12,
              "V100": 125e12, "L4": 121e12, "3090": 71e12, "T4": 65e12}
AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def setup_device(cfg):
	"""-> (ddp, rank, local_rank, world_size, device). torchrun sets RANK/LOCAL_RANK/WORLD_SIZE."""
	if "RANK" in os.environ:
		rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
		if torch.cuda.is_available():
			dist.init_process_group("nccl")
			device = torch.device(f"cuda:{local_rank}")
			torch.cuda.set_device(device)
		else:
			dist.init_process_group("gloo")
			device = torch.device("cpu")
		return True, rank, local_rank, world, device
	device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
	return False, 0, 0, 1, device


def peak_flops(device):
	if device.type != "cuda":
		return None
	name = torch.cuda.get_device_name(device)
	return next((v for k, v in PEAK_FLOPS.items() if k in name), None)


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--preset", required=True, choices=list(TRAIN_PRESETS))
	parser.add_argument("--resume", action="store_true", help="continue from <out_dir>/latest.pt")
	args, overrides = parser.parse_known_args()
	model_cfg, cfg = load_preset(args.preset, overrides)

	ddp, rank, local_rank, world, device = setup_device(cfg)
	master = rank == 0
	log = print if master else (lambda *a, **k: None)

	torch.manual_seed(cfg.seed)  # same init on every rank (DDP also broadcasts rank 0's weights)
	torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+

	# ---- tokenizer: the model's vocab size comes from it ----
	tokenizer = BPETokenizer(0)
	tokenizer.load(os.path.join(cfg.data_dir, "tokenizer.json"))
	model_cfg = replace(model_cfg, vocab_size=tokenizer.vocab_size)
	eot = tokenizer.special_tokens["<|endoftext|>"]

	# ---- batch geometry ----
	B, T = cfg.micro_batch_size, model_cfg.max_seq_len
	tokens_per_micro_step = B * T * world
	assert cfg.total_batch_tokens % tokens_per_micro_step == 0, (
		f"total_batch_tokens={cfg.total_batch_tokens} must be a multiple of "
		f"micro_batch_size*max_seq_len*world_size={tokens_per_micro_step}"
	)
	accum_steps = cfg.total_batch_tokens // tokens_per_micro_step

	train_loader = ShardLoader(cfg.data_dir, "train", B, T, rank, world)
	val_loader = ShardLoader(cfg.data_dir, "val", B, T, rank, world)

	# ---- precision ----
	amp_dtype = AMP_DTYPES[cfg.dtype]
	if cfg.dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
		raise SystemExit("this GPU has no bf16 (e.g. T4, V100): rerun with --dtype=fp16")
	if cfg.dtype == "fp16" and device.type != "cuda":
		raise SystemExit("--dtype=fp16 needs a GPU; use bf16 or fp32 on CPU")
	scaler = torch.amp.GradScaler("cuda") if cfg.dtype == "fp16" else None

	def autocast():
		return torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None)

	# ---- model + optimizer ----
	model = CausalLM(model_cfg).to(device)
	optimizer = configure_optimizer(model, cfg.weight_decay, cfg.max_lr, (cfg.beta1, cfg.beta2), device.type)
	nb_params = sum(p.numel() for p in model.parameters())

	start_step, best_val, wandb_id = 0, math.inf, None
	latest_path = os.path.join(cfg.out_dir, "latest.pt")
	if args.resume:
		if not os.path.exists(latest_path):
			raise SystemExit(f"--resume: no checkpoint at {latest_path}")
		ckpt = load_checkpoint(latest_path)  # to cpu: RNG states must stay cpu tensors
		if ckpt["model_cfg"] != model_cfg.to_dict():
			raise SystemExit(f"--resume: checkpoint was trained with a different model config:\n{ckpt['model_cfg']}")
		model.load_state_dict(ckpt["model"])
		optimizer.load_state_dict(ckpt["optimizer"])
		train_loader.load_state_dict(ckpt["loader"])
		set_rng_state(ckpt["rng"])
		if scaler is not None and ckpt.get("scaler"):
			scaler.load_state_dict(ckpt["scaler"])
		start_step, best_val, wandb_id = ckpt["step"], ckpt.get("best_val", math.inf), ckpt.get("wandb_id")
		log(f"resumed from {latest_path} at step {start_step}")
		del ckpt

	if cfg.compile:
		model = torch.compile(model)
	if ddp:
		model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
	raw_model = unwrap(model)

	log(f"device {device} x{world} | dtype {cfg.dtype} | compile {cfg.compile}")
	log(f"model {nb_params / 1e6:.1f}M params | {model_cfg}")
	log(f"batch {cfg.total_batch_tokens:,} tokens/step = {B} x {T} x {world} gpu x {accum_steps} accum | "
		f"{cfg.max_steps} steps = {cfg.max_steps * cfg.total_batch_tokens / 1e9:.2f}B tokens")

	# 6N per token for the matmuls forward+backward, plus the attention scores (PaLM appendix B)
	flops_per_token = 6 * nb_params + 12 * model_cfg.n_layers * model_cfg.dim * T
	peak = peak_flops(device)

	# ---- logging ----
	os.makedirs(cfg.out_dir, exist_ok=True)
	metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
	wandb_run = None
	if master and cfg.wandb:
		import wandb
		wandb_run = wandb.init(
			project=cfg.wandb_project, name=cfg.run_name or None, id=wandb_id, resume="allow",
			config={"preset": args.preset, "params": nb_params, "accum_steps": accum_steps,
					**model_cfg.to_dict(), **cfg.to_dict()},
		)
		wandb_id = wandb_run.id

	def log_metrics(step, metrics):
		if not master:
			return
		with open(metrics_path, "a") as fd:
			fd.write(json.dumps({"step": step, **metrics}) + "\n")
		if wandb_run is not None:
			wandb_run.log(metrics, step=step)

	# ---- eval / sample / checkpoint ----
	@torch.no_grad()
	def evaluate():
		model.eval()
		val_loader.reset()  # same val tokens every time, so evals are comparable
		total = torch.zeros((), device=device)
		for _ in range(cfg.eval_iters):
			x, y = val_loader.next_batch()
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
			with autocast():
				logits = model(x)
			total += torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
		if ddp:
			dist.all_reduce(total, op=dist.ReduceOp.AVG)
		model.train()
		return (total / cfg.eval_iters).item()

	def sample(prompt="The"):
		ids = torch.tensor([[eot] + tokenizer.encode(prompt)], device=device)
		with autocast():
			out = raw_model.generate(ids, cfg.sample_tokens, temperature=0.8, top_k=50, stop_token=eot)
		return tokenizer.decode(out[0, 1:].tolist())

	def checkpoint(step, path=latest_path, with_state=True):
		save_checkpoint(
			path, raw_model, optimizer if with_state else None, step, train_loader if with_state else None,
			model_cfg, cfg,
			extra={"best_val": best_val, "wandb_id": wandb_id,
				   "scaler": scaler.state_dict() if scaler is not None and with_state else None},
		)

	def eval_and_sample(step):
		nonlocal best_val
		val_loss = evaluate()
		log(f"step {step:6d} | val loss {val_loss:.4f} | ppl {math.exp(val_loss):.1f}")
		log_metrics(step, {"val/loss": val_loss})
		if master:
			if val_loss < best_val:
				best_val = val_loss
				checkpoint(step, os.path.join(cfg.out_dir, "best.pt"), with_state=False)
			log(f"  sample: {sample()!r}")

	# ---- training loop ----
	# `step` = number of optimizer steps completed so far; checkpoints store it so
	# --resume picks up at exactly the next batch.
	model.train()
	last_checkpoint_time = time.time()
	end_step = min(cfg.stop_step, cfg.max_steps) if cfg.stop_step else cfg.max_steps
	for step in range(start_step, end_step):
		if step % cfg.eval_interval == 0:
			eval_and_sample(step)

		t0 = time.time()
		lr = get_lr(step, cfg.warmup_steps, cfg.max_steps, cfg.max_lr, cfg.min_lr)
		for group in optimizer.param_groups:
			group["lr"] = lr

		loss, grad_norm = train_step(model, train_loader.next_batch, optimizer, accum_steps, device,
									 amp_dtype, cfg.grad_clip, scaler)
		if ddp:
			dist.all_reduce(loss, op=dist.ReduceOp.AVG)

		if step % cfg.log_interval == 0 or step == cfg.max_steps - 1:
			loss = loss.item()  # syncs with the GPU, so dt below is real compute time
			if not math.isfinite(loss):
				raise SystemExit(f"loss is {loss} at step {step}, stopping")
			dt = time.time() - t0
			tokens_per_s = cfg.total_batch_tokens / dt
			mfu = flops_per_token * tokens_per_s / (peak * world) if peak else None
			log(f"step {step:6d} | loss {loss:.4f} | lr {lr:.2e} | grad norm {float(grad_norm):.3f} | "
				f"{dt * 1000:.0f}ms | {tokens_per_s / 1e3:,.1f}k tok/s" + (f" | mfu {mfu:.1%}" if mfu else ""))
			log_metrics(step, {"train/loss": loss, "lr": lr, "grad_norm": float(grad_norm),
							   "tokens_per_s": tokens_per_s, "tokens_seen": (step + 1) * cfg.total_batch_tokens,
							   **({"mfu": mfu} if mfu else {})})

		completed = step + 1
		if master and (completed % cfg.checkpoint_interval == 0
					   or time.time() - last_checkpoint_time > cfg.checkpoint_minutes * 60):
			checkpoint(completed)
			last_checkpoint_time = time.time()

	if end_step < cfg.max_steps:
		if master:
			checkpoint(end_step)
		log(f"stopped at step {end_step} (stop_step), continue with --resume")
	elif start_step < cfg.max_steps:
		eval_and_sample(cfg.max_steps)
		if master:
			checkpoint(cfg.max_steps)
		log(f"done. best val loss {best_val:.4f}, checkpoints in {cfg.out_dir}")

	if wandb_run is not None:
		wandb_run.finish()
	if ddp:
		dist.destroy_process_group()


if __name__ == "__main__":
	main()
