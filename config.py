from dataclasses import dataclass, asdict, replace


@dataclass
class ModelConfig:
	dim: int = 768
	n_layers: int = 12
	n_heads: int = 12
	n_kv_heads: int = 4
	vocab_size: int = 32768
	max_seq_len: int = 1024
	dropout: float = 0.0
	rope_base: float = 10000.0

	# ablation switches -- the defaults are the Llama-style choices
	norm: str = "rms"          # "rms" | "layer"
	mlp: str = "swiglu"        # "swiglu" | "gelu"
	pos: str = "rope"          # "rope" | "learned"
	attn_impl: str = "manual"  # "manual" | "sdpa"

	def __post_init__(self):
		assert self.norm in ("rms", "layer"), self.norm
		assert self.mlp in ("swiglu", "gelu"), self.mlp
		assert self.pos in ("rope", "learned"), self.pos
		assert self.attn_impl in ("manual", "sdpa"), self.attn_impl
		assert self.dim % self.n_heads == 0, "dim % n_heads != 0"
		assert self.n_heads % self.n_kv_heads == 0, "n_heads % n_kv_heads != 0"

	@property
	def head_dim(self):
		return self.dim // self.n_heads

	def to_dict(self):
		return asdict(self)


MODEL_PRESETS = {
	# ~2M params, mostly embeddings. For checking the pipeline end to end on CPU.
	"debug_cpu": ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128),
	# ~40M params. Size used for the Phase 2 ablations.
	"small_ablation": ModelConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=1024),
	# ~100M params (25M of it the tied embedding). The main Phase 1 run.
	# Same width/depth/heads/context as GPT-2 small (124M), so results can be compared with it.
	"base_100m": ModelConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024),
}


def model_preset(name, **overrides):
	return replace(MODEL_PRESETS[name], **overrides)


@dataclass
class TrainConfig:
	data_dir: str = "data/fineweb_edu"
	out_dir: str = "checkpoints/run"
	run_name: str = ""
	seed: int = 1337

	# batch: micro_batch_size sequences per forward pass, accumulated until
	# total_batch_tokens tokens have been seen (across all GPUs) per optimizer step
	micro_batch_size: int = 16
	total_batch_tokens: int = 2 ** 19  # ~0.5M, GPT-3 small / GPT-2 replication setting

	max_steps: int = 5000
	stop_step: int = 0        # stop early (with a checkpoint) at this step, e.g. to end a Colab session cleanly; 0 = off
	warmup_steps: int = 700
	max_lr: float = 6e-4
	min_lr: float = 6e-5
	weight_decay: float = 0.1
	beta1: float = 0.9
	beta2: float = 0.95
	grad_clip: float = 1.0

	dtype: str = "bf16"      # "bf16" | "fp16" | "fp32"
	compile: bool = False
	device: str = ""         # "" = auto

	log_interval: int = 10
	eval_interval: int = 250
	eval_iters: int = 20
	sample_tokens: int = 64
	checkpoint_interval: int = 250  # steps
	checkpoint_minutes: float = 20  # also save at least this often (Colab disconnects)

	wandb: bool = False
	wandb_project: str = "transformer-from-scratch"

	def __post_init__(self):
		assert self.dtype in ("bf16", "fp16", "fp32"), self.dtype

	def to_dict(self):
		return asdict(self)


TRAIN_PRESETS = {
	"debug_cpu": TrainConfig(
		micro_batch_size=8, total_batch_tokens=8 * 128 * 2, max_steps=200, warmup_steps=20,
		max_lr=3e-3, min_lr=3e-4, dtype="fp32", log_interval=10, eval_interval=50,
		eval_iters=5, checkpoint_interval=50, out_dir="checkpoints/debug_cpu",
	),
	"small_ablation": TrainConfig(
		micro_batch_size=32, total_batch_tokens=2 ** 18, max_steps=2000, warmup_steps=200,
		max_lr=1.5e-3, min_lr=1.5e-4, compile=True, out_dir="checkpoints/small_ablation",
		data_dir="data/fineweb_edu_3b",
	),
	"base_100m": TrainConfig(
		micro_batch_size=16, total_batch_tokens=2 ** 19, max_steps=5000, warmup_steps=700,
		max_lr=6e-4, min_lr=6e-5, compile=True, out_dir="checkpoints/base_100m",
		data_dir="data/fineweb_edu_3b",
	),
}


def _parse_value(text, current):
	if isinstance(current, bool):
		assert text.lower() in ("true", "false", "1", "0"), text
		return text.lower() in ("true", "1")
	return type(current)(float(text)) if isinstance(current, int) else type(current)(text)


def load_preset(name, overrides=()):
	"""-> (ModelConfig, TrainConfig) for a preset, with "key=value" overrides
	applied to whichever config owns the key."""
	model_cfg = MODEL_PRESETS[name]
	train_cfg = TRAIN_PRESETS[name]
	model_changes, train_changes = {}, {}
	for item in overrides:
		key, _, text = item.lstrip("-").partition("=")
		key = key.replace("-", "_")
		if key in model_cfg.__dataclass_fields__:
			model_changes[key] = _parse_value(text, getattr(model_cfg, key))
		elif key in train_cfg.__dataclass_fields__:
			train_changes[key] = _parse_value(text, getattr(train_cfg, key))
		else:
			raise KeyError(f"unknown config key: {key}")
	return replace(model_cfg, **model_changes), replace(train_cfg, **train_changes)
