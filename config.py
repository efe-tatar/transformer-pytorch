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
	"gpt2_small": ModelConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024),
}


def model_preset(name, **overrides):
	return replace(MODEL_PRESETS[name], **overrides)
