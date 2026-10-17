import os
import random

import numpy as np
import torch


def unwrap(model):
	# DDP wraps in .module, torch.compile wraps in ._orig_mod
	model = getattr(model, "module", model)
	return getattr(model, "_orig_mod", model)


def rng_state():
	state = {
		"python": random.getstate(),
		"numpy": np.random.get_state(),
		"torch": torch.get_rng_state(),
	}
	if torch.cuda.is_available():
		state["cuda"] = torch.cuda.get_rng_state_all()
	return state


def set_rng_state(state):
	random.setstate(state["python"])
	np.random.set_state(state["numpy"])
	torch.set_rng_state(state["torch"])
	if "cuda" in state and torch.cuda.is_available():
		torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, step, loader, model_cfg, train_cfg, extra=None):
	"""Everything needed to continue a run as if it never stopped.
	Written to a temp file then renamed: os.replace is atomic, so a crash or a
	Colab disconnect mid-save never leaves a half-written latest.pt behind."""
	state = {
		"model": unwrap(model).state_dict(),
		"optimizer": optimizer.state_dict() if optimizer is not None else None,
		"step": step,
		"loader": loader.state_dict() if loader is not None else None,
		"rng": rng_state(),
		"model_cfg": model_cfg.to_dict(),
		"train_cfg": train_cfg.to_dict(),
		**(extra or {}),
	}
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	tmp = path + ".tmp"
	torch.save(state, tmp)
	os.replace(tmp, path)


def load_checkpoint(path, map_location="cpu"):
	# weights_only=False: the checkpoint holds numpy RNG state and plain dicts we wrote ourselves
	return torch.load(path, map_location=map_location, weights_only=False)
