import glob
import os

import numpy as np
import torch


class ShardLoader:
	"""Serves contiguous (x, y) batches of shape (B, T) from uint16 .npy shards.

	With DDP each rank reads a different B*T window of the same stream:
	rank r takes [pos + r*B*T, pos + (r+1)*B*T + 1), then everyone advances
	by world_size*B*T. state_dict() lets a resumed run continue exactly where
	the previous one stopped."""

	def __init__(self, data_dir, split, batch_size, seq_len, rank=0, world_size=1):
		self.files = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.npy")))
		assert self.files, f"no {split}_*.npy shards in {data_dir}"
		self.B = batch_size
		self.T = seq_len
		self.rank = rank
		self.world_size = world_size
		self.reset()

	def reset(self):
		self._load_shard(0)

	def _load_shard(self, index, pos=0):
		self.shard = index
		# mmap: the OS pages the file in lazily, shards never sit fully in RAM
		self.tokens = np.load(self.files[index], mmap_mode="r")
		step = self.B * self.T * self.world_size
		assert len(self.tokens) > step, (
			f"{self.files[index]} has {len(self.tokens)} tokens, "
			f"need more than B*T*world_size = {step}"
		)
		self.pos = pos

	def next_batch(self):
		B, T = self.B, self.T
		start = self.pos + self.rank * B * T
		chunk = torch.from_numpy(self.tokens[start:start + B * T + 1].astype(np.int64))
		x = chunk[:-1].view(B, T)
		y = chunk[1:].view(B, T)

		self.pos += B * T * self.world_size
		if self.pos + B * T * self.world_size + 1 > len(self.tokens):
			self._load_shard((self.shard + 1) % len(self.files))
		return x, y

	def state_dict(self):
		return {"shard": self.shard, "pos": self.pos}

	def load_state_dict(self, state):
		self._load_shard(state["shard"], state["pos"])
