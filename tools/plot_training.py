"""
Plot a training run: loss over the whole run, a zoom on the second half, and the LR schedule.
Reads either metrics.jsonl (written by train.py) or the printed training log.

Usage:
    python tools/plot_training.py "training output.txt" --out assets/base_100m_training
    python tools/plot_training.py checkpoints/base_100m/metrics.jsonl --tokens-per-step 524288

Writes <out>_light.png and <out>_dark.png.
"""
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
	"light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
			  "train": "#2a78d6", "val": "#eb6834"},
	"dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#383835",
			 "train": "#3987e5", "val": "#d95926"},
}

TRAIN_LINE = re.compile(r"step +(\d+) \| loss ([\d.]+) \| lr ([\d.e+-]+)")
VAL_LINE = re.compile(r"step +(\d+) \| val loss ([\d.]+)")
HEADER = re.compile(r"batch ([\d,]+) tokens/step")


def read_run(path):
	train, val, lr, tokens_per_step = [], [], [], None
	with open(path, encoding="utf-8") as fd:
		if path.endswith(".jsonl"):
			for line in fd:
				row = json.loads(line)
				if "train/loss" in row:
					train.append((row["step"], row["train/loss"]))
					lr.append((row["step"], row["lr"]))
				if "val/loss" in row:
					val.append((row["step"], row["val/loss"]))
		else:
			for line in fd:
				if m := HEADER.search(line):
					tokens_per_step = int(m[1].replace(",", ""))
				elif m := VAL_LINE.search(line):
					val.append((int(m[1]), float(m[2])))
				elif m := TRAIN_LINE.search(line):
					train.append((int(m[1]), float(m[2])))
					lr.append((int(m[1]), float(m[3])))
	# a resumed run re-logs some steps: keep the last value for each step
	dedupe = lambda pts: sorted(dict(pts).items())
	return dedupe(train), dedupe(val), dedupe(lr), tokens_per_step


def smooth(values, half_window=7):
	# centred moving average: unlike an EMA it doesn't lag behind a falling curve
	out = []
	for i in range(len(values)):
		# at the start the loss falls fast: shrink the window symmetrically so the curve
		# starts at the true initial loss. at the end the loss is flat, so a one-sided
		# window there is unbiased, and it keeps a single noisy last batch from showing as a dip.
		h = min(half_window, i)
		window = values[i - h:i + h + 1]
		out.append(sum(window) / len(window))
	return out


def plot(train, val, lr, tokens_per_step, theme, title, out_path):
	t = THEMES[theme]
	to_b = lambda step: step * tokens_per_step / 1e9
	tr_x = [to_b(s) for s, _ in train]
	tr_y = smooth([v for _, v in train])
	va_x = [to_b(s) for s, _ in val]
	va_y = [v for _, v in val]

	plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
	fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
	fig.patch.set_facecolor(t["surface"])

	def style(ax, subtitle, ylabel):
		ax.set_facecolor(t["surface"])
		ax.set_title(subtitle, color=t["ink"], loc="left", fontsize=11, pad=8)
		ax.set_xlabel("tokens seen (billions)", color=t["ink2"])
		ax.set_ylabel(ylabel, color=t["ink2"])
		ax.tick_params(colors=t["ink2"], length=0)
		ax.grid(True, color=t["grid"], linewidth=0.8)
		ax.set_axisbelow(True)
		for side in ("top", "right", "left"):
			ax.spines[side].set_visible(False)
		ax.spines["bottom"].set_color(t["grid"])

	zoom_from = va_x[len(va_x) // 4] if len(va_x) > 4 else 0
	for ax, subtitle, x_min in [(axes[0], "Loss, full run", 0), (axes[1], f"Loss after {zoom_from:.1f}B tokens", zoom_from)]:
		style(ax, subtitle, "cross-entropy loss")
		keep = [i for i, x in enumerate(tr_x) if x >= x_min]
		ax.plot([tr_x[i] for i in keep], [tr_y[i] for i in keep], color=t["train"], linewidth=2, label="train (smoothed)")
		keep_v = [i for i, x in enumerate(va_x) if x >= x_min]
		ax.plot([va_x[i] for i in keep_v], [va_y[i] for i in keep_v], color=t["val"], linewidth=2,
				marker="o", markersize=5, markeredgecolor=t["surface"], markeredgewidth=1.5, label="validation")
		if x_min:
			lows = [tr_y[i] for i in keep] + [va_y[i] for i in keep_v]
			ax.set_ylim(min(lows) - 0.05, max(lows) + 0.05)

	# direct label on the final validation point + legend
	axes[1].annotate(f"val {va_y[-1]:.3f}", (va_x[-1], va_y[-1]), xytext=(0, 28), textcoords="offset points",
					 ha="right", color=t["ink"], fontsize=10,
					 bbox={"boxstyle": "round,pad=0.25", "facecolor": t["surface"], "edgecolor": "none"})
	legend = axes[0].legend(frameon=False, loc="upper right")
	for text in legend.get_texts():
		text.set_color(t["ink"])

	style(axes[2], "Learning rate (warmup + cosine)", "learning rate")
	axes[2].plot([to_b(s) for s, _ in lr], [v for _, v in lr], color=t["train"], linewidth=2)
	axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
	axes[2].yaxis.get_offset_text().set_color(t["ink2"])

	fig.suptitle(title, color=t["ink"], x=0.01, ha="left", fontsize=13, fontweight="bold")
	fig.tight_layout()
	fig.savefig(out_path, dpi=150, facecolor=t["surface"])
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("log", help="metrics.jsonl or the printed training log")
	parser.add_argument("--out", default="assets/training")
	parser.add_argument("--tokens-per-step", type=int, default=None, help="needed for metrics.jsonl")
	parser.add_argument("--title", default="base_100m pretraining on FineWeb-Edu")
	args = parser.parse_args()

	train, val, lr, tokens_per_step = read_run(args.log)
	tokens_per_step = args.tokens_per_step or tokens_per_step
	assert tokens_per_step, "pass --tokens-per-step (not stored in metrics.jsonl)"
	import os
	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	for theme in THEMES:
		plot(train, val, lr, tokens_per_step, theme, args.title, f"{args.out}_{theme}.png")
		print(f"wrote {args.out}_{theme}.png")
	print(f"{len(train)} train points, {len(val)} val points, final val loss {val[-1][1]:.4f}")


if __name__ == "__main__":
	main()
