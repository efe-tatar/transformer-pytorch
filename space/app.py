"""
Gradio demo for the Hugging Face Space. Runs the model on this project's own inference code
(CausalLM + KV cache from network/), loaded from the Hub with tools/load_hf.py.
Run locally:  python space/app.py
"""
import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import torch

from tools.load_hf import load_hf

MODEL_ID = os.environ.get("MODEL_ID", "Efe-Tatar/base-100m-fineweb-edu")
EOT = "<|endoftext|>"

model, tokenizer = load_hf(MODEL_ID, token=os.environ.get("HF_TOKEN"))
eot = tokenizer.special_tokens[EOT]
lock = threading.Lock()  # the free CPU tier has 2 cores: one generation at a time


def complete(prompt, max_new_tokens, temperature, top_k):
	if not prompt.strip():
		yield ""
		return
	ids = [eot] + tokenizer.encode(prompt, allowed_special="none")
	ids = torch.tensor([ids[-(model.max_seq_len - int(max_new_tokens)):]])
	new = []
	with lock:
		for token in model.stream(ids, int(max_new_tokens), float(temperature), int(top_k), stop_token=eot):
			if token.item() == eot:
				break
			new.append(token.item())
			text = tokenizer.decode(new)
			if not text.endswith("�"):  # wait for split multi-byte characters
				yield prompt + text
	yield prompt + tokenizer.decode(new)


DESCRIPTION = """
A **100M-parameter Llama-style model** whose architecture, tokenizer, data pipeline and training loop were
written from scratch in PyTorch, pretrained on 3B tokens of FineWeb-Edu on one A100 (4.5 h).
HellaSwag: **27.7%** (GPT-2 124M: 29.6%, random: 25%). It runs here on its own inference code (KV cache, GQA).

This is a **base model**: it continues text rather than answering questions. Start the text you want it to write.

[Code and training logs](https://github.com/efe-tatar/transformer-pytorch) ·
[Model](https://huggingface.co/Efe-Tatar/base-100m-fineweb-edu)
"""

EXAMPLES = [
	["Photosynthesis is the process by which"],
	["The three main causes of the French Revolution were"],
	["In this lesson, students will learn how to"],
	["The human heart has four chambers:"],
	["Machine learning is a branch of artificial intelligence that"],
]

with gr.Blocks(title="base_100m: an LLM trained from scratch") as demo:
	gr.Markdown("# base_100m: an LLM trained from scratch")
	gr.Markdown(DESCRIPTION)
	with gr.Row():
		with gr.Column(scale=3):
			prompt = gr.Textbox(label="Beginning of a text", lines=3, placeholder="Photosynthesis is the process by which")
			output = gr.Textbox(label="Model continuation", lines=10)
			run = gr.Button("Generate", variant="primary")
		with gr.Column(scale=1):
			max_new_tokens = gr.Slider(16, 512, value=120, step=8, label="New tokens")
			temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature (0 = greedy)")
			top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k")
	gr.Examples(EXAMPLES, inputs=[prompt])
	run.click(complete, [prompt, max_new_tokens, temperature, top_k], output)
	prompt.submit(complete, [prompt, max_new_tokens, temperature, top_k], output)

if __name__ == "__main__":
	demo.queue().launch()
