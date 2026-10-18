"""
HTTP API for the model, running on this project's own inference code (CausalLM + KV cache).

    MODEL_ID=Efe-Tatar/base-100m-fineweb-edu uvicorn serve.app:app --port 8000

MODEL_ID is a Hub repo or a local folder in the Hugging Face format (tools/export_hf.py output).

Endpoints:
    GET  /health            model info
    POST /generate          {"prompt": "...", "max_new_tokens": 100} -> full text + timing
    POST /generate/stream   same body -> server-sent events, one per decoded chunk, then [DONE]
"""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

EOT = "<|endoftext|>"


class GenerateRequest(BaseModel):
	prompt: str = Field(..., max_length=8000)
	max_new_tokens: int = Field(100, ge=1, le=512)
	temperature: float = Field(0.8, ge=0.0, le=2.0, description="0 = greedy")
	top_k: int | None = Field(50, ge=1)
	seed: int | None = None


class IncrementalDecoder:
	"""Turns a growing list of token ids into text chunks. A multi-byte character can be split
	across tokens; it's held back until complete instead of being sent as U+FFFD."""

	def __init__(self, tokenizer):
		self.tokenizer = tokenizer
		self.ids = []
		self.sent = ""

	def push(self, token_id):
		self.ids.append(token_id)
		text = self.tokenizer.decode(self.ids)
		if text.endswith("�"):
			return ""
		chunk, self.sent = text[len(self.sent):], text
		return chunk


def create_app(model, tokenizer, model_id="local"):
	app = FastAPI(title="base_100m", description="A Llama-style model pretrained from scratch")
	eot = tokenizer.special_tokens[EOT]
	device = next(model.parameters()).device
	# one generation at a time: on CPU, parallel requests would just fight over the same cores
	lock = threading.Lock()

	def prepare(req):
		ids = [eot] + tokenizer.encode(req.prompt, allowed_special="none")
		# keep the end of a too-long prompt, leaving room for the answer
		ids = ids[-(model.max_seq_len - req.max_new_tokens):]
		if req.seed is not None:
			torch.manual_seed(req.seed)
		return torch.tensor([ids], device=device)

	def tokens(req):
		with lock:
			for token in model.stream(prepare(req), req.max_new_tokens, req.temperature, req.top_k, stop_token=eot):
				token = token.item()
				if token == eot:
					return
				yield token

	@app.get("/health")
	def health():
		return {"status": "ok", "model": model_id, "params": sum(p.numel() for p in model.parameters()),
				"context": model.max_seq_len, "device": str(device)}

	@app.post("/generate")
	def generate(req: GenerateRequest):
		t0 = time.perf_counter()
		ids = list(tokens(req))
		seconds = time.perf_counter() - t0
		return {"text": tokenizer.decode(ids), "tokens": len(ids), "seconds": round(seconds, 3),
				"tokens_per_s": round(len(ids) / seconds, 1) if ids else 0.0}

	@app.post("/generate/stream")
	def generate_stream(req: GenerateRequest):
		def events():
			decoder = IncrementalDecoder(tokenizer)
			for token in tokens(req):
				chunk = decoder.push(token)
				if chunk:
					yield f"data: {json.dumps({'text': chunk})}\n\n"
			yield "data: [DONE]\n\n"
		return StreamingResponse(events(), media_type="text/event-stream")

	return app


def app_from_env():
	from tools.load_hf import load_hf
	model_id = os.environ.get("MODEL_ID", "Efe-Tatar/base-100m-fineweb-edu")
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model, tokenizer = load_hf(model_id, device=device, token=os.environ.get("HF_TOKEN"))
	return create_app(model, tokenizer, model_id)


# `uvicorn serve.app:app` builds the app from MODEL_ID; tests call create_app() directly
app = app_from_env() if os.environ.get("SERVE_NO_AUTOLOAD") is None else None
