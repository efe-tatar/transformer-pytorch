---
language: en
license: mit
library_name: transformers
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
tags:
- llama
- from-scratch
- pretraining
---

# base_100m: a Llama-style model pretrained from scratch

A 100M-parameter decoder-only language model whose architecture, tokenizer, data pipeline and
training loop were written from scratch in PyTorch, then exported to the standard
`LlamaForCausalLM` format. Code, tests and training logs: https://github.com/efe-tatar/transformer-pytorch

This is a **base model**: it continues text, it does not follow instructions or chat.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained("REPO_ID")

# documents started with <|endoftext|> during training, so prompts should too
ids = tok("<|endoftext|>Photosynthesis is the process by which", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=60, do_sample=True, temperature=0.8, top_k=50)
print(tok.decode(out[0][1:]))
```

## Model

| | |
|---|---|
| Parameters | 100.7M (25.2M of them the tied input/output embedding) |
| Shape | 768 hidden, 12 layers, 12 attention heads, 4 KV heads (grouped-query attention), 1024 context |
| Blocks | pre-norm RMSNorm, RoPE (base 10000), SwiGLU feed-forward (2048 hidden) |
| Tokenizer | byte-level BPE, 32,768 tokens, trained from scratch on 300M characters of FineWeb-Edu |

The export is checked against the original implementation: identical token ids, logits equal
to within 1e-5, identical greedy generation.

## Training

| | |
|---|---|
| Data | 2.99B tokens of FineWeb-Edu (`sample-10BT`), one epoch |
| Optimizer | AdamW (β = 0.9, 0.95, weight decay 0.1 on matmul weights only), gradient clipping 1.0 |
| Schedule | 0.5M tokens per step, 5700 steps, LR 6e-4, 700-step warmup, cosine decay to 6e-5 |
| Hardware | 1× A100 80GB, bf16, 4.5 hours, 184k tokens/s (42% MFU) |
| Final validation loss | 3.223 |

## Evaluation

HellaSwag validation (10,042 questions), scored with identical code for both models:

| Model | Training tokens | acc | acc_norm |
|---|---|---|---|
| random guessing | | 25.0% | 25.0% |
| **base_100m** | 3B | 27.7% | **27.7%** |
| GPT-2 124M | far more (undisclosed) | 28.7% | 29.6% |

## Limitations

- Small and trained on 3B tokens: fluent and on topic, but makes up facts and repeats itself.
- English only, educational web text only (FineWeb-Edu); weak on code.
- No safety tuning or instruction tuning of any kind.
