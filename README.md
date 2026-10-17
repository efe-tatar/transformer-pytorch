# Llama-style language model from scratch (PyTorch)

A decoder-only transformer, BPE tokenizer and pretraining pipeline written from
the ground up in PyTorch, with no `transformers` or `tokenizers` library.
Each architecture choice sits behind a config flag, so it can be compared against
its alternative under the same training budget.

**Status:** the model, tokenizer, data pipeline and training loop are done
and tested; pretraining runs on FineWeb-Edu are next.

## What's implemented

**Model** ([network/](network/))
- Pre-norm decoder blocks with RMSNorm and a SwiGLU feed-forward
- Rotary position embeddings (RoPE), computed once and shared by all layers
- Grouped-query attention (GQA): any number of KV heads from 1 (MQA) up to `n_heads` (MHA)
- KV cache for incremental decoding (prefill + token-by-token), checked against the full forward pass
- Two attention paths: a hand-written reference and PyTorch's fused `scaled_dot_product_attention`, tested to match
- Weight tying, GPT-2-style scaled init on the residual projections
- `generate()` with temperature, top-k and stop token

**Tokenizer** ([tokenizer/bpe.py](tokenizer/bpe.py))
- Byte-level BPE with a GPT-style pre-tokenization regex
- Training uses incremental pair counts on a heap, with streaming input, so a corpus of hundreds of MB never has to sit in memory as one string
- Special tokens (`<|endoftext|>`, chat roles) and a word-level encode cache

**Training** ([train.py](train.py), [training/](training/))
- FineWeb-Edu streamed, tokenized in parallel and stored as `uint16` shards read with memmap
- AdamW with weight decay on matmul weights only; linear warmup + cosine decay
- Gradient accumulation to a fixed token batch size; gradient clipping
- Mixed precision: bf16 autocast, or fp16 with `GradScaler` on older GPUs
- Multi-GPU with `torchrun` (DDP); optional `torch.compile`
- Logging of tokens/s and MFU; Weights & Biases optional
- Checkpoints save the model, optimizer, data position and RNG state, written atomically. `--resume` continues a run **exactly** (tested: an interrupted run reproduces the uninterrupted one bit for bit)

### Ablation flags

| Flag | Default (Llama) | Alternative (GPT-2 era) |
|---|---|---|
| `--norm` | `rms` | `layer` |
| `--mlp` | `swiglu` | `gelu` |
| `--pos` | `rope` | `learned` |
| `--n_kv_heads` | `4` (GQA) | `12` (MHA), `1` (MQA) |
| `--attn_impl` | `manual` | `sdpa` |

## Quickstart

```bash
pip install -r requirements.txt

# 1. data: trains the tokenizer, then writes token shards to data/fineweb_edu/
python data/prepare_fineweb.py --max-tokens 5e6 --tokenizer-chars 20e6     # small debug set
python data/prepare_fineweb.py --max-tokens 3e9                            # pretraining set

# 2. train
python train.py --preset debug_cpu                                   # ~2M params, runs on a laptop CPU
python train.py --preset base_100m --attn_impl=sdpa --wandb=true    # ~100M params, GPU
torchrun --nproc_per_node=2 train.py --preset base_100m --dtype=fp16   # 2 GPUs without bf16 (e.g. T4)

# 3. pick up an interrupted run
python train.py --preset base_100m --resume
```

Every field in [config.py](config.py) can be overridden with `--key=value`.

| Preset | Params | Shape | Use |
|---|---|---|---|
| `debug_cpu` | ~2M | 64d × 2 layers, seq 128 | pipeline checks |
| `small_ablation` | ~40M | 512d × 8 layers, seq 1024 | architecture comparisons |
| `base_100m` | ~100M | 768d × 12 layers, 12 heads / 4 KV heads, seq 1024 | main pretraining run (GPT-2 small dimensions, for comparison) |

## Tests

```bash
python -m pytest tests
```

The 100 tests cover:
- KV cache vs full forward, for every architecture variant
- causal masking
- SDPA vs manual attention, with and without the cache
- initial loss ≈ ln(vocab)
- tokenizer round-trips and special tokens
- shard layout and DDP data splitting
- LR schedule and optimizer parameter groups
- accumulated gradients equal to a single large batch
- exact resume after an interruption

## Layout

```
config.py                  model / training configs and presets
network/transformer.py     attention block (RoPE, GQA, KV cache, manual + SDPA)
network/causal_lm.py       full model, init, generation
tokenizer/bpe.py           byte-level BPE
data/prepare_fineweb.py    download, tokenizer training, sharding
training/                  data loader, LR schedule, optimizer, train step, checkpoints
train.py                   training entry point
tests/                     pytest suite
```

## Roadmap

- [x] Llama-style architecture: RMSNorm, SwiGLU, RoPE, GQA, KV cache
- [x] BPE tokenizer trained from scratch
- [x] Pretraining pipeline: mixed precision, DDP, gradient accumulation, exact resume
- [ ] Pretrain ~100M model on FineWeb-Edu, evaluate on HellaSwag against GPT-2 124M
- [ ] Architecture ablations (GQA vs MHA, RoPE vs learned, SwiGLU vs GELU, RMSNorm vs LayerNorm)
- [ ] Tokenizer efficiency study on French and Turkish
- [ ] Instruction tuning (SFT) and LoRA implemented from scratch
- [ ] Export to Hugging Face `LlamaForCausalLM`, inference benchmarks, API + demo
