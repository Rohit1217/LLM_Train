# LLM_train

Training a ~1B decoder-only language model from scratch, on a small multi-GPU
setup (RTX A6000s). This repo has the full stack: tokenizer, data pipeline,
model, custom attention kernels, training loop, and generation.

I wrote most of the non-trivial parts myself to understand them end to end,
rather than pulling in a framework.

Runs / loss curves: [Run link]("https://wandb.ai/rohit_iisc-indian-institute-of-science/llm_overfit/workspace?nw=nwuserrohit_iisc")

## What's in here

- A custom Triton flash-attention kernel (forward and backward) for
  intra-document attention. Documents are packed into fixed-length sequences,
  and attention is masked so it never crosses a document boundary. RoPE
  positions reset at each document.
- Multi-token prediction (MTP) heads on top of the trunk, so the model is
  trained to predict a few tokens ahead (used later for speculative decoding).
- Mixed-precision training with fp32 master weights. 2D weight matrices are
  optimized with Muon, everything else (norms, embedding) with AdamW.
- GQA, SwiGLU, RMSNorm, WSD learning-rate schedule, z-loss, gradient
  accumulation, Liger fused cross-entropy, torch.compile.
- Multi-GPU (DDP) training that is resumable: checkpoints are written from a
  background thread (so a slow/stuck filesystem doesn't stall training), and a
  run can resume from the exact batch even if you change the number of GPUs.

## Model

Decoder-only transformer.

| | |
|---|---|
| params |1B |
| layers | 30 |
| d_model | 1536 |
| heads | 12 (GQA, 4 kv groups) |
| ffn hidden | 4096 |
| context | 1024 |
| vocab | 48011 (custom BPE + special tokens) |
| MTP heads | 2 |

Config lives in `Config/config.py`.

## Setup

```bash
pip install -r requirements.txt
```

Point the data dir at your tokenized `.bin` files:

```bash
export LLM_DATA_DIR=/path/to/data      # expects tokens.bin, overfit_10k_sequences.bin
export WANDB_ENTITY=your_wandb_entity  # optional
```

## Train

```bash
# N GPUs; OFFSET in train_ddp.py picks which physical GPUs
torchrun --nproc_per_node=4 --master_port=25678 train_ddp.py
```

Accumulation, checkpoint frequency, run name etc. are in the config. The run
name sets both the wandb id and the checkpoint folder, so set `RUN_NAME` to
resume a specific run.

## Eval / generate

```bash
# validation loss + MMLU + GSM8K on the latest checkpoint
CUDA_VISIBLE_DEVICES=3 python eval.py --once --wandb

# sample from the model
python Generate/generate_samples.py --prompt "The capital of France is"
```

<!-- ## Results

`<drop in a loss curve and a few sample generations>` -->

## Status

Work in progress. At this scale the model learns fluent text and basic facts;
it is a base model with no instruction tuning, so MMLU sits near chance and
GSM8K near zero. Next steps: finish the token budget, then SFT + GRPO for
math/reasoning.

## Layout

```
Config/            config
Data/              tokenizer output loading, dedup, download, shuffling
Tokenizers/        custom BPE tokenizer + vocab
Intradoc_kernels/  triton intra-doc attention (fwd/bwd) + torch wrapper
Generate/          sampling / generation
Misc/              profiling and scratch
model.py           the transformer
train_ddp.py       training loop (DDP)
load_save_ckpt.py  checkpoint save/load
eval.py            eval harness
wandb_log.py       logging
```
