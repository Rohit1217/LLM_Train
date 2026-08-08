# LLM_train

Training a ~1B decoder-only language model from scratch, on a small multi-GPU
setup (RTX A6000s). This repo has the full stack: tokenizer, data pipeline,
model, custom attention kernels, training loop, and generation.

I wrote most of the non-trivial parts myself to understand them end to end,
rather than pulling in a framework.

Runs / loss curves: [wandb](https://wandb.ai/rohit_iisc-indian-institute-of-science/llm_overfit/workspace?nw=nwuserrohit_iisc)

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

## Data

Target is ~20B tokens, which is roughly Chinchilla-optimal for a 1B model
(about 20 tokens per parameter). The corpus is a filtered, deduplicated mix
of code and math (subsets of The Stack — Python, C++, TeX, SQL — plus math),
so this is a base model biased toward code/math rather than general web text.

The tokenizer is a custom BPE (48k vocab) I trained on this corpus instead of
reusing an off-the-shelf one, so the vocabulary actually fits the code/math
distribution (see `Tokenizers/`). Data prep lives in `Data/`: StarCoder-style
quality filtering, exact-dup removal (SHA-256 over whitespace-normalized text),
then packing whole documents into fixed 1024-token sequences with an EOT
separator. At train time the intra-document attention mask keeps attention and
RoPE positions from leaking across document boundaries within a packed sequence.

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

## Things worth a look

If you're reading the code, these are the parts that took the most thought:

- **Intra-document attention kernels** (`Intradoc_kernels/`) — hand-written
  Triton forward and backward, with GQA and RoPE positions that reset per
  document, so packing many short docs into one sequence stays correct.
- **World-size-agnostic training** (`Config/config.py`, `Data/dataload.py`) —
  gradient accumulation auto-scales with the number of GPUs so the global
  batch is constant. The LR schedule and the exact data order come out
  identical whether you launch on 2 GPUs or 8, and a run resumes from the
  exact micro-batch even if the GPU count changed between runs.
- **Fault-tolerant checkpointing** (`load_save_ckpt.py`) — checkpoints are
  written from a background thread so a slow or stuck NFS mount can't freeze
  the GPUs, and resume falls back through checkpoint history if the latest
  file is bad.
- **Optimizer split** (`train_ddp.py`) — Muon on 2D weight matrices, AdamW on
  everything else, with fp32 master weights alongside the bf16 model.

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
