# LLM_train

A from-scratch **~1B-parameter transformer pretraining stack**, built to understand modern LLM training end-to-end rather than to wrap an existing framework. Almost every component — attention, RoPE, SwiGLU, RMSNorm, init, the multi-token-prediction heads, the loss/optimizer wiring, the instrumentation — is written and reasoned about directly, with SOTA techniques (Muon, Liger fused CE, MTP, `torch.compile`, manual mixed precision) layered on top.

The repo is currently in its **overfit-validation phase**: the full architecture and training loop are exercised by deliberately overfitting char-level tiny-shakespeare, to prove the pipeline (especially the MTP target alignment) is correct before scaling to a real tokenized corpus.

---

## Repository layout

| Path | Role |
|------|------|
| `main_models.py` | **Live model.** The clean, training-path implementation: `Transformer`, `Transformer_block`, `mhma`/`gqa`, `mtp_head`, RMSNorm, SwiGLU, RoPE, init. This is what `train.py` imports. |
| `model_def.py` | **From-scratch reference.** My own implementations of pieces not used on the hot path (manual cross-entropy, LayerNorm, sinusoidal embeddings, list-comprehension RoPE). Kept as a learning/reference artifact — *not* imported by training. |
| `train.py` | Training loop: data → model → Liger loss → backward → Muon/AdamW step → WSD schedule, with full wandb instrumentation. |
| `config.py` | Single `@dataclass` of all hyperparameters; derives step/schedule counts in `__post_init__`. |
| `wandb_log.py` | Observability: MFU, per-group grad/param/**update-ratio** norms, activation-norm hooks, run config. |
| `Data/` | `data.py` (Dataset/DataLoader), `process_tiny_shakespeare.py` (char-level corpus for the overfit harness). |
| `Generate/` | `generate.py` — inference/sampling against a trained checkpoint. |
| `Tokenizers/` | BPE / pretrained / slow tokenizer variants for the real (non-overfit) corpus. |
| `Misc/` | Scratch (`einsum.py`, `kernels.py`) kept out of the main namespace. |
| `profile_kernel.py` | Standalone kernel/throughput profiling. |

---

## Architecture

A pre-norm decoder-only transformer with current-generation components. At the default config (`d_model=1536`, `30` layers, `n_head=12`, `ffn=4096`, `vocab=48k`, `2` MTP heads) it is **≈0.99 B parameters** (~915 M non-embedding).

### Core blocks & why

- **Pre-norm RMSNorm.** Cheaper than LayerNorm (no mean subtraction / bias) and the standard for LLaMA-class models. **Computed in fp32** (upcast inside `forward`, downcast out) so the normalization statistic isn't corrupted by bf16 rounding.
- **SwiGLU FFN.** Gated activation (`silu(W1 x) * W2 x`) — the empirically strongest FFN for LLMs. Hidden dim `4096`.
- **RoPE**, precomputed once for `max_context` via an outer product (`precompute_rope_fast`) and applied in fp32. The slow list-comprehension version is kept in `model_def.py` for reference.
- **Attention via SDPA flash kernel** (`F.scaled_dot_product_attention`, `is_causal=True`), forced through the FLASH backend with `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`.
  - **MHA** (`mhma`) is the default; **GQA** (`gqa`, `enable_gqa=True`) is wired and selectable via `gqa_groups` for KV-cache/bandwidth savings at scale.
  - A causal `mask` buffer is threaded through the blocks but currently unused (SDPA handles causality) — reserved for **intra-document masking** later.
- **Tied embeddings.** Input embedding doubles as the output projection (`hidden @ embedding.weight.T`), saving ~74 M params and a known regularizer.

### Initialization (deliberate, not default)

- **Embedding:** `N(0, 1/√d_model)` so token-vector norms start at ~1.
- **Residual-path projections** (`linear_proj`): std scaled by `1/(2·num_layers)` — **layer-aware init** so the residual-stream variance doesn't grow with depth (the GPT-2/LLaMA residual-scaling trick).
- **SwiGLU/QKV projections** (`linear_swig`): `std = 1/√in_dim` (small init, LLaMA-style).

### Multi-Token Prediction (the centerpiece)

DeepSeek-V3-style **sequential MTP heads**. The trunk predicts the next token (offset +1); MTP head `i` predicts offset `+(i+2)`, and the heads chain — each consumes the previous head's hidden state.

Each `mtp_head` is a full `Transformer_block` plus a `proj: 2·d_model → d_model` that fuses the previous hidden state `h` with the **RMSNorm'd embedding of the future token** at the head's offset.

Key design choices that make this correct *and* cheap:

- **Teacher-forcing shift baked into the model.** The model is fed `EFF_SEQ_LEN = SEQ_LEN + MTP_HEADS + 1` tokens and internally uses `L = T − MTP_HEADS − 1` positions, so every head sees its own correctly-shifted future token without the training loop reshuffling targets.
- **Head-major output layout.** Forward returns hidden states stacked along the **batch dim** (`cat(dim=0)`): rows `[0:B]` are the main head, `[B:2B]` the first MTP head, etc. This exactly matches the flattened target order built in `train.py`, so the Liger loss consumes them with a single `view(-1, d_model)` and no gather.
- **Targets, head-major:** `y_mtp = x[:,2:].unfold(1, K, 1).permute(2,0,1)` — `unfold` is a zero-copy strided view that produces all `K` shifted windows; the permute reorders to head-major to line up with the stacked hidden states.
- **Inference is trunk-only.** The MTP heads are a *training signal*; `generate()` uses only the main next-token head (and pads the context by `MTP_HEADS+1` to undo the baked-in shift), so sampling cost is unchanged.

---

## Training stack

### Optimizers — Muon + AdamW split

- **Muon** for all 2-D weight matrices (attention/FFN projections). Muon orthogonalizes the momentum update via a 5-step Newton–Schulz iteration; `adjust_lr_fn="match_rms_adamw"` (Moonshot's RMS-matching) lets it reuse AdamW-tuned LR/WD.
- **AdamW** for 1-D params (norm scales) and the embedding.

Rationale: Muon's orthogonalized updates are strong on hidden matrices but undefined for 1-D tensors; the embedding is sparse-access and better served by Adam.

### Loss — Liger fused linear cross-entropy

`LigerFusedLinearCrossEntropyLoss` fuses the output projection + softmax + CE into one kernel that never materializes the full `(N, vocab)` logits — critical at vocab 48k. The main head additionally uses **z-loss** (`lse_square_scale=1e-4`, `return_z_loss=True`) to keep `logsumexp` bounded; it's logged as a stability signal.

### Scheduler — WSD

Warmup–Stable–Decay (5% / 85% / 10%) via `SequentialLR`, mirrored across both optimizers. Linear warmup from `1e-4×`, constant plateau, linear decay back to `1e-4×`.

### `torch.compile` — model forward only

The **model** is compiled (`mode="max-autotune-no-cudagraphs"`); the loss+backward run eager. Reason: Liger calls `.item()` internally → a Dynamo graph break, and with *two* Liger calls (main + MTP) the resume frame wraps a Triton kernel and crashes Inductor's `decompose_triton_kernel_wrapper_functional` pass. Compiling the model alone still captures ~all the FLOPs, and the compiled forward keeps its AOT-generated backward.

### Mixed precision *(in progress)*

The model runs in **bf16** compute. bf16 (not fp16) is chosen because it shares fp32's 8-bit exponent → the same ~1e-38 dynamic range, so **no loss scaling is needed** (gradients don't underflow).

bf16's cost is *precision*, not range: 7 mantissa bits → ~0.8% relative resolution. This surfaced concretely as a **frozen-scale bug** — RMSNorm scales initialized at `1.0` stopped updating, because an AdamW step of ~`2e-4` is below the bf16 ULP at 1.0 (`0.0078`) and rounds away. The fix is the standard manual-MP recipe, being wired in now:

- **fp32 master weights** + bf16 compute copies (the bf16 weights feed the matmuls; the optimizer accumulates into the fp32 master).
- bf16 grads cast to fp32 per-tensor (freed immediately, no persistent fp32 grad buffer).
- **fp32 optimizer state + fp32 Muon momentum** — falls out automatically once the optimizers own the fp32 masters (Muon's `zeros_like(grad)` buffer inherits fp32). This matches Moonshot's production Muon: **bf16 only inside Newton–Schulz**, fp32 for everything persistent.

> Mixed precision is mid-implementation. `train.py`'s optimizers are renamed `*_bf16` in anticipation; the master-weight wrapper and the step-call rename are the remaining work.

Numerically-sensitive ops kept in fp32 throughout: RMSNorm statistics, RoPE rotation, attention softmax (inside SDPA), and the CE/z-loss.

---

## Observability (`wandb_log.py`)

Beyond loss/throughput, the run logs the diagnostics that actually catch training pathologies:

- **MFU** against the A6000 bf16 peak (`6·B·T·N_nonembed / (peak·step_time)`).
- **Per-parameter-group** grad-norm, param-norm, and **true update-ratio** (`‖Δθ‖/‖θ‖` from a pre-/post-step snapshot), grouped into `embed / attn_qkv / attn_out / ffn_in / ffn_out / norm`. The update-ratio is what exposed the frozen-norm-scale bug above.
- **Activation norms/maxes** per block via forward hooks, attached only on an eager pass so they never perturb the compiled path.
- EMA (bias-corrected) loss and throughput; data-IO fraction of step time; z-loss; per-sequence shard IDs to trace loss spikes back to data.

---

## Configuration (defaults)

| | | | |
|---|---|---|---|
| `D_MODEL` 1536 | `NUM_LAYERS` 30 | `N_HEAD` 12 | `FFN_HIDDEN_DIM` 4096 |
| `VOCAB_SIZE` 48000 | `MAX_CONTEXT` 8192 | `SEQ_LEN` 1024 | `BATCH_SIZE` 20 |
| `MTP_HEADS` 2 | `MTP_LOSS_WEIGHT` 0.3 | `MAX_FREQ` 1e4 | `NUM_GROUPS` 4 (GQA) |
| `TOTAL_TOKENS` 1e8 | WSD 5/85/10% | `A6000_BF16_PEAK` 154 TF | `EFF_SEQ_LEN` = SEQ_LEN+MTP+1 |

---

## Running

```bash
# overfit-validation run (char-level tiny-shakespeare)
python train.py

# sample from a checkpoint
python -m Generate.generate
```

Single-device by default (`DEVICE="cuda:6"`); requires `torch` (w/ Muon), `liger-kernel`, `wandb`, `triton`.

---

## Status & roadmap

- [x] Architecture: RMSNorm / SwiGLU / RoPE / MHA+GQA / tied embeddings / layer-aware init
- [x] Sequential MTP heads with baked-in teacher-forcing shift + head-major Liger alignment
- [x] Muon+AdamW split, Liger fused CE + z-loss, WSD schedule, model-only `torch.compile`
- [x] Overfit validation of the MTP pipeline (verified verbatim recall of training text)
- [x] Full wandb instrumentation (MFU, update-ratio, activation hooks)
- [ ] **Manual mixed precision** (fp32 master + bf16 compute) — in progress; fixes frozen norm scales
- [ ] Validation loop (currently stubbed)
- [ ] Intra-document attention masking (mask buffer reserved)
- [ ] Distributed data parallel (single-device today; `DDP` imported but not yet wired)
- [ ] Scale to real tokenized corpus (the 48k vocab is sized for this; current char data is the overfit harness)
</content>
</invoke>