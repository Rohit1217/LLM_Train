"""
Cheap out-of-band eval harness: MMLU + GSM8K + held-out validation loss.

Runs SEPARATELY from training (its own process, one spare GPU) so it never touches the
training hot path. Loads a checkpoint (fp32 masters saved by train_ddp.save_training_checkpoint),
rebuilds the model, evaluates, and logs to wandb. Can run once (--ckpt) or poll latest.pt (--loop).

NOTE on scale: at ~875M params / base (no SFT), expect MMLU ~random (25%) and GSM8K low.
The harness is here so you can track those as the model/data grows; val-loss is your real signal.

Usage:
  # one checkpoint:
  CUDA_VISIBLE_DEVICES=3 python eval.py --ckpt Weights/checkpoints/<RUN>/latest.pt --once
  # poll for new checkpoints forever:
  CUDA_VISIBLE_DEVICES=3 python eval.py --loop
Set WORLD_SIZE=1 in the env is NOT needed (eval ignores world), but Config reads WORLD_SIZE for
ACCUM math only, which eval doesn't use.
"""
import os, re, sys, time, glob, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                 # repo root -> Config/, main_models_train
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tokenizers"))
from Config.config import Config
from model import Transformer          # registers the intradoc custom op on import
from tokenizer_fast import Tokenizer

cfg = Config()
EOT = 48000                                        # <|endoftext|> id (tokenizer appends it at vocab_size)
TOK_PATH = "Tokenizers/tokenizer_vocab.titoken"

# ---------------- model / tokenizer loading ----------------
def load_tokenizer():
    tok = Tokenizer(48000)
    tok.load_tokenizer(TOK_PATH)
    return tok

def build_and_load(ckpt_path, device):
    m = Transformer(vocab_size=cfg.VOCAB_SIZE, max_context=cfg.MAX_CONTEXT, max_freq=cfg.MAX_FREQ,
                    d_model=cfg.D_MODEL, n_heads=cfg.N_HEAD, num_layers=cfg.NUM_LAYERS, attn_dropout=0.0,
                    ffn_hidden_dim=cfg.FFN_HIDDEN_DIM, ffn_dropout=0.0, mtp_heads=cfg.MTP_HEADS,
                    group_size=cfg.NUM_GROUPS, max_len=cfg.MAX_CONTEXT, intradoc_att=cfg.INTRADOC_ATT).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # SAME split/order as train_ddp.load_optim -> the saved master lists align by named_parameters() order
    p2 = [p for n, p in m.named_parameters() if p.ndim == 2 and "embed" not in n]
    p1 = [p for n, p in m.named_parameters() if p.ndim == 1 or "embed" in n]
    with torch.no_grad():
        for p, s in zip(p2, ck["master_2d"]): p.copy_(s.to(device).to(p.dtype))
        for p, s in zip(p1, ck["master_1d"]): p.copy_(s.to(device).to(p.dtype))
    m = m.to(torch.bfloat16)
    m.buffers_to_float()                           # rope/rms buffers back to fp32 (matches training)
    m.eval()
    return m, int(ck.get("opt_step", 0))

# ---------------- core forward (padded main-head, single sequence) ----------------
@torch.no_grad()
def seq_logprobs(model, ids, device):
    """ids: 1D python list of token ids. Returns fp32 log-probs (S, vocab): row i predicts ids[i+1]."""
    S = len(ids)
    pad = cfg.MTP_HEADS + 1                         # model drops mtp+1 tail tokens -> pad so L == S (like generate())
    x = torch.tensor(ids + [EOT] * pad, dtype=torch.long, device=device).unsqueeze(0)   # (1, S+pad)
    cuseq = torch.tensor([0, S], dtype=torch.int32, device=device)                       # one document over the S trunk tokens
    token_pos = torch.arange(S, dtype=torch.long, device=device)
    hidden = model(x, cuseq, token_pos)            # (B*(mtp+1), S, D); main head = row 0 (B=1)
    logits = hidden[0].float() @ model.embedding.weight.float().t()                       # (S, vocab)
    return torch.log_softmax(logits, dim=-1)

@torch.no_grad()
def loglikelihood(model, ctx_ids, cont_ids, device):
    """summed log P(cont | ctx). forward-only, no generation."""
    ids = ctx_ids + cont_ids
    logp = seq_logprobs(model, ids, device)
    total = 0.0
    for j, t in enumerate(cont_ids):
        pos = len(ctx_ids) + j                      # ids[pos] is this cont token; predicted by logp[pos-1]
        total += logp[pos - 1, t].item()
    return total

@torch.no_grad()
def greedy_generate(model, prompt_ids, max_new, device, stop_text="\n\n", tok=None):
    """simple O(n^2) greedy decode (no KV cache) — fine for small GSM8K subsets."""
    ids = list(prompt_ids)
    start = len(ids)
    for _ in range(max_new):
        nxt = int(seq_logprobs(model, ids, device)[-1].argmax())
        ids.append(nxt)
        if nxt == EOT:
            break
        if stop_text and tok is not None and stop_text in tok.decode(ids[start:]):
            break
    return ids[start:]

# ---------------- tasks ----------------
@torch.no_grad()
def eval_val_loss(model, device, val_file, n_seqs, seq_len):
    """mean next-token CE over held-out token slices. IDEALLY point val_file at data EXCLUDED from training."""
    data = np.memmap(val_file, dtype=np.uint16, mode="r")
    total_seqs = data.shape[0] // seq_len
    n = min(n_seqs, total_seqs)
    tot_loss, tot_tok = 0.0, 0
    for i in range(n):
        chunk = np.asarray(data[i * seq_len:(i + 1) * seq_len]).astype(np.int64).tolist()
        logp = seq_logprobs(model, chunk, device)                 # (seq_len, vocab)
        tgt = torch.tensor(chunk[1:], device=device)              # predict positions 1..seq_len-1
        ll = logp[torch.arange(seq_len - 1, device=device), tgt]  # log-prob of each true next token
        tot_loss += (-ll.sum()).item(); tot_tok += (seq_len - 1)
    loss = tot_loss / max(tot_tok, 1)
    return {"val/loss": loss, "val/perplexity": float(np.exp(loss))}

def _mmlu_fmt(q, choices, answer_idx=None):
    s = q.strip() + "\n"
    for i, c in enumerate(choices):
        s += f"{chr(65+i)}. {c}\n"
    s += "Answer:"
    if answer_idx is not None:
        s += f" {chr(65+answer_idx)}\n\n"
    return s

@torch.no_grad()
def eval_mmlu(model, tok, device, subjects, n_per_subject):
    from datasets import load_dataset
    # 1-FORWARD FAST PATH: the answer is a single token " A".." D" predicted right after "Answer:",
    # so one forward of the prompt + reading the 4 letter logits at the last position replaces 4 forwards.
    # Falls back to full multi-token loglikelihood if " X" isn't a single distinct token in this tokenizer.
    letter_ids = [tok.encode(f" {chr(65+i)}") for i in range(4)]
    single_tok = all(len(t) == 1 for t in letter_ids) and len({t[0] for t in letter_ids}) == 4
    print(f"[mmlu] 1-forward fast path: {'ON' if single_tok else 'OFF (letters not single tokens) -> 4 fwd/q'}")
    accs = []
    for subj in subjects:
        try:
            dev = load_dataset("cais/mmlu", subj, split="dev")      # 5 canonical few-shot exemplars
            test = load_dataset("cais/mmlu", subj, split="test")
        except Exception as e:
            print(f"[mmlu] skip {subj}: {e}"); continue
        shots = "".join(_mmlu_fmt(d["question"], d["choices"], d["answer"]) for d in dev)
        correct = 0; total = 0
        for d in test.select(range(min(n_per_subject, len(test)))):
            nc = len(d["choices"])
            ctx = tok.encode(shots + _mmlu_fmt(d["question"], d["choices"]))
            if single_tok and nc <= 4:
                last = seq_logprobs(model, ctx, device)[-1]        # distribution for the token after "Answer:"
                scores = [last[letter_ids[i][0]].item() for i in range(nc)]
            else:
                scores = [loglikelihood(model, ctx, letter_ids[i], device) for i in range(nc)]
            correct += int(int(np.argmax(scores)) == d["answer"]); total += 1
        if total:
            acc = correct / total; accs.append(acc)
            print(f"[mmlu] {subj}: {acc:.3f} ({total} q)")
    return {"mmlu/acc": float(np.mean(accs)) if accs else 0.0}

GSM8K_SHOTS = (
    "Question: Natalia sold clips to 48 friends in April, and half as many in May. How many total?\n"
    "Answer: In April she sold 48. In May she sold 48/2 = 24. Total = 48+24 = 72. #### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday she babysat 50 minutes. How much did she earn?\n"
    "Answer: Per minute she earns 12/60 = 0.2 dollars. For 50 minutes = 50*0.2 = 10. #### 10\n\n"
)
def _extract_num(text):
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None

@torch.no_grad()
def eval_gsm8k(model, tok, device, n, max_new=256):
    from datasets import load_dataset
    try:
        test = load_dataset("gsm8k", "main", split="test")
    except Exception as e:
        print(f"[gsm8k] load failed: {e}"); return {"gsm8k/acc": 0.0}
    correct = 0; total = 0
    for d in test.select(range(min(n, len(test)))):
        prompt = GSM8K_SHOTS + f"Question: {d['question']}\nAnswer:"
        gen = greedy_generate(model, tok.encode(prompt), max_new, device, stop_text="\n\n", tok=tok)
        pred = _extract_num(tok.decode(gen))
        gold = _extract_num(d["answer"])
        correct += int(pred is not None and gold is not None and pred == gold); total += 1
    acc = correct / max(total, 1)
    print(f"[gsm8k] acc {acc:.3f} ({total} q)")
    return {"gsm8k/acc": acc}

# ---------------- driver ----------------
def run_eval(ckpt_path, device, args):
    t0 = time.time()
    model, opt_step = build_and_load(ckpt_path, device)
    metrics = {}
    metrics.update(eval_val_loss(model, device, args.val_file, args.val_seqs, cfg.SEQ_LEN))
    if args.mmlu_n > 0:
        metrics.update(eval_mmlu(model, tok=load_tokenizer(), device=device,
                                 subjects=args.mmlu_subjects, n_per_subject=args.mmlu_n))
    if args.gsm8k_n > 0:
        metrics.update(eval_gsm8k(model, tok=load_tokenizer(), device=device, n=args.gsm8k_n))
    metrics["eval/opt_step"] = opt_step
    metrics["eval/seconds"] = round(time.time() - t0, 1)
    del model; torch.cuda.empty_cache()
    return opt_step, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint .pt (single-shot). If omitted with --loop, watches latest.pt")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--device", default="cuda:0")   # set via CUDA_VISIBLE_DEVICES to a spare/cool GPU
    ap.add_argument("--val_file", default="/home/rohit1/LLM_train/Data/overfit_10k_sequences.bin", help="POINT AT A HELD-OUT FILE for a true val signal")
    ap.add_argument("--val_seqs", type=int, default=200)
    ap.add_argument("--mmlu_n", type=int, default=100, help="test questions per subject (0 to skip)")
    ap.add_argument("--mmlu_subjects", nargs="+", default=[
        "abstract_algebra", "high_school_mathematics", "college_computer_science",
        "elementary_mathematics", "world_religions", "high_school_physics"])
    ap.add_argument("--gsm8k_n", type=int, default=100, help="0 to skip")
    ap.add_argument("--poll_sec", type=int, default=300)
    ap.add_argument("--wandb", action="store_true", help="log to wandb (project eval-<RUN_NAME>)")
    args = ap.parse_args()

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="llm_overfit", name=f"eval-{cfg.RUN_NAME}", id=f"eval-{cfg.RUN_NAME}", resume="allow")

    latest = os.path.join(cfg.CKPT_DIR, "latest.pt")
    if args.once or args.ckpt:
        opt_step, m = run_eval(args.ckpt or latest, args.device, args)
        print("METRICS:", m)
        if run: run.log(m, step=opt_step)
        return

    # loop: eval each new checkpoint
    seen = -1
    while True:
        if os.path.lexists(latest):
            try:
                st = int(torch.load(latest, map_location="cpu", weights_only=False).get("opt_step", -1))
            except Exception as e:
                print(f"[loop] could not read {latest}: {e}"); st = -1
            if st != seen and st >= 0:
                seen = st
                try:
                    opt_step, m = run_eval(latest, args.device, args)
                    print(f"[loop] opt_step {opt_step}:", m)
                    if run: run.log(m, step=opt_step)
                except Exception as e:
                    print(f"[loop] eval failed: {e}")
        time.sleep(args.poll_sec)

if __name__ == "__main__":
    main()
