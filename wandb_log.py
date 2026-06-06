import subprocess
import torch
from collections import defaultdict


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


#RUN HYGIENE — log once at start so dashboards self-document
def log_run_config(run, *, model, cfg, num_params, num_non_embed_params):
    try:
        import triton
        triton_ver = triton.__version__
    except Exception:
        triton_ver = "n/a"
    dtype = str(next(model.parameters()).dtype)
    run.summary.update({
        "model/num_params": num_params,
        "model/num_non_embed_params": num_non_embed_params,
        "model/d_model": cfg.D_MODEL,
        "model/n_heads": cfg.N_HEAD,
        "model/num_layers": cfg.NUM_LAYERS,
        "model/ffn_hidden_dim": cfg.FFN_HIDDEN_DIM,
        "model/vocab_size": cfg.VOCAB_SIZE,
        "model/seq_len": cfg.SEQ_LEN,
        "model/batch_size": cfg.BATCH_SIZE,
        "model/dtype": dtype,
        "env/torch": torch.__version__,
        "env/cuda": torch.version.cuda,
        "env/triton": triton_ver,
        "env/git_commit": _git_commit(),
        "env/seed": cfg.SEED,
        "target/throughput_tok_s": cfg.THROUGHPUT_TARGET,
        "target/peak_mfu": cfg.PEAK_MFU_TARGET,
    })


def compute_mfu(step_time, seq_len, batch_size, num_params, peak_flops):
    return (6 * batch_size * seq_len * num_params) / (peak_flops * step_time)


#PER-STEP SCALARS (loss / perf / lr / global grad norm)
def log_step_metrics(run, step, *, loss, grad_norm, lr_adamw, lr_muon,
                     tokens_seen, throughput, step_time, data_io, mfu):
    run.log({
        "perf/tokens_seen": tokens_seen,
        "loss/ce_loss": loss,
        "loss/perplexity": torch.exp(loss + 1e-9),
        "grad/global_grad_norm": grad_norm,
        "opt/last_lr_adamw": lr_adamw,
        "opt/last_lr_muon": lr_muon,
        "perf/token_throughput": throughput,
        "perf/step_time": step_time,
        "perf/data_io": data_io,
        "perf/MFU": mfu,
    }, step=step, commit=False)


#ACTIVATION DIAGNOSTICS — eager forward, hooks attached only here (never touch compiled path)
def log_activations(run, step, model, x):
    act_norm, act_max, handles = {}, {}, []
    for i, b in enumerate(model.transformer_block_list):
        def make(i):
            def hook(m, inp, out):
                act_norm[f"res_norm/block_{i}"] = out.norm(dim=-1).mean().item()
                act_max[f"res_max/block_{i}"] = out.abs().max().item()
            return hook
        handles.append(b.register_forward_hook(make(i)))

    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    run.log({**act_norm, **act_max}, step=step, commit=False)


#PARAM-GROUP DIAGNOSTICS — grad / param / true update-ratio norms
def snapshot_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

def _group_of(n):
    if "embed" in n:                       return "embed"
    if "qkv" in n:                         return "attn_qkv"
    if "att" in n and "linear_proj" in n:  return "attn_out"
    if "swig" in n:                        return "ffn_in"
    if "fc" in n:                          return "ffn_out"
    if "norm" in n:                        return "norm"
    return "other"


def log_param_diagnostics(run, step, model, prev, device):
    z = lambda: defaultdict(lambda: torch.zeros((), device=device))
    gn, pn, un = z(), z(), z()
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = _group_of(n)
        gn[g] += p.grad.detach().float().pow(2).sum()
        pn[g] += p.detach().float().pow(2).sum()
        un[g] += (p.detach().float() - prev[n].float()).pow(2).sum()
    diag = {}
    for g in pn:
        pnorm = pn[g].sqrt().item()
        diag[f"grad_norm/{g}"] = gn[g].sqrt().item()
        diag[f"param_norm/{g}"] = pnorm
        diag[f"upd_ratio/{g}"] = un[g].sqrt().item() / (pnorm + 1e-12)
    run.log(diag, step=step, commit=False)


#FLUSH all commit=False logs accumulated for this step
def commit(run, step):
    run.log({}, step=step)
