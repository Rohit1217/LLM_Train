"""
Checkpoint save/load (NFS-safe, resumable across world-size changes).
Extracted from train_ddp.py. Public entry points:
    save_training_checkpoint(...)   -> background NFS-safe write, keyed by opt_step
    load_training_checkpoint(...)   -> robust resume with history fallback, returns opt_step
"""
import os
import glob
import shutil
import threading
import torch

from Config.config import Config

cfg = Config()   # same pattern as train_ddp; reads CKPT_DIR / MAX_CHECKPOINTS / WORLD_SIZE / ACCUMULATION_STEP / RESUME_CHECKPOINT / RUN_NAME


# ---------------- pruning ----------------
def _prune_checkpoints(ckpt_dir, max_keep):
    #delete old checkpoints, keep the most-recent max_keep by step
    def stepnum(f):
        try:
            return int(os.path.basename(f).split("step")[1].split(".pt")[0])
        except Exception:
            return -1

    files = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_step*.pt")), key=stepnum)

    if max_keep and len(files) > max_keep:
        for f in files[:len(files) - max_keep]:
            try: os.remove(f)
            except OSError: pass


# ---------------- save ----------------
def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _write_checkpoint(checkpoint, ckpt_dir, step, max_keep):
    #runs on a DAEMON thread: the slow, NFS-stall-prone disk write. If NFS hangs, only this thread
    #blocks (D-state) -> the training loop / NCCL barrier on the main thread keep running.
    named = os.path.join(ckpt_dir, f"checkpoint_step{step}.pt")
    latest = os.path.join(ckpt_dir, "latest.pt")
    try:
        tmp = named + ".tmp"
        torch.save(checkpoint, tmp)
        os.replace(tmp, named)
        tmpl = latest + ".lnk"
        try:
            if os.path.lexists(tmpl): os.remove(tmpl)
            os.symlink(os.path.basename(named), tmpl)
            os.replace(tmpl, latest)
        except OSError:
            shutil.copyfile(named, latest + ".tmp"); os.replace(latest + ".tmp", latest)
        _prune_checkpoints(ckpt_dir, max_keep)
    except Exception as e:
        print(f"[ckpt] background write failed (NFS?): {e}")


_save_thread = None
def save_training_checkpoint(master_param_1d, master_param_2d,
                             muon_optim_fp32, adamw_optim_fp32, wsd_scheduler_muon,
                             wsd_scheduler_adamw, opt_step, ckpt_dir=None, max_keep=None):
    #NFS-SAFE: prep a CPU snapshot on the MAIN thread (fast, bounded PCIe copy), then hand the slow
    #NFS torch.save to a daemon thread so an NFS stall can't D-state the training/NCCL loop.
    global _save_thread
    ckpt_dir = ckpt_dir if ckpt_dir is not None else cfg.CKPT_DIR
    max_keep = max_keep if max_keep is not None else cfg.MAX_CHECKPOINTS
    os.makedirs(ckpt_dir, exist_ok=True)

    #skip if the previous save is still writing (slow/stuck NFS) -> never queue up or block
    if _save_thread is not None and _save_thread.is_alive():
        print("[ckpt] previous save still in progress (slow NFS?), skipping this step")
        return

    #CPU SNAPSHOT ON MAIN THREAD (state is stable here: after optim step, before next) -> race-free
    checkpoint = {
        'opt_step': opt_step,
        'world_size': cfg.WORLD_SIZE,
        'accum': cfg.ACCUMULATION_STEP,
        'master_1d': [p.data.to("cpu", copy=True) for p in master_param_1d],
        'master_2d': [p.data.to("cpu", copy=True) for p in master_param_2d],
        'optimizer_muon_state_dict': _to_cpu(muon_optim_fp32.state_dict()),
        'optimizer_adam_state_dict': _to_cpu(adamw_optim_fp32.state_dict()),
        'admaw_scheduler_state_dict': wsd_scheduler_adamw.state_dict(),
        'muon_scheduler_state_dict': wsd_scheduler_muon.state_dict(),
        'rng_state': (torch.get_rng_state(), torch.cuda.get_rng_state().cpu()),
        'wandb_run_id': cfg.RUN_NAME,
    }

    _save_thread = threading.Thread(target=_write_checkpoint, args=(checkpoint, ckpt_dir, opt_step, max_keep), daemon=True)
    _save_thread.start()


# ---------------- load ----------------
def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _load_ckpt_dict(path, timeout_s=300):
    #threaded read w/ timeout so an NFS stall is a clear error, not an indefinite hang
    result = {}
    def _job():
        try: result["ckpt"] = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e: result["err"] = e

    t = threading.Thread(target=_job, daemon=True); t.start(); t.join(timeout_s)
    if t.is_alive(): raise TimeoutError(f"checkpoint read exceeded {timeout_s}s (NFS stall?): {path}")

    if "err" in result: raise result["err"]

    return result["ckpt"]


def _apply_checkpoint(ckpt, master_param_1d, master_param_2d, muon_optim_fp32, adamw_optim_fp32,
                      wsd_scheduler_muon, wsd_scheduler_adam, bf16_param_1d, bf16_param_2d, device):
    with torch.no_grad():
        for m, s in zip(master_param_1d, ckpt['master_1d']): m.data.copy_(s.to(device))
        for m, s in zip(master_param_2d, ckpt['master_2d']): m.data.copy_(s.to(device))
    #optim state loaded to CPU move to device so it matches the GPU params
    muon_optim_fp32.load_state_dict(_to_device(ckpt['optimizer_muon_state_dict'], device))
    adamw_optim_fp32.load_state_dict(_to_device(ckpt['optimizer_adam_state_dict'], device))

    with torch.no_grad():
        torch._foreach_copy_(bf16_param_2d, master_param_2d)
        torch._foreach_copy_(bf16_param_1d, master_param_1d)

    wsd_scheduler_adam.load_state_dict(ckpt["admaw_scheduler_state_dict"])
    wsd_scheduler_muon.load_state_dict(ckpt["muon_scheduler_state_dict"])

    torch.set_rng_state(ckpt['rng_state'][0].cpu())
    torch.cuda.set_rng_state(ckpt['rng_state'][1].cpu())

    return ckpt['opt_step']


def load_training_checkpoint(master_param_1d, master_param_2d, muon_optim_fp32, adamw_optim_fp32,
                             wsd_scheduler_muon, wsd_scheduler_adam, bf16_param_1d, bf16_param_2d, device,
                             ckpt_dir=None, resume_name=None):
    ckpt_dir = ckpt_dir if ckpt_dir is not None else cfg.CKPT_DIR
    resume_name = resume_name if resume_name is not None else cfg.RESUME_CHECKPOINT

    def stepnum(f):
        try: return int(os.path.basename(f).split("step")[1].split(".pt")[0])
        except Exception: return -1

    chosen = os.path.join(ckpt_dir, resume_name if resume_name else "latest.pt")
    named = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_step*.pt")), key=stepnum, reverse=True)
    candidates = []

    for c in [chosen] + named:
        if c not in candidates: candidates.append(c)

    existed = any(os.path.lexists(c) for c in candidates)

    for path in candidates:
        if not os.path.lexists(path): continue
        try:
            ckpt = _load_ckpt_dict(path)
            step = _apply_checkpoint(ckpt, master_param_1d, master_param_2d, muon_optim_fp32, adamw_optim_fp32,
                                     wsd_scheduler_muon, wsd_scheduler_adam, bf16_param_1d, bf16_param_2d, device)
            print(f"RESUMED from {path} at step {step}")
            return step
        except Exception as e:
            print(f"[ckpt] could not load {path}: {e} -> trying older")
    if existed:
        raise RuntimeError(f"checkpoints exist in {ckpt_dir} but NONE could be loaded; refusing to silently "
                           f"restart from step 0. Point RESUME_CHECKPOINT at a good one or move the bad files.")
    print(f"[ckpt] no checkpoint in {ckpt_dir} -> fresh start (step 0)")
    return 0