from experiment1 import make_vit,give_transforms,give_optim_scheduler,train_model
from models.vit import VIT
from models.data import load_dataloaders

import torch,torchvision
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from torch.optim import AdamW
import torchvision.transforms.v2 as tf
import config.config as cfg
import time

import os
from tqdm import tqdm
import matplotlib.pyplot as plt

print(torchvision.__version__)

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_inductor_cache"
torch.set_float32_matmul_precision('high')


Device="cuda:1"
num_epochs=cfg.NUM_EPOCHS
warmup_epochs=cfg.WARMUP_EPOCHS
weight_decay=cfg.WEIGHT_DECAY
lr=cfg.LR

batch_size=cfg.BATCH_SIZE
num_workers=cfg.NUM_WORKERS
data_frac=cfg.DATA_FRAC

def make_vit(patch_size):
    # Build ViT with the given patch_size; all other hyperparams from config
    vit_cifar=VIT(image_size=cfg.VIT_IMAGE_SIZE,patch_size=patch_size,
                  num_classes=cfg.VIT_NUM_CLASSES,num_blocks=cfg.VIT_NUM_BLOCKS,embed_dim=cfg.VIT_EMBED_DIM,
                  n_heads=cfg.VIT_N_HEADS,hidden_dim=cfg.VIT_HIDDEN_DIM,
                  max_seq_len=cfg.VIT_MAX_SEQ_LEN).to(Device)
    
    vit_cifar=torch.compile(vit_cifar)
    return vit_cifar





RESULTS_DIR = "results/experiment2"
NUM_TRIALS = 5  # run each patch size multiple times to get mean/std accuracy

# Experiment 2: Effect of patch size on ViT accuracy and training time.
# Trains ViT at patch sizes 4, 8, 16 for NUM_TRIALS each; all other params fixed.
def run(patch_size_list=[4,8,16]):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # {patch_size: [acc_trial1, acc_trial2, ...]}
    vit_acc_dict={}
    vit_loss_dict={}
    vit_time_dict={}

    for patch_size in patch_size_list:
        trial_accs=[]
        trial_losses=[]
        trial_times=[]

        for trial in range(NUM_TRIALS):
            s_time=time.time()

            print(f"\n{'='*60}")
            print(f"Patch Size: {patch_size} | Trial {trial+1}/{NUM_TRIALS}")
            print(f"{'='*60}")

            dataloaders=load_dataloaders(batch_size=batch_size,num_workers=num_workers)

            vit_cifar=make_vit(patch_size)

            vit_acc,vit_loss_list = train_model(vit_cifar,dataloaders,num_epochs,model_type="VIT",Device=Device)

            trial_accs.append(vit_acc)
            trial_losses.append(vit_loss_list)
            trial_times.append(time.time()-s_time)

            print(f"ViT  best acc (trial {trial+1}): {vit_acc:.4f}")

        vit_acc_dict[patch_size]=trial_accs
        vit_loss_dict[patch_size]=trial_losses
        vit_time_dict[patch_size]=trial_times

    # ── Plot: avg loss curves per patch size ───────────────────────────────────
    plt.figure(figsize=(8, 5))
    for patch_size in patch_size_list:
        losses=torch.tensor(vit_loss_dict[patch_size])  # (trials, epochs)
        avg_loss=losses.mean(dim=0).tolist()
        plt.plot(avg_loss, label=f"patch={patch_size}")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Training Loss")
    plt.title(f"Training Loss vs Epoch — ViT Patch Size Comparison ({NUM_TRIALS} trials)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(RESULTS_DIR, "experiment2_loss_curves.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.show()
    print(f"\nLoss plot saved to {loss_plot_path}")

    # ── Log results to file ────────────────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "results.txt")
    with open(log_path, "w") as f:
        f.write(f"{'Patch Size':>12} | {'Avg Val Acc':>12} | {'Std Val Acc':>12} | {'Avg Time (s)':>14}\n")
        f.write("-" * 58 + "\n")
        for patch_size in patch_size_list:
            accs=torch.tensor(vit_acc_dict[patch_size])
            times=torch.tensor(vit_time_dict[patch_size])
            f.write(f"{patch_size:>12} | {accs.mean():>12.4f} | {accs.std():>12.4f} | {times.mean():>14.1f}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Patch Size':>12} | {'Avg Val Acc':>12} | {'Std Val Acc':>12} | {'Avg Time (s)':>14}")
    print("-" * 58)
    for patch_size in patch_size_list:
        accs=torch.tensor(vit_acc_dict[patch_size])
        times=torch.tensor(vit_time_dict[patch_size])
        print(f"{patch_size:>12} | {accs.mean():>12.4f} | {accs.std():>12.4f} | {times.mean():>14.1f}")
    print(f"\nResults logged to {log_path}")



if __name__=="__main__":
    run()