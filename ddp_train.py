import torch

from models_fast import Transformer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW,Muon
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR,ConstantLR,StepLR,SequentialLR
from config import Config 

from tqdm import tqdm
from dataclasses import asdict
import time

import wandb
import wandb_log as wl

cfg=Config()
torch.manual_seed(cfg.SEED)

#MODEL

run = wandb.init(
    entity="rohit_iisc-indian-institute-of-science",
    project="llm_overfit",
    config=asdict(cfg)
)


transformer_model=Transformer(vocab_size=cfg.VOCAB_SIZE,max_context=cfg.MAX_CONTEXT,max_freq=cfg.MAX_FREQ,
                              d_model=cfg.D_MODEL,n_heads=cfg.N_HEAD,num_layers=cfg.NUM_LAYERS,attn_dropout=cfg.ATT_DROPOUT,
                              ffn_hidden_dim=cfg.FFN_HIDDEN_DIM,ffn_dropout=cfg.FFN_DROPOUT)

num_params=sum([p.numel() for p in transformer_model.parameters()])
num_non_embed_params=num_params-cfg.VOCAB_SIZE*cfg.D_MODEL

#log model dims, dtype, versions, git commit, seed, targets
wl.log_run_config(run,model=transformer_model,cfg=cfg,
                  num_params=num_params,num_non_embed_params=num_non_embed_params)

#PARAM SEPERATION FOR MUON AND ADAMW
param_2d=[p for n,p in transformer_model.named_parameters() if p.ndim==2 and "embed" not in n]
param_1d_embed=[p for n,p in transformer_model.named_parameters() if p.ndim==1 or "embed" in n]

#OPTIMIZER ADAMW AND MUON
adamw_optim=AdamW(param_1d_embed,lr=2e-4,betas=[0.9,0.95],weight_decay=0.1)
muon_optim=Muon(param_2d,lr=2e-4,weight_decay=0.1,momentum=0.95,adjust_lr_fn="match_rms_adamw")


#WSD SCHEDULER
warmup_scheduler_adam=LinearLR(adamw_optim,0,1,1,total_iters=cfg.WARMUP_STEPS)
stable_scheduler_adam=ConstantLR(adamw_optim,factor=1,total_iters=cfg.STABLE_STEPS)
decay_scheduler_adam=LinearLR(adamw_optim,start_factor=1,end_factor=0.01,total_iters=cfg.DECAY_STEPS)

wsd_scheduler_adam=SequentialLR(adamw_optim,[warmup_scheduler_adam,stable_scheduler_adam,decay_scheduler_adam],milestones=[cfg.WARMUP_STEPS,cfg.WARMUP_STEPS+cfg.STABLE_STEPS,cfg.WARMUP_STEPS+cfg.STABLE_STEPS+cfg.DECAY_STEPS])

warmup_scheduler_muon=LinearLR(muon_optim,0,1,1,total_iters=cfg.WARMUP_STEPS)
stable_scheduler_muon=ConstantLR(muon_optim,factor=1,total_iters=cfg.STABLE_STEPS)
decay_scheduler_muon=LinearLR(muon_optim,start_factor=1,end_factor=0.01,total_iters=cfg.DECAY_STEPS)

wsd_scheduler_muon=SequentialLR(muon_optim,[warmup_scheduler_muon,stable_scheduler_muon,decay_scheduler_muon],milestones=[cfg.WARMUP_STEPS,cfg.WARMUP_STEPS+cfg.STABLE_STEPS,cfg.WARMUP_STEPS+cfg.STABLE_STEPS+cfg.DECAY_STEPS])



#LIGER FUSED CE KERNEL
lse_square_scale=1e-4
liger_fused_ce=LigerFusedLinearCrossEntropyLoss(lse_square_scale=1e-4,return_z_loss=True)
#TRAIN STEP FOR TORCH COMPILE OPTIMIZATION
def train_step(transformer_model,y,x,d_model):
    hidden_state=transformer_model(x)
    loss,z_loss=liger_fused_ce(transformer_model.embedding.weight,hidden_state.view(-1,d_model),y)
    loss.backward()
    logsumexp_avg=z_loss/lse_square_scale
    return loss,logsumexp_avg

#TORCH COMPILE
optimized_train_step=torch.compile(train_step,mode="max-autotune-no-cudagraphs")

tokens_seen=0
dataloader=None
data_iter=iter(dataloader)

LOG_EVERY=2000
ema_loss=0
ema_token_throughput=0
alpha=0.05


for step in tqdm(range(cfg.TOTAL_STEPS)):
    s_time=time.time()

    #ZERO_GRAD
    muon_optim.zero_grad()
    adamw_optim.zero_grad()

    #DATA LOAD TO GPU
    x,y=next(data_iter)
    shard_id=x[0]    
    x,y=x[1].to(cfg.DEVICE),y.to(cfg.DEVICE)

    #OPTIMIZED FORWARD BACKWARD PASS
    fwd_start=time.time()
    loss,logsumexp_avg=optimized_train_step(transformer_model,y,x,d_model=cfg.D_MODEL)

    #CLIP NORM
    norm=torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0) #CLIP NORM

    #SNAPSHOT + ACTIVATIONS ON PRE-STEP WEIGHTS 
    if (step+1) % LOG_EVERY == 0:
        prev=wl.snapshot_params(transformer_model)
        wl.log_activations(run,step,transformer_model,x)

    #OPTIM STEP
    muon_optim.step()
    adamw_optim.step()

    #SCHEDULER STEP
    wsd_scheduler_adam.step()
    wsd_scheduler_muon.step()

    #LOGGING
    torch.cuda.synchronize()
    f_time=time.time()
    tokens_seen+=cfg.BATCH_SIZE*cfg.SEQ_LEN
    step_time=round(f_time-s_time,4)
    data_io_time=fwd_start-s_time
    mfu=wl.compute_mfu(step_time,cfg.SEQ_LEN,cfg.BATCH_SIZE,num_non_embed_params,cfg.A6000_BF16_PEAK)
    token_throughput=(cfg.BATCH_SIZE*cfg.SEQ_LEN)/step_time

    loss_val=loss.item()
    #RAW BIASED EMA CARRIES FORWARD; BIAS-CORRECT ONLY FOR LOGGING
    ema_loss=ema_loss*(1-alpha) + alpha*loss_val
    ema_token_throughput=ema_token_throughput*(1-alpha) + alpha*token_throughput
    bias_corr=1-(1-alpha)**(step+1)
    ema_loss_log=ema_loss/bias_corr
    ema_throughput_log=ema_token_throughput/bias_corr
    data_io_frac=data_io_time/step_time

    wl.log_step_metrics(run,step,loss=loss,grad_norm=norm,
                        lr_adamw=wsd_scheduler_adam.get_last_lr()[0],
                        lr_muon=wsd_scheduler_muon.get_last_lr()[0],
                        tokens_seen=tokens_seen,throughput=token_throughput,
                        step_time=step_time,data_io=data_io_frac,mfu=mfu)

    run.log({"loss/ema_ce":ema_loss_log,"perf/ema_throughput":ema_throughput_log,
             "loss/logsumexp_sq":logsumexp_avg},step=step,commit=False)

    #GROUPED GRAD / PARAM / TRUE UPDATE-RATIO NORMS
    if (step+1) % LOG_EVERY == 0:
        wl.log_param_diagnostics(run,step,transformer_model,prev,cfg.DEVICE)
        del prev

    run.log({"data/shard_id":shard_id},step=step,commit=False)
    wl.commit(run,step)


