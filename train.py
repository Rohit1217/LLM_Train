import os

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["TORCH_USE_CUDA_DSA"]="1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_VERBOSE"]="1"
import torch

from models_fast import Transformer
from Data.data import load_data
from Data.process_tiny_shakespeare import generate_shakespeare_dataset

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW,Muon
from torch.nn.attention import SDPBackend, sdpa_kernel

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR,ConstantLR,StepLR,SequentialLR
from config import Config 

from tqdm import tqdm
from dataclasses import asdict
from itertools import cycle
import time

import wandb
import wandb_log as wl



cfg=Config()
torch.manual_seed(cfg.SEED)

#MODEL
run_id="Run_overfit-1-mtp"
run = wandb.init(
    entity="rohit_iisc-indian-institute-of-science",
    project="llm_overfit",
    id=run_id,                                
    config=asdict(cfg)
)


dataset=generate_shakespeare_dataset(cfg.EFF_SEQ_LEN,cfg.BATCH_SIZE)
dataloader=load_data(dataset,cfg.BATCH_SIZE)
#MODEL DEFINIION
transformer_model=Transformer(vocab_size=cfg.VOCAB_SIZE,max_context=cfg.MAX_CONTEXT,
                              max_freq=cfg.MAX_FREQ,d_model=cfg.D_MODEL,n_heads=cfg.N_HEAD,
                              num_layers=cfg.NUM_LAYERS,attn_dropout=cfg.ATT_DROPOUT,
                              ffn_hidden_dim=cfg.FFN_HIDDEN_DIM,ffn_dropout=cfg.FFN_DROPOUT,mtp_heads=cfg.MTP_HEADS)

transformer_model=transformer_model.to(cfg.DEVICE)

num_params=sum([p.numel() for p in transformer_model.parameters()])
num_non_embed_params=num_params-cfg.VOCAB_SIZE*cfg.D_MODEL

#log model dims, dtype, versions, git commit, seed, targets
wl.log_run_config(run,model=transformer_model,cfg=cfg,
                  num_params=num_params,num_non_embed_params=num_non_embed_params)

#PARAM SEPERATION FOR MUON AND ADAMW
param_2d=[p for n,p in transformer_model.named_parameters() if p.ndim==2 and "embed" not in n]
param_1d_embed=[p for n,p in transformer_model.named_parameters() if p.ndim==1 or "embed" in n]

#OPTIMIZER ADAMW AND MUON
adamw_optim_bf16=AdamW(param_1d_embed,lr=2e-4,betas=[0.9,0.95],weight_decay=0.1)
muon_optim_bf16=Muon(param_2d,lr=2e-4,weight_decay=0.1,momentum=0.95,adjust_lr_fn="match_rms_adamw")

transformer_model=transformer_model.to(dtype=torch.bfloat16)
transformer_model.buffers_to_float()
# transformer_model.embedding=transformer_model.embedding.to(dtype=torch.float32)


#WSD SCHEDULER ADAM AND MUON
warmup_scheduler_adam=LinearLR(adamw_optim_bf16,start_factor=1e-4,end_factor=1,
                               total_iters=cfg.WARMUP_STEPS)
stable_scheduler_adam=ConstantLR(adamw_optim_bf16,factor=1,
                                 total_iters=cfg.STABLE_STEPS)
decay_scheduler_adam=LinearLR(adamw_optim_bf16,start_factor=1,
                              end_factor=1e-4,total_iters=cfg.DECAY_STEPS)

wsd_scheduler_adam=SequentialLR(adamw_optim_bf16,[warmup_scheduler_adam,stable_scheduler_adam,
                                             decay_scheduler_adam],
                                milestones=[cfg.WARMUP_STEPS,cfg.WARMUP_STEPS+cfg.STABLE_STEPS])

warmup_scheduler_muon=LinearLR(muon_optim_bf16,start_factor=1e-4,end_factor=1,
                               total_iters=cfg.WARMUP_STEPS)
stable_scheduler_muon=ConstantLR(muon_optim_bf16,factor=1,
                                 total_iters=cfg.STABLE_STEPS)
decay_scheduler_muon=LinearLR(muon_optim_bf16,start_factor=1,
                              end_factor=1e-4,total_iters=cfg.DECAY_STEPS)

wsd_scheduler_muon=SequentialLR(muon_optim_bf16,[warmup_scheduler_muon,stable_scheduler_muon,decay_scheduler_muon],
                                milestones=[cfg.WARMUP_STEPS,cfg.WARMUP_STEPS+cfg.STABLE_STEPS])



#LIGER FUSED CE KERNEL NAIN AND MTP
lse_square_scale=1e-4
liger_fused_ce_main=LigerFusedLinearCrossEntropyLoss(lse_square_scale=1e-4,return_z_loss=True)
liger_fused_ce_mtp=LigerFusedLinearCrossEntropyLoss()


#TRAIN STEP FOR TORCH COMPILE OPTIMIZATION
def train_step(transformer_model,x,d_model,mtp_k=0,mtp_weight=0):
    B,T=x.shape

    y_main=x[:,1:T-mtp_k].contiguous().view(-1) #SHIFT BY ONE TEACHER FORCING

    hidden_states=transformer_model(x)                    

    out=liger_fused_ce_main(transformer_model.embedding.weight,
                       hidden_states[:B,:,:].view(-1,d_model),y_main)
    main_loss,main_z_loss=out.loss,out.z_loss
    loss=main_loss
    
    if mtp_weight>0:
        y_mtp=x[:,2:].unfold(1,mtp_k,1).permute(2,0,1).contiguous().view(-1) # SHIFT BY ONE FOR EACH MTP THEN PERMUTE TO GET MTP0|MTP!.. ORDERIING

        mtp_loss=liger_fused_ce_mtp(transformer_model.embedding.weight,
                            hidden_states[B:,:,:].view(-1,d_model),y_mtp)
        loss=loss+mtp_weight*mtp_loss
    
    loss.backward()

    logsumexp_avg=main_z_loss/lse_square_scale
    return loss,logsumexp_avg

#TORCH COMPILE — COMPILE MODEL FORWARD ONLY. Liger graph-breaks on its internal .item();
#with 2 Liger calls (main+mtp) the resume frame wraps a Triton kernel and crashes Inductor's
#decompose_triton_kernel_wrapper_functional pass. Compiling the model alone captures ~all FLOPs;
#loss+backward run eager (Liger is its own fused kernel; compiled fwd keeps its AOT backward).
optimized_model=torch.compile(transformer_model,mode="max-autotune-no-cudagraphs")

tokens_seen=0
data_iter=cycle(dataloader)

LOG_EVERY=2000
ema_loss=0
ema_token_throughput=0
alpha=0.05


def save_model(model):
    torch.save(model.state_dict(), "model_weights.pth")



with sdpa_kernel(SDPBackend.FLASH_ATTENTION):

    for global_step in tqdm(range(cfg.TOTAL_STEPS)):
        s_time=time.time()

        #ZERO_GRAD
        muon_optim_bf16.zero_grad()
        adamw_optim_bf16.zero_grad()

        #DATA LOAD TO GPU
        x=next(data_iter)
        shard_ids=x[0]    
        x=x[1].to(cfg.DEVICE)

        #OPTIMIZED FORWARD BACKWARD PASS
        fwd_start=time.time()
        loss,logsumexp_avg=train_step(optimized_model,x,d_model=cfg.D_MODEL,
                                      mtp_weight=cfg.MTP_LOSS_WEIGHT,mtp_k=cfg.MTP_HEADS)

        #CLIP NORM
        norm=torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0) #CLIP NORM

        #SNAPSHOT + ACTIVATIONS ON PRE-STEP WEIGHTS 
        if (global_step+1) % LOG_EVERY == 0:
            save_model(transformer_model)
            prev=wl.snapshot_params(transformer_model)
            wl.log_activations(run,global_step,transformer_model,x)

        #OPTIM STEP
        muon_optim.step()
        adamw_optim.step()

        #SCHEDULER STEP
        wsd_scheduler_adam.step()
        wsd_scheduler_muon.step()

        # #VALIDATION
        # if (step%1000)==0:
        #     val_loss=eval(model)

        #LOGGING
        # torch.cuda.synchronize()
        f_time=time.time()
        tokens_seen+=cfg.BATCH_SIZE*cfg.EFF_SEQ_LEN
        step_time=round(f_time-s_time,4)
        data_io_time=fwd_start-s_time
        mfu=wl.compute_mfu(step_time,cfg.EFF_SEQ_LEN,cfg.BATCH_SIZE,num_non_embed_params,cfg.A6000_BF16_PEAK)
        token_throughput=(cfg.BATCH_SIZE*cfg.EFF_SEQ_LEN)/step_time

        loss_train=loss.item()
        #RAW BIASED EMA CARRIES FORWARD; BIAS-CORRECT ONLY FOR LOGGING
        ema_loss=ema_loss*(1-alpha) + alpha*loss_train
        ema_token_throughput=ema_token_throughput*(1-alpha) + alpha*token_throughput
        bias_corr=1-(1-alpha)**(global_step+1)
        ema_loss_log=ema_loss/bias_corr
        ema_throughput_log=ema_token_throughput/bias_corr
        data_io_frac=data_io_time/step_time

        wl.log_step_metrics(run,global_step,loss=loss,grad_norm=norm,
                            lr_adamw=wsd_scheduler_adam.get_last_lr()[0],
                            lr_muon=wsd_scheduler_muon.get_last_lr()[0],
                            tokens_seen=tokens_seen,throughput=token_throughput,
                            step_time=step_time,data_io=data_io_frac,mfu=mfu)

        run.log({"loss/ema_ce":ema_loss_log,"perf/ema_throughput":ema_throughput_log,
                "loss/logsumexp_sq":logsumexp_avg},step=global_step,commit=False)

        #GROUPED GRAD / PARAM / TRUE UPDATE-RATIO NORMS
        if (global_step+1) % LOG_EVERY == 0:
            wl.log_param_diagnostics(run,global_step,transformer_model,prev,cfg.DEVICE)
            del prev

        run.log({"data/shard_ids":shard_ids},step=global_step,commit=False)
        wl.commit(run,global_step)


