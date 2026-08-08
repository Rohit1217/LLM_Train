import os
from contextlib import nullcontext

# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["TORCH_USE_CUDA_DSA"]="1"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_VERBOSE"]="1"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler,DataLoader

from model import Transformer
from Data.data import load_data,sharded_dataset
from Data.dataload import make_dataloader
from Data.process_tiny_shakespeare import generate_shakespeare_dataset

from torch.optim import AdamW,Muon
from torch.nn.attention import SDPBackend, sdpa_kernel

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from torch.optim.lr_scheduler import LinearLR,ConstantLR,SequentialLR
from Config.config import Config

import numpy as np

from tqdm import tqdm
from dataclasses import asdict
import time
import glob
import shutil
import threading
from datetime import timedelta

import wandb
import wandb_log as wl
from load_save_ckpt import save_training_checkpoint, load_training_checkpoint

cfg=Config()
torch.manual_seed(cfg.SEED)

OFFSET=1            
HOST_LOCAL_RANK=1   
def setup():
    local_rank=int(os.environ["LOCAL_RANK"])+OFFSET
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",device_id=torch.device(f"cuda:{local_rank}"),
                            timeout=timedelta(minutes=30))
    return local_rank

def cleanup():
    dist.destroy_process_group()

def infinite(dataloader):
    while True:
        for batch in dataloader:
            yield batch

#MODEL
def setup_wandb(run_id=cfg.RUN_NAME,entity=os.environ.get("WANDB_ENTITY"),project_id=os.environ.get("WANDB_PROJECT","llm_train")):
    run_id=run_id
    run = wandb.init(
        entity=entity,
        project=project_id,
        id=run_id,
        mode="offline" if cfg.WANDB_OFFLINE else "online",  
        config=asdict(cfg)
    )
    return run

def load_dataset(rank,type="shakespeare",eot_id=None,resume_step=0):
    if type=="shakespeare":
        rank=rank-OFFSET
        tokens, shard_ids = generate_shakespeare_dataset(cfg.EFF_SEQ_LEN, cfg.BATCH_SIZE)
        ds = sharded_dataset(tokens, shard_ids)              # len = n_seq (~1086)
        sampler = DistributedSampler(ds, num_replicas=dist.get_world_size(),
                                    rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(ds, batch_size=cfg.BATCH_SIZE,
                                sampler=sampler, drop_last=True, pin_memory=True)
    elif type=="tokens_overfit":
        rank=rank-OFFSET
        world=dist.get_world_size()
        data = np.fromfile(cfg.OVERFIT_FILE_PATH, dtype=np.uint16)
        total_seqs=data.shape[-1]//(cfg.EFF_SEQ_LEN)

        perm=np.random.default_rng(cfg.SEED).permutation(total_seqs).astype(np.int32)        
        dataloader=make_dataloader(cfg.EFF_SEQ_LEN,cfg.BATCH_SIZE,eot_id,rank,world,perm,cfg.TOTAL_STEPS,
                                   start_idx=resume_step*cfg.BATCH_SIZE,arr=data,mtp_k=cfg.MTP_HEADS)
    elif type=="main_run":
        rank=rank-OFFSET   #GLOBAL DEVICE INDEX -> LOCAL rank 0..world-1
        world=dist.get_world_size()

        #ONE SHARED PHYSICAL COPY IN RAM: 
        data=np.memmap(cfg.MAIN_FILE_PATH,dtype=np.uint16,mode="r")

        total_seqs=data.shape[-1]//cfg.EFF_SEQ_LEN
        #GLOBAL permutation, identical on every rank(seed).Consumed as a single prefix stream 
        perm=np.random.default_rng(cfg.SEED).permutation(total_seqs).astype(np.int32)

        start_idx=resume_step*cfg.BATCH_SIZE
        #WARM WHOLE FILE INTO PAGE CACHE ONCE.
        n=data.shape[0]
        data[rank*n//world:(rank+1)*n//world].sum()
        dist.barrier()   #all pages loaded before ddp starts

        dataloader=make_dataloader(cfg.EFF_SEQ_LEN,cfg.BATCH_SIZE,eot_id,rank,world,perm,cfg.TOTAL_STEPS,
                                   start_idx=start_idx,arr=data,mtp_k=cfg.MTP_HEADS)
    return dataloader

def load_model(run,device):
    transformer_model=Transformer(vocab_size=cfg.VOCAB_SIZE,max_context=cfg.MAX_CONTEXT,
                                max_freq=cfg.MAX_FREQ,d_model=cfg.D_MODEL,n_heads=cfg.N_HEAD,
                                num_layers=cfg.NUM_LAYERS,attn_dropout=cfg.ATT_DROPOUT,
                                ffn_hidden_dim=cfg.FFN_HIDDEN_DIM,ffn_dropout=cfg.FFN_DROPOUT,mtp_heads=cfg.MTP_HEADS,
                                grad_checkpoint_every=cfg.GRAD_CHECKPOINT_EVERY,group_size=cfg.NUM_GROUPS,max_len=cfg.MAX_CONTEXT,
                                intradoc_att=cfg.INTRADOC_ATT)

    transformer_model=transformer_model.to(device)

    num_params=sum([p.numel() for p in transformer_model.parameters()])
    num_non_embed_params=num_params-cfg.VOCAB_SIZE*cfg.D_MODEL

    if run is not None:   
        wl.log_run_config(run,model=transformer_model,cfg=cfg,
                        num_params=num_params,num_non_embed_params=num_non_embed_params)
    
    return transformer_model,num_params,num_non_embed_params



def load_optim(transformer_model):
    # PARAM SEPERATION FOR MUON AND ADAMW
    master_param_2d=[p.detach().clone().float().requires_grad_(True) for n,p in transformer_model.named_parameters() if p.ndim==2 and "embed" not in n]
    master_param_1d=[p.detach().clone().float().requires_grad_(True) for n,p in transformer_model.named_parameters() if p.ndim==1 or "embed" in n]

    adamw_optim_fp32=AdamW(master_param_1d,lr=cfg.LR,betas=[0.9,0.95],weight_decay=cfg.WEIGHT_DECAY)
    muon_optim_fp32=Muon(master_param_2d,lr=cfg.LR,weight_decay=cfg.WEIGHT_DECAY,momentum=0.95,adjust_lr_fn="match_rms_adamw",ns_steps=3)

    transformer_model=transformer_model.to(dtype=torch.bfloat16)
    transformer_model.buffers_to_float()

    bf16_param_2d=[p for n,p in transformer_model.named_parameters() if p.ndim==2 and "embed" not in n]
    bf16_param_1d=[p for n,p in transformer_model.named_parameters() if p.ndim==1 or "embed" in n]

    return master_param_2d,master_param_1d,adamw_optim_fp32,muon_optim_fp32,bf16_param_1d,bf16_param_2d,transformer_model
    

    #WSD SCHEDULER ADAM AND MUON
def load_warmup_scheduler(adamw_optim_fp32,muon_optim_fp32):
    warmup_scheduler_adam=LinearLR(adamw_optim_fp32,start_factor=1e-4,end_factor=1,
                                total_iters=cfg.WARMUP_STEPS)
    stable_scheduler_adam=ConstantLR(adamw_optim_fp32,factor=1,
                                    total_iters=cfg.STABLE_STEPS)
    decay_scheduler_adam=LinearLR(adamw_optim_fp32,start_factor=1,
                                end_factor=1e-4,total_iters=cfg.DECAY_STEPS)

    wsd_scheduler_adam=SequentialLR(adamw_optim_fp32,[warmup_scheduler_adam,stable_scheduler_adam,
                                                decay_scheduler_adam],
                                    milestones=[cfg.WARMUP_STEPS,cfg.WARMUP_STEPS+cfg.STABLE_STEPS])

    warmup_scheduler_muon=LinearLR(muon_optim_fp32,start_factor=1e-4,end_factor=1,
                                total_iters=cfg.WARMUP_STEPS)
    stable_scheduler_muon=ConstantLR(muon_optim_fp32,factor=1,
                                    total_iters=cfg.STABLE_STEPS)
    decay_scheduler_muon=LinearLR(muon_optim_fp32,start_factor=1,
                                end_factor=1e-4,total_iters=cfg.DECAY_STEPS)

    wsd_scheduler_muon=SequentialLR(muon_optim_fp32,[warmup_scheduler_muon,stable_scheduler_muon,decay_scheduler_muon],
                                    milestones=[cfg.WARMUP_STEPS,cfg.WARMUP_STEPS+cfg.STABLE_STEPS])
    
    return wsd_scheduler_adam,wsd_scheduler_muon
#LIGER FUSED CE KERNEL NAIN AND MTP
def build_liger_ce(scale=1e-4,z_loss=True):
    liger_fused_ce_main=LigerFusedLinearCrossEntropyLoss(lse_square_scale=scale,return_z_loss=z_loss)
    liger_fused_ce_mtp=LigerFusedLinearCrossEntropyLoss()
    return  liger_fused_ce_main,liger_fused_ce_mtp

#TRAIN STEP FOR TORCH COMPILE OPTIMIZATION
def train_step(transformer_model,input,d_model,embed_weight,ce_main,ce_mtp,device,lse_square_scale=1e-4,mtp_k=0,mtp_weight=0): #CE EXPECTS LIGER FUSED KERNELS
 
    token_pos=input["rope_pos"].to(device)
    cuseq=input["cuseq"].to(device)
    y_main=input["y_main"].to(device)
    y_mtp=input["y_mtp"].to(device)
    x=input["tokens"].to(device)

    B,T=x.shape

    hidden_states=transformer_model(x,cuseq,token_pos)                    

    out=ce_main(embed_weight,hidden_states[:B,:,:].view(-1,d_model),y_main)
    main_loss,main_z_loss=out.loss,out.z_loss
    loss=main_loss
    
    if mtp_weight>0:
        # y_mtp=x[:,2:].unfold(1,mtp_k,1).permute(2,0,1).contiguous().view(-1) # SHIFT BY ONE FOR EACH MTP THEN PERMUTE TO GET MTP0|MTP!.. ORDERIING

        mtp_loss=ce_mtp(embed_weight,hidden_states[B:,:,:].view(-1,d_model),y_mtp)
        loss=loss+mtp_weight*mtp_loss
        mtp_loss=mtp_loss.detach()        #DETACH FOR LOGGING
    else:
        mtp_loss=main_loss.new_zeros(())  #0 when MTP disabled

    logsumexp_avg=main_z_loss/lse_square_scale
    return loss,logsumexp_avg,main_loss.detach(),mtp_loss   #ALSO RETURN main/mtp CE SEPARATELY

#LIGER BREAK COMPILE WITH .item() SINCE IT IS OPTIMIZED KERNEL WE COMPILE FORWARD UPTO IT AND KEEP LIGER IN EAGER MODE
#2LIGER 2 GRAPH BREAK COMPILE THROWS ERROR SO CAN'T DO TORCH COMPILE ON LIGER KEEP IT TILL FORWARD LIGER GIVES GRAD USE IT AOT BACKWARD
def save_model(model):
    torch.save(model.state_dict(), "Weights/model_weights.pth")

#checkpoint save/load extracted to load_save_ckpt.py (imported at top as save_training_checkpoint / load_training_checkpoint)

@torch.no_grad()   #IN-PLACE COPY INTO bf16 LEAF PARAMS + OPTIM STEP MUST RUN OUTSIDE AUTOGRAD
def mp_opt_step(muon_optim,master_param_2d,master_param_1d,bf16_param_2d,
                bf16_param_1d,adamw_optim,wsd_scheduler_muon,wsd_scheduler_adam,
                keep_grads=False):

    #COPY bf16 GRADS to fp32 MASTER GRADS  
    for m,p in zip(master_param_2d,bf16_param_2d):
        if p.grad is not None:
            m.grad=p.grad.float()
            if not keep_grads:
                p.grad=None

    for m,p in zip(master_param_1d,bf16_param_1d):
        if p.grad is not None:
            m.grad=p.grad.float()
            if not keep_grads:
                p.grad=None

    #GRAD CLIP
    norm2d=torch.nn.utils.clip_grad_norm_(master_param_2d, max_norm=1.0)
    norm1d=torch.nn.utils.clip_grad_norm_(master_param_1d, max_norm=1.0)

    #OPTIM STEP ON fp32 MASTERS
    muon_optim.step()
    adamw_optim.step()

    for m in master_param_2d: 
        m.grad = None      
    for m in master_param_1d: 
        m.grad = None   

    #fp32 MASTER COPIED TO BF16
    torch._foreach_copy_(bf16_param_2d,master_param_2d)
    torch._foreach_copy_(bf16_param_1d,master_param_1d)

    #SCHEDULER STEP
    wsd_scheduler_adam.step()
    wsd_scheduler_muon.step()

    return norm2d,norm1d

def train():
    local_rank=setup()
    device=f"cuda:{local_rank}"

    print(f"local_rank here is {local_rank}")
    run=setup_wandb() if local_rank==OFFSET+HOST_LOCAL_RANK else None 

    transformer_model,num_params,num_non_embed_params=load_model(run,device)

    master_param_2d,master_param_1d,adamw_optim_fp32,muon_optim_fp32,\
        bf16_param_1d,bf16_param_2d,transformer_model=load_optim(transformer_model)

    wsd_scheduler_adam,wsd_scheduler_muon=load_warmup_scheduler(adamw_optim_fp32,muon_optim_fp32)

    liger_fused_ce_main,liger_fused_ce_mtp=build_liger_ce()

    embed_weight=transformer_model.embedding.weight

    #LOAD CHECKPOINT
    resume_opt_step=load_training_checkpoint(master_param_1d,master_param_2d,
                                  muon_optim_fp32,adamw_optim_fp32,
                                  wsd_scheduler_muon,wsd_scheduler_adam,
                                  bf16_param_1d,bf16_param_2d,device)
    #opt step to global step
    step=resume_opt_step*cfg.ACCUMULATION_STEP

    dataloader=load_dataset(local_rank,type="main_run",eot_id=48000,resume_step=step)
    data_iter=infinite(dataloader)

    transformer_model=DDP(module=transformer_model,
                          device_ids=[local_rank],
                          gradient_as_bucket_view=True,
                          #KEEP False: static_graph=True Causes conflice with no sync since it requires syncing.
                          static_graph=False,
                          bucket_cap_mb=50,
                          broadcast_buffers=False)

    optimized_model=torch.compile(transformer_model,mode="max-autotune-no-cudagraphs")

    tokens_seen=0

    LOG_EVERY=cfg.ACCUMULATION_STEP*300
    CKPT_EVERY=cfg.CKPT_EVERY_OPT*cfg.ACCUMULATION_STEP   
    ema_loss=0
    ema_token_throughput=0
    alpha=0.05


    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):

        for global_step in tqdm(range(step,cfg.TOTAL_STEPS)):
            # if global_step>5:
            #     torch.cuda.profiler.start()
            if (global_step)%(cfg.ACCUMULATION_STEP)==0:
                s_time=time.time(); io_accum=0.0   

            is_sync_step = (global_step + 1) % cfg.ACCUMULATION_STEP == 0


            #DATA LOAD TO GPU
            io_t=time.time()
            input=next(data_iter)
            shard_ids=0
            # x=x[1].to(device)
            io_accum+=time.time()-io_t   #  TRUE data-load time across the window

            #OPTIMIZED FORWARD BACKWARD PASS
            fwd_start=time.time()
            loss,logsumexp_avg,main_loss,mtp_loss=train_step(optimized_model,input,embed_weight=embed_weight,d_model=cfg.D_MODEL,ce_main=liger_fused_ce_main,
                                          ce_mtp=liger_fused_ce_mtp,mtp_weight=cfg.MTP_LOSS_WEIGHT,mtp_k=cfg.MTP_HEADS,device=device)


            #CONTEXT MANAGEMENT TO FIRE DDP ONLY WHEN ACCUMULATION STEP ARRIVES FOR OPTIM UPDATE
            loss_for_log=loss.detach()  # keep UNSCALED loss for logging
            ctx = transformer_model.no_sync() if not is_sync_step else nullcontext()
            with ctx:
                loss=loss/cfg.ACCUMULATION_STEP
                loss.backward()

            if is_sync_step:
                #SNAPSHOT PRE-STEP WEIGHTS (true update-ratio); grads still live from backward
                is_log_step=(global_step+1) % LOG_EVERY == 0
                is_ckpt_step=(global_step+1) % CKPT_EVERY == 0  
                prev=None
                if is_log_step and run:
                    print(f"[mem step{global_step+1}] live={torch.cuda.memory_allocated()/1e9:.1f}GB "
                          f"reserved={torch.cuda.memory_reserved()/1e9:.1f}GB")   
                    try:
                        prev=wl.snapshot_params(transformer_model.module)   #pre-step weights for update-ratio
                    except Exception as e:
                        print(f"[rank0] snapshot_params failed, skipping param-diag: {e}"); prev=None

                norm2d,norm1d=mp_opt_step(muon_optim=muon_optim_fp32,master_param_2d=master_param_2d,master_param_1d=master_param_1d,
                                bf16_param_1d=bf16_param_1d,bf16_param_2d=bf16_param_2d,adamw_optim=adamw_optim_fp32,
                                wsd_scheduler_adam=wsd_scheduler_adam,wsd_scheduler_muon=wsd_scheduler_muon,
                                keep_grads=is_log_step and run is not None)   #ONLY MASTER RETAINS GRADS (FOR param_diagnostics); NON-MASTER MUST NULL ELSE STALE GRADS ACCUMULATE NEXT STEP

                act_stats={}
                if (is_log_step or is_ckpt_step) and run:
                    try:
                        torch.cuda.empty_cache()   
                        #SAVE fp32 MASTERS + OPTIM/SCHED/RNG STATE 
                        if is_ckpt_step:           

                            save_training_checkpoint(master_param_1d,master_param_2d,
                                                     muon_optim_fp32,adamw_optim_fp32,
                                                     wsd_scheduler_muon,wsd_scheduler_adam,
                                                     opt_step=(global_step+1)//cfg.ACCUMULATION_STEP)
                        if is_log_step and cfg.RUN_ACT_STATS:   #GATED: eager forward is the big VRAM spike; off by default
                            token_pos=input["rope_pos"].to(device)
                            cuseq=input["cuseq"].to(device)
                            x=input["tokens"].to(device)
                            act_stats=wl.activation_stats(transformer_model.module,x,cuseq,token_pos)
                    except Exception as e:
                        print(f"[rank0] logging/checkpoint step failed, continuing: {e}")
                        act_stats={}
                    finally:
                        torch.cuda.empty_cache()


                # #VALIDATION
                # if (step%1000)==0:
                #     val_loss=eval(model)

                #LOGGING — SINGLE run.log PER STEP 
                if run:
                    loss_train=loss_for_log.item()   #unscaled loss

                    f_time=time.time()
                    tokens_seen+=cfg.BATCH_SIZE*cfg.EFF_SEQ_LEN*cfg.ACCUMULATION_STEP*dist.get_world_size()  
                    step_time=round(f_time-s_time,4)
                    data_io_time=io_accum   # summed data-load seconds across the window
                    mfu=wl.compute_mfu(step_time,cfg.EFF_SEQ_LEN,cfg.BATCH_SIZE,cfg.ACCUMULATION_STEP,num_non_embed_params,cfg.A6000_BF16_PEAK)
                    token_throughput=(cfg.BATCH_SIZE*cfg.EFF_SEQ_LEN*cfg.ACCUMULATION_STEP)/step_time

                    #RAW BIASED EMA CARRIES FORWARD; BIAS-CORRECT ONLY FOR LOGGING
                    ema_loss=ema_loss*(1-alpha) + alpha*loss_train
                    ema_token_throughput=ema_token_throughput*(1-alpha) + alpha*token_throughput
                    n_opt=(global_step+1)//cfg.ACCUMULATION_STEP   # EMA updates once per OPTIMIZER step, not per micro-step
                    bias_corr=1-(1-alpha)**n_opt
                    ema_loss_log=ema_loss/bias_corr
                    ema_throughput_log=ema_token_throughput/bias_corr
                    data_io_frac=data_io_time/step_time

                    logs=wl.step_metrics(loss=loss_for_log,grad_norm_muon=norm2d,grad_norm_adamw=norm1d,
                                        lr_adamw=wsd_scheduler_adam.get_last_lr()[0],
                                        lr_muon=wsd_scheduler_muon.get_last_lr()[0],
                                        tokens_seen=tokens_seen,throughput=token_throughput,
                                        step_time=step_time,data_io=data_io_frac,mfu=mfu)
                    logs.update({"loss/ema_ce":ema_loss_log,"perf/ema_throughput":ema_throughput_log,
                                "loss/logsumexp_sq":logsumexp_avg,"data/shard_ids":shard_ids,
                                "loss/ce_main":main_loss,"loss/mtp":mtp_loss})  
                    logs.update(act_stats)

                #GROUPED GRAD / PARAM / TRUE UPDATE-RATIO NORMS 
                    if is_log_step:
                        try:
                            if prev is not None:
                                logs.update(wl.param_diagnostics(transformer_model.module,prev,device))
                        except Exception as e:
                            print(f"[rank0] param_diagnostics failed, continuing: {e}")
                        finally:
                            transformer_model.zero_grad(set_to_none=False)   #NULL RETAINED GRADS -> NO ACCUMULATION NEXT STEP
                            prev=None

                    try:
                        run.log(logs)   #ONE COMMIT PER STEP
                    except Exception as e:
                        print(f"[rank0] run.log failed, continuing: {e}")

                #RESYNC ALL RANKS after rank-0-only work (logging AND/OR checkpoint). 
                if is_log_step or is_ckpt_step:
                    dist.barrier()
        # torch.cuda.synchronize(device)
        # torch.cuda.profiler.stop()
        cleanup()



if __name__=="__main__":
    train()