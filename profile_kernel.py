import logging
import torch._logging
from torch.nn.attention import SDPBackend, sdpa_kernel
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


# os.environ["MLIR_ENABLE_DUMP"] = "1"
# os.environ["LLVM_IR_ENABLE_DUMP"] = "1"

torch._logging.set_logs(
    recompiles=True, 
    output_code=True, 
    graph_breaks=True
)

from models_fast import Transformer
from einops import repeat


seq_len=1024
batch_size=21

d_model=1536
n_head=12
num_layers=32
vocab_size=48000

x=torch.arange(seq_len)
x = repeat(x, '... -> b ...', b=batch_size).to(device="cuda:6", dtype=torch.long)
targets = torch.randint(0, vocab_size, (batch_size* seq_len,), device="cuda:6", dtype=torch.long)

loss_ce=LigerFusedLinearCrossEntropyLoss()


trans=Transformer(48000,8192,10000,d_model,n_head,num_layers,0.1,4096,0)
trans=trans.to(torch.bfloat16).to("cuda:6")

count=0
for p in trans.parameters():
    count+=p.numel()

print(f"params:{count/1e9:.3f} B")

def train_step(model, loss_fn, inputs, targets,hidden_dim):
    hidden_states = model(inputs)
    loss = loss_fn(model.embedding.weight, hidden_states.view(-1,hidden_dim), targets)
    loss.backward()
    return loss


optimized_train_step = torch.compile(train_step, mode="max-autotune-no-cudagraphs")             


import torch
import time

# ---- config ----
DEVICE = "cuda:6"
A6000_BF16_PEAK = 154e12   
WARMUP = 5                 
MEASURE = 20               


B, T = x.shape                     
N_params = sum(p.numel() for p in trans.parameters())
tokens_per_step = B * T


with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    # per-layer attention score+context FLOPs, summed over the batch
    attn_flops = 3 * 2 * (2 * T * T * 1536) * 32 * batch_size    
    flops_per_step = 6 * N_params * tokens_per_step + 0

    def step():
        loss=optimized_train_step(trans,loss_ce,x,targets,d_model)
        trans.zero_grad(set_to_none=True)

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize(DEVICE)

    t0 = time.perf_counter()
    torch.cuda.profiler.start()
    for _ in range(MEASURE):
        step()
    torch.cuda.synchronize(DEVICE) 
    torch.cuda.profiler.stop()

    elapsed = time.perf_counter() - t0

    step_time = elapsed / MEASURE
    achieved = flops_per_step / step_time          
    mfu = achieved / A6000_BF16_PEAK

    print(f"params:           {N_params/1e9:.3f} B")
    print(f"tokens/step:      {tokens_per_step}")
    print(f"step time:        {step_time*1e3:.2f} ms")
    print(f"throughput:       {tokens_per_step/step_time:,.0f} tok/s")
    print(f"achieved FLOP/s:  {achieved/1e12:.1f} TFLOP/s")
    print(f"MFU:              {mfu*100:.1f} %")