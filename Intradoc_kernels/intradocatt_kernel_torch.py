import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so sibling kernel imports resolve (standalone + as package)
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

import torch
import triton
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch._functorch.config

# Turn off buffer donation so the end-to-end benchmark loop runs safely
# torch._functorch.config.donated_buffer = False

from intradocatt_fwd_kernels import triton_intradoc_attention
from intradocatt_backwd_kernels import flash_att_backward_split


# 1. Register the operator in PyTorch's dispatcher
@torch.library.custom_op("intradoc_attn::triton_flash_attn_fwd", mutates_args=())
def triton_flash_attn_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cuseq:torch.Tensor,max_len:int,
    group_size:int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_contiguous(), "Query tensor must be contiguous"
    assert k.is_contiguous(), "Key tensor must be contiguous"
    assert v.is_contiguous(), "Value tensor must be contiguous"

    out,lse=triton_intradoc_attention(q,k,v,cuseq,max_len,group_size)
    return out,lse


@triton_flash_attn_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cuseq:torch.tensor, max_len:int,
      group_size:int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tells torch.compile the shape and dtype of the outputs.
    Adjust the LSE shape based on how Triton kernel stores it.
    """
    out_fake=torch.empty_like(q)

    seq_len, num_heads, _=q.shape
    lse_fake = torch.empty(
        (num_heads, seq_len),
        dtype=torch.float32,
        device=q.device
    )
    return out_fake,lse_fake

@torch.library.custom_op("intradoc_attn::triton_flash_attn_bwd", mutates_args=())
def triton_flash_attn_bwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    o: torch.Tensor, grad_out: torch.Tensor, lse: torch.Tensor,
    cuseq: torch.Tensor, max_seq_len: int,
    group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    dq,dk,dv = flash_att_backward_split(q,k,v,o,grad_out,lse,cuseq,max_seq_len,group_size)
    return dq,dk,dv

@triton_flash_attn_bwd.register_fake
def _(q, k, v, o, grad_out, lse, cuseq, max_seq_len,group_size):

    dq=torch.empty_like(q)
    dk=torch.empty_like(k)
    dv=torch.empty_like(v)
    return dq, dk, dv


def setup_context(ctx, inputs, output):
    q,k,v,cuseq,max_len,group_size =inputs
    o,lse=output
    ctx.mark_non_differentiable(lse)

    # Save the necessary tensors for the Triton backward pass
    ctx.save_for_backward(q,k,v,o,lse,cuseq)
    ctx.max_len = max_len
    ctx.group_size=group_size

def triton_flash_attn_bwd(ctx, grad_out,grad_lse):
    # We ignore grad_lse because the loss is only calculated on `out`
    q,k,v,o,lse,cuseq=ctx.saved_tensors
    max_len=ctx.max_len
    group_size=ctx.group_size

    with torch.no_grad():
        dq,dk,dv = torch.ops.intradoc_attn.triton_flash_attn_bwd(q,k,v,o,grad_out,lse,cuseq,max_len,group_size)
    # Return gradients for q, k, and v,cuseq
    return dq,dk,dv,None,None,None

torch.library.register_autograd(
    "intradoc_attn::triton_flash_attn_fwd",
    triton_flash_attn_bwd,
    setup_context=setup_context
)

class IntraDocAttention(nn.Module):
    def __init__(self,max_len,group_size):
        super().__init__()
        self.max_len=max_len
        self.group_size=group_size

    def forward(self,q,k,v,cuseq):
        out,lse=torch.ops.intradoc_attn.triton_flash_attn_fwd(q,k,v,cuseq,self.max_len,self.group_size)
        return out

if __name__ == "__main__":

    # --- Configuration ---
    B = 14          # Batch / Docs
    S = 1024        # Sequence length
    H_Q = 12        # Q Heads
    H_KV = 4        # KV Heads
    D = 128         # Head Dimension
    GROUP_SIZE = H_Q // H_KV
    DTYPE = torch.bfloat16
    DEVICE = "cuda:0"
    torch.cuda.set_device(DEVICE)

    # --- 1. Data Setup ---
    torch.manual_seed(42)
    total_tokens = B * S
    cuseq = torch.arange(0, total_tokens + 1, S, dtype=torch.int32, device=DEVICE)

    q_flat = torch.randn(total_tokens, H_Q, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k_flat = torch.randn(total_tokens, H_KV, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v_flat = torch.randn(total_tokens, H_KV, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    dout_flat = torch.randn(total_tokens, H_Q, D, device=DEVICE, dtype=DTYPE)

    # Initialize your custom module
    fused_attn_module = IntraDocAttention(max_len=S, group_size=GROUP_SIZE)

    # --- 2. Clean GQA Baselines ---

    # We view the flat tensors as [B, S, H, D] for the PyTorch baselines
    q_pt = q_flat.view(B, S, H_Q, D).clone().detach().requires_grad_(True)
    k_pt = k_flat.view(B, S, H_KV, D).clone().detach().requires_grad_(True)
    v_pt = v_flat.view(B, S, H_KV, D).clone().detach().requires_grad_(True)
    dout_pt = dout_flat.view(B, S, H_Q, D).transpose(1, 2)

    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    from torch.nn.attention import sdpa_kernel, SDPBackend


    def run_sdpa(q, k, v):
        # Transpose to [B, H, S, D] layout
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)

        # Using enable_gqa=True bypasses broadcasting completely
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                is_causal=True,
                enable_gqa=True
            )
        return out.transpose(1, 2)

    # Flex mask compilation
    def causal_mask_fn(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    flex_block_mask = create_block_mask(causal_mask_fn, B=None, H=None, Q_LEN=S, KV_LEN=S, device=DEVICE)

    def run_flex(q, k, v):
        q_flex = q.transpose(1, 2)
        k_flex = k.transpose(1, 2)
        v_flex = v.transpose(1, 2)

        out = flex_attention(
            q_flex, k_flex, v_flex,
            block_mask=flex_block_mask,
            enable_gqa=True
        )
        return out.transpose(1, 2)

    compiled_flex = torch.compile(run_flex, dynamic=False)

    # --- 3. Benchmark Harness ---
    def benchmark():
        print(f"--- Shapes: Docs={B}, S={S}, Q_Heads={H_Q}, KV_Heads={H_KV}, D={D} ---")

        # Custom Tensors
        q_cust = q_flat.clone().detach().requires_grad_(True)
        k_cust = k_flat.clone().detach().requires_grad_(True)
        v_cust = v_flat.clone().detach().requires_grad_(True)

        # 1. Forward Pass Execution
        out_sdpa = run_sdpa(q_pt, k_pt, v_pt)
        _ = compiled_flex(q_pt, k_pt, v_pt) # Warmup
        out_flex = compiled_flex(q_pt, k_pt, v_pt)

        # Note: Flattening tensors to match the custom Triton ops layout expectations
        out_cust_flat = fused_attn_module(q_cust, k_cust, v_cust, cuseq)
        out_cust = out_cust_flat.view(B, S, H_Q, D)

        # 2. Backward Pass Execution
        out_sdpa.backward(dout_pt.transpose(1, 2), retain_graph=True)
        sdpa_dq = q_pt.grad.clone()
        sdpa_dk = k_pt.grad.clone()

        out_cust_flat.backward(dout_flat, retain_graph=True)
        cust_dq = q_cust.grad.view(B, S, H_Q, D)
        cust_dk = k_cust.grad.view(B, S, H_KV, D)

        # 3. Correctness Verification
        print("\n[CORRECTNESS vs SDPA]")
        print(f"Output Max Diff: {(out_sdpa - out_cust).abs().max().item():.5f}")
        print(f"dQ Max Diff:     {(sdpa_dq - cust_dq).abs().max().item():.5f}")
        print(f"dK Max Diff:     {(sdpa_dk - cust_dk).abs().max().item():.5f}")

        # Zero out grads before benchmarking speed limits accumulation overhead
        q_pt.grad, k_pt.grad, v_pt.grad = None, None, None
        q_cust.grad, k_cust.grad, v_cust.grad = None, None, None

        # 4. Performance Benchmarks
        print("\n[PERFORMANCE BENCHMARK]")
        ms_sdpa_fw = triton.testing.do_bench(lambda: run_sdpa(q_pt, k_pt, v_pt))
        ms_flex_fw = triton.testing.do_bench(lambda: compiled_flex(q_pt, k_pt, v_pt))
        ms_cust_fw = triton.testing.do_bench(lambda: fused_attn_module(q_cust, k_cust, v_cust, cuseq))

        ms_sdpa_bw = triton.testing.do_bench(lambda: out_sdpa.backward(dout_pt.transpose(1, 2), retain_graph=True))
        ms_flex_bw = triton.testing.do_bench(lambda: out_flex.backward(dout_pt.transpose(1, 2), retain_graph=True))
        ms_cust_bw = triton.testing.do_bench(lambda: out_cust_flat.backward(dout_flat, retain_graph=True))

        print(f"{'Kernel':<20} | {'FWD (ms)':>10} | {'BWD (ms)':>10} | {'Total (ms)':>10}")
        print("-" * 59)
        print(f"{'Native SDPA GQA':<20} | {ms_sdpa_fw:>10.4f} | {ms_sdpa_bw:>10.4f} | {ms_sdpa_fw + ms_sdpa_bw:>10.4f}")
        print(f"{'FlexAttention GQA':<20} | {ms_flex_fw:>10.4f} | {ms_flex_bw:>10.4f} | {ms_flex_fw + ms_flex_bw:>10.4f}")
        print(f"{'Custom Fused Triton':<20} | {ms_cust_fw:>10.4f} | {ms_cust_bw:>10.4f} | {ms_cust_fw + ms_cust_bw:>10.4f}")

    benchmark()
