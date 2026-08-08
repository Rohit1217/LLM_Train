import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# Try to import flex_attention (requires PyTorch >= 2.5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ],
    key=['NUM_HEADS', 'HEAD_DIM', 'group_size'],
)
@triton.jit
def backwd_dkv(q_ptr,k_ptr,v_ptr,dk_ptr,dv_ptr,do_ptr,lse_ptr,odo_sum_ptr,cuseq_ptr,total_tokens,sm_scale:tl.constexpr,
                group_size:tl.constexpr,HEAD_DIM:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr):
    pid=tl.program_id(axis=0)
    pid_s=tl.program_id(axis=1)
    pid_h=tl.program_id(axis=2)

    start=tl.load(cuseq_ptr+pid_s)
    end=tl.load(cuseq_ptr+pid_s+1)
    doc_len=end-start

    if pid * BLOCK_N >= doc_len:
        return

    stride_tok_q=tl.num_programs(axis=2)*HEAD_DIM*group_size
    stride_tok_k=stride_tok_q//group_size
    stride_h=total_tokens
    pid_kh=pid_h

    base_offset_k=start*stride_tok_k + pid_kh*HEAD_DIM
    qk_scale = sm_scale * 1.44269504

    k_block_ptr=tl.make_block_ptr(k_ptr + base_offset_k,shape=(doc_len,HEAD_DIM),strides=(stride_tok_k,1),
                                       offsets=(pid*BLOCK_N,0),block_shape=(BLOCK_N,HEAD_DIM),order=(1,0))

    v_block_ptr=tl.make_block_ptr(v_ptr + base_offset_k,shape=(doc_len,HEAD_DIM),strides=(stride_tok_k,1),
                                       offsets=(pid*BLOCK_N,0),block_shape=(BLOCK_N,HEAD_DIM),order=(1,0))

    dv_block_ptr=tl.make_block_ptr(dv_ptr + base_offset_k,shape=(doc_len,HEAD_DIM),strides=(stride_tok_k,1),
                            offsets=(pid*BLOCK_N,0),block_shape=(BLOCK_N,HEAD_DIM),order=(1,0))

    dk_block_ptr=tl.make_block_ptr(dk_ptr + base_offset_k,shape=(doc_len,HEAD_DIM),strides=(stride_tok_k,1),
                                      offsets=(pid*BLOCK_N,0),block_shape=(BLOCK_N,HEAD_DIM),order=(1,0))

    k=tl.load(k_block_ptr,boundary_check=(0,),padding_option="zero")

    v=tl.load(v_block_ptr,boundary_check=(0,),padding_option="zero")

    acc_dv=tl.zeros((BLOCK_N,HEAD_DIM),tl.float32)
    acc_dk=tl.zeros((BLOCK_N,HEAD_DIM),tl.float32)

    indices_n=pid*BLOCK_N+tl.arange(0,BLOCK_N)

    diagonal_end = tl.minimum((pid + 1) * BLOCK_N, doc_len)
    start_i = (pid * BLOCK_N // BLOCK_M) * BLOCK_M

    for g in range(group_size):

        pid_h_q=pid_h*group_size+g
        base_offset_q=start*stride_tok_q + pid_h_q*HEAD_DIM

        lse_ptr_b=lse_ptr+pid_h_q*stride_h + start
        odo_sum_ptr_b=odo_sum_ptr+pid_h_q*stride_h + start

        q_block_ptr=tl.make_block_ptr(q_ptr + base_offset_q,shape=(doc_len,HEAD_DIM),strides=(stride_tok_q,1),
                                    offsets=(start_i,0),block_shape=(BLOCK_M,HEAD_DIM),order=(1,0))

        do_block_ptr=tl.make_block_ptr(do_ptr + base_offset_q,shape=(doc_len,HEAD_DIM),strides=(stride_tok_q,1),
                                    offsets=(start_i,0),block_shape=(BLOCK_M,HEAD_DIM),order=(1,0))

        odo_sum_block_ptr=tl.make_block_ptr(base=odo_sum_ptr_b,shape=(doc_len,),
                                strides=(1,),offsets=(start_i,),
                                block_shape=(BLOCK_M,),order=(0,))
        lse_block_ptr=tl.make_block_ptr(base=lse_ptr_b,shape=(doc_len,),
                                strides=(1,),offsets=(start_i,),
                                block_shape=(BLOCK_M,),order=(0,))

        for i in range(start_i,diagonal_end,BLOCK_M):
            do=tl.load(do_block_ptr,boundary_check=(0,),padding_option="zero")

            q = tl.load(q_block_ptr,boundary_check=(0,),padding_option="zero")

            odo_sum=tl.load(odo_sum_block_ptr,boundary_check=(0,),padding_option="zero")
            lse=tl.load(lse_block_ptr,boundary_check=(0,),padding_option="zero")

            pos_q=i+tl.arange(0,BLOCK_M)

            s=tl.dot(q,tl.trans(k))
            s=s*qk_scale

            att_mask=pos_q[:,None]>=indices_n[None,:]
            att_mask=att_mask
            s=tl.where(att_mask,s,float("-inf"))

            p=tl.math.exp2(s-lse[:,None])
            dp=tl.dot(do,tl.trans(v),out_dtype=tl.float32)
            ds=p*(dp-odo_sum[:,None])

            acc_dv=tl.dot(tl.trans(p).to(tl.bfloat16),do,acc_dv)
            acc_dk=tl.dot(tl.trans(ds).to(tl.bfloat16),q,acc_dk)

            q_block_ptr=tl.advance(q_block_ptr,(BLOCK_M,0))

            do_block_ptr=tl.advance(do_block_ptr,(BLOCK_M,0))
            odo_sum_block_ptr=tl.advance(odo_sum_block_ptr,(BLOCK_M,))
            lse_block_ptr=tl.advance(lse_block_ptr,(BLOCK_M,))

        for i in range(diagonal_end,doc_len,BLOCK_M):
            do=tl.load(do_block_ptr,boundary_check=(0,),padding_option="zero")

            q = tl.load(q_block_ptr,boundary_check=(0,),padding_option="zero")

            odo_sum=tl.load(odo_sum_block_ptr,boundary_check=(0,),padding_option="zero")
            lse=tl.load(lse_block_ptr,boundary_check=(0,),padding_option="zero")

            pos_q=i+tl.arange(0,BLOCK_M)

            s=tl.dot(q,tl.trans(k))
            s=s*qk_scale

            doc_mask= pos_q<doc_len
            s=tl.where(doc_mask[:,None],s,float("-inf"))

            p=tl.math.exp2(s-lse[:,None])
            dp=tl.dot(do,tl.trans(v),out_dtype=tl.float32)
            ds=p*(dp-odo_sum[:,None])

            acc_dv=tl.dot(tl.trans(p).to(tl.bfloat16),do,acc_dv)
            acc_dk=tl.dot(tl.trans(ds).to(tl.bfloat16),q,acc_dk)

            q_block_ptr=tl.advance(q_block_ptr,(BLOCK_M,0))

            do_block_ptr=tl.advance(do_block_ptr,(BLOCK_M,0))
            odo_sum_block_ptr=tl.advance(odo_sum_block_ptr,(BLOCK_M,))
            lse_block_ptr=tl.advance(lse_block_ptr,(BLOCK_M,))

    acc_dk=acc_dk*sm_scale

    tl.store(dk_block_ptr,acc_dk.to(tl.bfloat16), boundary_check=(0,))

    tl.store(dv_block_ptr,acc_dv.to(tl.bfloat16), boundary_check=(0,))

@triton.jit
def calc_odo_sum(
    o_ptr, do_ptr, odo_sum_ptr, cuseq_ptr,
    total_tokens: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_s = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    start = tl.load(cuseq_ptr+pid_s)
    end = tl.load(cuseq_ptr+pid_s+1)
    doc_len = end - start

    if pid * BLOCK_K >= doc_len:
        return

    num_heads = tl.num_programs(axis=2)
    stride_tok = num_heads * HEAD_DIM
    stride_h = total_tokens

    o_ptr_b = o_ptr + start * stride_tok + pid_h * HEAD_DIM
    do_ptr_b = do_ptr + start * stride_tok + pid_h * HEAD_DIM

    token_offsets = pid * BLOCK_K + tl.arange(0, BLOCK_K)
    odo_sum_ptr_b = odo_sum_ptr + pid_h * stride_h + start + token_offsets
    mask = token_offsets < doc_len

    o_block_ptr = tl.make_block_ptr(
        base=o_ptr_b, shape=(doc_len, HEAD_DIM),
        strides=(stride_tok, 1), offsets=(pid * BLOCK_K, 0),
        block_shape=(BLOCK_K, HEAD_DIM), order=(1, 0)
    )

    do_block_ptr = tl.make_block_ptr(
        base=do_ptr_b, shape=(doc_len, HEAD_DIM),
        strides=(stride_tok, 1), offsets=(pid * BLOCK_K, 0),
        block_shape=(BLOCK_K, HEAD_DIM), order=(1, 0)
    )

    o = tl.load(o_block_ptr, boundary_check=(0,), padding_option="zero")
    do = tl.load(do_block_ptr, boundary_check=(0,), padding_option="zero")

    odo = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    tl.store(odo_sum_ptr_b, odo, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['NUM_HEADS', 'HEAD_DIM', 'group_size'],
)
@triton.jit
def backwd_dq(q_ptr,k_ptr,v_ptr,dq_ptr,do_ptr,lse_ptr,odo_sum_ptr,cuseq_ptr,total_tokens,sm_scale:tl.constexpr,
              group_size:tl.constexpr,HEAD_DIM:tl.constexpr,BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr):
    pid=tl.program_id(axis=0)
    pid_s=tl.program_id(axis=1)
    pid_h=tl.program_id(axis=2)

    start=tl.load(cuseq_ptr+pid_s)
    end=tl.load(cuseq_ptr+pid_s+1)
    doc_len=end-start

    if pid * BLOCK_M >= doc_len:
        return

    stride_tok_q=tl.num_programs(axis=2)*HEAD_DIM
    stride_tok_k=stride_tok_q//group_size
    stride_h=total_tokens
    pid_kh=pid_h//group_size

    base_offset_q=start*stride_tok_q + pid_h*HEAD_DIM
    base_offset_k=start*stride_tok_k + pid_kh*HEAD_DIM

    # FIX 2: Corrected pid_s * total_tokens to start
    lse_ptr_b=lse_ptr+pid_h*stride_h + start
    odo_sum_ptr_b=odo_sum_ptr+pid_h*stride_h + start

    qk_scale = sm_scale * 1.44269504

    q_block_ptr=tl.make_block_ptr(q_ptr + base_offset_q,shape=(doc_len,HEAD_DIM),strides=(stride_tok_q,1),
                                offsets=(pid*BLOCK_M,0),block_shape=(BLOCK_M,HEAD_DIM),order=(1,0))


    k_block_ptr=tl.make_block_ptr(k_ptr + base_offset_k,shape=(doc_len,HEAD_DIM),strides=(stride_tok_k,1),
                                       offsets=(0,0),block_shape=(BLOCK_N,HEAD_DIM),order=(1,0))

    v_block_ptr=tl.make_block_ptr(v_ptr + base_offset_k,shape=(doc_len,HEAD_DIM),strides=(stride_tok_k,1),
                                       offsets=(0,0),block_shape=(BLOCK_N,HEAD_DIM),order=(1,0))
    do_block_ptr=tl.make_block_ptr(do_ptr + base_offset_q,shape=(doc_len,HEAD_DIM),strides=(stride_tok_q,1),
                                offsets=(pid*BLOCK_M,0),block_shape=(BLOCK_M,HEAD_DIM),order=(1,0))

    dq_block_ptr=tl.make_block_ptr(dq_ptr + base_offset_q,shape=(doc_len,HEAD_DIM),strides=(stride_tok_q,1),
                                offsets=(pid*BLOCK_M,0),block_shape=(BLOCK_M,HEAD_DIM),order=(1,0))

    odo_sum_block_ptr=tl.make_block_ptr(base=odo_sum_ptr_b,shape=(doc_len,),
                               strides=(1,),offsets=(pid*BLOCK_M,),
                               block_shape=(BLOCK_M,),order=(0,))
    lse_block_ptr=tl.make_block_ptr(base=lse_ptr_b,shape=(doc_len,),
                            strides=(1,),offsets=(pid*BLOCK_M,),
                            block_shape=(BLOCK_M,),order=(0,))


    do=tl.load(do_block_ptr,boundary_check=(0,),padding_option="zero")
    odo_sum=tl.load(odo_sum_block_ptr,boundary_check=(0,),padding_option="zero")
    lse=tl.load(lse_block_ptr,boundary_check=(0,),padding_option="zero")

    acc_dq=tl.zeros((BLOCK_M,HEAD_DIM),tl.float32)

    q = tl.load(q_block_ptr,boundary_check=(0,),padding_option="zero")


    for i in range(0,pid*BLOCK_M,BLOCK_N):
        k=tl.load(k_block_ptr,boundary_check=(0,),padding_option="zero")

        v=tl.load(v_block_ptr,boundary_check=(0,),padding_option="zero")

        s=tl.dot(q,tl.trans(k))
        s=s*qk_scale

        p=tl.math.exp2(s-lse[:,None])
        dp=tl.dot(do,tl.trans(v),out_dtype=tl.float32)
        ds=p*(dp-odo_sum[:,None])

        acc_dq=tl.dot(ds.to(tl.bfloat16),k,acc_dq)

        k_block_ptr=tl.advance(k_block_ptr,(BLOCK_N,0))

        v_block_ptr=tl.advance(v_block_ptr,(BLOCK_N,0))

    indices_m=pid*BLOCK_M + tl.arange(0,BLOCK_M)
    diagonal_end = tl.minimum((pid + 1) * BLOCK_M, doc_len)

    for i in range(pid*BLOCK_M,diagonal_end,BLOCK_N):
        k=tl.load(k_block_ptr,boundary_check=(0,),padding_option="zero")

        v=tl.load(v_block_ptr,boundary_check=(0,),padding_option="zero")

        pos_k=i+tl.arange(0,BLOCK_N)

        s=tl.dot(q,tl.trans(k))
        s=s*qk_scale

        att_mask=indices_m[:,None]>=pos_k[None,:]
        doc_mask= pos_k<doc_len
        att_mask=att_mask & doc_mask[None,:]
        s=tl.where(att_mask,s,float("-inf"))

        p=tl.math.exp2(s-lse[:,None])
        dp=tl.dot(do,tl.trans(v),out_dtype=tl.float32)
        ds=p*(dp-odo_sum[:,None])

        acc_dq=tl.dot(ds.to(tl.bfloat16),k,acc_dq)

        k_block_ptr=tl.advance(k_block_ptr,(BLOCK_N,0))

        v_block_ptr=tl.advance(v_block_ptr,(BLOCK_N,0))

    acc_dq=acc_dq*sm_scale

    tl.store(dq_block_ptr,acc_dq.to(tl.bfloat16), boundary_check=(0,))


def flash_att_backward_split(q,k,v,o,do,lse_sum,cuseq,max_seq_len,group_size):
    total_tokens, heads, head_dim = q.shape

    num_docs=cuseq.shape[0]-1

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    odo_sum = torch.empty_like(lse_sum)
    sm_scale = 1 / (head_dim ** 0.5)

    kv_heads=heads//group_size
    # lse_sum=lse_sum*1.44269504

    calc_odo_sum[(triton.cdiv(max_seq_len, 256), num_docs, heads)](o,do,odo_sum,cuseq,total_tokens,head_dim,BLOCK_K=256)

    grid_dq = lambda META: (triton.cdiv(max_seq_len, META['BLOCK_M']), num_docs, heads)
    grid_dkdv = lambda META: (triton.cdiv(max_seq_len, META['BLOCK_N']), num_docs, kv_heads)

    backwd_dq[grid_dq](q,k,v,dq,do,lse_sum,odo_sum,cuseq,total_tokens,sm_scale,group_size,head_dim)
    backwd_dkv[grid_dkdv](q, k, v, dk, dv, do, lse_sum, odo_sum, cuseq, total_tokens, sm_scale,group_size, head_dim)

    return dq,dk,dv


#Doesnt take into account gqa have to fix it albeit group_size=1 should work

if __name__=="__main__":

    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        HAS_FLEX = True
    except ImportError:
        HAS_FLEX = False
        print("FlexAttention not found. Please upgrade to PyTorch 2.5+")

    DEVICE = "cuda:0"
    torch.cuda.set_device(DEVICE)
    DTYPE = torch.bfloat16

    def benchmark_backwd(seq_len=1024,num_docs=8,num_heads=12,head_dim=128):
        seq_len = 1024
        num_docs = 8
        num_heads = 12
        head_dim = 128
        total_tokens = num_docs * seq_len


        q = torch.randn(total_tokens, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        k = torch.randn(total_tokens, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        v = torch.randn(total_tokens, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        do = torch.randn(total_tokens, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
        cuseq = torch.arange(0, (num_docs + 1) * seq_len, seq_len, dtype=torch.int32, device=DEVICE)

        # 2. Math Generator to get valid `o` and `lse_sum`
        q_val = q.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).float()
        k_val = k.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).float()
        v_val = v.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).float()

        sm_scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(q_val, k_val.transpose(-2, -1)) * sm_scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=DEVICE)) == 0
        scores.masked_fill_(causal_mask, float('-inf'))

        m = scores.max(dim=-1, keepdim=True)[0]
        scores_shifted = scores - m
        exp_scores = torch.exp(scores_shifted)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)
        probs = exp_scores / sum_exp

        o_exact = torch.matmul(probs, v_val).to(DTYPE)
        lse_exact = (m + torch.log(sum_exp)).squeeze(-1).to(DTYPE)

        o = o_exact.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)
        lse_sum = lse_exact.transpose(0, 1).reshape(num_heads, total_tokens)

        # PyTorch SDPA Reference
        print("Warming up SDPA...")
        q_sdpa = q.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).detach().clone().requires_grad_(True)
        k_sdpa = k.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).detach().clone().requires_grad_(True)
        v_sdpa = v.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).detach().clone().requires_grad_(True)
        do_sdpa = do.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).detach().clone()

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)

        out_sdpa.backward(do_sdpa,retain_graph=True)
        dq_sdpa, dk_sdpa, dv_sdpa = q_sdpa.grad, k_sdpa.grad, v_sdpa.grad

        def bench_sdpa():
            q_sdpa.grad, k_sdpa.grad, v_sdpa.grad = None, None, None
            out_sdpa.backward(do_sdpa, retain_graph=True)

        # PyTorch FlexAttention

        if HAS_FLEX:
            print("Compiling FlexAttention...")
            def causal_mask_fn(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            block_mask = create_block_mask(causal_mask_fn, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=DEVICE)

            q_flex = q.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).detach().clone().requires_grad_(True)
            k_flex = k.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).detach().clone().requires_grad_(True)
            v_flex = v.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2).detach().clone().requires_grad_(True)

            compiled_flex = torch.compile(flex_attention)
            out_flex = compiled_flex(q_flex, k_flex, v_flex, block_mask=block_mask)

            out_flex.backward(do_sdpa,retain_graph=True)
            dq_flex, dk_flex, dv_flex = q_flex.grad, k_flex.grad, v_flex.grad

            def bench_flex():
                q_flex.grad, k_flex.grad, v_flex.grad = None, None, None
                out_flex.backward(do_sdpa, retain_graph=True)

        # Triton Split Backward
        print("Warming up Custom Triton...")
        dq_tri, dk_tri, dv_tri = flash_att_backward_split(q, k, v, o, do, lse_sum, cuseq)

        dq_tri_fmt = dq_tri.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2)
        dk_tri_fmt = dk_tri.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2)
        dv_tri_fmt = dv_tri.view(num_docs, seq_len, num_heads, head_dim).transpose(1, 2)


        # VERIFICATION & TIMING RESULTS
        print("\n--- Correctness Check (Max Absolute Diff against SDPA) ---")
        if HAS_FLEX:
            print(f"Flex vs SDPA -> dQ: {torch.max(torch.abs(dq_sdpa - dq_flex)):.5f}, dK: {torch.max(torch.abs(dk_sdpa - dk_flex)):.5f}, dV: {torch.max(torch.abs(dv_sdpa - dv_flex)):.5f}")
        print(f"Tri vs SDPA  -> dQ: {torch.max(torch.abs(dq_sdpa - dq_tri_fmt)):.5f}, dK: {torch.max(torch.abs(dk_sdpa - dk_tri_fmt)):.5f}, dV: {torch.max(torch.abs(dv_sdpa - dv_tri_fmt)):.5f}")
        print("*(Values below ~0.05 are considered numerically equivalent due to bf16 summation differences)*")

        print("\n--- Running Benchmarks ---")
        ms_flash = triton.testing.do_bench(bench_sdpa)
        ms_triton = triton.testing.do_bench(lambda: flash_att_backward_split(q, k, v, o, do, lse_sum, cuseq))
        if HAS_FLEX:
            ms_flex = triton.testing.do_bench(bench_flex)

        flops_bwd = 10 * num_docs * num_heads * (seq_len ** 2) * head_dim
        tflops_flash = flops_bwd / (ms_flash * 1e-3) / 1e12
        tflops_triton = flops_bwd / (ms_triton * 1e-3) / 1e12

        print("\n--- Backward Pass Benchmark ---")
        print(f"{'Implementation':<17} | {'Time (ms)':>10} | {'TFLOPs':>10}")
        print("-" * 45)
        print(f"{'SDPA FlashAttn-2':<17} | {ms_flash:>10.4f} | {tflops_flash:>10.1f}")
        if HAS_FLEX:
            tflops_flex = flops_bwd / (ms_flex * 1e-3) / 1e12
            print(f"{'FlexAttention':<17} | {ms_flex:>10.4f} | {tflops_flex:>10.1f}")
        print(f"{'Triton Custom':<17} | {ms_triton:>10.4f} | {tflops_triton:>10.1f}")


    benchmark_backwd()
