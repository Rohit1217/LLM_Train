import torch
import triton
import triton.language as tl
from torch.nn.attention.flex_attention import flex_attention, create_block_mask



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
    ],
    key=[ 'NUM_HEADS','HEAD_DIM','group_size'],
)
@triton.jit
def flash_att_intra_doc_masked(q_ptr,k_ptr,v_ptr,o_ptr,cuseq_ptr,lse_ptr,total_tokens,sm_scale:tl.constexpr,
                               group_size:tl.constexpr,HEAD_DIM:tl.constexpr,
                                BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr):

    #Read grid
    pid=tl.program_id(axis=0)
    pid_doc=tl.program_id(axis=1)
    pid_h=tl.program_id(axis=2)

    #Assume data in shape (num_tokens,num_head,head_dim)
    #Load the doc start and end idx and calculate doc length using them

    start=tl.load(cuseq_ptr+pid_doc)
    end=tl.load(cuseq_ptr+pid_doc+1)

    doc_len=end-start

    if pid*BLOCK_M >= doc_len:
        return

    #Load q,k,v ptrs
    stride_tok_q=tl.num_programs(axis=2) * HEAD_DIM
    stride_tok_k=stride_tok_q//group_size
    pid_kh=pid_h//group_size

    q_ptr_b=q_ptr+start*stride_tok_q+ pid_h*HEAD_DIM
    k_ptr_b=k_ptr+start*stride_tok_k + pid_kh*HEAD_DIM
    v_ptr_b=v_ptr+start*stride_tok_k + pid_kh*HEAD_DIM
    o_ptr_b=o_ptr+start*stride_tok_q+ pid_h*HEAD_DIM

    lse_ptr_b=lse_ptr+pid_h*total_tokens + start

    q_block_ptr=tl.make_block_ptr(base=q_ptr_b,
                                  shape=(doc_len,HEAD_DIM),
                                  strides=(stride_tok_q,1),
                                  offsets=(pid*BLOCK_M,0),
                                  block_shape=(BLOCK_M,HEAD_DIM),
                                  order=(1,0))

    #Load k transposed,now changed since tl.trans(k) inside tl.dot is faster
    k_block_ptr=tl.make_block_ptr(k_ptr_b,shape=(doc_len,HEAD_DIM),strides=(stride_tok_k,1),
                                       offsets=(0,0),block_shape=(BLOCK_N,HEAD_DIM),order=(1,0))

    v_block_ptr=tl.make_block_ptr(base=v_ptr_b,
                                shape=(doc_len,HEAD_DIM),
                                strides=(stride_tok_k,1),
                                offsets=(0,0),
                                block_shape=(BLOCK_N,HEAD_DIM),
                                order=(1,0))

    o_block_ptr=tl.make_block_ptr(base=o_ptr_b,
                                  shape=(doc_len,HEAD_DIM),
                                  strides=(stride_tok_q,1),
                                  offsets=(pid*BLOCK_M,0),
                                  block_shape=(BLOCK_M,HEAD_DIM),
                                  order=(1,0))

    lse_block_ptr=tl.make_block_ptr(base=lse_ptr_b,shape=(doc_len,),
                            strides=(1,),offsets=(pid*BLOCK_M,),
                            block_shape=(BLOCK_M,),order=(0,))

    #scale trick: fold 1/ln(2) into qk so we can use exp2 (hardware-accelerated) instead of exp
    qk_scale = sm_scale * 1.44269504

    q = tl.load(q_block_ptr,boundary_check=(0,),padding_option="zero")

    # q = (q * qk_scale).to(tl.bfloat16)

    #Buffers to accumulate res ,max and sums
    acc=tl.zeros((BLOCK_M,HEAD_DIM),tl.float32)
    max_val=tl.full((BLOCK_M,),float("-inf"),tl.float32)
    sum_val=tl.zeros((BLOCK_M,),tl.float32)

    q_offset=pid * BLOCK_M + tl.arange(0, BLOCK_M)
    doc_bias=tl.where(q_offset<doc_len,0.0,-10000.0) # offsets for  doc masking which q crossing boundary


    #safe part no causality got till start of query block
    for i in range(0,pid*BLOCK_M,BLOCK_N): #There will be some tail  q # We can assume that q block size will be greater than kv

        k=tl.load(k_block_ptr,boundary_check=(0,),padding_option="zero")
        v=tl.load(v_block_ptr,boundary_check=(0,),padding_option="zero")

        scores=tl.dot(q,tl.trans(k))
        scores=scores*qk_scale
        scores+=doc_bias[:,None]

        new_max=tl.maximum(max_val,tl.max(scores,axis=1))
        scores_exp=tl.math.exp2(scores-new_max[:,None])

        update_factor=tl.math.exp2(max_val-new_max)

        acc=acc*update_factor[:,None]

        acc=tl.dot(scores_exp.to(tl.bfloat16),v,acc)

        sum_val=sum_val*update_factor + tl.sum(scores_exp,axis=1)
        max_val=new_max

        k_block_ptr=tl.advance(k_block_ptr,(BLOCK_N,0))
        v_block_ptr=tl.advance(v_block_ptr,(BLOCK_N,0))


    #Causal and doc masking(Need to do block masking padding for q,k,v)

    limit_causal = tl.minimum((pid + 1) * BLOCK_M, doc_len)
    for i in range(pid*BLOCK_M,limit_causal,BLOCK_N): #We ensure that BLOCK_N divides BLOCK_M so no tailwind,(Offcourse doc masking remains)
        k=tl.load(k_block_ptr,boundary_check=(0,),padding_option="zero")
        v=tl.load(v_block_ptr,boundary_check=(0,),padding_option="zero")

        scores=tl.dot(q,tl.trans(k))
        scores=scores*qk_scale

        k_offset=i+tl.arange(0,BLOCK_N)

        causal_bias=tl.where(q_offset[:, None] >= k_offset[None, :], 0.0, -10000.0)
        #This creates an M,N causal mask similar to float inf mask in standard torch inf
        scores=scores + causal_bias + doc_bias[:,None]

        new_max=tl.maximum(max_val,tl.max(scores,axis=1))
        scores_exp=tl.math.exp2(scores-new_max[:,None])
        update_factor=tl.math.exp2(max_val-new_max)

        acc=acc*update_factor[:,None]
        acc=tl.dot(scores_exp.to(tl.bfloat16),v,acc)

        sum_val=sum_val*update_factor + tl.sum(scores_exp,axis=1)
        max_val=new_max

        k_block_ptr=tl.advance(k_block_ptr,(BLOCK_N,0))
        v_block_ptr=tl.advance(v_block_ptr,(BLOCK_N,0))

    acc=acc/sum_val[:,None]
    lse=max_val + tl.math.log2(sum_val)

    tl.store(o_block_ptr, acc.to(tl.bfloat16), boundary_check=(0,))
    tl.store(lse_block_ptr, lse, boundary_check=(0,))

    return


def triton_intradoc_attention(q,k,v,cuseq,max_len,group_size):
    out=torch.empty_like(q)

    num_docs=cuseq.shape[0]-1
    total_tokens,num_heads,head_dim=q.shape   #DERIVE FROM q: MODULE LEVEL HEAD_DIM/NUM_HEADS ONLY EXIST UNDER __main__
    lse=torch.empty((num_heads,total_tokens),device=q.device) #num heads,num_tokens for better coalesced mem access

    # BLOCK_M,BLOCK_N=128,32
    sm_scale=1/(head_dim**0.5)

    grid=lambda META: (triton.cdiv(max_len,META["BLOCK_M"]),num_docs,num_heads)
    flash_att_intra_doc_masked[grid](q,k,v,out,cuseq,lse,total_tokens,sm_scale,group_size,head_dim)
    return out,lse



if __name__=="__main__":
    DEVICE = "cuda:2"
    DTYPE = torch.bfloat16
    HEAD_DIM = 128
    NUM_HEADS = 12
    NUM_DOCS = 8
    SEQ_LEN = 1024
    torch.cuda.set_device(DEVICE)
    # 2. COMPILED FLEX ATTENTION SETUP
    #torch compile
    compiled_flex = torch.compile(flex_attention, dynamic=False)

    def setup_flex_attention(q, k, v, doc_ids):
        q_f = q.unsqueeze(0).transpose(1, 2)
        k_f = k.unsqueeze(0).transpose(1, 2)
        v_f = v.unsqueeze(0).transpose(1, 2)

        def doc_causal_mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])

        total_seq = q_f.shape[2]
        block_mask = create_block_mask(doc_causal_mask, B=1, H=1, Q_LEN=total_seq, KV_LEN=total_seq, device=DEVICE)

        def run_flex():
            return compiled_flex(q_f, k_f, v_f, block_mask=block_mask)

        for _ in range(3):
            _ = run_flex()
        print("Warmup Complete.")

        return run_flex

    def benchmark():
        # 3. BENCHMARK HARNESS
        total_tokens = SEQ_LEN * NUM_DOCS

        lengths = torch.full((NUM_DOCS,), SEQ_LEN, dtype=torch.int32, device=DEVICE)
        cu_seq_lens = torch.zeros(NUM_DOCS + 1, dtype=torch.int32, device=DEVICE)
        cu_seq_lens[1:] = torch.cumsum(lengths, dim=0)

        doc_ids = torch.zeros(total_tokens, dtype=torch.int32, device=DEVICE)
        for i in range(NUM_DOCS):
            doc_ids[cu_seq_lens[i]:cu_seq_lens[i+1]] = i

        q = torch.randn(total_tokens, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        k = torch.randn(total_tokens, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        v = torch.randn(total_tokens, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)

        # Setup
        _ = triton_intradoc_attention(q, k, v, cu_seq_lens)
        run_flex = setup_flex_attention(q, k, v, doc_ids)

        # Correctness verification (Using standard 2e-2 tolerance for BF16)
        out_triton,lse = triton_intradoc_attention(q, k, v, cu_seq_lens)
        out_flex_raw = run_flex()
        out_flex = out_flex_raw.transpose(1, 2).squeeze(0)

        max_err = (out_triton - out_flex).abs().max().item()
        ok = torch.allclose(out_triton, out_flex, atol=2e-2, rtol=2e-2)

        # Benchmark Iterations
        ms_triton = triton.testing.do_bench(lambda: triton_intradoc_attention(q, k, v, cu_seq_lens))
        ms_flex = triton.testing.do_bench(lambda: run_flex())

        # Causal FLOPs for multiple stacked documents
        flops = NUM_DOCS * (2 * NUM_HEADS * (SEQ_LEN ** 2) * HEAD_DIM)
        tflops_triton = flops / (ms_triton * 1e-3) / 1e12
        tflops_flex = flops / (ms_flex * 1e-3) / 1e12

        print(f"\n{'Seq':>6} | {'Triton(ms)':>10} {'Flex(ms)':>9} | {'Tri TFLOPs':>10} {'Flex TFLOPs':>11} | {'Ratio':>6} {'MaxErr':>8} {'OK':>4}")
        print("-" * 80)
        print(f"{SEQ_LEN:>6} | {ms_triton:>10.4f} {ms_flex:>9.4f} | {tflops_triton:>10.1f} {tflops_flex:>11.1f} | {ms_triton/ms_flex:>6.2f} {max_err:>8.4f} {str(ok):>4}")

    benchmark()
