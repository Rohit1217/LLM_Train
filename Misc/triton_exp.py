import torch
import triton
import triton.language as tl
import torch.nn.functional as F

torch.cuda.set_device(2)



@triton.jit
def flash_att_kernel_batched(q_ptr,k_ptr,v_ptr,out_ptr,qn,kvn,sm_scale,NUM_HEADS:tl.constexpr,
                                HEAD_DIM:tl.constexpr,BLOCK_SIZE_Q:tl.constexpr,BLOCK_SIZE_KV:tl.constexpr):
    
    pid_b=tl.program_id(axis=0)
    pid_h=tl.program_id(axis=1)
    pid=tl.program_id(axis=2)

    q_ptr_b=q_ptr + pid_b*NUM_HEADS*qn*HEAD_DIM + pid_h*qn*HEAD_DIM  
    k_ptr_b=k_ptr + pid_b*NUM_HEADS*kvn*HEAD_DIM + pid_h*HEAD_DIM*kvn  
    v_ptr_b=v_ptr +  pid_b*NUM_HEADS*kvn*HEAD_DIM + pid_h*HEAD_DIM*kvn 

    out_ptr_b=out_ptr + pid_b*NUM_HEADS*qn*HEAD_DIM + pid_h*HEAD_DIM*qn 



    q_block_ptr=tl.make_block_ptr(base=q_ptr_b,
                                  shape=(qn,HEAD_DIM),
                                  strides=(HEAD_DIM,1),
                                  offsets=(pid*BLOCK_SIZE_Q,0),
                                  block_shape=(BLOCK_SIZE_Q,HEAD_DIM),
                                 order=(1,0))
    k_block_ptr=tl.make_block_ptr(base=k_ptr_b,
                                  shape=(kvn,HEAD_DIM),
                                  strides=(HEAD_DIM,1),
                                  offsets=(0,0),
                                  block_shape=(BLOCK_SIZE_KV,HEAD_DIM),
                                 order=(1,0))
    v_block_ptr=tl.make_block_ptr(base=v_ptr_b,
                                  shape=(kvn,HEAD_DIM),
                                  strides=(HEAD_DIM,1),
                                  offsets=(0,0),
                                  block_shape=(BLOCK_SIZE_KV,HEAD_DIM),
                                 order=(1,0))
    
    q=tl.load(q_block_ptr)
    sum_val=tl.zeros((BLOCK_SIZE_Q,),tl.float32)
    acc=tl.zeros((BLOCK_SIZE_Q,HEAD_DIM),tl.float32)
    max_val=tl.full((BLOCK_SIZE_Q,) ,float("-inf"),tl.float32)

    for i in range(0,kvn,BLOCK_SIZE_KV):
        k,v=tl.load(k_block_ptr),tl.load(v_block_ptr) #M,H N,H

        scores=tl.dot(q,tl.trans(k))*sm_scale  #M,N
        new_max=tl.maximum(max_val,tl.max(scores,axis=1))  # M

        scores_exp=tl.exp(scores-new_max[:,None]) #M,N
        update_factor=tl.exp(max_val-new_max) #M

        acc=acc*update_factor[:,None] + tl.dot(scores_exp.to(tl.bfloat16), v) #M,H

        sum_val=sum_val*update_factor + tl.sum(scores_exp,axis=1) #M

        max_val=new_max
            
        k_block_ptr=tl.advance(k_block_ptr,(BLOCK_SIZE_KV,0))
        v_block_ptr=tl.advance(v_block_ptr,(BLOCK_SIZE_KV,0))

    acc=acc/sum_val[:,None]

    out_block_ptr=tl.make_block_ptr(base=out_ptr_b,
                                    shape=(qn,HEAD_DIM),
                                    strides=(HEAD_DIM,1),
                                    offsets=(pid*BLOCK_SIZE_Q,0),
                                    block_shape=(BLOCK_SIZE_Q,HEAD_DIM),
                                   order=(1,0))
    
    tl.store(out_block_ptr,acc.to(tl.bfloat16))



def flash_att_block(q, k, v, BLOCK_M=64, BLOCK_N=64):
    batch_size,num_head,num_q, head_dim = q.shape
    num_kv = k.shape[-2]
    sm_scale = 1.0 / (head_dim ** 0.5)
    out = torch.empty_like(q)
    grid = (batch_size,num_head,triton.cdiv(num_q, BLOCK_M),)
    flash_att_kernel_batched[grid](q, k, v, out, num_q, num_kv, sm_scale,NUM_HEADS=num_head,
                           HEAD_DIM=head_dim, BLOCK_SIZE_Q=BLOCK_M, BLOCK_SIZE_KV=BLOCK_N)
    return out

# q,k,v=torch.randn(1024,64,device="cuda:2"),torch.randn(1024,64,device="cuda:2"),torch.randn(1024,64,device="cuda:2")

q,k,v=torch.randn(1,4,1024,64,device="cuda:2").to(torch.bfloat16).contiguous(),torch.randn(1,4,1024,64,device="cuda:2").to(torch.bfloat16).contiguous(),torch.randn(1,4,1024,64,device="cuda:2").to(torch.bfloat16).contiguous()
ms_triton = triton.testing.do_bench(lambda: flash_att_block(q,k,v))

print(ms_triton)

