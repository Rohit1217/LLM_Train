
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Intradoc_kernels"))  # kernels moved into Intradoc_kernels/
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["TORCH_USE_CUDA_DSA"]="1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from intradocatt_kernel_torch import IntraDocAttention

#LINEAR WITH HE INITALIZATION
class linear_swig(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        #const_var=torch.tensor(2**0.5/in_dim)  #HE INTIALIZATION SCALED VAR CACLULATION FOR SWIGLU TO  KEEP VAR 1 IN FORWARD PASS
        #Istd=torch.pow(const_var,0.5)

        std=torch.sqrt(torch.tensor(1/in_dim)) #SMALL INIT AS USED IN LLAMA AND OTHER RECENT WORKS

        weight=torch.randn(in_dim,out_dim)*std
        self.weight=nn.Parameter(weight)        
    
    def forward(self,x):
        out=x@self.weight 
        return out

## LINEAR PROJ
class linear_proj(nn.Module):
    def __init__(self,in_dim,out_dim,num_layers):
        super().__init__()
        const_dmodel=1/in_dim
        const_var=torch.tensor(const_dmodel/(2*num_layers))  #LAYER AWARE INIT TO MAKE SURE RESIDUAL VAR DOESNT EXPLODE
        std=torch.sqrt(const_var)

        weight=torch.randn(in_dim,out_dim)*std
        self.weight=nn.Parameter(weight)    

    def forward(self,x):
        out=x@self.weight 
        return out

def silu(x):
    return F.silu(x)

#RMS NORM
class rms_norm(nn.Module):
    def __init__(self,num_dim):
        super().__init__()
        scale=torch.ones(num_dim)

        self.scale=nn.Parameter(scale)

        rootn=torch.pow(torch.tensor(num_dim),0.5)
        eps=torch.tensor(2e-08)

        self.register_buffer("rootn",rootn)
        self.register_buffer("eps",eps)

    def forward(self,x):
        xdtype=x.dtype
        x=x.float()
        norm_rms=torch.norm(x,dim=-1,keepdim=True)
        
        x_norm=(x*self.rootn)/(norm_rms+self.eps)
        x_norm=x_norm*self.scale 
        x_norm=x_norm.to(dtype=xdtype)
        return x_norm


#SWIGLU
class swiglu(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.linear1=linear_swig(in_dim,out_dim)
        self.linear2=linear_swig(in_dim,out_dim)

    def forward(self,x):
        return silu(self.linear1(x))*self.linear2(x)
            
#FFN ROUTER IN TRANSFORMER BLOCK
class ffn_router(nn.Module):
    def __init__(self,in_dim,hidden_dim,num_layers,ffn_dropout):
        super().__init__()
        self.swig=swiglu(in_dim,hidden_dim)
        self.fc=linear_proj(hidden_dim,in_dim,num_layers)
        self.drop=nn.Dropout(ffn_dropout)
    
    def forward(self,x):
        return self.fc(self.drop(self.swig(x)))


#ROPE FAST VECTORIZED
def precompute_rope_fast(max_context,max_freq,head_dim): # USES OUTER PRODUCT TO BUILD THETA VALUES AND APPLY SIN COS AT ONCE
    seq_pos_tensor=torch.arange(max_context)
    i=torch.arange(0,head_dim//2)
    
    theta=1/torch.pow(max_freq,(2*i.float())/head_dim)
    seq_pos_theta=torch.outer(seq_pos_tensor,theta)
    return torch.cos(seq_pos_theta).repeat(1,2),torch.sin(seq_pos_theta).repeat(1,2)


def apply_rope_varlen(x,pos,cos_embed,sin_embed):

    BT,D=x.shape[0]
    x_type=x.dtype
    xf=x.to(torch.float32)
    
    new_pos=torch.ones(pos[-1]).long()
    lengths=torch.diff(pos,prepend=torch.tensor([0]))
    new_pos[pos[:-1]]-=lengths[:-1]
    new_pos=torch.cumsum(new_pos,dim=0)-1

    cos,sin=cos_embed[new_pos,:],sin_embed[new_pos,:]
    cos,sin=cos.unsqueeze(1),sin.unsqueeze(1)

    return (xf*cos + rotate_half(xf)*sin).to(x_type)


def apply_rope_varlen2(x,cos_embed,sin_embed,cuseq=None):

    BT=x.shape[-2]
    x_type=x.dtype
    xf=x.to(torch.float32)

    if cuseq is None:
        cos,sin=cos_embed[BT,:],sin_embed[BT,:]
    else:
        pos=cuseq[1:]-cuseq[:-1]
        cos,sin=cos_embed[pos,:],sin_embed[pos,:]

    cos,sin=cos.unsqueeze(1),sin.unsqueeze(1)

    return (xf*cos + rotate_half(xf)*sin).to(x_type)


def apply_rope(x,cos_embed,sin_embed): #EITHER COMPUTE ROPE ONCE FOR MAX CONTEXT OR COMPUTE FOR EACH CONTEXT LENGTH PER BATCH CHOICE 
    T=x.shape[-2]
    cos,sin=cos_embed[:T,:],sin_embed[:T,:]
    x_type=x.dtype
    xf=x.to(torch.float32)
    return (xf*cos + rotate_half(xf)*sin).to(x_type)

def apply_rope_seq_pos(q,cos_embed,sin_embed,seq_pos):
    T=q.shape[-2]
    cos,sin=cos_embed[seq_pos:seq_pos+T,:],sin_embed[seq_pos:seq_pos+T,:]
    q_type=q.dtype
    qf=q.to(torch.float32)
    return (qf*cos + rotate_half(qf)*sin).to(q_type)

def rotate_half(x):
    x1,x2=x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)

def apply_rope_intradoc(q,k,token_positions,cos,sin):
    qf,kf=q.to(torch.float32),k.to(torch.float32)
    cos,sin=cos[token_positions],sin[token_positions]

    q_rot,k_rot=(qf*cos + rotate_half(qf)*sin).to(q.dtype),(kf*cos + rotate_half(kf)*sin).to(k.dtype)
    return q_rot,k_rot

def intradoc_positions(cuseq,device): #PER DOC POSITIONS FROM CUSEQ SO ROPE RESETS AT EACH DOC BOUNDARY
    lengths=cuseq[1:]-cuseq[:-1]
    doc_start=torch.repeat_interleave(cuseq[:-1],lengths)
    tok=torch.arange(cuseq[-1],device=device)
    return (tok-doc_start).long().unsqueeze(1)
    

#CAUSAL MASKED MULTI HEAD SELF ATTENTION IMLEMENTATION
class mhma(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,attn_dropout):
        super().__init__()

        self.qkv_proj=linear_swig(d_model,d_model*3)
        self.rms_norm_att=rms_norm(d_model)
        self.attn_drop = nn.Dropout(p=attn_dropout)

        self.linear_proj=linear_proj(d_model,d_model,num_layers)

        self.head_dim=d_model//n_heads
        self.n_heads=n_heads
    
    def forward(self,x,cos,sin,cuseq=None,token_positions=None):
        B,T,D=x.shape
        residual=x
        x=self.rms_norm_att(residual)

        qkv=self.qkv_proj(x)
        qkv=qkv.view(B,T,3,self.n_heads,self.head_dim)
        q,k,v=qkv.permute(0,3,2,1,4).unbind(dim=2)

        q,k=apply_rope(q,cos,sin),apply_rope(k,cos,sin)

        x = F.scaled_dot_product_attention(q, k, v,is_causal=True)
        x=x.permute(0,2,1,3)
        x=x.contiguous().view(B,T,D)
        x=self.linear_proj(x)
        x+=residual
        return x

#TRANSFORMER BLOCK WITH FFN AND SELF ATTENTION BLOCK
class Transformer_block(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,attn_dropout,ffn_hidden_dim,ffn_dropout,max_len,group_size,intradoc_att=False):
        super().__init__()

        if group_size<=1:
            self.att=mhma(d_model,n_heads,num_layers,attn_dropout)
        elif intradoc_att:
            self.att=gqa_intradoc(d_model,n_heads,num_layers,attn_dropout,max_len,group_size)
        else:
            self.att=gqa(d_model,n_heads,num_layers,attn_dropout,group_size)

        self.intradoc=intradoc_att

        self.ffn=ffn_router(d_model,ffn_hidden_dim,num_layers,ffn_dropout)
        self.rms_norm_ffn=rms_norm(d_model)

    def forward(self,x,cos,sin,cuseq=None,token_positions=None):
        x=self.att(x,cos,sin,cuseq,token_positions)
        residual=x
        x=self.ffn(self.rms_norm_ffn(x)) +residual
        return x

    def decode_inference(self,x,k,v,cos,sin,seq_pos,layer_idx,verif=False):
        x=self.att.decode_inference(x,k,v,cos,sin,seq_pos,layer_idx,verif)
        residual=x
        x=self.ffn(self.rms_norm_ffn(x)) +residual
        return x

    def prefill_inference(self,x,cos,sin):
        x,k,v=self.att.prefill_inference(x,cos,sin)
        residual=x
        x=self.ffn(self.rms_norm_ffn(x)) +residual
        return x,k,v

class gqa_intradoc(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,attn_dropout,max_len,group_size):
        super().__init__()

        self.q_proj=linear_swig(d_model,d_model)
        self.kv_proj=linear_swig(d_model,(2*d_model)//group_size)
        self.rms_norm_att=rms_norm(d_model)
        self.attn_drop = nn.Dropout(p=attn_dropout)

        self.linear_proj=linear_proj(d_model,d_model,num_layers)

        self.intradoc_atten=IntraDocAttention(max_len,group_size)

        self.head_dim=d_model//n_heads
        self.kvn_heads=n_heads//group_size
        self.group_size=group_size

        self.max_len=max_len

    def forward(self,x,cos,sin,cuseq,token_positions=None):
        B,T,D=x.shape
        residual=x
        x=self.rms_norm_att(x)

        q =self.q_proj(x)
        q=q.view(B*T,self.kvn_heads*self.group_size,self.head_dim).contiguous()

        kv=self.kv_proj(x)
        kv=kv.view(B*T,2,self.kvn_heads,self.head_dim)
        k,v=kv.unbind(dim=1)
        k,v=k.contiguous(),v.contiguous()

        if token_positions is None:                       #FALLBACK: derive per-doc positions from cuseq if not supplied
            token_positions=intradoc_positions(cuseq,x.device)
        
        q,k=apply_rope_intradoc(q,k,token_positions,cos,sin)
        q,k=q.contiguous(),k.contiguous()

        x = self.intradoc_atten(q,k,v,cuseq)
        x=x.view(B,T,D)

        x=self.linear_proj(x)

        x+=residual
        return x


class gqa(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,attn_dropout,groups):
        super().__init__()

        self.q_proj=linear_swig(d_model,d_model)
        self.kv_proj=linear_swig(d_model,(2*d_model)//groups)
        self.rms_norm_att=rms_norm(d_model)
        self.attn_drop = nn.Dropout(p=attn_dropout)

        self.linear_proj=linear_proj(d_model,d_model,num_layers)

        self.head_dim=d_model//n_heads
        self.kvn_heads=n_heads//groups
        self.groups=groups
    
    def forward(self,x,cos,sin,cuseq=None,token_positions=None):
        B,T,D=x.shape
        residual=x
        x=self.rms_norm_att(x)

        q =self.q_proj(x)
        q=q.view(B,T,self.kvn_heads*self.groups,self.head_dim)
        q=q.permute(0,2,1,3)

        kv=self.kv_proj(x)
        kv=kv.view(B,T,2,self.kvn_heads,self.head_dim)
        k,v=kv.permute(0,2,3,1,4).unbind(dim=1)

        q,k=apply_rope(q,cos,sin),apply_rope(k,cos,sin)
        x = F.scaled_dot_product_attention(q, k, v,is_causal=True,enable_gqa=True)
        x=x.permute(0,2,1,3).contiguous()
        x=x.view(B,T,D)
        
        x=self.linear_proj(x)

        x+=residual
        return x  

    def decode_inference(self,x,k,v,cos,sin,seq_pos,layer_idx,verif=False):
        B,T,D=x.shape
        
        residual=x
        x=self.rms_norm_att(x)
        
        q=self.q_proj(x)
        q=q.view(B,T,self.kvn_heads*self.groups,self.head_dim)
        q=q.permute(0,2,1,3)

        x_kv=self.kv_proj(x)
        x_kv=x_kv.view(B,T,2,self.kvn_heads,self.head_dim)
        xk,xv=x_kv.permute(0,2,3,1,4).unbind(dim=1)

        # print(xk.shape,k.shape,xk.unsqueeze(0).shape)
        offset=xk.shape[-2]
        S=seq_pos+offset
        q=apply_rope_seq_pos(q,  cos, sin, seq_pos)        
        xk=apply_rope_seq_pos(xk, cos, sin, seq_pos)        
        k[layer_idx,:,:,seq_pos:seq_pos+offset,:]=xk       
        v[layer_idx,:,:,seq_pos:seq_pos+offset,:]=xv

        if verif:
            m = torch.zeros(offset, S, dtype=torch.bool, device=q.device)
            m[:, seq_pos:] = torch.triu(torch.ones(offset, offset, device=q.device), 1).bool()  # mask future new toks
            x = F.scaled_dot_product_attention(q,k[layer_idx,:,:,:seq_pos+offset,:], v[layer_idx,:,:,:seq_pos+offset,:], attn_mask=~m, enable_gqa=True)
        else:
             x = F.scaled_dot_product_attention(q, k[layer_idx,:,:,:seq_pos+offset,:], v[layer_idx,:,:,:seq_pos+offset,:],is_causal=False,enable_gqa=True)
        
        x=x.permute(0,2,1,3).contiguous()
        x=x.view(B,T,D)
        
        x=self.linear_proj(x)
        x+=residual
        return x   

    def prefill_inference(self,x,cos,sin):
        B,T,D=x.shape
        residual=x
        x=self.rms_norm_att(x)

        q =self.q_proj(x)
        q=q.view(B,T,self.kvn_heads*self.groups,self.head_dim)
        q=q.permute(0,2,1,3)

        kv=self.kv_proj(x)
        kv=kv.view(B,T,2,self.kvn_heads,self.head_dim)
        k,v=kv.permute(0,2,3,1,4).unbind(dim=1)

        q,k=apply_rope(q,cos,sin),apply_rope(k,cos,sin)
        x = F.scaled_dot_product_attention(q, k, v,is_causal=True,enable_gqa=True)
        x=x.permute(0,2,1,3).contiguous()
        x=x.view(B,T,D)
        
        x=self.linear_proj(x)

        x+=residual
        return x,k,v  


class mtp_head(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,attn_dropout,ffn_hidden_dim,
                 ffn_dropout,group_size,max_len,intradoc_att):
        super().__init__()
        self.proj=linear_proj(2*d_model,d_model,num_layers=1)
        
        self.rms_embed=rms_norm(d_model)
        self.rms_out=rms_norm(d_model)

        self.trans_block=Transformer_block(d_model,n_heads,num_layers,attn_dropout,
                                           ffn_hidden_dim,ffn_dropout,max_len,group_size,intradoc_att)
    
    def forward(self,h,embed,cos,sin,cuseq=None,token_positions=None):
        x=torch.cat([h,self.rms_embed(embed)],dim=-1)
        x=self.proj(x)
        x=self.trans_block(x,cos,sin,cuseq,token_positions)
        return self.rms_out(x)

    def prefill_inference(self,h,embed,cos,sin):
        x=torch.cat([h,self.rms_embed(embed)],dim=-1)
        x=self.proj(x)
        x,k,v=self.trans_block.prefill_inference(x,cos,sin)
        return self.rms_out(x),k,v
    
    def decode_inference(self,h,embed,k,v,cos,sin,seq_pos,layer_idx):

        if len(h.shape)!=len(embed.shape):
            x=torch.cat([h.unsqueeze(0),self.rms_embed(embed)],dim=-1)
        else:
            x=torch.cat([h,self.rms_embed(embed)],dim=-1)

        x=self.proj(x)
        x=self.trans_block.decode_inference(x,k,v,cos,sin,seq_pos,layer_idx)
        return self.rms_out(x)

#TRANSFORMER WITH ALL THE BLOCKS BUILT EARLIERREJECTED
class Transformer(nn.Module):
    def __init__(self,vocab_size,max_context,max_freq,d_model,n_heads,num_layers,attn_dropout,
                 ffn_hidden_dim,ffn_dropout,mtp_heads=None,group_size=1,grad_checkpoint_every=0,
                 max_len=1024,intradoc_att=False):
        super().__init__()

        self.grad_checkpoint_every=grad_checkpoint_every  #0=off; else checkpoint every Nth trunk block (recompute its activations in backward to save memory)
        self.embedding=nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**(-0.5)) #N(0,1/sqrt(d)) init so output norm is 1

        self.transformer_block_list=nn.ModuleList([Transformer_block(d_model,n_heads,num_layers,attn_dropout,
                                                                     ffn_hidden_dim,ffn_dropout,max_len,
                                                                     group_size,intradoc_att) for _ in range(num_layers)])
        self.rms_out=rms_norm(d_model)
        
        if mtp_heads:
            self.mtp_heads_list=nn.ModuleList([mtp_head(d_model,n_heads,num_layers,attn_dropout,
                                               ffn_hidden_dim,ffn_dropout,group_size,max_len,intradoc_att) for _ in range(mtp_heads)])
        else:
            self.mtp_heads_list=[]        

        self.head_dim=d_model//n_heads
        self.nkv_heads=n_heads//group_size
        self.d_model=d_model
        self.num_mtp_heads=len(self.mtp_heads_list)
        self.num_layers=num_layers
        self.intradoc_att=intradoc_att

        # perm,alt_bit=permute_indcies_rope(head_dim)
        cos,sin=precompute_rope_fast(max_context,max_freq,self.head_dim)
        mask=torch.triu(torch.ones(max_context,max_context),diagonal=1).bool()

        self.register_buffer("cos",cos,persistent=True)
        self.register_buffer("sin",sin,persistent=True)
        self.register_buffer("mask",mask)

    def forward(self,x,cuseq=None,token_positions=None):
        x_embed=self.embedding(x.int()) #B,T+mtp_heads+1,D  #tokens arrive as uint16 (memmap);  embedding needs Int/Long
        B,T,C=x_embed.shape
        L=T-self.num_mtp_heads-1 #L seq length model sees

        x=x_embed[:,:L,:].contiguous()

        if token_positions is not None:                #(total_tokens,)->(total_tokens,1) long, for broadcasting cos/sin over heads
            token_positions=token_positions.view(-1,1).long()

        trunk_cos,trunk_sin=(self.cos,self.sin) if self.intradoc_att else (self.cos[:L],self.sin[:L])

        #Activation checkpointing(not used in training)
        for i,trans_block in enumerate(self.transformer_block_list):
            if self.grad_checkpoint_every and i % self.grad_checkpoint_every == 0:
                x=checkpoint(trans_block,x,trunk_cos,trunk_sin,cuseq,token_positions,
                             use_reentrant=False,preserve_rng_state=False)
            else:
                x=trans_block(x,trunk_cos,trunk_sin,cuseq,token_positions)

        x=self.rms_out(x)
        prev_h_rms=x

        #MTP HEADS
        for i,mtp_head in enumerate(self.mtp_heads_list):
            embed=x_embed[:,1+i:L+1+i,:].contiguous()
            if self.intradoc_att:                      #per-doc rope from full table; std path shifts window by 1+i
                x=mtp_head(x,embed,self.cos,self.sin,cuseq,token_positions)
            else:
                x=mtp_head(x,embed,self.cos[1+i:1+L+i],self.sin[1+i:1+L+i],cuseq)
            prev_h_rms=torch.cat([prev_h_rms,x],dim=0)

        return prev_h_rms #OUTPUT B*(mtp_heads+1),T,D_MODEL

    def generate(self,x,seq_len,temperature=1.0):
        self.eval()
        max_ctx=self.mask.shape[0]
        pad=self.num_mtp_heads+1  
        B=x.shape[0]

        with torch.no_grad():
            for _ in range(seq_len):
                x_cond=x[:,-(max_ctx-pad):]
                x_in=F.pad(x_cond,(0,pad))                       
                out=self.forward(x_in)                           
                logits=out[:B,-1,:]@self.embedding.weight.T
                prob=F.softmax(logits.float()/temperature,dim=-1)
                pred=torch.multinomial(prob,num_samples=1)        
                x=torch.cat([x,pred],dim=1)                        
        
        return x
    
    def buffers_to_float(self):
        for name, buffer in list(self.named_buffers()):       
            if torch.is_floating_point(buffer):               
                parent = self.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self
                attr   = name.rsplit(".", 1)[-1]
                setattr(parent, attr, buffer.float()) 
    
    def make_buffers_none(self):
        for name, buffer in list(self.named_buffers()):       
            parent = self.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self
            attr   = name.rsplit(".", 1)[-1]
            setattr(parent, attr,None) 

    def sample(self,temperature,top_k,top_p):
        return
    
    def prefill_inference(self,x,temperature):
        embed=self.embedding(x).contiguous()
        
        x=embed
        B,T,D=x.shape
        k,v=torch.zeros(self.num_layers+self.num_mtp_heads,B,self.nkv_heads,T,self.head_dim),torch.zeros(self.num_layers+self.num_mtp_heads,B,self.nkv_heads,T,self.head_dim)
        k,v=k.to(x.device).to(x.dtype),v.to(x.device).to(x.dtype)

        for i,trans_block in enumerate(self.transformer_block_list):
            x,k[i,:,:,:,:],v[i,:,:,:,:]=trans_block.prefill_inference(x,self.cos[:T],self.sin[:T])
        
        mtp_x=x
        for i,mtp_head in enumerate(self.mtp_heads_list):
            mtp_x,k[i+self.num_layers,:,:,:,:],v[i+self.num_layers,:,:,:,:]=mtp_head.prefill_inference(mtp_x,embed,self.cos[1+i:1+T+i],self.sin[1+i:1+T+i])
            # prev_h_rms=torch.cat([prev_h_rms,x],dim=0)
            
        hidden_x=self.rms_out(x[:,-1,:])
        logits=hidden_x@self.embedding.weight.T

        main_out=torch.multinomial(F.softmax(logits.float()/temperature,dim=-1),num_samples=1)
        return main_out,k,v
    
    def decode_inference(self,x,k,v,seq_pos,temperature):
        x=x.unsqueeze(0)
        x=self.embedding(x)
        B,T,D=x.shape
        
        Tk=k.shape[-2]
        
        #MAIN MODEL
        for i,trans_block in enumerate(self.transformer_block_list):
            x=trans_block.decode_inference(x,k,v,self.cos[:Tk+T+self.num_mtp_heads],self.sin[:Tk+T+self.num_mtp_heads],seq_pos,i,False)
            
        # print("MAIN MODEL DONE")
        hidden_x=self.rms_out(x[:,-1,:])
        logits=hidden_x@self.embedding.weight.T
        main_out=torch.multinomial(F.softmax(logits.float()/temperature,dim=-1),num_samples=1)
        
        embed_out=self.embedding(main_out)
        draft_embeddings=embed_out

        #MTP DRAFT
        for i,mtp_head in enumerate(self.mtp_heads_list):
            hidden_x=mtp_head.decode_inference(hidden_x,embed_out,k,v,self.cos[:Tk+T+self.num_mtp_heads],
                                               self.sin[:Tk+T+self.num_mtp_heads],seq_pos+i+1,self.num_layers+i).squeeze(0)
            
            logits=hidden_x@self.embedding.weight.T
            out_dist=F.softmax(logits.float()/temperature,dim=-1)
            
            x_out=torch.multinomial(out_dist,num_samples=1)
    
            embed_out=self.embedding(x_out)
            draft_embeddings=torch.cat([draft_embeddings,embed_out],dim=1)

            if i==0:
                draft_dist=out_dist
                draft_idx=x_out
            else:
                draft_dist=torch.cat([draft_dist,out_dist],dim=-2)
                draft_idx=torch.cat([draft_idx,x_out],dim=-2)

        # print("DRAFT DONE",x.shape,draft_embeddings.shape,draft_dist.shape)
        
        #VERIFIER DIST
        x=draft_embeddings[:,:-1,:]

        for i,trans_block in enumerate(self.transformer_block_list):
            x=trans_block.decode_inference(x,k,v,self.cos[:Tk+T+self.num_mtp_heads],self.sin[:Tk+T+self.num_mtp_heads],seq_pos+1,i,True)

        # mtp_x=x
        # embed_out=draft_embeddings[:,1:,:]
        # # print("VERIF TRANS DONE",x.shape,mtp_x.shape,embed_out.shape)
        
        # for i,mtp_head in enumerate(self.mtp_heads_list):
        #     mtp_x=mtp_head.decode_inference(mtp_x,embed_out,k,v,self.cos[:Tk+T+self.num_mtp_heads],
        #                                     self.sin[:Tk+T+self.num_mtp_heads],seq_pos+1+i,self.num_layers+i).squeeze(0)

        # print(mtp_x.shape,embed_out.shape,"SHAPES")

        hidden_x=self.rms_out(x)
        logits=hidden_x@self.embedding.weight.T
        verf_dist=(F.softmax(logits.float()/temperature,dim=-1)).squeeze(0)

        #P(X)/Q(X)>R  where R U[0,1]

        # print(verf_dist.shape,draft_dist.shape,draft_idx.shape,"ADD")
        p_verf=torch.gather(verf_dist, dim=-1, index=draft_idx)
        p_draft=torch.gather(draft_dist, dim=-1, index=draft_idx)
        
        p_verf = p_verf.squeeze(-1)
        p_draft = p_draft.squeeze(-1)

        vd_ratio= p_verf/(p_draft+1e-10)
        uniform=torch.rand(self.num_mtp_heads,device=x.device)
        decisions=torch.where(vd_ratio>uniform,1,0)

        #UPDATE OUTPUT BASED ON ACCEPTANCE
        for i in range(self.num_mtp_heads):
            if decisions[i]:
                continue
            else:
                #REJECT
                x_out=torch.multinomial(F.relu(verf_dist[i]-draft_dist[i])+1e-12,num_samples=1)
                # print(x_out.shape,main_out.shape)
                out_tokens=torch.cat([main_out,draft_idx[:i],x_out.unsqueeze(0)])
                return out_tokens
        
        out_tokens=torch.cat([main_out,draft_idx])
        return out_tokens
        

if __name__=="__main__":
    device="cuda:0"
    torch.manual_seed(0)

    HEAD_DIM,N_HEADS,GROUP=128,12,3
    D_MODEL=N_HEADS*HEAD_DIM                       #1536
    KV_HEADS=N_HEADS//GROUP                        #4
    MTP,NUM_LAYERS,L,B=2,2,8,2
    T=L+MTP+1                                      #model slices x[:,:L]; loader packs EFF_SEQ_LEN=L+MTP+1
    MAX_LEN,MAX_CTX,VOCAB=16,64,50

    x=torch.randint(0,VOCAB,(B,T),device=device).long()

    #DOCS OVER THE B*L TOKENS ATTENTION ACTUALLY SEES (row-major flatten of x[:,:L])
    lengths=torch.tensor([8,3,5],dtype=torch.int32,device=device)
    assert int(lengths.sum())==B*L, "cuseq lengths must cover the B*L attended tokens"
    cuseq=torch.zeros(lengths.shape[0]+1,dtype=torch.int32,device=device)
    cuseq[1:]=torch.cumsum(lengths,dim=0)
    token_positions=torch.cat([torch.arange(int(n),device=device) for n in lengths])  #per-doc local positions

    trans=Transformer(VOCAB,MAX_CTX,10000,D_MODEL,N_HEADS,NUM_LAYERS,0,512,0,
                      mtp_heads=MTP,group_size=GROUP,max_len=MAX_LEN,
                      intradoc_att=True).to(device).to(torch.bfloat16)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out=trans(x,cuseq,token_positions)
        print("out:",out.shape,"expected:",(B*(MTP+1),L,D_MODEL),
              "all_finite:",bool(torch.isfinite(out).all()))

        out.float().sum().backward()               #exercise the custom intradoc autograd (bwd kernel)
        g=trans.embedding.weight.grad
        print("backward ok — embed grad finite:",bool(torch.isfinite(g).all()))