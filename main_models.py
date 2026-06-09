import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

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
    pos_tensor=torch.arange(max_context)
    i=torch.arange(0,head_dim//2)
    
    theta=1/torch.pow(max_freq,(2*i.float())/head_dim)
    pos_theta=torch.outer(pos_tensor,theta)
    return torch.cos(pos_theta).repeat(1,2),torch.sin(pos_theta).repeat(1,2)


def apply_rope(x,cos_embed,sin_embed): #EITHER COMPUTE ROPE ONCE FOR MAX CONTEXT OR COMPUTE FOR EACH CONTEXT LENGTH PER BATCH CHOICE 
    T=x.shape[-2]
    cos,sin=cos_embed[:T,:],sin_embed[:T,:]
    x_type=x.dtype
    xf=x.to(torch.float32)
    return (xf*cos + rotate_half(xf)*sin).to(x_type)

def rotate_half(x):
    x1,x2=x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)

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
    
    def forward(self,x,mask,cos,sin):
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
    def __init__(self,d_model,n_heads,num_layers,attn_dropout,ffn_hidden_dim,ffn_dropout,gqa_groups=0):
        super().__init__()

        if gqa_groups<=0:
            self.att=mhma(d_model,n_heads,num_layers,attn_dropout)
        else:
            self.att=gqa(d_model,n_heads,num_layers,attn_dropout,gqa_groups)

        self.ffn=ffn_router(d_model,ffn_hidden_dim,num_layers,ffn_dropout)
        self.rms_norm_ffn=rms_norm(d_model)

    def forward(self,x,mask,cos,sin):
        x=self.att(x,mask,cos,sin)
        residual=x
        x=self.ffn(self.rms_norm_ffn(x)) +residual
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
    
    def forward(self,x,mask,cos,sin):
        B,T,D=x.shape
        residual=x
        x=self.rms_norm_att(residual)

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

class mtp_head(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,attn_dropout,ffn_hidden_dim,ffn_dropout,gqa_groups):
        super().__init__()
        self.proj=linear_proj(2*d_model,d_model,num_layers=1)
        
        self.rms_embed=rms_norm(d_model)
        self.rms_out=rms_norm(d_model)

        self.trans_block=Transformer_block(d_model,n_heads,num_layers,attn_dropout,ffn_hidden_dim,ffn_dropout,gqa_groups)
    
    def forward(self,h,embed,mask,cos,sin):
        x=torch.cat([h,self.rms_embed(embed)],dim=-1)
        x=self.proj(x)
        x=self.trans_block(x,mask,cos,sin)
        return self.rms_out(x)


#TRANSFORMER WITH ALL THE BLOCKS BUILT EARLIER
class Transformer(nn.Module):
    def __init__(self,vocab_size,max_context,max_freq,d_model,n_heads,num_layers,attn_dropout,ffn_hidden_dim,ffn_dropout,mtp_heads=None,gqa_groups=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**(-0.5)) #N(0,1/sqrt(d)) init so output norm is 1

        self.transformer_block_list=nn.ModuleList([Transformer_block(d_model,n_heads,num_layers,attn_dropout,
                                                                     ffn_hidden_dim,ffn_dropout,gqa_groups) for idx in range(num_layers)])
        self.rms_out=rms_norm(d_model)
        
        head_dim=d_model//n_heads
        self.d_model=d_model
        self.num_mtp_heads=len(self.mtp_heads_list)

        if mtp_heads:
            self.mtp_heads_list=nn.ModuleList([mtp_head(d_model,n_heads,num_layers,attn_dropout,
                                                          ffn_hidden_dim,ffn_dropout,gqa_groups) for idx in range(mtp_heads)])
        else:
            self.mtp_heads_list=[]
        # perm,alt_bit=permute_indcies_rope(head_dim)
        cos,sin=precompute_rope_fast(max_context,max_freq,head_dim)
        mask=torch.triu(torch.ones(max_context,max_context),diagonal=1).bool()

        self.register_buffer("cos",cos)
        self.register_buffer("sin",sin)
        self.register_buffer("mask",mask)




    def forward(self,x):
        x_embed=self.embedding(x) #B,T+mtp_heads+1,D
        B,T,C=x_embed.shape
        L=T-self.num_mtp_heads-1 #L seq length model sees

        x=x_embed[:,:L,:].contiguous() 

        #MAIN  BLOCK
        for i,trans_block in enumerate(self.transformer_block_list):
            x=trans_block(x,self.mask[:L,:L],self.cos[:L],self.sin[:L])
        
        x=self.rms_out(x)
        prev_h_rms=x

        #MTP HEADS
        for i,mtp_head in enumerate(self.mtp_heads_list):
            embed=x_embed[:,1+i:L+1+i,:].contiguous() 
            x=mtp_head(x,embed,self.mask[:L,:L],self.cos[:L],self.sin[:L])
            prev_h_rms=torch.cat([prev_h_rms,x],dim=0)

        return prev_h_rms #OUTPUT B*(mtp_heads+1),T,D_MODEL

    def generate(self,x,seq_len,temperature=1.0):
        self.eval()
        max_ctx=self.mask.shape[0]
        pad=self.num_mtp_heads+1   #FORWARD DROPS num_mtp_heads+1 TOKENS (TEACHER-FORCING SHIFT); PAD SO LAST REAL TOKEN EMITS NEXT-TOKEN LOGITS
        B=x.shape[0]
        with torch.no_grad():
            for _ in range(seq_len):
                x_cond=x[:,-(max_ctx-pad):]
                x_in=F.pad(x_cond,(0,pad))                       #APPEND pad DUMMY TOKENS (CAUSAL-MASKED, DONT AFFECT hidden[n-1])
                out=self.forward(x_in)                           #(B*(num_mtp+1),n,D) -> MAIN HEAD IS FIRST B ROWS
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


if __name__=="__main__":
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):

        x=torch.tensor([33]).to("cuda:7")
        x=torch.ones((4,5)).to("cuda:7").long()
        # x=x.view(1,-1)

        trans=Transformer(50,400,100,128,4,8,0,512,0,None,2).to("cuda:7").to(torch.bfloat16)
        out=trans(x)
        print(out,out.shape)