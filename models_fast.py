import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
from torch.nn.attention import SDPBackend, sdpa_kernel


prof_schedule = schedule(wait=1, warmup=2, active=3, repeat=1)

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
#RELU
def relu(x):
    return torch.clamp(x,0.0)
    
#SILU
def my_silu(x):
    return (x*(torch.sigmoid(x.float()))).to(x.dtype)

def silu(x):
    return F.silu(x)

#LAYER NORM
class layer_norm(nn.Module):
    def __init__(self,num_dim):
        super().__init__()
        shift=torch.zeros(num_dim)
        scale=torch.ones(num_dim)

        self.shift=nn.Parameter(shift)
        self.scale=nn.Parameter(scale)

        eps=torch.tensor(2e-08)
        self.register_buffer("eps",eps)

    def forward(self,x):
        xdtype=x.dtype
        x=x.float()
        x_mean,x_std=torch.mean(x,dim=-1,keepdim=True),torch.std(x,dim=-1,correction=0,keepdim=True)
        x_norm=(x-x_mean)/(x_std+self.eps)
        
        x_norm=x_norm*self.scale + self.shift
        x_norm=x_norm.to(dtype=xdtype)
        return x_norm

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


#ROPE
def precompute_rope(max_context,max_freq,head_dim): #LIST COMPREHENSION ROPE
    k=max_freq

    rope_sin=[[torch.sin(pos/torch.pow(k,torch.tensor((i-i%2))/head_dim)) for i in range(head_dim)] for pos in range(max_context)]
    rope_cos=[[torch.cos(pos/torch.pow(k,torch.tensor((i-i%2))/head_dim)) for i in range(head_dim)] for pos in range(max_context)]
    return torch.tensor(rope_cos),torch.tensor(rope_sin)


#ROPE FAST VECTORIZED
def precompute_rope_fast(max_context,max_freq,head_dim): # USES OUTER PRODUCT TO BUILD THETA VALUES AND APPLY SIN COS AT ONCE
    pos_tensor=torch.arange(max_context)
    i=torch.arange(0,head_dim//2)
    
    theta=1/torch.pow(max_freq,(2*i.float())/head_dim)
    pos_theta=torch.outer(pos_tensor,theta)
    return torch.cos(pos_theta).repeat(1,2),torch.sin(pos_theta).repeat(1,2)

#X PERMUTE INDICES(SWAP ALT) AND BIT
# def permute_indcies_rope(head_dim): #CREATE PERM INDICES USING LOGIC i=i+1  if ODD i=i-1 if EVEN IT PERMUTES ALTERNATIVE INDEX
#     i=torch.arange(head_dim)
#     alt_bit=torch.tensor([1,-1]).repeat(head_dim//2)
#     return i+alt_bit,-alt_bit

def apply_rope(x,cos_embed,sin_embed): #EITHER COMPUTE ROPE ONCE FOR MAX CONTEXT OR COMPUTE FOR EACH CONTEXT LENGTH PER BATCH CHOICE 
    T=x.shape[-2]
    cos,sin=cos_embed[:T,:].to(x.dtype),sin_embed[:T,:].to(x.dtype)
    return x*cos + rotate_half(x)*sin

def rotate_half(x):
    x1,x2=x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)

def sinusoidal_embeddings(max_context,max_freq,d_model): #SINCE SIN COS ALTERNATE IT BUILDS USING FOR LOOP ONE TIME COST SO DOESNT MATTER
    k=max_freq
    pos=torch.zeros((max_context,d_model))
    for i in range(max_context):
        for j in range(d_model):
            num=i
            exp=torch.tensor((j-j%2)/d_model)
            denom=torch.pow(k,exp)
            pos_embed=num/denom
            
            if j%2==0:
                pos[i][j]=torch.sin(pos_embed)
            else:
                pos[i][j]=torch.cos(pos_embed)                
    return pos



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

        # att=q@k.transpose(2,3)/(self.head_dim**0.5)

        # if mask is not None:
        #     att=att.masked_fill(mask,-torch.inf)

        # att=F.softmax(att.float(),dim=-1).to(x.dtype)
        # att = self.attn_drop(att)

        # x=(att@v).permute(0,2,1,3).contiguous()
        # x=x.view(B,T,D)
        # x=self.linear_proj(x)

        # x+=residual
        # return x

#TRANSFORMER BLOCK WITH FFN AND SELF ATTENTION BLOCK
class transformer_block(nn.Module):
    def __init__(self,d_model,n_heads,num_layers,attn_dropout,ffn_hidden_dim,ffn_dropout):
        super().__init__()

        self.att=mhma(d_model,n_heads,num_layers,attn_dropout)
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
        q=q.view(B,T,self.kvn_heads,self.groups,self.head_dim)
        q=q.permute(0,2,3,1,4)

        kv=self.kv_proj(x)
        kv=kv.view(B,T,2,self.kvn_heads,self.head_dim).unsqueeze(-2)
        k,v=kv.permute(0,2,3,4,1,5).unbind(dim=1)

        q,k=apply_rope(q,cos,sin),apply_rope(k,cos,sin)
        att=q@k.transpose(-2,-1)/(self.head_dim**0.5)

        if mask is not None:
            att=att.masked_fill(mask,-torch.inf)

        att=F.softmax(att.float(),dim=-1).to(x.dtype)
        att = self.attn_drop(att)

        x=(att@v).permute(0,3,1,2,4).contiguous()
        x=x.view(B,T,D)
        x=self.linear_proj(x)

        x+=residual
        return x



def cross_entropy_loss(x,y): #MULTICLASS CROSS ENTROPY WITH LOGITS COMING AND Y IS LABEL BOTH TENSOR SHAPE (B,T,D) and B,T
    B,T,C=x.shape
    x,y=x.view(B*T,C),y.view(B*T)
    x=x-torch.max(x,dim=-1,keepdims=True).values
    
    log_num=-x[torch.arange(B*T),y]
    log_denom=torch.log(torch.sum(torch.exp(x),dim=-1))

    loss=torch.mean(log_num+log_denom)    
    return loss

def get_perplexity(x,y):
    loss=cross_entropy_loss(x,y)
    return 2**loss

#EMBEDDING WITH Sqrt(1/d_Model) SINCE WE TIE UNEMBEDDING AND EMBEDDING
class embedding(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        std=1/(d_model**0.5)
        weight=torch.randn(vocab_size,d_model)*std
        self.weight=nn.Parameter(weight)

    def forward(self,x):
        return self.weight[x]
    

#TRANSFORMER WITH ALL THE BLOCKS BUILT EARLIER
class Transformer(nn.Module):
    def __init__(self,vocab_size,max_context,max_freq,d_model,n_heads,num_layers,attn_dropout,ffn_hidden_dim,ffn_dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_block_list=nn.ModuleList([transformer_block(d_model,n_heads,num_layers,attn_dropout,ffn_hidden_dim,ffn_dropout) for idx in range(num_layers)])
        
        self.rms_norm=rms_norm(d_model)

        head_dim=d_model//n_heads
        # perm,alt_bit=permute_indcies_rope(head_dim)
        cos,sin=precompute_rope_fast(max_context,max_freq,head_dim)
        mask=torch.triu(torch.ones(max_context,max_context),diagonal=1).bool()

        self.register_buffer("cos",cos)
        self.register_buffer("sin",sin)
        self.register_buffer("mask",mask)


    def forward(self,x,step_count):
        x=self.embedding(x)
        B,T,C=x.shape

        for i,trans_block in enumerate(self.transformer_block_list):
            x=trans_block(x,self.mask[:T,:T],self.cos[:T],self.sin[:T])

        x=self.rms_norm(x)
        return x



# if __name__=="__main__":

#     x=torch.randn(5,3,2)
#     y=torch.ones(5,3).long()

#     loss=cross_entropy_loss(x,y.long())


#     rope_cos,rope_sin=precompute_rope_fast(5,10000,4)
#     perm,alt_bit=permute_indcies_rope(4)

#     mask=torch.ones(5,5)
#     mask=torch.triu(mask,1).bool()

#     x=torch.randn(5,5,16)
#     att=gqa(16,4,4,0.1,2)
#     x=att(x,mask,rope_cos,rope_sin)
#     print(x.shape)

#     # x=torch.arange(1024)
#     # x=torch.stack((x,x,x,x,x,x),dim=0).to("cuda:3")
#     # x=x.to(torch.bfloat16)

#     trans=Transformer(50000,4096,10000,1536,12,32,0.1,5120,0.1)
#     trans=trans.to(torch.bfloat16).to("cuda:3")

#     print("IOKK")
#     count=0
#     for p in trans.parameters():
#         count+=p.numel()
#     print(count)
