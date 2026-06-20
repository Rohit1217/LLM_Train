import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch

from config import Config
from main_models import Transformer
from Data.process_tiny_shakespeare import load_text, build_vocab
from torch.nn.attention import SDPBackend, sdpa_kernel

cfg = Config()
device = "cuda:3"

SMALL_NUM_LAYERS=8
SMALL_MTP_HEADS=2
SMALL_MAX_CONTEXT=600
SMALL_N_HEAD=4
SMALL_N_GROUPS=2
SMALL_D_MODEL=128
SMALL_HEAD_DIM=SMALL_D_MODEL//SMALL_N_HEAD
SMALL_KVN_HEADS=SMALL_N_HEAD//SMALL_N_GROUPS

LARGE_NUM_LAYERS=cfg.NUM_LAYERS
LARGE_MTP_HEADS=cfg.MTP_HEADS   
LARGE_MAX_CONTEXT=cfg.MAX_CONTEXT
LARGE_N_HEAD=cfg.N_HEAD
LARGE_N_GROUPS=cfg.NUM_GROUPS
LARGE_D_MODEL=cfg.D_MODEL
LARGE_HEAD_DIM=LARGE_D_MODEL//LARGE_N_HEAD
LARGE_KVN_HEADS=LARGE_N_HEAD//LARGE_N_GROUPS


text = load_text()
chars, stoi, itos = build_vocab(text)


def encode_prompt(s):
    return torch.tensor([[stoi[c] for c in s]], dtype=torch.long, device=device)

def decode(ids):
    # pull off-GPU once (iterating a cuda tensor syncs per element); guard ids outside char vocab
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return "".join(itos.get(int(i), "") for i in ids)

def load_model(weights="Weights/model_weights.pth"):

    model=Transformer(vocab_size=cfg.VOCAB_SIZE,max_context=cfg.MAX_CONTEXT,
                                max_freq=cfg.MAX_FREQ,d_model=cfg.D_MODEL,n_heads=cfg.N_HEAD,
                                num_layers=cfg.NUM_LAYERS,attn_dropout=cfg.ATT_DROPOUT,
                                ffn_hidden_dim=cfg.FFN_HIDDEN_DIM,ffn_dropout=cfg.FFN_DROPOUT,mtp_heads=cfg.MTP_HEADS,
                                grad_checkpoint_every=cfg.GRAD_CHECKPOINT_EVERY,gqa_groups=cfg.NUM_GROUPS)
    # drop MTP-head weights; remaining keys match the trunk 1:1 (strict load catches anything else)
    # state = {k: v for k, v in state.items() if not k.startswith("mtp_heads_list.")}
    state = torch.load(weights, map_location=device)

    model.load_state_dict(state)
    return model.to(device).to(torch.bfloat16).eval()

def sample(model,prompt="ROMEO",temperature=1.0,n_new=200,type="LARGE"):
    with torch.no_grad():
        x = encode_prompt(prompt)
        if type=="SMALL":
            k=torch.zeros(SMALL_NUM_LAYERS+SMALL_MTP_HEADS,1,SMALL_KVN_HEADS,SMALL_MAX_CONTEXT,SMALL_HEAD_DIM).to(x.device).to(torch.bfloat16)
            v=torch.zeros(SMALL_NUM_LAYERS+SMALL_MTP_HEADS,1,SMALL_KVN_HEADS,SMALL_MAX_CONTEXT,SMALL_HEAD_DIM).to(x.device).to(torch.bfloat16)
        else:
            k=torch.zeros(LARGE_NUM_LAYERS+LARGE_MTP_HEADS,1,LARGE_KVN_HEADS,LARGE_MAX_CONTEXT,LARGE_HEAD_DIM).to(x.device).to(torch.bfloat16)
            v=torch.zeros(LARGE_NUM_LAYERS+LARGE_MTP_HEADS,1,LARGE_KVN_HEADS,LARGE_MAX_CONTEXT,LARGE_HEAD_DIM).to(x.device).to(torch.bfloat16)


        out,k_pref,v_pref = model.prefill_inference(x,temperature=temperature)

        print("PREFILL DONE")
        next_pos=k_pref.shape[3]


        k[:,:,:,:next_pos,:]=k_pref
        v[:,:,:,:next_pos,:]=v_pref

        while next_pos<(n_new+k_pref.shape[3]):
            new_out=model.decode_inference(out[-1],k,v,next_pos,temperature)
            out=torch.cat([out,new_out])
            next_pos+=new_out.shape[0]
            # print(new_out.shape[0])
        
        # print(out.shape,out)
        return decode(out.squeeze())    


if __name__ == "__main__":
    # model = load_model()
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.MATH,SDPBackend.EFFICIENT_ATTENTION]):
        print("LOADING")
        # model=Transformer(288,600,100,128,4,8,0,512,0,2,2).to(device=device).to(torch.bfloat16)
        model=load_model()
        print("MODEL LOADED")

        prompts = ["ROMEO:", "To be, or not", "\n"]
        for p in prompts:
            print("=" * 60)
            print(f"prompt: {p!r}")
            print(sample(model, prompt=p, n_new=200,temperature=1,type="LARGE"))

        print(sample(model,prompt="Before we proceed any further, hear me speak.",n_new=200,temperature=0.01,type="LARGE"))    