import torch

from config import Config
from main_models import Transformer
from Data.process_tiny_shakespeare import load_text, build_vocab

cfg = Config()
device = cfg.DEVICE

# ---- vocab (same deterministic char vocab used in training) ----
text = load_text()
chars, stoi, itos = build_vocab(text)


def encode_prompt(s):
    return torch.tensor([[stoi[c] for c in s]], dtype=torch.long, device=device)


def decode(ids):
    # pull off-GPU once (iterating a cuda tensor syncs per element); guard ids outside char vocab
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return "".join(itos.get(int(i), "") for i in ids)


def load_model(weights="model_weights.pth"):
    # MTP heads are train-only: the main next-token head is just trunk + rms_out and is NOT
    # touched by the MTP loop, so we build the TRUNK ALONE -> identical main-head output, faster
    # generation (no MTP block per token), and generate() pads by 1 (num_mtp_heads=0).
    model = Transformer(vocab_size=cfg.VOCAB_SIZE, max_context=cfg.MAX_CONTEXT,
                        max_freq=cfg.MAX_FREQ, d_model=cfg.D_MODEL, n_heads=cfg.N_HEAD,
                        num_layers=cfg.NUM_LAYERS, attn_dropout=cfg.ATT_DROPOUT,
                        ffn_hidden_dim=cfg.FFN_HIDDEN_DIM, ffn_dropout=cfg.FFN_DROPOUT)
    state = torch.load(weights, map_location=device)
    # drop MTP-head weights; remaining keys match the trunk 1:1 (strict load catches anything else)
    state = {k: v for k, v in state.items() if not k.startswith("mtp_heads_list.")}
    model.load_state_dict(state)
    return model.to(device).to(torch.bfloat16).eval()


def sample(model, prompt="ROMEO:", n_new=200, temperature=1.0):
    x = encode_prompt(prompt)
    out = model.generate(x, n_new, temperature=temperature)
    return decode(out[0])


if __name__ == "__main__":
    model = load_model()

    prompts = ["ROMEO:", "To be, or not", "\n"]
    for p in prompts:
        print("=" * 60)
        print(f"prompt: {p!r}")
        print(sample(model, prompt=p, n_new=200, temperature=0.5))

    print(sample(model,"Before we proceed any further, hear me speak.",200,temperature=0.01))    