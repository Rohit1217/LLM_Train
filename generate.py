import torch

from config import Config
from models_fast import Transformer
from process_tiny_shakespeare import load_text, build_vocab

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
    model = Transformer(vocab_size=cfg.VOCAB_SIZE, max_context=cfg.MAX_CONTEXT,
                        max_freq=cfg.MAX_FREQ, d_model=cfg.D_MODEL, n_heads=cfg.N_HEAD,
                        num_layers=cfg.NUM_LAYERS+2, attn_dropout=cfg.ATT_DROPOUT,
                        ffn_hidden_dim=cfg.FFN_HIDDEN_DIM, ffn_dropout=cfg.FFN_DROPOUT)
    state = torch.load(weights, map_location=device)
    # old checkpoints named the final norm "rms_norm"; it's "rms_out" now
    state = {("rms_out." + k[len("rms_norm."):] if k.startswith("rms_norm.") else k): v
             for k, v in state.items()}
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