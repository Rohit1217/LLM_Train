import os
import urllib.request
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE = "tiny_shakespeare.txt"


def load_text():
    if not os.path.exists(CACHE):
        urllib.request.urlretrieve(URL, CACHE)
    with open(CACHE, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return chars, stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batches(data, seq_len):
    # chop into fixed-length sequences; targets are derived in train_step via shift
    n_seq = len(data) // seq_len
    usable = n_seq * seq_len

    tokens = data[:usable].view(n_seq, seq_len)     # [n_seq, seq_len]
    shard_ids = torch.arange(n_seq)                 # per-sequence id (traces spikes)
    return tokens, shard_ids


def generate_shakespeare_dataset(seq_len, batch_size=None):
    # batch_size unused here; DataLoader handles batching
    text = load_text()
    _, stoi, _ = build_vocab(text)
    data = encode(text, stoi)
    return get_batches(data, seq_len)               # (tokens, shard_ids)


if __name__ == "__main__":
    seq_len = 128

    text = load_text()
    chars, stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    tokens, shard_ids = get_batches(data, seq_len)

    print(f"chars: {len(text)}  vocab_size: {len(chars)}")
    print(f"tokens: {tuple(tokens.shape)}  shard_ids: {tuple(shard_ids.shape)}  (n_seq, seq_len)")
    print("seq 0 decode:", repr("".join(itos[c.item()] for c in tokens[0, :40])))
    print("shard_ids[:8]:", shard_ids[:8].tolist())
 