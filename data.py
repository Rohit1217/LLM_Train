import torch
from torch.utils.data import Dataset, DataLoader


class sharded_dataset(Dataset):
    def __init__(self, tokens, shard_ids):
        self.tokens=tokens          # [n_seq, seq_len]
        self.shard_ids=shard_ids    # [n_seq]

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return self.shard_ids[idx], self.tokens[idx]


def load_data(dataset, batch_size, shuffle=False):
    tokens,shard_ids=dataset
    ds=sharded_dataset(tokens, shard_ids)
    dataloader=DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                            pin_memory=True, drop_last=True)
    return dataloader     