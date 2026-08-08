from torch.utils.data import Dataset,DataLoader,Sampler
import numpy as np
import torch
from tqdm import tqdm

class chunks_dataset(Dataset):
    #Local sample s -> local batch b=s//B (== the microstep this rank is on), position k=s%B.
    # data access is of fromat [b1l1,b1l2..b1ln,b2l1] (B,World_size,seq_len)
    #Global micro-batch ordinal = b*world + rank, so "equences consumed after opt-step t = perm[0 : t*GLOBAL_BATCH]
    def __init__(self,seq_len,batch_size,eot_id,rank,world,perm,total_micro,mtp_k=None,arr=None):
        self.seq_len=seq_len
        self.batch_size=batch_size
        self.eot_id=eot_id
        self.rank=rank                 #LOCAL rank 0..world-1
        self.world=world
        self.perm=perm                 #index permutations
        self.total_seqs=len(perm)
        self.total_micro=total_micro   #num forward pass per rank
        self.mtp_k=mtp_k
        self.arr=arr

    def load_arr(self,arr):
        if self.arr is None:
            self.arr=arr

    def __len__(self):
        return self.total_micro*self.batch_size

    def __getitem__(self,s):
        b=s//self.batch_size # get batch and the position of s in curr batch
        k=s%self.batch_size 
        global_mb=b*self.world+self.rank                       
        pos=(global_mb*self.batch_size+k)%self.total_seqs      
        seq=int(self.perm[pos]) #get shuffled indices.
        s0=seq*self.seq_len
        return self.arr[s0:s0+self.seq_len]


    #SOME PREPROCESSING OFLOADED HERE AS DATALOADER PREPARE BATCHES WHILE GPU IS DOING COMPUTATION SO THERE IS COMPUTATION OVERLAP SAVING US TIME.
    def data_collate(self,x):
        x=np.array(x)
        B,seq_len=x.shape #seq_len=(ctx_len+1+num_mtp)

        #MODEL DROPS num_mtp_heads+1 COLS PER ROW (TEACHER FORCING SHIFT + MTP), SO MAINs ONLY SEES FIRST L COLS.
        #rope_pos/cuseq MUST DESCRIBE THOSE L COLS ELSE THEY MISMATCH q AT B*L

        drop=1 if self.mtp_k is None else self.mtp_k+1
        L=seq_len-drop
        total=B * L
        flat=x[:,:L].reshape(-1) #intradoc attention requires flat arr

        #flatten x for pos and cuseq calc. Find where doc ends
        eot_indices = np.where(flat==self.eot_id)[0]

        #batch wise end of doc
        row_boundaries=np.arange(L,total,L,dtype=np.int32)

        #append all ending position get unique and sort
        cuseq=np.unique(np.concatenate(([0], eot_indices + 1, row_boundaries,[total])))

        cuseq.sort()

        #lenghts cuseq[1:]-cuseq[:-1] and then offset using np. repeat it repeats cuseq length times
        #pos is given by arange - offset, For each doc reduce the previous doc end id to get its current position in current doc
        lengths=np.diff(cuseq).astype(np.uint32)
        offsets=np.repeat(cuseq[:-1], lengths).astype(np.int32)
        rope_pos=(np.arange(total, dtype=np.int32)-offsets).astype(np.int16)

        #shift by 1 for teacher forcing make contiguous flatten for liger ce kernel
        if self.mtp_k is None:
            y_main=torch.from_numpy(x[:,1:seq_len-0]).contiguous().view(-1)
        else:
            y_main=torch.from_numpy(x[:,1:seq_len-self.mtp_k]).contiguous().view(-1)

        
        #Then we do unfold to create mtp_heads,B*T seqences and finally make them coontiguous for liger ce kernel
        if self.mtp_k is not None:
            y_mtp=torch.from_numpy(x[:,2:])
            y_mtp=y_mtp.unfold(1,self.mtp_k,1).permute(2,0,1).contiguous().view(-1) 
        else:
            y_mtp=None
            

        return {"rope_pos":torch.from_numpy(rope_pos).contiguous(),
                "cuseq":torch.from_numpy(cuseq.astype(np.int32)).contiguous(),
                "tokens":torch.from_numpy(x).contiguous(),
                "y_main":y_main,
                "y_mtp":y_mtp,}


class ResumeSampler(Sampler):
    #FIRST pass (post-resume) starts at start_idx; every later epoch starts at 0. Without the flag
    #each re-iteration would permanently skip the head start_idx sequences. Doesnt matter much for us since we only do one epoch training.
    def __init__(self,data_len,start_idx=0):
        self.data_len=data_len
        self.start_idx=start_idx
        self.first=True
    def __len__(self):
        return self.data_len

    def __iter__(self):
        start=self.start_idx if self.first else 0
        self.first=False
        yield from range(start,self.data_len)

def make_dataloader(seq_len,batch_size,eot_id,rank,world,perm,total_micro,start_idx=0,arr=None,mtp_k=None,num_workers=4,
                      pin_memory=True,prefetch_factor=2,persistent_workers=True):
    ds=chunks_dataset(seq_len,batch_size,eot_id,rank,world,perm,total_micro,mtp_k=mtp_k,arr=arr)

    if arr is not None:
        ds.load_arr(arr)

    sampler=ResumeSampler(len(ds),start_idx)   #continuous from start_idx (=resume microstep*B); run never wraps its one pass
    #drop_last: keep every batch size B no torch.compile recompile on a short final batch
    trainloader=DataLoader(ds,batch_size=batch_size,num_workers=num_workers,sampler=sampler,drop_last=True,pin_memory=pin_memory,collate_fn=ds.data_collate,prefetch_factor=prefetch_factor,persistent_workers=persistent_workers)
    return trainloader
        

if __name__=="__main__":
    #tiny smoke test of the global-prefix loader
    EFF,B,WORLD=1027,4,2
    total_seqs=200
    data=np.arange(total_seqs*EFF,dtype=np.uint16)
    perm=np.random.default_rng(0).permutation(total_seqs).astype(np.int32)
    dl=make_dataloader(EFF,B,48000,rank=0,world=WORLD,perm=perm,total_micro=50,arr=data,mtp_k=2,num_workers=0,persistent_workers=False,prefetch_factor=None)
    for x in tqdm(dl):
        print(x["rope_pos"].shape,x["tokens"].shape); break
