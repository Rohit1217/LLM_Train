from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch

class chunks_dataset(Dataset):
    def __init__(self,chunk_path,seq_len,batch_size,eot_id,rank,data_len):
        self.chunk_path=chunk_path
        self.seq_len=seq_len
        self.batch_size=batch_size
        self.data_len=data_len
        self.eot_id=eot_id
        self.rank=rank
        self.arr=None
        self.mtp_k=None

    def warmup(self):
        if not self.arr:
            self.arr=np.memmap(self.chunk_path,dtype=np.uint16,mode="r") # ALL WORKERS FETCH FROM A COMMON PHYSICAL COPY WHICH FOR OUR CASE IS LOADED INTO RAM. SAVES SPACE AND ALLOWS US TO UTILIZE NUMWORKERS
            shard = self.arr[self.rank*self.data_len*self.seq_len : (self.rank+1)*self.data_len*self.seq_len]
            shard.sum() 
            print(f"WARMED UP CACHE FOR RAM,RANK: {self.rank}")  #WARMUP CACHE FOR FAST I/O (HAD ACCESS TO RAM SO I CAN HAVE WHOLE DATA AS WARM)
        return 

    def __len__(self):
        return self.data_len
    
    def __getitem__(self,idx):
        if not self.arr:
            self.warmup()

        s=self.rank*self.data_len*self.seq_len*self.batch_size + idx*(self.seq_len*self.batch_size)
        return np.asarray(self.arr[s:s+self.seq_len*self.batch_size])


    #SOME PREPROCESSING OFLOADED HERE AS DATALOADER PREPARE BATCHES WHILE GPU IS DOING COMPUTATION SO THERE IS OVERLAP OF BOTH SAVING US TIME.
    def data_collate(self,x):
        eot_indicies=np.where(x[:-1]==self.eot_id)[0]
        cuseq=np.zeros(eot_indicies.shape[0]+2)
        cuseq[-1]=x.shape[0]
        idx=np.arange(eot_indicies.shape[0])

        cuseq[idx+1]=eot_indicies[idx]
        cumsum_rope=np.concatenate([np.arange(cuseq[i+1]-cuseq[i]) for i in range(len(cuseq)-1)]).astype(np.int16)

        y_main=torch.from_numpy(x[1:x.shape[0]-self.mtp_k])

        if self.mtp_k>0:
            y_mtp=torch.from_numpy(x[2:])
            y_mtp=y_mtp.unfold(0,self.mtp_k,1).permute(1,0).contiguous() 
        else:
            y_mtp=None

        cumsum_rope,cuseq
        
        return {"rope_pos":torch.from_numpy(cumsum_rope).contiguous(),
                "cuseq":torch.from_numpy(cuseq.astype(np.int32)).contiguous(),
                "tokens":torch.from_numpy(x).contiguous(),
                "y_main":y_main.contiguous(),
                "y_mtp":y_mtp,
                "seqlen":self.seq_len}




def make_dataloader(chunk_path,seq_len,batch_size,eot_id,rank,data_len):
    ds=chunks_dataset(chunk_path,seq_len,batch_size,eot_id,rank,data_len)
    trainloader=DataLoader(ds,1,num_workers=4,pin_memory=True,collate_fn=ds.data_collate,prefetch_factor=2,persistent_workers=True)
    return trainloader
        
