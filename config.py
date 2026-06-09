from dataclasses import dataclass


@dataclass
class Config:
    BATCH_SIZE:int = 20
    SEQ_LEN:int = 1024

    D_MODEL:int = 1536
    VOCAB_SIZE:int = 48000
    MAX_CONTEXT:int = 8192
    MAX_FREQ:int = 10000
    N_HEAD:int = 12
    NUM_LAYERS:int = 30
    FFN_HIDDEN_DIM:int = 4096

    ATT_DROPOUT:float = 0
    FFN_DROPOUT:float = 0

    NUM_GROUPS:int=4
    MTP_HEADS:int=2
    MTP_LOSS_WEIGHT:float=0.3

    WEIGHT_DECAY:float=0.1
    LR:float=2e-4

    DEVICE:str = "cuda:6"
    SEED:int = 133721

    TOTAL_TOKENS:int = 10**8
    A6000_BF16_PEAK :float= 154e12
    THROUGHPUT_TARGET:float = 15805   # tok/s baseline
    PEAK_MFU_TARGET:float = 0.56


    def __post_init__(self):
        self.EFF_SEQ_LEN:int=self.SEQ_LEN+self.MTP_HEADS+1
        self.TOKENS_PER_STEP = self.BATCH_SIZE*self.SEQ_LEN
        self.TOTAL_STEPS = self.TOTAL_TOKENS//self.TOKENS_PER_STEP

        self.WARMUP_STEPS = int(0.05*self.TOTAL_STEPS)
        self.STABLE_STEPS = int(0.85*self.TOTAL_STEPS)
        self.DECAY_STEPS = int(0.10*self.TOTAL_STEPS)