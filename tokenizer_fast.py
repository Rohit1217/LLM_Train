import pandas as pd
from collections import defaultdict

df=pd.read_parquet("../data/fineweb-edu-10BT/sample/10BT/000_00000.parquet")
texts=df['text'].tolist()

class BPE():
    def __init__(self,vocab_size):
        self.vocab=[str(idx) for idx in range(256)]
        self.vocab_size=vocab_size
        self.token_idx=defaultdict(str)

    def get_max_pair(self,sample):
        pair_count=defaultdict(int)
        max_count=0
        max_pair=None
        
        for doc in sample:
            for idx in range(0,len(doc)-1):
                pair=doc[idx] + "|" +doc[idx+1]
                pair_count[pair]+=1

                if pair_count[pair]>max_count:
                    max_pair=pair
                    max_count=pair_count[pair]
        return max_pair
    
    def update_token_idx(self):
        for idx in range(len(self.vocab)):
            self.token_idx[self.vocab[idx]]=idx

    def update_encoding(self,sample,max_pair):
        res_sample=[]
        
        for doc in sample:
            curr_res=[]
            idx=0
            
            while idx<len(doc)-1:
                pair=doc[idx] + "|" +doc[idx+1]
                if max_pair==pair:
                    curr_res.append(pair) 
                    idx+=2
                else:
                    curr_res.append(doc[idx])
                    idx+=1
                    
            if idx==len(doc)-1:
                curr_res.append(doc[-1])

            res_sample.append(curr_res)
        return res_sample

    def train(self,sample):
        sample=[[str(byte) for byte in doc] for doc in sample]
        
        while len(self.vocab)<self.vocab_size:
            max_pair=self.get_max_pair(sample)
            sample=self.update_encoding(sample,max_pair)
            self.vocab.append(max_pair)
        
        self.update_token_idx()

    def decode_ids(self,idx):
        word=(self.vocab[idx]).split("|")   
        word=[int(byte) for byte in word]
        word=bytes(word).decode("utf-8",errors="replace")
        return word
    
    def decode(self,tokens):
        word=[self.vocab[token] for token in tokens]
        res_word=[]
        for char in word:
            res_word+=char.split("|")

        word=[int(byte) for byte in res_word]
        word=bytes(word).decode("utf-8")
        return word
        
    
    def encode(self,s):
        s_byte=list(s.encode("utf-8"))
        s_byte=[str(byte) for byte in s_byte]
                
        while len(s_byte)>1:
            merged=[str(s_byte[idx])+"|"+s_byte[idx+1] for idx in range(len(s_byte)-1)]
            max_pair=None

            # Going in descending order of count as vocab is populated in that way .merge max pairs until none left to merge or we get one word.
            
            for pair in self.vocab: 
                if pair in merged:
                    max_pair=pair
                    break
            
            if max_pair is None:
                return [self.token_idx[byte] for byte in s_byte]
            
            curr_res=[]
            idx=0
            
            while idx<len(s_byte)-1:
                pair=s_byte[idx] + "|" +s_byte[idx+1]
                if max_pair==pair:
                    curr_res.append(pair) 
                    idx+=2
                else:
                    curr_res.append(s_byte[idx])
                    idx+=1
            
            if idx==len(s_byte)-1:
                curr_res.append(s_byte[-1])
        
            s_byte=curr_res
            
        tokens=[self.token_idx[byte] for byte in s_byte]
        return tokens
                       


if __name__=="__main__":
    
    sample = [list(text.encode('utf-8')) for text in texts[:1000]]
    sample_s=sample[:100]

    bpe=BPE(1000)
    bpe.train(sample_s)

    total_len=sum([len(char.split("|")) for char in bpe.vocab])
    print(f"Bytes_per_token{total_len/1000}")

    tokens=bpe.encode("héllo 日本語 🙂")
    print(f"token ids {tokens}")
    decoded_text=bpe.decode(tokens)
    print(f"decoded text {decoded_text}")
              