import pandas as pd
import os
from collections import defaultdict,Counter
import regex as re
import heapq
from tqdm import tqdm
import tiktoken


df=pd.read_parquet("../data/fineweb-edu-10BT/sample/10BT/000_00000.parquet")
texts=df['text'].tolist()


GPT2_SPLIT_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
GPT4_SPLIT_PATTERN = re.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")


def get_word_counts(texts):
    GPT4_SPLIT_PATTERN = re.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
    word_counts = Counter()
    
    for text in tqdm(texts):
        word_counts.update(re.findall(GPT4_SPLIT_PATTERN, text))
    
    return word_counts


class WordNode():
    def __init__(self,token=None,next=None,prev=None,count=None):
        self.token=token
        self.next=next
        self.prev=prev
        self.count=count    
        
class WordList():
    def __init__(self,text=None,count=None,index_map=None):
        text=text.encode('utf-8')
        byte=list(bytes(text))
        self.head=None
        self.count=count
        self.head=WordNode((byte[0]),count=self.count)
        curr=self.head
        
        if len(byte)>1:
            i=0
            index_map[(byte[i],byte[i+1])]["pos"].append(self.head)
            index_map[(byte[i],byte[i+1])]["count"]+=count

            for i in range(1,len(byte)-1):
                node=WordNode((byte[i]),count=self.count)
                curr.next=node
                node.prev=curr
                curr=node
                
                index_map[(byte[i],byte[i+1])]["pos"].append(node)
                index_map[(byte[i],byte[i+1])]["count"]+=count
            
            node=WordNode((byte[len(byte)-1]),count=self.count)
            curr.next=node
            node.prev=curr
            curr=node
                            

class Index_Heap():
    def __init__(self,word_counts):
        self.heap=[]
        self.index_map=defaultdict(lambda: {"pos": [], "count": 0,"time":0})        
        self.wordchain=[]    
        self.counter=0
        self.time_hash={}
        self.modified=set()
        

        for text in word_counts:
            self.wordchain.append(WordList(text,word_counts[text],self.index_map))

        for token_pair in self.index_map:
            count=-self.index_map[token_pair]['count']
            self.heap.append((count,0,token_pair))

        heapq.heapify(self.heap)

    
    def heap_add(self,token_pair,count):
        self.counter+=1
        heapq.heappush(self.heap,(-count,-self.counter,token_pair))
        self.time_hash[token_pair]=-self.counter
        

    def merge(self,node,token_pair):
            token1,token2=token_pair
            if node.token==token1 and node.next.token==token2:
                node.token=(token1,token2)
                right=node.next

                if node.prev:
                    self.index_map[(node.prev.token,token1)]["count"]-=node.prev.count                    
                    self.index_map[(node.prev.token,node.token)]["pos"].append(node.prev)
                    self.index_map[(node.prev.token,node.token)]["count"]+=node.prev.count
                    self.modified.add((node.prev.token,node.token))
                    self.modified.add((node.prev.token,token1))
                
                if node.next.next:
                    self.index_map[(token2,node.next.next.token)]["count"]-=node.next.next.count                    
                    node.next.next.prev=node

                    self.index_map[(node.token,node.next.next.token)]["pos"].append(node)
                    self.index_map[(node.token,node.next.next.token)]["count"]+=node.next.next.count
                    self.modified.add((node.token,node.next.next.token))
                    self.modified.add((token2,node.next.next.token))
                
                node.next=node.next.next
                right.next=None
                right.prev=None
                return
            

    def get_max(self):
        while self.heap:
            
            count,time,token_pair=heapq.heappop(self.heap)          
            if count<0 and time==self.time_hash.get(token_pair,0):
                    return token_pair
        return None
    
    
    def update_heap(self):
        for token_pair in self.modified:
            self.heap_add(token_pair,self.index_map[token_pair]["count"])
    
    
    def update_index(self,token_pair):
        self.modified=set()

        for node in self.index_map[token_pair]["pos"]:
            if node.next and node.token==token_pair[0] and node.next.token==token_pair[1]:
                self.merge(node,token_pair)
        
        self.modified.discard(token_pair)
        self.update_heap()
        del self.index_map[token_pair]
        return


def arr_from_tuples_rec(tuple):
    if type(tuple)==int:
        return [tuple]
    else:
        tuple1,tuple2=tuple
        return arr_from_tuples_rec(tuple1) + arr_from_tuples_rec(tuple2)

arr=arr_from_tuples_rec( (((104, 101), 108), ((108, 111), 32)))
bytes(arr),bytes([125])


def build_vocab(texts,vocab_size=48000):
    word_counts=get_word_counts(texts)
    index_heap=Index_Heap(word_counts)    
    vocab={}

    for i in range(256):
        vocab[bytes([i])]=i	
    
    for j in tqdm(range(i,vocab_size)):
        maxm=index_heap.get_max()

        if maxm:
            maxm_bytes=bytes(arr_from_tuples_rec(maxm))
            vocab[maxm_bytes]=j
        else:
            break

        index_heap.update_index(maxm)
    return vocab        
        

class Tokenizer():
    def __init__(self,vocab_size):
        self.vocab_size=vocab_size
        self.vocab=None
        self.encoding=None
        self.pat_str=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def train(self,texts):
        self.vocab=build_vocab(texts,self.vocab_size)
    
    def create_tiktoken_encdoing(self):
        if self.vocab:
            self.encoding = tiktoken.Encoding(
            name="custom_encoding",
            pat_str=self.pat_str,
            mergeable_ranks=self.vocab,
            special_tokens={
                "<|endoftext|>": len(self.vocab)
            } 
        )
    
    def encode_text(self,text):
        if self.encoding:
            tokens = self.encoding.encode(text)
            return tokens
        else:
            print("TRAIN THE TOKENIZER FIRST ")
    
    def decode_text(self,text):
        if self.encoding:
            text = self.encoding.decode(tokens)
            return text
        else:
            print("TRAIN THE TOKENIZER FIRST ")        


tokenizer=Tokenizer(48000)
tokenizer.train(texts[:10000])
text = "BIRTH OF THE TOKENIZATION GOD"
tokens = tokenizer.encode(text)
print("Token IDs:", tokens)

decoded_text = tokenizer.decode(tokens)
print("Decoded Text:", decoded_text)
