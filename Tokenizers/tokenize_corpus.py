import os,io
from tokenizer_fast import Tokenizer
import json
import zstandard as zstd,numpy as np,pandas as pd


tokenizer=Tokenizer(48000)
tokenizer.load_tokenizer("tokenizer_vocab.titoken")

TOTAL_TOKEN_COUNT=int(2e10)
EOT_ID=tokenizer.encoding._special_tokens["<|endoftext|>"]

data_splits={"fineweb_edu":{"tok_frac":0.60,"path":"../raw_corpus_optimized/fineweb-edu-10BT/sample/10BT/000_00000.parquet","num_parquets":14}, 
             "fine_math":{"tok_frac":0.25,"path":"../raw_corpus_optimized/finemath_4plus.jsonl.zst"},
              "stack_python":{"tok_frac":0.25*0.5,"path":"../raw_corpus_optimized/stack_python_dedup.jsonl.zst"},
              "stack_cpp":{"tok_frac":0.25*0.2,"path":"../raw_corpus_optimized/stack_cpp_dedup.jsonl.zst" },
              "stack_tex":{"tok_frac":0.25*0.2,"path":"../raw_corpus_optimized/stack_tex_dedup.jsonl.zst" },
              "stack_sql":{"tok_frac":0.25*0.1,"path":"../raw_corpus_optimized/stack_sql_dedup.jsonl.zst" },}



def create_token_arr(data_splits, total_token_count, output_path):
    count = 0
    with open(output_path, "wb") as fout:
        def add(text):
            nonlocal count
            ids = tokenizer.encode(text) 
            ids.append(EOT_ID)
            fout.write(np.asarray(ids, dtype=np.uint16).tobytes())
            count += len(ids)
        
        for name, split in data_splits.items():
            print(count, name, "START")
            tok_frac = split["tok_frac"] 
            curr = count
            target = curr+tok_frac*total_token_count
            
            if "fineweb_edu" in name:
                for i in range(split["num_parquets"]):
                    if count > target:
                        break
                    df = pd.read_parquet(f"../raw_corpus_optimized/fineweb-edu-10BT/sample/10BT/{i:03d}_00000.parquet", columns=["text"])
                    
                    for line in df["text"]:
                        if count > target: 
                            break
                        add(line)
            else:
                with open(split["path"], "rb") as fh:
                    reader = zstd.ZstdDecompressor().stream_reader(fh)
                    for line in io.TextIOWrapper(reader, encoding="utf-8"):
                        
                        if count > target: 
                            break
                        if not line.strip(): 
                            continue
                        
                        add(json.loads(line)["text"])
            print(f"{name}: {count} tokens")
    return count
    
