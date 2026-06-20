import zstandard as zstd
import io
import json
import pandas as pd
import regex as re
import json, pickle
from collections import Counter
from tqdm import tqdm


#GETTING 2B TOKENS SEPERATELY WITH PROPORTION 50 TEXT(FINE WEB) 25 MATH 25 CODE (INCLUDES 5 LATEX)

#ESTIMATES OF CHAR PER TOKENS METRICS
cpt = {"eng": 4.0, "codemath": 3.0}

total_token_count=2e9

data_splits={"fineweb_edu1":{"tok_frac":0.50,"path":"../raw_corpus_optimized/fineweb-edu-10BT/sample/10BT/000_00000.parquet"}, # COMBINED 0.5 REPETITION JUST FOR DICT
             "fineweb_edu2":{"tok_frac":0.50,"path":"../raw_corpus_optimized/fineweb-edu-10BT/sample/10BT/001_00000.parquet"},
             "fine_math":{"tok_frac":0.25,"path":"../raw_corpus_optimized/finemath_4plus.jsonl.zst"},
              "stack_python":{"tok_frac":0.25*0.5,"path":"../raw_corpus_optimized/stack_python_dedup.jsonl.zst"},
              "stack_cpp":{"tok_frac":0.25*0.2,"path":"../raw_corpus_optimized/stack_cpp_dedup.jsonl.zst" },
              "stack_tex":{"tok_frac":0.25*0.2,"path":"../raw_corpus_optimized/stack_tex_dedup.jsonl.zst" },
              "stack_sql":{"tok_frac":0.25*0.1,"path":"../raw_corpus_optimized/stack_sql_dedup.jsonl.zst" },}


def convert_count(type,count):
    if type=="eng":
        return count//4
    elif type=="codemath":
        return count//3

def create_doc_list(data_splits,total_token_count):
#WE WILL UTILISE THE FACT THAT FINEWEB-EDU PARQUET1 IS 1BT
    count=0
    doc_list=[]
    for name,split in data_splits.items():
        print(count,name,"START")
        path=split["path"]
        tok_frac=split["tok_frac"]

        if "fineweb_edu" in name:
            df=pd.read_parquet(path)
            texts=df['text'].tolist()
            
            #COMBINED FINEWEB-EDU
            for line in texts:
                if count>tok_frac*total_token_count:
                    break
                doc_list.append(line)
                count+=convert_count("eng",len(line))
        
        else:
            with open(path, "rb") as fh:
                #READ ZSTD FILE
                curr_count=count
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:     
                    #DOMAIN FRAC DATA APPEND
                    text = io.TextIOWrapper(reader, encoding="utf-8")
                    for line in text:        
                        if count>curr_count+tok_frac*total_token_count:
                            break
                                
                        obj = json.loads(line)
                        doc_list.append(obj["text"])
                        count+=convert_count("codemath",len(obj["text"]))

    print(count,len(doc_list))          
    return doc_list
    


O200_K_PATTERN=r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}|_[\p{L}\p{N}]+| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|        |    |   |  |\s+(?!\S)|\s+"""
_SPLIT = re.compile(O200_K_PATTERN)


def attribute_markers(docs, markers=("_id","_slot","_prop","_get","_set",' ":','",','"),',"(try")):
    """Find whether cluster tokens come from a few generated files (contamination)
    or are spread across many docs (representative). High top-3 share = contamination."""
    per_marker = {m: [] for m in markers}      # (count, doc_id) per doc
    suspects = []                               # generated/minified files
    for i, text in enumerate(docs):
        toks = _SPLIT.findall(text)
        n = len(toks)
        if n == 0:
            continue
        tc = Counter(toks)
        for m in markers:
            if tc.get(m):
                per_marker[m].append((tc[m], i))
        uniq_ratio = len(tc) / n
        punct_ratio = sum(v for t,v in tc.items() if not re.search(r"[\p{L}\p{N}]", t)) / n
        if n > 2000 and uniq_ratio < 0.25 and punct_ratio > 0.30:
            suspects.append((i, n, round(uniq_ratio,3), round(punct_ratio,3)))

    print(f"{'marker':>8} | {'total':>7} | top1  top3  top5 | top doc_ids")
    for m in markers:
        c = sorted(per_marker[m], reverse=True)
        tot = sum(x for x,_ in c)
        if not tot:
            continue
        f = lambda k: sum(x for x,_ in c[:k]) / tot * 100
        print(f"{m:>8} | {tot:>7} | {f(1):4.0f}% {f(3):4.0f}% {f(5):4.0f}% | {[d for _,d in c[:5]]}")

    print(f"\nsuspected generated docs (n>2000, uniq<0.25, punct>0.30): {len(suspects)}")
    for s in sorted(suspects, key=lambda s: -s[1])[:10]:
        print(f"  doc {s[0]:>5}: {s[1]:>6} tok, uniq={s[2]}, punct={s[3]}")
    return per_marker, suspects


def get_word_counts(texts, ):
    word_counts = Counter()
    total = 0
    for text in tqdm(texts):
        toks = _SPLIT.findall(text)
        word_counts.update(toks)
        total += len(toks)

    return word_counts

def save_pretokenized(res, counts_path, text_path=None):
    word_counts = get_word_counts(res)

    with open(counts_path, "wb") as f:
        pickle.dump(dict(word_counts), f)          
    print(f"saved {len(word_counts)} unique pretokens "
          f"({sum(word_counts.values())/1e6:.0f}M total) -> {counts_path}")

    if text_path:
        import pyarrow as pa, pyarrow.parquet as pq
        pq.write_table(pa.table({"text": res}), text_path, compression="zstd")
        print(f"saved {len(res)} docs -> {text_path}")

    return word_counts

if __name__=="__main__":
    doc_list=create_doc_list(data_splits,total_token_count)
    save_pretokenized(doc_list,"/home/rohit1/LLM_train/raw_corpus_optimized/count.pkl")
