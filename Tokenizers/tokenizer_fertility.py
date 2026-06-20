import io, json, regex, numpy as np, pandas as pd, zstandard as zstd
from tokenizer_fast import Tokenizer

tokenizer = Tokenizer(48000)
tokenizer.load_tokenizer("tokenizer_vocab.titoken")

# word splitter for the denominator (whitespace words; \p{L}+ if you prefer)
_WORD = regex.compile(r"\S+")

eval_splits = {
    "fineweb_edu": {"kind": "parquet",
                    "path": "../raw_corpus_optimized/fineweb-edu-10BT/sample/10BT/{i:03d}_00000.parquet",
                    "heldout_files": [12, 13]},          # files NOT used for tokenizer training
    "fine_math":   {"kind": "zst", "path": "../raw_corpus_optimized/finemath_4plus.jsonl.zst"},
    "stack_python":{"kind": "zst", "path": "../raw_corpus_optimized/stack_python_dedup.jsonl.zst"},
    "stack_cpp":   {"kind": "zst", "path": "../raw_corpus_optimized/stack_cpp_dedup.jsonl.zst"},
    "stack_tex":   {"kind": "zst", "path": "../raw_corpus_optimized/stack_tex_dedup.jsonl.zst"},
    "stack_sql":   {"kind": "zst", "path": "../raw_corpus_optimized/stack_sql_dedup.jsonl.zst","skip_docs": 8000},
}

def iter_heldout(split, skip_docs, max_docs):
    """Yield text from documents AFTER skip_docs (held-out region), up to max_docs."""
    if split["kind"] == "parquet":
        # use dedicated held-out parquet files the tokenizer never saw
        for i in split["heldout_files"]:
            df = pd.read_parquet(split["path"].format(i=i), columns=["text"])
            for t in df["text"]:
                yield t
    else:
        with open(split["path"], "rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            seen = 0
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if not line.strip(): continue
                seen += 1
                if seen <= skip_docs:        # skip the region tokenizer trained on
                    continue
                if seen > skip_docs + max_docs:
                    break
                yield json.loads(line)["text"]

def measure_fertility(split, skip_docs=200_000, max_docs=20_000, min_words=500_000):
    """Per-document fertility samples -> mean + bootstrap CI. Stops once min_words reached."""
    per_doc_tokens, per_doc_words, per_doc_bytes = [], [], []
    total_words = 0
    skip_d=split.get("skip_docs",skip_docs)
    for text in iter_heldout(split, skip_d, max_docs):
        nw = len(_WORD.findall(text))
        if nw == 0: continue
        nt = len(tokenizer.encode(text))
        nb = len(text.encode("utf-8"))
        per_doc_tokens.append(nt); per_doc_words.append(nw); per_doc_bytes.append(nb)
        total_words += nw
        if total_words >= min_words:        # statistical-significance floor
            break

    T = np.array(per_doc_tokens); W = np.array(per_doc_words); B = np.array(per_doc_bytes)
    fertility = T.sum() / W.sum()            # tokens per word (corpus-level)
    bpt       = B.sum() / T.sum()            # bytes per token (compression)

    # bootstrap CI over documents (accounts for doc-length variation)
    rng = np.random.default_rng(0)
    boots = []
    idx = np.arange(len(T))
    for _ in range(1000):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots.append(T[s].sum() / W[s].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"fertility": fertility, "ci": (lo, hi), "bytes_per_tok": bpt,
            "n_docs": len(T), "n_words": int(W.sum())}

if __name__ == "__main__":
    print(f"{'source':>14} | {'fertility':>9} | {'95% CI':>16} | {'bytes/tok':>9} | {'words':>10} | docs")
    all_T = all_W = 0
    for name, split in eval_splits.items():
        r = measure_fertility(split)
        print(f"{name:>14} | {r['fertility']:>9.3f} | "
              f"[{r['ci'][0]:.3f},{r['ci'][1]:.3f}] | {r['bytes_per_tok']:>9.3f} | "
              f"{r['n_words']:>10,} | {r['n_docs']}")