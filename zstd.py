import zstandard as zstd
import io
import json

with open("/home/rohit1/LLM_train/raw_corpus_optimized/finemath_4plus.jsonl.zst", "rb") as fh:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(fh) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text:            # one JSON record per line for FineWeb/JSONL
            obj = json.loads(line)
            print(line)
        
            # exit()