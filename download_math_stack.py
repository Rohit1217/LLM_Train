import os
import json
import zstandard as zstd
from datasets import load_dataset
from torch.utils.data import DataLoader

# --- TARGET LIMITS (Tokens converted to character estimates via * 4) ---
FINEMATH_CHAR_TARGET = 5_000_000_000 * 4  # 20 Billion chars
STACK_CHAR_TARGET = 10_000_000_000 * 4     # 40 Billion chars will dedup

OUTPUT_DIR = "./raw_corpus_optimized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- POLITE BACKGROUND WORKER SETTINGS ---
# HF `datasets` partitions the shard (parquet file) list across DataLoader
# workers, so each shard is streamed by exactly one worker -- no duplication.
# Cap effective parallelism at the dataset's shard count (extra workers idle).
NUM_WORKERS = 16

# Restrict background library thread explosions
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["RAYON_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def polite_extractor(dataset_id, subset, target_chars, base_name):
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}.jsonl.zst")
    print(f"\n[Polite Mode] Processing {base_name} -> Targeting {target_chars:,} chars.")

    # The Stack v1 (dedup) ships code inline in the `content` column, keyed by
    # a per-language data_dir. Gated dataset -> requires `huggingface-cli login`
    # plus accepting the terms on the dataset page; token=True forwards the auth.
    dataset = load_dataset(
        dataset_id, data_dir=f"data/{subset}", split="train", streaming=True, token=True
    )
    text_key = "content"

    # Drop the Stack's metadata columns (some are nullable, e.g. max_stars_count)
    # so the default DataLoader collate never chokes on a None value.
    dataset = dataset.select_columns([text_key])

    dataloader = DataLoader(dataset, batch_size=1, num_workers=NUM_WORKERS)
    
    chars_saved = 0
    cctx = zstd.ZstdCompressor(level=3)
    
    with open(out_path, 'wb') as fh:
        with cctx.stream_writer(fh) as compressor:
            for batch in dataloader:
                text_data = batch[text_key]
                text_str = text_data[0] if isinstance(text_data, list) else text_data
                
                if not text_str or not isinstance(text_str, str):
                    continue
                    
                text_str = text_str.strip()
                chars_saved += len(text_str)
                
                # Format to JSONL
                record = json.dumps({"text": text_str}, ensure_ascii=False) + "\n"
                compressor.write(record.encode('utf-8'))
                
                # Print sparse progress lines so we don't spam stdout
                if chars_saved % 100_000_000 < len(record):
                    print(f" -> {base_name}: Gathered {chars_saved:,} / {target_chars:,} characters...")

                if chars_saved >= target_chars:
                    break
                    
    print(f"Finished: Saved compressed file to {out_path}")

# =====================================================================
# RUN THE DOWNLOADING PIPELINES
# =====================================================================

# All languages come from The Stack v1 (dedup): it embeds `content` inline, so
# it needs only an HF login (no AWS). Directory names must match the repo
# exactly -- in the-stack-dedup C++ is under data/cpp (the non-dedup repo
# uses data/c++ instead).
STACK_DATASET = "bigcode/the-stack-dedup"
stack_languages = [
    {"lang": "python", "chars": int(STACK_CHAR_TARGET * 0.60)},
    {"lang": "cpp",    "chars": int(STACK_CHAR_TARGET * 0.20)},
    {"lang": "tex",    "chars": int(STACK_CHAR_TARGET * 0.15)},
    {"lang": "sql",    "chars": int(STACK_CHAR_TARGET * 0.05)},
]

for config in stack_languages:
    polite_extractor(
        dataset_id=STACK_DATASET,
        subset=config["lang"],
        target_chars=config["chars"],
        base_name=f"stack_v1_{config['lang']}",
    )

print("\nAll targeted data subsets successfully saved!")
