import numpy as np

CONTEXT_LENGTH = 1024
NUM_SEQUENCES = 10000
FILE_PATH="/home/rohit1/LLM_train/raw_corpus_optimized/tokens.bin"


print("Loading data...")
data = np.fromfile(FILE_PATH, dtype=np.uint16)

print("Structuring into 1024-token blocks...")
total_possible_seqs = data.shape[0] // CONTEXT_LENGTH

structured_data = data[:total_possible_seqs * CONTEXT_LENGTH].reshape(-1, CONTEXT_LENGTH)

print("Shuffling sequence orders...")
rng = np.random.default_rng(seed=42)
rng.shuffle(structured_data)  

print(structured_data.shape)

print(f"Extracting {NUM_SEQUENCES} sequences...")
overfit_matrix = structured_data[:NUM_SEQUENCES]

output_file = "overfit_10k_sequences.bin"
print(f"Saving to {output_file}...")
with open(output_file, "wb") as f:
    f.write(overfit_matrix.tobytes())

print("Done! Verified shape:", overfit_matrix.shape)

