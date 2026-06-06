import numpy as np
import tiktoken

data = np.fromfile('/home/rohit1/data/fineweb-edu-10BT/sample/10BT/chunk_0.bin', dtype=np.uint16)

enc = tiktoken.get_encoding("gpt2")
text=enc.decode(data[40000:])

EOT=enc.eot_token

print(f"tokens: {len(data):,}")
print(f"EOT count: {(data == EOT).sum()}")  # should be ≈ #docs in chunk
print(f"avg tokens/doc: {len(data) / (data == EOT).sum():.1f}")
print(text)