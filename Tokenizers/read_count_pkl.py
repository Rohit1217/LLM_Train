from collections import Counter
import pickle
from operator import itemgetter
from tokenizer_fast import Tokenizer

tokenizer=Tokenizer(48000)
tokenizer.load_tokenizer("tokenizer_vocab.titoken")

text = "BIRTH OF THE TOKENIZATION GOD 123456 123"
tokens = tokenizer.encode(text)
print("Token IDs:", tokens)

# print(tokenizer.vocab)
decoded_text = tokenizer.decode(tokens)
print("Decoded Text:", decoded_text)
print(tokenizer.encoding._special_tokens["<|endoftext|>"],tokenizer.vocab_size)




# word_counts = Counter(pickle.load(open("/home/rohit1/LLM_train/raw_corpus_optimized/count.pkl", "rb")))

# word_counts_desc = dict(sorted(word_counts.items(), key=itemgetter(1), reverse=True))
# count=0
# for word,countw in word_counts_desc.items():
#     if count>100:
#         break
#     print(word,countw)
#     count+=1