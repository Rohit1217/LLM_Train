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
