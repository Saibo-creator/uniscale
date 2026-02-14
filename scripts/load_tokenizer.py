from transformers import AutoTokenizer

# Load BPE (already in HF format)
bpe_tokenizer = AutoTokenizer.from_pretrained("out/tokenizers/bpe_v80k")

# Load UnigramLM (now also compatible with AutoTokenizer!)
unigram_tokenizer = AutoTokenizer.from_pretrained("out/tokenizers/unigram_v80k")

# Unified interface
text = "Hello world!"
bpe_tokens = bpe_tokenizer.tokenize(text)
print(f"BPE {bpe_tokens}")
unigram_tokens = unigram_tokenizer.tokenize(text)
print(f"Unigram {unigram_tokens}")