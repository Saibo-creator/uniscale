from transformers import AutoTokenizer

# 加载BPE（已经是HF格式）
bpe_tokenizer = AutoTokenizer.from_pretrained("out/tokenizers/bpe_v80k")

# 加载UnigramLM（现在也能用AutoTokenizer！）
unigram_tokenizer = AutoTokenizer.from_pretrained("out/tokenizers/unigram_v80k")

# 统一的接口
text = "Hello world!"
bpe_tokens = bpe_tokenizer.tokenize(text)
print(f"BPE {bpe_tokens}")
unigram_tokens = unigram_tokenizer.tokenize(text)
print(f"Unigram {unigram_tokens}")