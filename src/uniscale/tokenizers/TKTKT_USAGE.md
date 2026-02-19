# TkTkT Backend Usage Guide

This guide explains how to use the TkTkT backend to train various tokenizer algorithms.

## Supported Algorithms

### BPE Variants
- **classic-bpe**: Classical Byte-Pair Encoding
- **picky-bpe**: PickyBPE with selective merge/split operations
- **scaffold-bpe**: ScaffoldBPE that removes low-frequency tokens
- **trimmed-bpe**: TrimmedBPE with recursive decomposition
- **bpe-dropout**: BPE with dropout during inference

### Unigram
- **kudopiece**: Unigram Language Model (native TkTkT implementation)

### Other Algorithms
- **sage**: SaGe vocabularisation algorithm
- **ngram**: Character/byte N-gram tokenizer
- **lzw**: Lempel-Ziv-Welch compression-based tokenizer

## Basic Usage

### List Available Algorithms

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm list
```

### Train Classic BPE

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm classic-bpe \
  --vocab_size 32000 \
  --data_file data/raw/tokenizer_data.jsonl \
  --output_dir tokenizers/trained/classic_bpe_32k
```

### Train PickyBPE

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm picky-bpe \
  --vocab_size 32000 \
  --data_file data/raw/tokenizer_data.jsonl \
  --picky_threshold 0.5 \
  --output_dir tokenizers/trained/picky_bpe_32k
```

### Train ScaffoldBPE

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm scaffold-bpe \
  --vocab_size 32000 \
  --data_file data/raw/tokenizer_data.jsonl \
  --output_dir tokenizers/trained/scaffold_bpe_32k
```

### Train TrimmedBPE

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm trimmed-bpe \
  --vocab_size 32000 \
  --data_file data/raw/tokenizer_data.jsonl \
  --output_dir tokenizers/trained/trimmed_bpe_32k
```

### Train BPE-Dropout

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm bpe-dropout \
  --vocab_size 32000 \
  --data_file data/raw/tokenizer_data.jsonl \
  --dropout_probability 0.1 \
  --output_dir tokenizers/trained/bpe_dropout_32k
```

### Train KudoPiece (Unigram)

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm kudopiece \
  --vocab_size 32000 \
  --data_file data/raw/tokenizer_data.jsonl \
  --initial_vocab_size 1000000 \
  --shrinking_factor 0.75 \
  --output_dir tokenizers/trained/kudopiece_32k
```

### Train N-gram Tokenizer

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm ngram \
  --vocab_size 32000 \
  --data_file data/raw/tokenizer_data.jsonl \
  --ngram_n 3 \
  --output_dir tokenizers/trained/ngram_3_32k
```

### Train LZW Tokenizer

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm lzw \
  --vocab_size 32000 \
  --data_file data/raw/tokenizer_data.jsonl \
  --output_dir tokenizers/trained/lzw_32k
```

### Train SaGe

```bash
python src/uniscale/tokenizers/train_tokenizer_beta.py \
  --algorithm sage \
  --vocab_size 32000 \
  --data_file data/raw/tokenizer_data.jsonl \
  --output_dir tokenizers/trained/sage_32k
```

## Algorithm-Specific Parameters

### BPE Parameters
- `--character_coverage`: Character coverage (default: 1.0)
- `--max_type_length`: Maximum token length (default: 16)
- `--min_frequency`: Minimum frequency for tokens (default: 2)

### PickyBPE Parameters
- `--picky_threshold`: Threshold for merge/split decisions (0.0-1.0, default: 0.5)

### BPE-Dropout Parameters
- `--dropout_probability`: Probability of dropping merges (default: 0.1)

### KudoPiece Parameters
- `--initial_vocab_size`: Initial vocabulary size before pruning (default: 1,000,000)
- `--shrinking_factor`: Factor for vocabulary reduction (default: 0.75)
- `--num_sub_iterations`: Number of EM sub-iterations (default: 2)

### N-gram Parameters
- `--ngram_n`: N for N-gram tokenizer (default: 3)

## Testing

Run the test script to verify all algorithms work correctly:

```bash
python scripts/test_tktkt_algorithms.py
```

This will train a small tokenizer with each algorithm and verify the outputs.

## Loading Trained Tokenizers

### BPE-based Tokenizers

```python
from transformers import AutoTokenizer

# Load HF-exported tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizers/trained/classic_bpe_32k/hf")

# Use tokenizer
tokens = tokenizer.encode("Hello world!")
```

### KudoPiece Tokenizer

```python
from transformers import AutoTokenizer

# Load KudoPiece tokenizer (uses SentencePiece model)
tokenizer = AutoTokenizer.from_pretrained("tokenizers/trained/kudopiece_32k/hf")

# Use tokenizer
tokens = tokenizer.encode("Hello world!")
```

## Notes

1. **Data Format**: Training data should be in JSONL format with a `text` field:
   ```json
   {"text": "This is a sample sentence."}
   {"text": "Another sentence for training."}
   ```

2. **Output Structure**: Each trained tokenizer produces:
   - `config.json`: Training configuration
   - `vocab.json`: Vocabulary mapping
   - `merges.txt` (BPE variants): Merge operations
   - `tokenizer.model` (KudoPiece): SentencePiece model
   - `hf/`: HuggingFace-compatible export (if `--no_export_hf` not specified)

3. **Memory Requirements**: Some algorithms (especially KudoPiece with large initial vocab) may require significant memory.

4. **Caching**: TkTkT caches training results. If you retrain with the same parameters, it may load from cache.

## Troubleshooting

### TkTkT Not Found

```bash
# Install TkTkT from the included submodule
pip install -e tmp/TkTkT
```

### Missing Dependencies

```bash
# Install required packages
pip install sentencepiece
pip install pickybpe
```

### Out of Memory

- Reduce `--initial_vocab_size` for KudoPiece
- Use smaller training data for testing
- Reduce `--vocab_size`

## References

- [TkTkT Repository](https://github.com/bauwenst/TkTkT)
- [PickyBPE Paper](https://aclanthology.org/2024.emnlp-main.925/)
- [ScaffoldBPE Paper](https://dl.acm.org/doi/10.1609/aaai.v39i23.34633)
- [TrimmedBPE Paper](https://arxiv.org/abs/2404.00397)
- [BPE-Knockout Paper](https://aclanthology.org/2024.naacl-long.324/)
- [SaGe Paper](https://aclanthology.org/2023.eacl-main.45/)
