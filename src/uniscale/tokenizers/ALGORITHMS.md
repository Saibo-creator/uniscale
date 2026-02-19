# Supported Tokenizer Algorithms

This document describes all supported tokenizer training algorithms and their implementations.

## Overview

| Algorithm | Backend | Training Method | Pretokenization |
|-----------|---------|-----------------|-----------------|
| **bpe** | HuggingFace or Parity-aware | Single-step | whitespace + bytelevel |
| **super-bpe** | Parity-aware | Two-step | Phase1: whitespace+bytelevel → Phase2: bytelevel only |
| **unigram** | SentencePiece | Single-step | Character-based |
| **pa-bpe** | Parity-aware | Single-step | whitespace + bytelevel |
| **pa-super-bpe** | Parity-aware | Two-step | Phase1: whitespace+bytelevel → Phase2: bytelevel only |

## Algorithm Details

### 1. BPE (Byte-Pair Encoding)

**Standard BPE with whitespace pretokenization**

- **Backend**: HuggingFace (fast) or Parity-aware
- **Training**: Single-step
- **Pretokenization**: Whitespace + ByteLevel
- **Description**: Standard BPE that respects word boundaries. Cannot merge across spaces.

**Usage:**
```bash
# Fast implementation (HuggingFace)
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm bpe \
    --vocab_size 80000 \
    --backend huggingface \
    --data_file data/raw/tokenizer_data.jsonl

# Alternative implementation (Parity-aware)
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm bpe \
    --vocab_size 80000 \
    --backend parity-aware-bpe \
    --data_file data/raw/tokenizer_data.jsonl
```

---

### 2. Super-BPE

**Two-step BPE training for aggressive cross-word merging**

- **Backend**: Parity-aware
- **Training**: Two-step (Phase 1 + Phase 2)
- **Description**:
  - **Phase 1**: Train standard BPE with whitespace+bytelevel pretokenization
  - **Phase 2**: Load Phase 1 merges, continue training with bytelevel-only pretokenization
  - This allows the tokenizer to merge across word boundaries in Phase 2

**Training Process:**
1. Phase 1 trains `phase1_merges` (default: 60% of vocab_size) using standard BPE
2. Phase 2 continues from Phase 1, training up to `vocab_size` total merges using bytelevel-only

**Usage:**
```bash
# SuperBPE with default phase1 (60% of vocab_size)
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm super-bpe \
    --vocab_size 80000 \
    --data_file data/raw/tokenizer_data.jsonl

# SuperBPE with custom phase1
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm super-bpe \
    --vocab_size 80000 \
    --phase1_merges 48000 \
    --data_file data/raw/tokenizer_data.jsonl
```

**Output Files:**
- `phase1_merges.txt` - Merges from Phase 1
- `merges.txt` - Final merges (includes Phase 1 + Phase 2)
- `config.json` - Training configuration

---

### 3. Unigram

**Unigram Language Model tokenization**

- **Backend**: SentencePiece
- **Training**: Single-step
- **Description**: Probabilistic subword tokenization based on unigram language model

**Usage:**
```bash
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm unigram \
    --vocab_size 128000 \
    --data_file data/raw/tokenizer_data.jsonl
```

---

### 4. PA-BPE (Parity-Aware BPE)

**Cross-lingual fairness-aware BPE**

- **Backend**: Parity-aware
- **Training**: Single-step
- **Pretokenization**: Whitespace + ByteLevel
- **Description**: BPE that optimizes for parity (fairness) in token lengths across languages

**Usage:**
```bash
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm pa-bpe \
    --vocab_size 80000 \
    --data_file data/raw/tokenizer_data.jsonl
```

**Note**: For true parity computation, you should provide development files per language. See parity-aware-bpe documentation for details.

---

### 5. PA-Super-BPE (Parity-Aware Super-BPE)

**Two-step parity-aware BPE with cross-word merging**

- **Backend**: Parity-aware
- **Training**: Two-step (Phase 1 + Phase 2)
- **Description**:
  - Combines parity-aware training with SuperBPE's two-step approach
  - **Phase 1**: Train PA-BPE with whitespace+bytelevel
  - **Phase 2**: Continue with bytelevel-only for cross-word merging

**Usage:**
```bash
# PA-SuperBPE with default phase1 (60% of vocab_size)
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm pa-super-bpe \
    --vocab_size 80000 \
    --data_file data/raw/tokenizer_data.jsonl

# PA-SuperBPE with custom phase1
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm pa-super-bpe \
    --vocab_size 80000 \
    --phase1_merges 48000 \
    --data_file data/raw/tokenizer_data.jsonl
```

---

## Common Parameters

### Required
- `--algorithm`: Algorithm to use (bpe, super-bpe, unigram, pa-bpe, pa-super-bpe)
- `--vocab_size`: Target vocabulary size

### Optional
- `--data_file`: Path to training data (default: data/raw/tokenizer_data.jsonl)
- `--output_dir`: Output directory (default: tokenizers/trained/{algorithm}_v{vocab_size}k)
- `--backend`: Force specific backend (auto-detected if not specified)
- `--min_frequency`: Minimum frequency for tokens (default: 2)
- `--phase1_merges`: For super-bpe variants, number of merges in phase1 (default: 60% of vocab_size)
- `--no_export_hf`: Skip exporting to HuggingFace format

### Examples

```bash
# List all available algorithms
python -m src.uniscale.tokenizers.train_tokenizer --algorithm list

# Train BPE (fast HF implementation)
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm bpe \
    --vocab_size 80000 \
    --data_file data/raw/tokenizer_data.jsonl

# Train SuperBPE
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm super-bpe \
    --vocab_size 80000 \
    --phase1_merges 48000 \
    --data_file data/raw/tokenizer_data.jsonl

# Train Unigram
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm unigram \
    --vocab_size 128000 \
    --data_file data/raw/tokenizer_data.jsonl

# Train PA-BPE
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm pa-bpe \
    --vocab_size 80000 \
    --data_file data/raw/tokenizer_data.jsonl

# Train PA-SuperBPE
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm pa-super-bpe \
    --vocab_size 80000 \
    --phase1_merges 48000 \
    --data_file data/raw/tokenizer_data.jsonl
```

## Output Format

All tokenizers are exported to HuggingFace format for consistent usage:

```
tokenizers/trained/{algorithm}_v{vocab_size}k/
├── config.json              # Training configuration
├── [algorithm-specific files]
└── hf/                      # HuggingFace format
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.json
```

Load trained tokenizer:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tokenizers/trained/bpe_v80k/hf")
```

## References

- **BPE**: Sennrich et al., 2016. "Neural Machine Translation of Rare Words with Subword Units"
- **SuperBPE**: Liu et al., 2025. (Two-step BPE training)
- **Unigram**: Kudo, 2018. "Subword Regularization: Improving Neural Network Translation Models"
- **Parity-aware BPE**: [arxiv.org/abs/2508.04796](https://arxiv.org/abs/2508.04796)
