# Parity-Aware BPE Usage Guide

## Overview

Parity-Aware BPE is designed for **multi-lingual** tokenizer training to ensure fairness across languages.

## Key Concepts

### Algorithms

1. **pa-bpe** - Parity-Aware BPE (multi-lingual)
2. **pa-super-bpe** - Parity-Aware Super-BPE with two-phase training (multi-lingual)

### Compression Ratio

The `ratio` parameter specifies the desired compression rate for each language:
- **1.0** = No compression (keep original length)
- **1.5** = Compress to 2/3 of original length
- **2.0** = Compress to 1/2 of original length (50% compression)
- **3.0** = Compress to 1/3 of original length (66% compression)

**Recommended range:** 1.0 to 3.0

## Data Preparation

All data must be in JSONL format with `text` and `language` fields.

Data is organized as consecutive blocks per language:

```
[First N lines: language 1]
{"text": "Hello world", "language": "en"}
{"text": "This is a sentence", "language": "en"}
...

[Next N lines: language 2]
{"text": "Bonjour le monde", "language": "fr"}
{"text": "C'est une phrase", "language": "fr"}
...
```

**Important**: The j-th sample of each language should be parallel translations.

Dev files follow the same format.

## Usage Examples

### 1. PA-BPE with Dev File

```bash
python -m uniscale.tokenizers.train_tokenizer_beta \
    --algorithm pa-bpe \
    --data_file data/train.jsonl \
    --dev_file data/dev.jsonl \
    --vocab_size 32000 \
    --num_workers 8
```

Data format:
```json
{"text": "Hello world", "language": "en"}
{"text": "Another sentence", "language": "en"}
{"text": "Bonjour le monde", "language": "fr"}
{"text": "Une autre phrase", "language": "fr"}
```

### 2. PA-BPE with Compression Ratio

```bash
python -m uniscale.tokenizers.train_tokenizer_beta \
    --algorithm pa-bpe \
    --data_file data/train.jsonl \
    --ratio 1.5 1.8 1.2 \
    --vocab_size 32000 \
    --num_workers 8
```

**Note:** Ratio values correspond to languages in order of appearance in the JSONL file.

### 3. PA-Super-BPE

```bash
python -m uniscale.tokenizers.train_tokenizer_beta \
    --algorithm pa-super-bpe \
    --data_file data/train.jsonl \
    --dev_file data/dev.jsonl \
    --vocab_size 128000 \
    --phase1_merges 80000 \
    --num_workers 32
```

## Parameters

### Common Parameters

- `--algorithm`: Tokenization algorithm (pa-bpe, pa-super-bpe)
- `--vocab_size`: Target vocabulary size (number of merge operations)
- `--data_file`: Path to JSONL training data with multiple languages (required)
- `--output_dir`: Output directory (auto-generated if not specified)
- `--num_workers`: Number of parallel workers (default: 1)
- `--min_frequency`: Minimum frequency for merges (default: 2)

### Parity-Aware Parameters

- `--dev_file`: Path to JSONL development data (alternative to ratio)
- `--ratio`: Compression ratios per language (alternative to dev_file)

### PA-Super-BPE Parameters

- `--phase1_merges`: Number of merges in phase 1 (default: 60% of vocab_size)

## Common Errors

### Error: "Parity-aware BPE requires multi-lingual data"

**Cause:** Using `pa-bpe` or `pa-super-bpe` with only one language in JSONL

**Solution:** Provide JSONL with multiple languages (at least 2)

### Error: "Dev languages must match training languages"

**Cause:** Mismatch between languages in train and dev JSONL files

**Solution:** Ensure both files have the same set of languages:
- train.jsonl: languages ["en", "fr", "zh"]
- dev.jsonl: languages ["en", "fr", "zh"] (same order)

### Error: Missing "language" field

**Cause:** JSONL missing the "language" field

**Solution:** Ensure all JSONL lines include "language" field:
```json
{"text": "...", "language": "en"}
```

## Best Practices

1. **Use appropriate algorithm:**
   - Basic multi-lingual → `pa-bpe`
   - Advanced multi-lingual → `pa-super-bpe` (two-phase training)

2. **Data format:**
   - Always include "language" field in JSONL
   - Organize as consecutive blocks per language

3. **Choose ratio carefully:**
   - Start with equal ratios (e.g., all 1.5)
   - Adjust based on language characteristics
   - Higher ratio = more compression

4. **Language codes:**
   - Use consistent ISO language codes (en, fr, zh, etc.)
   - Keep same language order in train and dev files

5. **Performance:**
   - Use `--num_workers` to parallelize (recommended: 8-32)
   - Larger vocab_size requires more memory and time

## Examples

### Prepare Multi-Lingual Data

Convert parallel corpus to JSONL format:

```python
import json

# Prepare training data
with open("data/train.jsonl", "w") as f:
    # English samples (N lines)
    for line in open("corpus.en"):
        f.write(json.dumps({"text": line.strip(), "language": "en"}) + "\n")

    # French samples (N lines, parallel translations)
    for line in open("corpus.fr"):
        f.write(json.dumps({"text": line.strip(), "language": "fr"}) + "\n")

    # Chinese samples (N lines, parallel translations)
    for line in open("corpus.zh"):
        f.write(json.dumps({"text": line.strip(), "language": "zh"}) + "\n")

# Prepare dev data (same structure)
with open("data/dev.jsonl", "w") as f:
    # Same structure as train, with dev samples
    ...
```

### Train PA-Super-BPE

```bash
python -m uniscale.tokenizers.train_tokenizer_beta \
    --algorithm pa-super-bpe \
    --data_file data/train.jsonl \
    --dev_file data/dev.jsonl \
    --vocab_size 128000 \
    --phase1_merges 76800 \
    --min_frequency 2 \
    --num_workers 32 \
    --output_dir tokenizers/multilingual_128k
```

This will create:
- `tokenizers/multilingual_128k/merges.txt` - Final merge operations
- `tokenizers/multilingual_128k/phase1_merges.txt` - Phase 1 merges (Super-BPE only)
- `tokenizers/multilingual_128k/config.json` - Training configuration
- `tokenizers/multilingual_128k/hf/` - HuggingFace tokenizer (if export enabled)
