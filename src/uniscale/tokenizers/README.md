# Tokenizer Training with Multiple Backends

This module provides a unified interface for training tokenizers using different backend libraries. All tokenizers are exported to HuggingFace format for consistent usage.

## Architecture

```
src/uniscale/tokenizers/
├── train_tokenizer.py      # Main training script
├── backends/
│   ├── base.py            # Abstract base class
│   ├── huggingface.py     # HF tokenizers backend
│   ├── sentencepiece.py   # SentencePiece backend
│   ├── parity_aware_bpe.py # Parity-aware BPE backend
│   └── tktkt.py           # TkTkT backend
```

## Supported Backends & Algorithms

### 1. HuggingFace Tokenizers
- **bpe**: Byte-level BPE

### 2. SentencePiece
- **unigram**: Unigram Language Model
- **bpe**: Character-level BPE (via SentencePiece)

### 3. Parity-aware BPE
- **parity-bpe**: Cross-lingual fairness BPE (base variant)
- **parity-bpe-window**: Moving-window balancing variant

### 4. TkTkT
- **classic-bpe**: Classical BPE
- **bpe-dropout**: BPE with dropout
- **bpe-knockout**: BPE with knockout
- **picky-bpe**: PickyBPE
- **unigram**: Unigram LM (KudoPiece)

## Installation

Install the required backend libraries:

```bash
# Core dependencies (already in requirements)
pip install tokenizers transformers sentencepiece

# Parity-aware BPE
pip install -e tmp/parity-aware-bpe

# TkTkT
pip install -e tmp/TkTkT
```

## Usage

### Command Line

```bash
# List all available algorithms
python -m src.uniscale.tokenizers.train_tokenizer --algorithm list

# Train BPE tokenizer (auto-detects backend)
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm bpe \
    --vocab_size 80000 \
    --data_file data/raw/tokenizer_data.jsonl

# Train Unigram tokenizer
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm unigram \
    --vocab_size 128000 \
    --data_file data/raw/tokenizer_data.jsonl

# Train parity-aware BPE
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm parity-bpe \
    --vocab_size 80000 \
    --data_file data/raw/tokenizer_data.jsonl

# Train with specific backend
python -m src.uniscale.tokenizers.train_tokenizer \
    --algorithm bpe \
    --vocab_size 80000 \
    --backend huggingface \
    --data_file data/raw/tokenizer_data.jsonl
```

### Python API

```python
from src.uniscale.tokenizers.train_tokenizer import train_tokenizer

# Train a tokenizer
output_dir = train_tokenizer(
    algorithm="bpe",
    data_file="data/raw/tokenizer_data.jsonl",
    vocab_size=80000,
    export_hf=True,  # Export to HuggingFace format
)

# Load the trained tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(f"{output_dir}/hf")
```

## Output Structure

Each training run produces the following structure:

```
tokenizers/trained/{algorithm}_v{vocab_size}k/
├── config.json              # Training configuration
├── [backend-specific files] # E.g., tokenizer.json, merges.txt, tokenizer.model
└── hf/                      # HuggingFace format (if export_hf=True)
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── ...
```

## Adding a New Backend

To add support for a new tokenizer library:

1. Create a new backend class in `backends/`:

```python
from .base import TokenizerBackend

class MyBackend(TokenizerBackend):
    def get_supported_algorithms(self) -> List[str]:
        return ["my-algorithm"]

    def train(self, algorithm: str, data_file: str, vocab_size: int,
              output_dir: str, **kwargs) -> Dict[str, Any]:
        # Train tokenizer
        # Save artifacts to output_dir
        return config

    def export_to_hf(self, artifacts_dir: str, output_dir: str,
                     **kwargs) -> PreTrainedTokenizerFast:
        # Convert to HuggingFace format
        # Save to output_dir
        return tokenizer
```

2. Register the backend in `train_tokenizer.py`:

```python
from .backends import MyBackend

BACKEND_REGISTRY = {
    # ... existing backends
    "my-backend": MyBackend(),
}
```

## Design Principles

1. **Separation of Concerns**: Each backend encapsulates its library-specific logic
2. **Unified Interface**: All backends export to HuggingFace format
3. **Extensibility**: Easy to add new backends and algorithms
4. **Auto-detection**: Backends are automatically selected based on algorithm
5. **Consistency**: Same data format (JSONL with 'text' field) for all backends

## Benefits

- ✅ Support multiple tokenizer training libraries
- ✅ Consistent HuggingFace output format
- ✅ Easy to switch between algorithms
- ✅ Clean separation between training and inference
- ✅ Extensible architecture for future backends
