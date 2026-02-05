# Scale-Invariant Tokenizer Selection

## Research Question

Can we use small language models (e.g., 50M-200M params) to choose the tokenizer for our larger language models?

More formally: Does the performance (in terms of perplexity per byte and/or downstream task performance) of small language models trained on a set of different tokenizers follow the same patterns of the performance of larger language models trained on those same tokenizers?

We define "follow the same patterns" as there being a high rank-correlation in the performance of the small models and large models, where our observations are (small model performance, large model performance) for each tokenizer.

## Project Structure

```
scale-invariant-tokenizer-pick/
├── data/                      # Data downloading and preprocessing
│   ├── download_data.py      # FineWeb 2 subset download script
│   └── raw/                  # Raw downloaded data
├── tokenizers/               # Tokenizer training
│   ├── train_tokenizer.py   # Tokenizer training script
│   └── configs/             # Tokenizer configurations
├── models/                   # Language model training
│   ├── train_lm.py          # LM training script
│   ├── architectures/       # Model architectures
│   └── configs/             # Model configurations
├── evaluation/               # Evaluation scripts
│   ├── evaluate.py          # Main evaluation script
│   └── metrics.py           # Metrics computation
├── analysis/                 # Analysis and visualization
│   └── correlation_analysis.py  # Rank correlation analysis
├── experiments/              # Experiment configurations
│   └── configs/
├── notebooks/                # Jupyter notebooks for exploration
└── scripts/                  # Utility scripts
```

## Experimental Setup

### Tokenizers
- **Algorithms**: BPE, UnigramLM (byte-level)
- **Vocab Sizes**: 80k, 128k, 256k
- **Training Data**: ~2GB of text from FineWeb 2
- **Implementation**:
  - Hugging Face tokenizers for BPE
  - SentencePiece for UnigramLM
  - Consistent pretokenization across all tokenizers

### Language Models
- **Architecture**: Apertus (scaled down from swiss-ai/Apertus-8B-2509)
- **Model Sizes**: 50M, 100M, 300M, 1B params (3B if feasible)
- **Random Seeds**: 3 seeds per configuration
- **Training Data**: LANG_SET_20 languages from FineWeb 2
- **Training Framework**: Hugging Face Trainer

### Evaluation
- **Primary Metric**: Perplexity per byte
- **Secondary**: Downstream task performance
- **Analysis**: Rank correlation between small and large model performance

## Installation

This project is organized as a Python package called `uniscale`. Install it in editable mode:

```bash
# Clone the repository
git clone https://github.com/yourusername/scale-invariant-tokenizer-pick.git
cd scale-invariant-tokenizer-pick

# Install the package in editable mode
pip install -e .

# Or install with development tools
pip install -e ".[dev]"
```

After installation, you can import the package anywhere:

```python
from uniscale.tokenizers.train_tokenizer import train_bpe_tokenizer
from uniscale.models.train_lm import load_tokenizer
from uniscale.evaluation.metrics import compute_perplexity_per_byte
```

## Quick Start

The entire experimental pipeline is managed through YAML configuration files. Here's the complete workflow:

### 1. Download Data

Download training data from FineWeb 2:

```bash
python -m uniscale.data.download_data \
  --lang_set LANG_SET_20 \
  --total_size_gb 10 \
  --tokenizer_size_gb 2
```

This will create:
- `data/raw/train_data.jsonl` - Full training data (~10GB)
- `data/raw/tokenizer_data.jsonl` - Subset for tokenizer training (~2GB)

### 2. Train Tokenizers

Train all tokenizers based on the YAML configuration:

```bash
python scripts/train_all_tokenizers.py \
  --config experiments/configs/tokenizer_training.yaml
```

Configuration file: [experiments/configs/tokenizer_training.yaml](experiments/configs/tokenizer_training.yaml)

This will train:
- BPE tokenizers with vocab sizes: 80k, 128k, 256k
- UnigramLM tokenizers with vocab sizes: 80k, 128k, 256k

Output: `tokenizers/trained/{algorithm}_v{size}k/`

### 3. Train Language Models

Train all model size and tokenizer combinations:

```bash
python scripts/train_all_models.py \
  --config experiments/configs/model_training.yaml
```

Configuration file: [experiments/configs/model_training.yaml](experiments/configs/model_training.yaml)

This will train models for:
- Model sizes: 50M, 100M, 300M, 1B
- All trained tokenizers (6 total)
- 3 random seeds each
- Total: 72 training runs (4 sizes × 6 tokenizers × 3 seeds)

**Training a specific combination:**

```bash
# Train only 50M models
python scripts/train_all_models.py --model_size 50M

# Train only with a specific tokenizer
python scripts/train_all_models.py --tokenizer tokenizers/trained/bpe_v80k

# Train with specific seed
python scripts/train_all_models.py --seed 42

# Dry run to see what will be trained
python scripts/train_all_models.py --dry_run
```

Output: `models/checkpoints/{model_size}_{tokenizer}_{seed}/`

### 4. Evaluate Models

Evaluate all trained models on test data:

```bash
python scripts/evaluate_all_models.py \
  --eval_file data/raw/test_data.jsonl \
  --output_dir evaluation/results
```

This computes perplexity per byte for all trained models.

**Evaluate specific models:**

```bash
# Evaluate only 50M models
python scripts/evaluate_all_models.py \
  --eval_file data/raw/test_data.jsonl \
  --model_filter 50M
```

Output: `evaluation/results/{model_name}_results.json`

### 5. Analyze Rank Correlation

Compute rank correlation between small and large model performance:

```bash
python -m uniscale.analysis.correlation_analysis \
  --results_dir evaluation/results \
  --output_dir analysis/results
```

This will:
- Compute Spearman and Kendall rank correlations
- Generate correlation plots
- Save summary statistics

Output:
- `analysis/results/correlation_*.png` - Scatter plots
- `analysis/results/correlation_summary.csv` - Summary statistics
- `analysis/results/correlation_summary.json` - JSON results

## Advanced Usage

### Training Individual Components

**Train a single tokenizer:**

```bash
python -m uniscale.tokenizers.train_tokenizer \
  --algorithm bpe \
  --vocab_size 80000 \
  --data_file data/raw/tokenizer_data.jsonl
```

**Train a single model:**

```bash
python -m uniscale.models.train_lm \
  --model_size 50M \
  --tokenizer_path tokenizers/trained/bpe_v80k \
  --train_file data/raw/train_data.jsonl \
  --seed 42 \
  --bf16
```

**Evaluate a single model:**

```bash
python -m uniscale.evaluation.evaluate \
  --model_path models/checkpoints/50M_bpe_v80k_seed42 \
  --eval_file data/raw/test_data.jsonl \
  --output_file evaluation/results/result.json
```

## Configuration

### Tokenizer Training Configuration

Edit [experiments/configs/tokenizer_training.yaml](experiments/configs/tokenizer_training.yaml) to customize:
- Data file path
- Tokenizer algorithms (bpe, unigram)
- Vocabulary sizes
- Training parameters

### Model Training Configuration

Edit [experiments/configs/model_training.yaml](experiments/configs/model_training.yaml) to customize:
- Model sizes
- Tokenizers to use
- Random seeds
- Training hyperparameters (learning rate, batch size, etc.)
- Weights & Biases integration

## Expected Results

The core research question will be answered by analyzing the rank correlation results:

- **High correlation (ρ > 0.8)**: Small models are excellent predictors of tokenizer performance for larger models
- **Moderate correlation (0.5 < ρ < 0.8)**: Small models provide useful guidance but may not always predict the best tokenizer
- **Low correlation (ρ < 0.5)**: Small models are not reliable for tokenizer selection

The analysis will produce:
1. Correlation plots showing performance relationship between model sizes
2. Statistical significance tests (p-values)
3. Rank comparisons across different tokenizers

## Project Structure Details

```
scale-invariant-tokenizer-pick/
├── src/
│   └── uniscale/                     # Main Python package
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── download_data.py   # FineWeb 2 download script
│       ├── tokenizers/
│       │   ├── __init__.py
│       │   └── train_tokenizer.py # Core tokenizer training
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train_lm.py        # LM training with HF Trainer
│       │   └── architectures/
│       │       ├── __init__.py
│       │       └── model_config.py # Apertus config for different sizes
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── evaluate.py        # Evaluation script
│       │   └── metrics.py         # Perplexity per byte computation
│       └── analysis/
│           ├── __init__.py
│           └── correlation_analysis.py # Rank correlation analysis
├── scripts/
│   ├── train_all_tokenizers.py   # Batch tokenizer training
│   ├── train_all_models.py       # Batch model training
│   └── evaluate_all_models.py    # Batch evaluation
├── experiments/
│   └── configs/
│       ├── tokenizer_training.yaml
│       └── model_training.yaml
├── data/
│   └── raw/                       # Downloaded data
├── tokenizers/
│   └── trained/                   # Trained tokenizers (output)
│       ├── bpe_v80k/
│       ├── bpe_v128k/
│       ├── bpe_v256k/
│       ├── unigram_v80k/
│       ├── unigram_v128k/
│       └── unigram_v256k/
├── models/
│   └── checkpoints/               # Trained models (output)
├── evaluation/
│   └── results/                   # Evaluation results (output)
├── analysis/
│   └── results/                   # Plots and statistics (output)
├── notebooks/                     # Jupyter notebooks
├── pyproject.toml                 # Package configuration
├── .gitignore
└── README.md
```

## References

- **Apertus Model**: https://huggingface.co/swiss-ai/Apertus-8B-2509
- **FineWeb 2**: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- **Hugging Face Tokenizers**: https://huggingface.co/docs/tokenizers/
- **SentencePiece**: https://github.com/google/sentencepiece

## Citation

If you use this code for your research, please cite:

```bibtex
@misc{scale-invariant-tokenizer-pick,
  title={Scale-Invariant Tokenizer Selection Using Small Language Models},
  author={Saibo},
  year={2026},
  url={https://github.com/saibo/scale-invariant-tokenizer-pick}
}
```
