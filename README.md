# Scale-Invariant Tokenizer Selection

## Research Question

Can we use small language models (e.g., 50M-200M params) to choose the tokenizer for our larger language models?

More formally: Does the performance (in terms of perplexity per byte and/or downstream task performance) of small language models trained on a set of different tokenizers follow the same patterns of the performance of larger language models trained on those same tokenizers?

We define "follow the same patterns" as there being a high rank-correlation in the performance of the small models and large models, where our observations are (small model performance, large model performance) for each tokenizer.

## Project Structure

```
scale-invariant-tokenizer-pick/
├── data/                      # Data downloading and preprocessing
│   ├── corpus_downloader.py  # FineWeb 2 data downloader
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

### Docker Environment (Recommended)

For a consistent development environment with GPU support, we recommend using:

```bash
docker pull nvcr.io/nvidia/pytorch:25.09-py3
```
This image includes PyTorch 2.10, which is required for this project.

### Package Installation

This project is organized as a Python package called `uniscale`. Clone the repository and install in editable mode:

```bash
# Clone the repository
git clone https://github.com/Saibo-creator/uniscale.git
cd uniscale

# Install in editable mode
pip install -e .
```

This installs the package in editable mode, allowing you to:
- Run scripts from the `scripts/` directory
- Use experiment configurations from `experiments/configs/`
- Import the package anywhere:

```python
from uniscale.tokenizers.train_tokenizer import train_bpe_tokenizer
from uniscale.models.train_lm import load_tokenizer
from uniscale.evaluation.metrics import compute_perplexity_per_byte
```

## Quick Start

The entire experimental pipeline is managed through YAML configuration files. Here's the complete workflow:

### 1. Download Data

Download training data from FineWeb 2 using the corpus downloader:

```bash
python -m uniscale.data.corpus_downloader \
  --dataset fineweb \
  --lang_set L20 \
  --total_size_gb 10 \
  --tokenizer_size_gb 2 \
  --output_dir data/raw
```

This creates per-language text files in:
- `data/raw/tokenizer_corpus/{lang}/train.txt` - Tokenizer training data (~2GB)
- `data/raw/main_corpus/{lang}/{train,val,test}.txt` - Main corpus (~10GB)

Then convert to JSONL format:

```bash
python scripts/convert_corpus_to_jsonl.py
```

This will create:
- `data/raw/tokenizer_data.jsonl` - Tokenizer training data
- `data/raw/train_data.jsonl` - Training data
- `data/raw/val_data.jsonl` - Validation data
- `data/raw/test_data.jsonl` - Test data

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

### 3. Evaluate Tokenizers (Intrinsic Metrics)

Evaluate trained tokenizers with comprehensive intrinsic metrics:

```bash
python scripts/evaluate_tokenizers.py \
  --tokenizers_dir out/tokenizers \
  --eval_data_file data/raw/test_data.jsonl \
  --output_dir out/tokenizer_eval
```

This will compute:
- **Compression Rate**: Text encoding efficiency (bytes/chars per token)
- **Fertility**: Tokens per word/character (granularity measure)
- **Vocabulary Utilization**: Fraction of vocabulary actually used
- **Multilingual Fairness (Gini)**: Cross-language equity (0=perfect fairness, 1=max unfairness)
- **Entropy**: Information-theoretic token distribution analysis

**Key Outputs**:
- `out/tokenizer_eval/analysis_results.json` - Comprehensive metrics
- `out/tokenizer_eval/*.png` - Comparison plots
- `out/tokenizer_eval/latex_tables/` - Publication-ready tables

**Understanding Results**:
- **Lower fertility**: More efficient tokenization
- **Higher compression rate**: Better text compression
- **Lower Gini**: Fairer treatment across languages
- **Higher vocab utilization**: Less wasted vocabulary space

### 4. Train Language Models

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

**Training with multiple GPUs (DDP):**

```bash
# Single GPU (default)
python scripts/train_all_models.py \
  --config experiments/configs/model_training.yaml \
  --num_gpus 1

# Multi-GPU with Distributed Data Parallel
python scripts/train_all_models.py \
  --config experiments/configs/model_training.yaml \
  --num_gpus 2  # or 4, 8, etc.
```

**Note on DDP training:**
- The script automatically uses `torchrun` when `--num_gpus > 1`
- Only the main process (rank 0) logs to wandb and saves checkpoints
- Dataset preparation is done once by the main process, then shared via cache
- If you encounter NCCL shared memory errors in containers, clean up `/dev/shm/nccl-*` files or set `NCCL_SHM_DISABLE=1`

**Training a specific combination:**

```bash
# Train only 50M models
python scripts/train_all_models.py --model_size 50M

# Train only with a specific tokenizer
python scripts/train_all_models.py --tokenizer tokenizers/trained/bpe_v80k

# Train with specific seed
python scripts/train_all_models.py --seed 42

# Combine with multi-GPU
python scripts/train_all_models.py --model_size 50M --num_gpus 2

# Customize output directory
python scripts/train_all_models.py --output_dir experiments/run1

# Dry run to see what will be trained
python scripts/train_all_models.py --dry_run
```

**Default output structure:**
- Models are saved to `out/models/{model_size}_{tokenizer}_{seed}/`
- Use `--output_dir` to change the base directory (e.g., `--output_dir experiments/run1`)

### 5. Evaluate Models

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

### 6. Analyze Rank Correlation

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
# Single GPU
python src/uniscale/models/train_lm.py \
  --model_size 50M \
  --tokenizer_path out/tokenizers/bpe_v80k \
  --train_file data/raw/train_data.jsonl \
  --eval_file data/raw/val_data.jsonl \
  --seed 42 \
  --bf16

# Multi-GPU with DDP (use torchrun)
torchrun --nproc_per_node=2 \
  src/uniscale/models/train_lm.py \
  --model_size 50M \
  --tokenizer_path out/tokenizers/bpe_v80k \
  --train_file data/raw/train_data.jsonl \
  --eval_file data/raw/val_data.jsonl \
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
│       │   ├── corpus_downloader.py  # FineWeb 2 data downloader
│       │   └── __init__.py
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
