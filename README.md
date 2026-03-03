# Scale-Invariant Tokenizer Selection

## Research Question

Can we use small language models (e.g., 50M-200M params) to choose the tokenizer for our larger language models?

More formally: Does the performance (in terms of perplexity per byte and/or downstream task performance) of small language models trained on a set of different tokenizers follow the same patterns of the performance of larger language models trained on those same tokenizers?

We define "follow the same patterns" as there being a high rank-correlation in the performance of the small models and large models, where our observations are (small model performance, large model performance) for each tokenizer.

## Project Structure

```
scale-invariant-tokenizer-pick/
├── data/                      # Data downloading and preprocessing
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
- **Model Sizes**: 30M, 100M, 300M, 1B params
- **Token Budget**: Chinchilla scaling law — 20 tokens per parameter (derived at runtime from actual param count)
- **Random Seeds**: 3 seeds per configuration
- **Training Data**: LANG_SET_20 languages from FineWeb 2
- **Training Framework**: Hugging Face Trainer + torchrun (DDP)

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

<details>
<summary>Data from Corpus Downloader (old)</summary>

Download training data from FineWeb 2 using the corpus downloader:

```bash
python scripts/corpus_downloader.py \
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

</details>

<details>
<summary>Parquets Data on Clariden (new)</summary>

Inspect and convert parquet files (e.g. from `data/tokenizer_training_dataset`) to JSONL.
Language is taken from each parquet file's stem name. Outputs a 99/1 train/dev split.

```bash
# Inspect
python scripts/process_parquets.py --input_dir data/tokenizer_training_dataset

# Convert all data
python scripts/process_parquets.py \
    --input_dir data/tokenizer_training_dataset \
    --output_dir out/all
```

This will create:
- `out/all/train.jsonl` - Training data (~99%)
- `out/all/dev.jsonl` - Dev data (~1%)
- `out/all/dev.txt` - Dev data in .txt format (for SuperBPE)
- `out/all/train.txt` - Train data in .txt format (for SuperBPE)

</details>

### 2. Train Tokenizers

Train all tokenizers based on the YAML configuration:

```bash
python scripts/train_all_tokenizers.py \
  --config experiments/configs/tokenizer_training.yaml
```

Configuration file: [experiments/configs/tokenizer_training.yaml](experiments/configs/tokenizer_training.yaml)

For superbpe, use:

```bashpython scripts/train_all_tokenizers.py \
  --config experiments/configs/tokenizer_training_superbpe.yaml
```

Output: `out/tokenizers`

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
  --config experiments/configs/model_training_scalinglaw_new.yaml \
  --num_gpus 4
```

Configuration file: [experiments/configs/model_training_scalinglaw_new.yaml](experiments/configs/model_training_scalinglaw_new.yaml)

**Token budget (Chinchilla):** instead of hardcoding steps, the config specifies `tokens_per_param: 20`. At launch, the script calls `estimate_parameters_for_size(model_size, tokenizer_vocab_size)` to get the actual parameter count, then derives:
```
total_tokens = tokens_per_param × param_count
max_steps    = total_tokens / tokens_per_step
```

**Training with multiple GPUs (DDP):**

```bash
# Single GPU
python scripts/train_all_models.py \
  --config experiments/configs/model_training_scalinglaw_new.yaml \
  --num_gpus 1

# Multi-GPU with Distributed Data Parallel (uses torchrun automatically)
python scripts/train_all_models.py \
  --config experiments/configs/model_training_scalinglaw_new.yaml \
  --num_gpus 4

# With torch.compile for ~10-30% throughput gain (adds JIT warmup ~2 min)
python scripts/train_all_models.py \
  --config experiments/configs/model_training_scalinglaw_new.yaml \
  --num_gpus 4 \
  --torch_compile
```

**Note on DDP training:**
- The script automatically uses `torchrun` when `--num_gpus > 1`
- Must be launched via `train_all_models.py` (or `torchrun` directly) — running `train_lm.py` with plain `python` and multiple visible GPUs will trigger `DataParallel` instead, which may cause NCCL errors
- If you encounter NCCL errors, try `NCCL_P2P_DISABLE=1` as a workaround

**Training a specific combination:**

```bash
# Train only 30M models
python scripts/train_all_models.py \
  --config experiments/configs/model_training_scalinglaw_new.yaml \
  --model_size 30M --num_gpus 2

# Train only with a specific tokenizer
python scripts/train_all_models.py \
  --config experiments/configs/model_training_scalinglaw_new.yaml \
  --tokenizer out/tokenizers/bpe_apertus_128k

# Train with a specific seed
python scripts/train_all_models.py \
  --config experiments/configs/model_training_scalinglaw_new.yaml \
  --seed 42

# Dry run to preview commands and step counts
python scripts/train_all_models.py \
  --config experiments/configs/model_training_scalinglaw_new.yaml \
  --dry_run --num_gpus 4
```

**Default output structure:**
- Models are saved to `out/models/{model_size}_{tokenizer}_{seed}/`
- Use `--output_dir` to change the base directory

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
  --model_size 30M \
  --tokenizer_path out/tokenizers/bpe_apertus_128k \
  --train_dir data/tokenizer_training_dataset \
  --max_steps 5000 \
  --seed 42 \
  --bf16

# Multi-GPU with DDP (must use torchrun)
torchrun --nproc_per_node=4 \
  src/uniscale/models/train_lm.py \
  --model_size 30M \
  --tokenizer_path out/tokenizers/bpe_apertus_128k \
  --train_dir data/tokenizer_training_dataset \
  --max_steps 5000 \
  --seed 42 \
  --bf16 \
  --torch_compile
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

Edit [experiments/configs/model_training_scalinglaw_new.yaml](experiments/configs/model_training_scalinglaw_new.yaml) to customize:
- `model_sizes`: which sizes to train (`30M`, `100M`, `300M`, `1B`)
- `tokenizers`: list of tokenizer paths
- `seeds`: random seeds for statistical significance
- `tokens_per_param`: Chinchilla ratio (default `20`); `max_steps` is auto-derived at runtime
- `tokens_per_step`: global batch size in tokens (controls `gradient_accumulation_steps`)
- Other hyperparameters (learning rate, warmup, etc.)
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
│   ├── corpus_downloader.py      # FineWeb 2 data downloader
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
