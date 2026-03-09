#!/bin/bash
#SBATCH --job-name=scale-inv-tokenizer
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks-per-node=4          # 4 tasks per node (one per GPU)
#SBATCH --gpus-per-node=4            # 4 GPUs per node
#SBATCH --cpus-per-task=8            # CPUs per task (32 / 4 = 8 per GPU)
#SBATCH --time=11:30:00              # Maximum runtime
#SBATCH --partition=normal
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ============================================================================
# SLURM Multi-Node Distributed Training Script
# One SLURM job = one (model_size, tokenizer, seed) combination.
#
# Usage (must be run from the repo root directory):
#   sbatch --export=ALL scripts/slurm_train.sbatch
#   sbatch --export=ALL,MODEL_SIZE=100M,TOKENIZER_PATH=/capstor/store/cscs/swissai/a139/saibo_uniscale/out/tokenizers/unigram_apertus_128k scripts/slurm_train.sbatch
#
# Overridable environment variables:
#   MODEL_SIZE        - e.g. 30M, 100M, 300M
#   TOKENIZER_PATH    - path to tokenizer directory
#   SEED              - random seed
#   OUTPUT_DIR        - base output directory
#   CONTAINER_ENV     - Enroot container environment name
# ============================================================================

# ── Job info ────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "SLURM Job Information"
echo "======================================================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_NNODES"
echo "Node List:      $SLURM_JOB_NODELIST"
echo "GPUs per node:  $SLURM_GPUS_ON_NODE"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "======================================================================"

export PYTHONUNBUFFERED=1
mkdir -p logs

# ── Networking ──────────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# ── Experiment identity (override via --export) ──────────────────────────────
MODEL_SIZE=${MODEL_SIZE:-"30M"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/capstor/store/cscs/swissai/a139/saibo_uniscale/out/tokenizers/bpe_apertus_128k"}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-"out/experiments"}
CONTAINER_ENV=${CONTAINER_ENV:-"pytorch_210_env"}

# ── Hyperparameters (from experiments/configs/model_training_scalinglaw_new.yaml) ──
TRAIN_DIR="/capstor/store/cscs/swissai/a139/datasets/tokenizer_training/downstream_lm_training_data"
EVAL_DOCS=3000
MAX_LENGTH=2048
PER_DEVICE_BATCH=32
TOKENS_PER_STEP=262144   # target tokens consumed per optimizer step across all GPUs
TOKENS_PER_PARAM=20      # Chinchilla ratio
LEARNING_RATE="5e-4"
WARMUP_RATIO="0.01"
SAVE_STEPS=100
EVAL_STEPS=100
LOGGING_STEPS=10
WANDB_PROJECT="uniscale"

# ── Derived: total GPUs and gradient_accumulation_steps ─────────────────────
TOTAL_GPUS=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
export WORLD_SIZE=$TOTAL_GPUS
GRAD_ACCUM=$((TOKENS_PER_STEP / (PER_DEVICE_BATCH * MAX_LENGTH * TOTAL_GPUS)))
GRAD_ACCUM=$(( GRAD_ACCUM < 1 ? 1 : GRAD_ACCUM ))
ACTUAL_TPS=$((PER_DEVICE_BATCH * MAX_LENGTH * TOTAL_GPUS * GRAD_ACCUM))
WARMUP_RATIO="0.01"   # 1% of max_steps; handled natively by HF TrainingArguments

echo "Total GPUs:               $TOTAL_GPUS (${SLURM_NNODES} nodes × ${SLURM_GPUS_ON_NODE} GPUs/node)"
echo "gradient_accumulation:    $GRAD_ACCUM  (actual tokens/step = $ACTUAL_TPS)"
echo "tokens_per_param:         $TOKENS_PER_PARAM  (max_steps derived inside train_lm.py)"
echo "Model:                    $MODEL_SIZE"
echo "Tokenizer:                $TOKENIZER_PATH"
echo "Seed:                     $SEED"
echo "Output:                   $OUTPUT_DIR"
echo "======================================================================"


# Set Python path to pre-installed venv
PYTHON_PATH="/iopsstor/scratch/cscs/saibogeng/venvs/uniscale/bin/python"
echo "Using Python: $PYTHON_PATH"

# ── Training ─────────────────────────────────────────────────────────────────
echo "Starting distributed training..."
srun --environment=$CONTAINER_ENV \
     --container-mounts=/capstor/store/cscs/swissai/a139:/capstor/store/cscs/swissai/a139 \
     bash -c "
    echo '[Rank \$SLURM_PROCID / LocalRank \$SLURM_LOCALID] Starting...' && \
    cd $SLURM_SUBMIT_DIR || { echo 'ERROR: cd to $SLURM_SUBMIT_DIR failed'; exit 1; } && \
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $PYTHON_PATH src/uniscale/models/train_lm.py \
        --model_size $MODEL_SIZE \
        --tokenizer_path $TOKENIZER_PATH \
        --train_dir $TRAIN_DIR \
        --eval_docs $EVAL_DOCS \
        --seed $SEED \
        --output_dir ${OUTPUT_DIR}/${MODEL_SIZE}_\$(basename $TOKENIZER_PATH)_seed${SEED} \
        --per_device_train_batch_size $PER_DEVICE_BATCH \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --learning_rate $LEARNING_RATE \
        --max_length $MAX_LENGTH \
        --tokens_per_param $TOKENS_PER_PARAM \
        --warmup_ratio $WARMUP_RATIO \
        --save_steps $SAVE_STEPS \
        --eval_steps $EVAL_STEPS \
        --logging_steps $LOGGING_STEPS \
        --bf16 \
        --wandb_project $WANDB_PROJECT \
        --run_name ${MODEL_SIZE}_\$(basename $TOKENIZER_PATH)_seed${SEED}
"

echo "======================================================================"
echo "Training completed!"
echo "======================================================================"
