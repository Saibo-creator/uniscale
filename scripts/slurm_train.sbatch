#!/bin/bash
#SBATCH --job-name=scale-inv-tokenizer
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks-per-node=4          # 4 tasks per node (one per GPU)
#SBATCH --gpus-per-node=4            # 4 GPUs per node
#SBATCH --cpus-per-task=8            # CPUs per task (32 / 4 = 8 per GPU)
#SBATCH --time=11:30:00              # Maximum runtime
# Memory allocation: Let SLURM use default per-GPU memory, or specify like --mem=256G if needed
#SBATCH --partition=normal              # GPU partition name (adjust for your cluster)
#SBATCH --output=logs/slurm-%j.out   # Output log
#SBATCH --error=logs/slurm-%j.err    # Error log

# ============================================================================
# SLURM Multi-Node Distributed Training Script
# ============================================================================
# Usage:
#   sbatch scripts/slurm_train.sbatch [--export=ALL,MODEL_SIZE=50M,...]
#
# Environment variables to customize:
#   MODEL_SIZE       - Model size (e.g., 50M, 100M, 200M)
#   TOKENIZER_PATH   - Path to tokenizer
#   SEED             - Random seed
#   CONFIG_FILE      - Path to training config YAML
#   OUTPUT_DIR       - Base output directory
# ============================================================================

# Print job information
echo "======================================================================"
echo "SLURM Job Information"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_NNODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Current Node: $SLURMD_NODENAME"
echo "======================================================================"

export PYTHONUNBUFFERED=1  # Disable Python output buffering for real-time logs

# Create logs directory
mkdir -p logs

# Set environment variables (if needed)
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0  # Adjust for your network interface

# Get master node information
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "======================================================================"

# From environment variables or use defaults
MODEL_SIZE=${MODEL_SIZE:-"50M"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"out/tokenizers/bpe_v128k"}
SEED=${SEED:-42}
CONFIG_FILE=${CONFIG_FILE:-"experiments/configs/model_training.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"out/models"}
CONTAINER_ENV=${CONTAINER_ENV:-"pytorch_210_env"}  # Enroot environment name

echo "Training Configuration:"
echo "  Model Size: $MODEL_SIZE"
echo "  Tokenizer: $TOKENIZER_PATH"
echo "  Seed: $SEED"
echo "  Config: $CONFIG_FILE"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Container: $CONTAINER_ENV"
echo "======================================================================"

# Activate conda environment (if using)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Or use virtualenv
# source /path/to/venv/bin/activate

# Calculate total GPUs and set WORLD_SIZE
TOTAL_GPUS=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
export WORLD_SIZE=$TOTAL_GPUS
echo "Total GPUs: $TOTAL_GPUS (${SLURM_NNODES} nodes × ${SLURM_GPUS_ON_NODE} GPUs/node)"
echo "World Size: $WORLD_SIZE"
echo "======================================================================"

# Set Python path to pre-installed venv
PYTHON_PATH="/iopsstor/scratch/cscs/saibogeng/venvs/uniscale/bin/python"
echo "Using Python: $PYTHON_PATH"
echo "======================================================================"

# Use srun to directly launch Python training on all GPUs with Enroot container
# srun will launch one process per GPU (SLURM_NNODES × SLURM_GPUS_ON_NODE total processes)
# SLURM automatically provides SLURM_PROCID (global rank) and SLURM_LOCALID (local rank)
echo "Starting distributed training with pure srun (no torchrun)..."
srun --environment=$CONTAINER_ENV \
     --container-mounts=/capstor/store/cscs/swissai/infra01/users/$USER:/capstor/store/cscs/swissai/infra01/users/$USER,/iopsstor/scratch/cscs/$USER:/iopsstor/scratch/cscs/$USER \
     bash -c "
    echo '[Rank \$SLURM_PROCID / LocalRank \$SLURM_LOCALID] Starting training...' && \
    cd ~/uniscale/uniscale_code || { echo '[Rank \$SLURM_PROCID] ERROR: Failed to cd to ~/uniscale/uniscale_code'; exit 1; } && \
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $PYTHON_PATH src/uniscale/models/train_lm.py \
        --model_size $MODEL_SIZE \
        --tokenizer_path $TOKENIZER_PATH \
        --seed $SEED \
        --output_dir ${OUTPUT_DIR}/${MODEL_SIZE}_\$(basename $TOKENIZER_PATH)_seed${SEED} \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-4 \
        --max_length 2048 \
        --save_steps 1000 \
        --logging_steps 100 \
        --bf16 \
        --wandb_project scale-invariant-tokenizer \
        --max_steps 10000
"

echo "======================================================================"
echo "Training completed!"
echo "======================================================================"
