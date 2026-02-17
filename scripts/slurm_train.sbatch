#!/bin/bash
#SBATCH --job-name=scale-inv-tokenizer
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks-per-node=1          # 1 task per node
#SBATCH --gpus-per-node=4            # 4 GPUs per node
#SBATCH --cpus-per-task=32           # CPUs per task (adjust for your cluster)
#SBATCH --time=11:30:00              # Maximum runtime
#SBATCH --mem=0                      # Use all node memory (or specify like --mem=256G)
#SBATCH --partition=gpu              # GPU partition name (adjust for your cluster)
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
TOKENIZER_PATH=${TOKENIZER_PATH:-"out/tokenizers/unigram_32k"}
SEED=${SEED:-42}
CONFIG_FILE=${CONFIG_FILE:-"experiments/configs/model_training.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"out/models"}

echo "Training Configuration:"
echo "  Model Size: $MODEL_SIZE"
echo "  Tokenizer: $TOKENIZER_PATH"
echo "  Seed: $SEED"
echo "  Config: $CONFIG_FILE"
echo "  Output Dir: $OUTPUT_DIR"
echo "======================================================================"

# Activate conda environment (if using)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# Or use virtualenv
# source /path/to/venv/bin/activate

# Calculate total GPUs
TOTAL_GPUS=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
echo "Total GPUs: $TOTAL_GPUS (${SLURM_NNODES} nodes × ${SLURM_GPUS_ON_NODE} GPUs/node)"
echo "======================================================================"

# Use srun to launch torchrun on all nodes
# Each node will launch a torchrun instance to manage all GPU processes on that node
srun bash -c "
    torchrun \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --nproc_per_node=$SLURM_GPUS_ON_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        src/uniscale/models/train_lm.py \
        --model_size $MODEL_SIZE \
        --tokenizer_path $TOKENIZER_PATH \
        --train_file \$(grep '^train_file:' $CONFIG_FILE | awk '{print \$2}') \
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
