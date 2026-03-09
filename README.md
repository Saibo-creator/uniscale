# Uniscale


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

### Clariden (CSCS SLURM Cluster) Setup

Step 1 (Enroot Container Configuration):

Create the following Enroot container configuration file to use the NVIDIA PyTorch 25.12 image with GPU support and appropriate mounts for your home, store, and scratch directories: 
Name the file `pytorch_210_env.toml` and place it in `~/.edf/` on the cluster.

```toml
image = "nvcr.io#nvidia/pytorch:25.12-py3"

# "src_path:trg_path" mounts the src_path on the host inside the container at the trg_path.
mounts = [
    "/users/<USERNAME>:/users/<USERNAME>",
    "/capstor/store/cscs/swissai/infra01/users/<USERNAME>:/capstor/store/cscs/swissai/infra01/users/<USERNAME>",
    "/iopsstor/scratch/cscs/<USERNAME>:/iopsstor/scratch/cscs/<USERNAME>"
]
# The initial directory in the container.
workdir = "/users/<USERNAME>"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
```

Step 2 (Dependency Installation):

On Clariden, myself decided to mange Python dependencies via a **persistent venv that lives on the scratch filesystem** (`/iopsstor/`). The venv must be created **inside the Enroot container** (`pytorch_210_env`), because:

1. The container provides the correct Python version (3.12) and GPU/CUDA libraries.
2. The login node has a different (older) system Python — a venv created there is incompatible with the compute nodes.
3. Once created, the venv is reused by every SLURM job without reinstalling anything, keeping job startup fast.

In order to set this up, run the following commands to have one interactive job to create the venv and install the project dependencies. This only needs to be done once — after that, the venv will persist on the scratch storage and be available for all future jobs.

```bash
# 1. Start an interactive job inside the container, mounting scratch storage
srun --container-writable \
  --time=1:30:00 \
  --account=a139 \
  --partition=normal \
  --environment=pytorch_210_env \
  --container-mounts=/capstor/store/cscs/swissai/a139:/capstor/store/cscs/swissai/a139 \
  --pty bash

# 2. Inside the container: create the venv on scratch (persists after job ends)
python3 -m venv --system-site-packages /iopsstor/scratch/cscs/$USER/venvs/uniscale

# 3. Install the project and all dependencies in editable mode
cd ~/uniscale/uniscale_code   # path to this repo on the cluster
/iopsstor/scratch/cscs/$USER/venvs/uniscale/bin/pip install -e .

# 4. Exit the interactive job
exit
```

<!-- After this, `scripts/slurm_train.sbatch` will use this venv automatically via the hardcoded `PYTHON_PATH`. -->

In the future, we will simply activate this venv in the SLURM job script (`scripts/slurm_train.sbatch`) to ensure all dependencies are available for training:



Step 3 (Interative Mode Training):

This mode limites the training to a single node (4 GPUs) and is useful for testing and debugging. 

Start an interactive job with the same configuration as above indicated, and once in the container, run the training script:

```bash
cd uniscale/uniscale_code
/iopsstor/scratch/cscs/saibogeng/venvs/uniscale/bin/python scripts/train_all_models.py   --config experiments/configs/model_training_scalinglaw_new.yaml  --num_gpus 4
```

Configuration file: [experiments/configs/model_training_scalinglaw_new.yaml](experiments/configs/model_training_scalinglaw_new.yaml)



Step 4 (Distributed Training with SLURM):

This mode allows training across multiple nodes and is suitable for large-scale experiments. The SLURM job script (`scripts/slurm_train.sbatch`) is already configured to use the Enroot container and the persistent venv. To submit a training job, simply run:
```bash
sbatch --nodes=2 --account=a139 -p normal  --time=01:30:00  scripts/slurm_train.sbatch
```

You can adjust the `--nodes` and `--time` parameters as needed for your specific training run. The job script will handle the rest, including activating the venv and launching the distributed training across the allocated GPUs.