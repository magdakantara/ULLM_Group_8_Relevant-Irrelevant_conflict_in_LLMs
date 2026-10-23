#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=snellius-70b-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source snellius/activate_env.sh
# 70B weights (~140 GB) live on scratch, not in the home quota.
export HF_HOME="/scratch-shared/$USER/hf_cache"
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/run_model_long_native.py "$@"
