#!/bin/bash
# Long-context (131k) run for an 8B model with native 128k context, on one A100 (same GPU as the Qwen runs).
# Example: sbatch snellius/run_long_native.sh --config config/experiment_long_native.llama8b.json --prompt baseline --all --out outputs/llama8b_long_baseline
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=snellius-long-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source snellius/activate_env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/run_model_long_native.py "$@"
