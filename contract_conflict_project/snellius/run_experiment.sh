#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=snellius-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source snellius/activate_env.sh
# Pass the same arguments as src/run_model.py; no dataset subset is hardcoded.
# Example: sbatch snellius/run_experiment.sh --prompt baseline --owner D --contract Driven --limit 6 --out outputs/driven_baseline_v3
python src/run_model.py "$@"
