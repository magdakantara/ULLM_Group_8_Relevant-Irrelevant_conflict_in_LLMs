#!/bin/bash
# Source this file: source snellius/activate_env.sh
# Reuse the environment already installed in the original Snellius project.
module load 2023 || return 1
module load Python/3.11.3-GCCcore-12.3.0 || return 1
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
elif [[ -f "$HOME/contract_conflict_project/.venv/bin/activate" ]]; then
    source "$HOME/contract_conflict_project/.venv/bin/activate"
else
    echo "No environment found. See README.md for first-time setup."
    return 1
fi
