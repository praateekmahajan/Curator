#!/bin/bash
#SBATCH --job-name=embedding-try
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gpus-per-node=4
#SBATCH --mem=920G
#SBATCH --time=04:00:00
#SBATCH --exclusive

set -euo pipefail

CURATOR_DIR="${CURATOR_DIR:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}"
export CURATOR_DIR
exec bash "${CURATOR_DIR}/benchmarking/embedding_generation/embedding_try_worker_environment.sh"
