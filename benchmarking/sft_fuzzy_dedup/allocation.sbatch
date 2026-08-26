#!/usr/bin/env bash
#SBATCH --job-name=sft-fuzzy-dedup
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/benchmarking/slurm_logs/allocation-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/benchmarking/slurm_logs/allocation-%j.err

set -euo pipefail

echo "allocation_job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
date --iso-8601=seconds
nvidia-smi -L

sleep 14400 &
wait
