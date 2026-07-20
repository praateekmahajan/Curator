#!/bin/bash
#SBATCH --job-name=semdedup-kmeans-dev
#SBATCH --account=nemotron_n4_pre
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --gpus-per-node=4
#SBATCH --mem=920G
#SBATCH --time=04:00:00
#SBATCH --exclusive

set -euo pipefail

echo "Held allocation ${SLURM_JOB_ID} ready on ${SLURM_JOB_NODELIST}"
while true; do
    sleep 60
done
