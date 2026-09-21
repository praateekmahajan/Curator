#!/bin/bash
# Supply cluster account, partition, QOS, CPUs, memory, array range and log paths
# on the sbatch command line. This example assumes a four-GPU node.
#SBATCH --job-name=manifest-inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:30:00
set -euo pipefail
: "${WORKTREE:?}"
srun --ntasks=1 /usr/bin/env bash "$WORKTREE/tutorials/text/manifest_inference/step.sh"
