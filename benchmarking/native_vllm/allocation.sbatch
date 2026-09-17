#!/bin/bash
# Supply partition, account, resources, time and log path to sbatch (see README).
set -euo pipefail
printf 'job=%s nodes=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST"
while true; do sleep 300; done
