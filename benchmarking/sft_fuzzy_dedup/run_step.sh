#!/usr/bin/env bash
set -euo pipefail

: "${JOB_ID:?Set JOB_ID to the held allocation ID}"
: "${ENTRY:?Set ENTRY to the exact benchmark entry name}"
: "${SESSION_NAME:?Set SESSION_NAME to the stable benchmark session name}"

WORKTREE=/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/benchmarking/sft_fuzzy_dedup/worktree
RUN_DIR="$WORKTREE/benchmarking/sft_fuzzy_dedup"
LOG_DIR=/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/benchmarking/slurm_logs/$SESSION_NAME

mkdir -p "$LOG_DIR"

exec srun \
  --jobid="$JOB_ID" \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=128 \
  --gres=gpu:8 \
  --output="$LOG_DIR/$ENTRY.out" \
  --error="$LOG_DIR/$ENTRY.err" \
  bash "$RUN_DIR/worker_step.sh" run \
  python benchmarking/run.py \
  --config benchmarking/sft_fuzzy_dedup/fuzzy_deduplication.yaml \
  --entries-exact "$ENTRY" \
  --session-name "$SESSION_NAME"
