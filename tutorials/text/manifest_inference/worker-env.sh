#!/bin/bash
# Source on a worker; environment creation is a separate preparation step.
set -euo pipefail
: "${WORKTREE:?}" "${BENCHMARK_ROOT:?}" "${SESSION_NAME:?}" "${MODEL_KEY:?}"
: "${INPUT_DIR:?}" "${MANIFEST_PATH:?}" "${OUTPUT_DIR:?}" "${TOTAL_SHARDS:?}"
source "$WORKTREE/.venv/bin/activate"
case "$(command -v python)" in
  "$WORKTREE/.venv/"*) ;;
  *) echo 'Python must resolve inside WORKTREE/.venv' >&2; exit 1 ;;
esac
export PYTHONPATH="$WORKTREE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CHECKPOINT_PATH="$BENCHMARK_ROOT/$SESSION_NAME/.nemo_curator_checkpoint_dir"
physical_index=${SLURM_ARRAY_TASK_ID:?}
export NEMO_CURATOR_SLURM_ARRAY_ENABLED=1
export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="${SHARD_INDEX:-$((physical_index + ${SHARD_INDEX_OFFSET:-0}))}"
export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="$TOTAL_SHARDS"
export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX=0
export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:8000/v1}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"
case "$MODEL_KEY" in
  qwen) export REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-128}" ;;
  deepseek) export REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-80}" ;;
  *) echo 'MODEL_KEY must be qwen or deepseek' >&2; exit 1 ;;
esac
# Inherit shared tool caches unchanged. Only sockets need a short worker-local root.
export RAY_TMPDIR="${RAY_TMPDIR:-${SLURM_TMPDIR:-/tmp}/mi-${SLURM_JOB_ID:?}}"
mkdir -p "$RAY_TMPDIR"
cd "$WORKTREE"
