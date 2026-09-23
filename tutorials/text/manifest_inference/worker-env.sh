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
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$BENCHMARK_ROOT/$SESSION_NAME/.nemo_curator_checkpoint_dir}"
export NEMO_CURATOR_SLURM_ARRAY_ENABLED="${NEMO_CURATOR_SLURM_ARRAY_ENABLED:-1}"
case "${NEMO_CURATOR_SLURM_ARRAY_ENABLED,,}" in
  1|true|yes|on)
    physical_index=${SLURM_ARRAY_TASK_ID:?}
    export ENTRY_INDEX="$physical_index"
    export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="${SHARD_INDEX:-$((physical_index + ${SHARD_INDEX_OFFSET:-0}))}"
    export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="$TOTAL_SHARDS"
    export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX=0
    ;;
  0|false|no|off)
    export ENTRY_INDEX="${ENTRY_INDEX:-bundle}"
    unset NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX
    unset NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS
    unset NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX
    ;;
  *)
    echo 'NEMO_CURATOR_SLURM_ARRAY_ENABLED must be a true or false value' >&2
    exit 1
    ;;
esac
restart_count=${SLURM_RESTART_COUNT:-0}
if [[ ! "$restart_count" =~ ^[0-9]+$ ]]; then
  echo 'SLURM_RESTART_COUNT must be a non-negative integer' >&2
  exit 1
fi
default_entry="${MODEL_KEY}_${ENTRY_INDEX}_${SLURM_JOB_ID:?}"
if (( restart_count > 0 )); then
  default_entry="${default_entry}_restart_${restart_count}"
fi
export ENTRY="$default_entry"
export MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:8000/v1}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="${no_proxy:-},127.0.0.1,localhost"
case "$MODEL_KEY" in
  qwen) export REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-128}" ;;
  deepseek) export REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY:-80}" ;;
  *) echo 'MODEL_KEY must be qwen or deepseek' >&2; exit 1 ;;
esac
# Inherit shared tool caches unchanged. Only sockets need a short worker-local root.
export RAY_TMPDIR="${RAY_TMPDIR:-${SLURM_TMPDIR:-/tmp}/mi-${SLURM_JOB_ID:?}-${ENTRY_INDEX}-r${SLURM_RESTART_COUNT:-0}}"
mkdir -p "$RAY_TMPDIR"
cd "$WORKTREE"
