#!/bin/bash

set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
    echo "This launcher must run as a Slurm array task." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_DIR="${CURATOR_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
: "${ARRAY_RUNTIME_ROOT:?Set ARRAY_RUNTIME_ROOT to a protected shared directory}"
RUNTIME_ROOT="${ARRAY_RUNTIME_ROOT}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SHORT_RUNTIME="/tmp/et_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/cache" "${RUNTIME_ROOT}/ray"
if [[ ! -e "${SHORT_RUNTIME}" ]]; then
    ln -s "${RUNTIME_ROOT}" "${SHORT_RUNTIME}"
fi

export TMPDIR="${SHORT_RUNTIME}/tmp"
export XDG_CACHE_HOME="${SHORT_RUNTIME}/cache"
export RAY_TMPDIR="${SHORT_RUNTIME}/ray"

SHARD_INDEX_OFFSET="${SHARD_INDEX_OFFSET:-0}"
SHARD_INDEX="${SHARD_INDEX:-$((SLURM_ARRAY_TASK_ID + SHARD_INDEX_OFFSET))}"
TOTAL_SHARDS="${TOTAL_SHARDS:-${SLURM_ARRAY_TASK_COUNT}}"
MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX:-0}"

export NEMO_CURATOR_SLURM_ARRAY_ENABLED=1
export NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX="${SHARD_INDEX}"
export NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS="${TOTAL_SHARDS}"
export NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX}"

exec bash "${CURATOR_DIR}/benchmarking/embedding_generation/embedding_try_step.sh"
