#!/usr/bin/env bash
# =============================================================================
# Submit / status / retry / worker for SLURM array DataDesigner runs.
#
# Usage:
#   submit_array.sh submit          — sbatch a fresh array job (0..ARRAY_SIZE-1)
#   submit_array.sh status          — list completed / missing shards
#   submit_array.sh retry-missing   — resubmit only shards missing _SUCCESS markers
#   submit_array.sh worker          — per-shard srun entrypoint (called by sbatch)
#
# Required env (submit/status/retry):
#   INPUT_PATH, OUTPUT_PATH, DD_CONFIG, and one of MODELS_JSON / MODELS_JSON_FILE
#
# Common knobs (see README for full list):
#   ARRAY_SIZE, MAX_CONCURRENT, NODES, GPUS_PER_NODE, TIME_LIMIT,
#   ACCOUNT, PARTITION, INPUT_FORMAT, OUTPUT_FORMAT, OUTPUT_LAYOUT,
#   SERVER_BACKEND, CONTAINER_IMAGE, USE_CONTAINER
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_DIR="${CURATOR_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "${CURATOR_DIR}/.." && pwd)}"

PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${SCRIPT_DIR}/seeded_sdg_pipeline.py}"
PYTHON_CMD="${PYTHON_CMD:-python}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${WORKSPACE_ROOT}/container-images/nemo_curator_nightly-26_05_04.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre}"
USE_CONTAINER="${USE_CONTAINER:-1}"

JOB_NAME="${JOB_NAME:-curator-slurm-ndd}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
ARRAY_SIZE="${ARRAY_SIZE:-${CURATOR_NUM_SHARDS:-1}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-${ARRAY_SIZE}}"

usage() { sed -n '2,18p' "$0"; exit "${1:-1}"; }

require_env() {
  local missing=()
  for v in INPUT_PATH OUTPUT_PATH DD_CONFIG; do
    [[ -z "${!v:-}" ]] && missing+=("${v}")
  done
  if [[ -z "${MODELS_JSON:-}" && -z "${MODELS_JSON_FILE:-}" ]]; then
    missing+=("MODELS_JSON or MODELS_JSON_FILE")
  fi
  if (( ${#missing[@]} > 0 )); then
    echo "Missing required env var(s): ${missing[*]}" >&2
    exit 2
  fi
}

# ---- marker helpers -------------------------------------------------------

shard_marker() { printf "%s/_SUCCESS/shard_%05d.json" "${OUTPUT_PATH%/}" "$1"; }

collect_status() {
  COMPLETED_SHARDS=(); MISSING_SHARDS=()
  for ((i = 0; i < ARRAY_SIZE; i++)); do
    if [[ -f "$(shard_marker "${i}")" ]]; then COMPLETED_SHARDS+=("${i}"); else MISSING_SHARDS+=("${i}"); fi
  done
}

print_status() {
  require_env
  collect_status
  echo "Output root: ${OUTPUT_PATH}"
  echo "Completed (${#COMPLETED_SHARDS[@]}/${ARRAY_SIZE}): ${COMPLETED_SHARDS[*]:-none}"
  echo "Missing   (${#MISSING_SHARDS[@]}/${ARRAY_SIZE}): ${MISSING_SHARDS[*]:-none}"
}

# ---- sbatch ----------------------------------------------------------------

submit_array() {  # $1: array spec
  require_env
  mkdir -p "${OUTPUT_PATH%/}/_logs"

  export CURATOR_ORIGINAL_ARRAY_SIZE="${ARRAY_SIZE}"
  export INPUT_PATH OUTPUT_PATH DD_CONFIG
  export MODELS_JSON="${MODELS_JSON:-}" MODELS_JSON_FILE="${MODELS_JSON_FILE:-}"
  export INPUT_FORMAT="${INPUT_FORMAT:-jsonl}" OUTPUT_FORMAT="${OUTPUT_FORMAT:-jsonl}" OUTPUT_LAYOUT="${OUTPUT_LAYOUT:-flat}"
  export FILES_PER_PARTITION="${FILES_PER_PARTITION:-}" BLOCKSIZE="${BLOCKSIZE:-}" FIELDS="${FIELDS:-}"
  export PROVIDER_NAME="${PROVIDER_NAME:-local}"
  export SERVER_BACKEND="${SERVER_BACKEND:-dynamo}" SERVE_PORT="${SERVE_PORT:-8000}"
  export HEALTH_CHECK_TIMEOUT_S="${HEALTH_CHECK_TIMEOUT_S:-600}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-}" VLLM_ENGINE_KWARGS_JSON="${VLLM_ENGINE_KWARGS_JSON:-}"
  export RAY_WORKER_CONNECT_TIMEOUT_S="${RAY_WORKER_CONNECT_TIMEOUT_S:-600}"
  export IGNORE_HEAD_NODE_FOR_REPLICAS="${IGNORE_HEAD_NODE_FOR_REPLICAS:-0}" VERBOSE="${VERBOSE:-0}"
  export CURATOR_DIR SCRIPT_DIR PIPELINE_SCRIPT PYTHON_CMD
  export CONTAINER_IMAGE CONTAINER_MOUNTS USE_CONTAINER

  local sbatch_args=(
    "--job-name=${JOB_NAME}" "--nodes=${NODES}" "--ntasks-per-node=1"
    "--gpus-per-node=${GPUS_PER_NODE}" "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}" "--array=$1"
    "--output=${OUTPUT_PATH%/}/_logs/%x-%A_%a.out"
    "--error=${OUTPUT_PATH%/}/_logs/%x-%A_%a.err"
    "--export=ALL,CURATOR_ORIGINAL_ARRAY_SIZE=${ARRAY_SIZE}"
  )
  [[ -n "${ACCOUNT:-}" ]] && sbatch_args+=("--account=${ACCOUNT}")
  [[ -n "${PARTITION:-}" ]] && sbatch_args+=("--partition=${PARTITION}")
  if [[ -n "${SBATCH_EXTRA:-}" ]]; then
    read -r -a extra <<< "${SBATCH_EXTRA}"
    sbatch_args+=("${extra[@]}")
  fi
  echo "Submitting array $1; each element uses ${NODES} node(s)"
  sbatch "${sbatch_args[@]}" "$0" worker
}

submit_new() {
  (( ARRAY_SIZE >= 1 )) || { echo "ARRAY_SIZE must be >= 1" >&2; exit 2; }
  submit_array "0-$((ARRAY_SIZE - 1))%${MAX_CONCURRENT}"
}

retry_missing() {
  require_env
  collect_status
  if (( ${#MISSING_SHARDS[@]} == 0 )); then
    echo "All ${ARRAY_SIZE} shards complete; nothing to retry."
    return 0
  fi
  local spec; spec="$(IFS=, ; echo "${MISSING_SHARDS[*]}")%${MAX_CONCURRENT}"
  echo "Retrying missing shards: ${MISSING_SHARDS[*]}"
  submit_array "${spec}"
}

# ---- worker (called from sbatch) -------------------------------------------

run_worker() {
  [[ -n "${OUTPUT_PATH:-}" ]] || { echo "OUTPUT_PATH must be exported into the worker job" >&2; exit 2; }
  export CURATOR_SHARD_INDEX="${CURATOR_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
  export CURATOR_NUM_SHARDS="${CURATOR_NUM_SHARDS:-${CURATOR_ORIGINAL_ARRAY_SIZE:-${ARRAY_SIZE}}}"
  export RAY_PORT_BROADCAST_DIR="${RAY_PORT_BROADCAST_DIR:-${OUTPUT_PATH%/}/_ray_ports/${SLURM_JOB_ID:-local}_${CURATOR_SHARD_INDEX}}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_curator_${SLURM_JOB_ID:-local}_${CURATOR_SHARD_INDEX}_${SLURM_NODEID:-0}}"
  mkdir -p "${RAY_PORT_BROADCAST_DIR}"

  local srun_args=("--ntasks-per-node=1")
  if [[ "${USE_CONTAINER}" == "1" ]]; then
    srun_args+=("--container-image=${CONTAINER_IMAGE}")
    [[ -n "${CONTAINER_MOUNTS}" ]] && srun_args+=("--container-mounts=${CONTAINER_MOUNTS}")
    srun_args+=("--container-workdir=${CURATOR_DIR}")
  fi

  echo "Worker shard ${CURATOR_SHARD_INDEX}/${CURATOR_NUM_SHARDS} on SLURM job ${SLURM_JOB_ID:-local}"
  srun "${srun_args[@]}" bash -lc \
    'set -euo pipefail; cd "${CURATOR_DIR}"; export PYTHONPATH="${CURATOR_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"; ${PYTHON_CMD} "${PIPELINE_SCRIPT}"'
}

# ---- dispatch -------------------------------------------------------------

case "${1:-}" in
  submit)        submit_new ;;
  status)        print_status ;;
  retry-missing) retry_missing ;;
  worker)        run_worker ;;
  -h|--help|help|"") usage 0 ;;
  *) echo "Unknown mode: $1" >&2; usage 2 ;;
esac
