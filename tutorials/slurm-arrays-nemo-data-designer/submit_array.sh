#!/usr/bin/env bash
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
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION:-}"
NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-128}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
ARRAY_SIZE="${ARRAY_SIZE:-${CURATOR_NUM_SHARDS:-1}}"
MAX_CONCURRENT="${MAX_CONCURRENT:-${ARRAY_SIZE}}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

MODE="${1:-submit}"

usage() {
  cat <<'USAGE'
Usage:
  submit_array.sh submit
  submit_array.sh status
  submit_array.sh retry-missing
  submit_array.sh worker

Required for submit/retry/status:
  INPUT_PATH=/path/to/seed/jsonl_or_parquet
  OUTPUT_PATH=/path/to/output_root
  DD_CONFIG=/path/to/data_designer.yaml
  MODEL=/path/or/hf/model        Single-model shorthand, or use MODELS_JSON/MODELS_JSON_FILE.

Common knobs:
  ARRAY_SIZE=32                 Logical shard count H.
  MAX_CONCURRENT=8              SLURM array throttle.
  NODES=4                       Nodes V allocated to each array element.
  GPUS_PER_NODE=8
  TIME_LIMIT=04:00:00
  ACCOUNT=...
  PARTITION=...
  INPUT_FORMAT=jsonl|parquet
  OUTPUT_FORMAT=jsonl|parquet
  OUTPUT_LAYOUT=flat|by_shard
  TP=8
  REPLICAS=auto
  MODELS_JSON='[{"model":"...","served_model_name":"...","tp":1,"alias":"dd_alias"}]'
  MODELS_JSON_FILE=tutorials/slurm-arrays-nemo-data-designer/configs/two_qwen_models.json
  SERVER_BACKEND=dynamo|ray-serve
  CONTAINER_IMAGE=/path/to/nemo_curator_nightly-26_05_04.sqsh
  USE_CONTAINER=0               Run without Pyxis/Enroot container flags.
USAGE
}

require_env() {
  local missing=()
  for name in INPUT_PATH OUTPUT_PATH DD_CONFIG; do
    if [[ -z "${!name:-}" ]]; then
      missing+=("${name}")
    fi
  done
  if [[ -z "${MODEL:-}" && -z "${MODELS_JSON:-}" && -z "${MODELS_JSON_FILE:-}" ]]; then
    missing+=("MODEL or MODELS_JSON or MODELS_JSON_FILE")
  fi
  if (( ${#missing[@]} > 0 )); then
    echo "Missing required env var(s): ${missing[*]}" >&2
    exit 2
  fi
}

shard_marker() {
  local shard_idx="$1"
  printf "%s/_SUCCESS/shard_%05d.json" "${OUTPUT_PATH%/}" "${shard_idx}"
}

collect_status() {
  COMPLETED_SHARDS=()
  MISSING_SHARDS=()
  for ((idx = 0; idx < ARRAY_SIZE; idx++)); do
    if [[ -f "$(shard_marker "${idx}")" ]]; then
      COMPLETED_SHARDS+=("${idx}")
    else
      MISSING_SHARDS+=("${idx}")
    fi
  done
}

print_status() {
  require_env
  collect_status
  echo "Output root: ${OUTPUT_PATH}"
  echo "Completed shards (${#COMPLETED_SHARDS[@]}/${ARRAY_SIZE}): ${COMPLETED_SHARDS[*]:-none}"
  echo "Missing shards (${#MISSING_SHARDS[@]}/${ARRAY_SIZE}): ${MISSING_SHARDS[*]:-none}"
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

submit_array() {
  local array_spec="$1"
  require_env
  mkdir -p "${OUTPUT_PATH%/}/_logs"

  export CURATOR_ORIGINAL_ARRAY_SIZE="${ARRAY_SIZE}"
  export INPUT_PATH OUTPUT_PATH DD_CONFIG
  export MODEL="${MODEL:-}"
  export MODELS_JSON="${MODELS_JSON:-}"
  export MODELS_JSON_FILE="${MODELS_JSON_FILE:-}"
  export INPUT_FORMAT="${INPUT_FORMAT:-jsonl}"
  export OUTPUT_FORMAT="${OUTPUT_FORMAT:-jsonl}"
  export OUTPUT_LAYOUT="${OUTPUT_LAYOUT:-flat}"
  export FILES_PER_PARTITION="${FILES_PER_PARTITION:-}"
  export BLOCKSIZE="${BLOCKSIZE:-}"
  export FIELDS="${FIELDS:-}"
  export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
  export PROVIDER_NAME="${PROVIDER_NAME:-local}"
  export TP="${TP:-1}"
  export REPLICAS="${REPLICAS:-auto}"
  export SERVER_BACKEND="${SERVER_BACKEND:-dynamo}"
  export SERVE_PORT="${SERVE_PORT:-8000}"
  export HEALTH_CHECK_TIMEOUT_S="${HEALTH_CHECK_TIMEOUT_S:-600}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
  export VLLM_ENGINE_KWARGS_JSON="${VLLM_ENGINE_KWARGS_JSON:-}"
  export RAY_WORKER_CONNECT_TIMEOUT_S="${RAY_WORKER_CONNECT_TIMEOUT_S:-600}"
  export IGNORE_HEAD_NODE_FOR_REPLICAS="${IGNORE_HEAD_NODE_FOR_REPLICAS:-0}"
  export VERBOSE="${VERBOSE:-0}"
  export CURATOR_DIR SCRIPT_DIR PIPELINE_SCRIPT PYTHON_CMD
  export CONTAINER_IMAGE CONTAINER_MOUNTS USE_CONTAINER

  local sbatch_args=(
    "--job-name=${JOB_NAME}"
    "--nodes=${NODES}"
    "--ntasks-per-node=1"
    "--gpus-per-node=${GPUS_PER_NODE}"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--time=${TIME_LIMIT}"
    "--array=${array_spec}"
    "--output=${OUTPUT_PATH%/}/_logs/%x-%A_%a.out"
    "--error=${OUTPUT_PATH%/}/_logs/%x-%A_%a.err"
    "--export=ALL,CURATOR_ORIGINAL_ARRAY_SIZE=${ARRAY_SIZE}"
  )
  if [[ -n "${ACCOUNT}" ]]; then
    sbatch_args+=("--account=${ACCOUNT}")
  fi
  if [[ -n "${PARTITION}" ]]; then
    sbatch_args+=("--partition=${PARTITION}")
  fi
  if [[ -n "${SBATCH_EXTRA}" ]]; then
    read -r -a extra_args <<< "${SBATCH_EXTRA}"
    sbatch_args+=("${extra_args[@]}")
  fi

  echo "Submitting array ${array_spec}; each element uses ${NODES} node(s)"
  sbatch "${sbatch_args[@]}" "$0" worker
}

submit_new() {
  if (( ARRAY_SIZE < 1 )); then
    echo "ARRAY_SIZE must be >= 1" >&2
    exit 2
  fi
  submit_array "0-$((ARRAY_SIZE - 1))%${MAX_CONCURRENT}"
}

retry_missing() {
  require_env
  collect_status
  if (( ${#MISSING_SHARDS[@]} == 0 )); then
    echo "All ${ARRAY_SIZE} shards have success markers; nothing to retry."
    return 0
  fi
  local array_spec
  array_spec="$(join_by_comma "${MISSING_SHARDS[@]}")%${MAX_CONCURRENT}"
  echo "Retrying missing shards: ${MISSING_SHARDS[*]}"
  submit_array "${array_spec}"
}

run_worker() {
  if [[ -z "${OUTPUT_PATH:-}" ]]; then
    echo "OUTPUT_PATH must be exported into the worker job" >&2
    exit 2
  fi

  export CURATOR_SHARD_INDEX="${CURATOR_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
  export CURATOR_NUM_SHARDS="${CURATOR_NUM_SHARDS:-${CURATOR_ORIGINAL_ARRAY_SIZE:-${ARRAY_SIZE}}}"
  export RAY_PORT_BROADCAST_DIR="${RAY_PORT_BROADCAST_DIR:-${OUTPUT_PATH%/}/_ray_ports/${SLURM_JOB_ID:-local}_${CURATOR_SHARD_INDEX}}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_curator_${SLURM_JOB_ID:-local}_${CURATOR_SHARD_INDEX}_${SLURM_NODEID:-0}}"
  mkdir -p "${RAY_PORT_BROADCAST_DIR}" "${OUTPUT_PATH%/}/_ray_tmp"

  local worker_command
  worker_command='set -euo pipefail; cd "${CURATOR_DIR}"; export PYTHONPATH="${CURATOR_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"; ${PYTHON_CMD} "${PIPELINE_SCRIPT}"'

  local srun_args=("--ntasks-per-node=1")
  if [[ "${USE_CONTAINER}" == "1" ]]; then
    srun_args+=("--container-image=${CONTAINER_IMAGE}")
    if [[ -n "${CONTAINER_MOUNTS}" ]]; then
      srun_args+=("--container-mounts=${CONTAINER_MOUNTS}")
    fi
    srun_args+=("--container-workdir=${CURATOR_DIR}")
  fi

  echo "Worker shard ${CURATOR_SHARD_INDEX}/${CURATOR_NUM_SHARDS} on SLURM job ${SLURM_JOB_ID:-local}"
  srun "${srun_args[@]}" bash -lc "${worker_command}"
}

case "${MODE}" in
  submit)
    submit_new
    ;;
  status)
    print_status
    ;;
  retry-missing)
    retry_missing
    ;;
  worker)
    run_worker
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
