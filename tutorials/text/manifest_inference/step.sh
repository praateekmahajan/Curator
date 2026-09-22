#!/bin/bash
# Run inside a held allocation for validation, then unchanged from array.sbatch.
set -euo pipefail
source "${WORKTREE:?}/tutorials/text/manifest_inference/worker-env.sh"
SERVER_SCRIPT="${SERVER_SCRIPT:-$WORKTREE/tutorials/text/manifest_inference/serve.sh}"
entry="${MODEL_KEY}_${SLURM_ARRAY_TASK_ID:?}_${SLURM_ARRAY_JOB_ID:?}"
attempt="$BENCHMARK_ROOT/$SESSION_NAME/$entry/logs/restart_${SLURM_RESTART_COUNT:-0}"
mkdir -p "$attempt"
export SERVER_CLI_FILE="$attempt/server-cli.json"
if [[ -e "$attempt/worker-environment.txt" ]]; then
  echo "Attempt already exists: $attempt; use a new submission or restart count" >&2
  exit 1
fi
{
  hostname
  command -v python
  python --version
  git rev-parse HEAD
  git diff --stat
  printf 'model=%s\ninput=%s\nmanifest=%s\noutput=%s\ncheckpoint=%s\nshard=%s/%s\nserver=%s\n' \
    "$MODEL_KEY" "$INPUT_DIR" "$MANIFEST_PATH" "$OUTPUT_DIR" "$CHECKPOINT_PATH" \
    "$NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX" "$TOTAL_SHARDS" "$SERVER_SCRIPT"
} > "$attempt/worker-environment.txt"
if curl --fail --silent --max-time 3 "${MODEL_ENDPOINT%/v1}/health" > /dev/null; then
  echo 'A server is already healthy at MODEL_ENDPOINT; refusing to start a second server' >&2
  exit 1
fi
bash "$SERVER_SCRIPT" > "$attempt/server.out" 2> "$attempt/server.err" &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
deadline=$((SECONDS + ${SETUP_TIMEOUT_SECONDS:-1800}))
until curl --fail --silent --max-time 3 "${MODEL_ENDPOINT%/v1}/health" > /dev/null; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Server exited; inspect $attempt/server.err" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo 'Server readiness budget exhausted' >&2
    exit 1
  fi
  sleep 5
done
python benchmarking/run.py --config benchmarking/manifest-inference.yaml \
  --session-name "$SESSION_NAME" --strict-config-check
