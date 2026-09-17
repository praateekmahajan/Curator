#!/bin/bash
set -euo pipefail
source "${WORKTREE:?}/benchmarking/native_vllm/worker-env.sh"
case "${1:?qwen, deepseek, benchmark or collect}" in
  qwen)
    exec vllm serve Qwen/Qwen3.8-27B --host 0.0.0.0 --port 8000 \
      --dtype bfloat16 --tensor-parallel-size 1 --data-parallel-size "${QWEN_DP:-8}" \
      --max-num-seqs "${QWEN_MAX_NUM_SEQS:-256}" \
      --enable-auto-tool-choice --tool-call-parser qwen3_xml \
      --reasoning-parser qwen3 --language-model-only "${@:2}"
    ;;
  deepseek)
    export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_V2_MODEL_RUNNER=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    extra=()
    if [[ ${DEEPSEEK_H100_WORKAROUND:-0} == 1 ]]; then
      export VLLM_ALLREDUCE_USE_SYMM_MEM=0
      extra+=(--disable-custom-all-reduce)
    fi
    exec vllm serve deepseek-ai/DeepSeek-V4.1-Flash --host 0.0.0.0 --port 8000 \
      --tokenizer-mode deepseek_v41 --engram-config '{"cpu_offload":true}' \
      --max-num-seqs "${DEEPSEEK_MAX_NUM_SEQS:-64}" \
      --max-num-batched-tokens 4096 --gpu-memory-utilization 0.92 \
      --tensor-parallel-size "${DEEPSEEK_TP:-8}" \
      --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
      --reasoning-parser deepseek_v41 --language-model-only "${extra[@]}" "${@:2}"
    ;;
  benchmark)
    : "${RESULTS_PATH:?}" "${INPUT_PATH:?}" "${SESSION:?}" "${CONFIG:?}" "${TELEMETRY_DIR:?}"
    export CUDA_VISIBLE_DEVICES=""
    # Run one entry at a time; keep the same SESSION for all three entries.
    entry=${2:?exact YAML entry name required}
    test ! -e "$RESULTS_PATH/$SESSION/$entry"
    for server in "${QWEN_HOST:?}" "${DEEPSEEK_HOST:?}"; do
      curl --fail --silent --show-error --max-time 30 "http://$server:8000/health"
      curl --fail --silent --show-error --max-time 60 --request POST \
        "http://$server:8000/reset_prefix_cache" | \
        python -c 'import json,sys; assert json.load(sys.stdin)["success"], "Prefix cache reset failed"'
    done
    python benchmarking/run.py --config "$CONFIG" --session-name "$SESSION" \
      --entries-exact "$entry" --strict-config-check
    python benchmarking/scripts/collect_native_vllm_gpustats.py \
      "$RESULTS_PATH/$SESSION" --telemetry-dir "$TELEMETRY_DIR" \
      --qwen-node "$QWEN_HOST" --deepseek-node "$DEEPSEEK_HOST"
    ;;
  collect)
    exec python benchmarking/scripts/collect_native_vllm_gpustats.py \
      "${RESULTS_PATH:?}/${SESSION:?}" --telemetry-dir "${TELEMETRY_DIR:?}" \
      --qwen-node "${QWEN_HOST:?}" --deepseek-node "${DEEPSEEK_HOST:?}"
    ;;
  *) echo "Unknown step: $1" >&2; exit 2 ;;
esac
