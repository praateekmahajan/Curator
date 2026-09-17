#!/usr/bin/env bash
set -euo pipefail

: "${HF_HOME:?Set HF_HOME to the existing Hugging Face cache}"
: "${SERVING_CACHE:?Set SERVING_CACHE to a persistent cache directory}"
image=${SERVING_IMAGE:-vllm/vllm-openai@sha256:c4392d76e3eec8983fa152651365158cb062e348fd40398963f499d5867b9e28}

case "${1:-}" in
  qwen)
    model=Qwen/Qwen3.8-27B
    devices=${GPUS:-0}
    port=${PORT:-18101}
    args=(--reasoning-parser qwen3 --gpu-memory-utilization 0.85 --max-num-seqs 4)
    ;;
  deepseek)
    model=deepseek-ai/DeepSeek-V4.1-Flash
    devices=${GPUS:-1,2,3,4}
    port=${PORT:-18102}
    args=(--tensor-parallel-size 4 --engram-config '{"cpu_offload":true}'
          --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41
          --tool-call-parser deepseek_v41 --enable-auto-tool-choice
          --gpu-memory-utilization 0.9 --max-num-seqs 1 --max-num-batched-tokens 1024)
    ;;
  *) echo "Usage: $0 qwen|deepseek" >&2; exit 2 ;;
esac

mkdir -p "$SERVING_CACHE"/{cuda,triton,vllm}
exec docker run --rm --init --name "${CONTAINER_NAME:-native-$1-env-smoke}" \
  --gpus "\"device=$devices\"" --shm-size=32g \
  -p "127.0.0.1:$port:8000" \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -e CUDA_CACHE_PATH=/cache/cuda -e TRITON_CACHE_DIR=/cache/triton \
  -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_DEEP_GEMM_WARMUP=skip \
  -v "$HF_HOME:/hf:ro" -v "$SERVING_CACHE:/cache" \
  "$image" "$model" --served-model-name "$model" --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 --enforce-eager --language-model-only "${args[@]}"
