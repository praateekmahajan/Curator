#!/bin/bash
# Reference four-GB300 server profiles; run in your validated vLLM environment.
set -euo pipefail
: "${MODEL_KEY:?qwen or deepseek}"
export VLLM_USE_RUST_FRONTEND=1

serve_qwen() {
  local model=$1
  shift
  # Each Qwen engine owns exactly one GPU (TP1), so keep its worker in-process.
  # This also avoids the nightly's ShmRingBuffer race in the redundant worker
  # subprocess when several DP engines finish loading at different times.
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  exec vllm serve "$model" --host 0.0.0.0 --port 8000 \
    --language-model-only --tensor-parallel-size 1 \
    --data-parallel-size "${QWEN_DP:-4}" \
    --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION:-0.95}" \
    --max-model-len "${QWEN_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs "${QWEN_MAX_NUM_SEQS:-1024}" \
    --max-num-batched-tokens "${QWEN_MAX_NUM_BATCHED_TOKENS:-8192}" \
    --max-cudagraph-capture-size "${QWEN_MAX_CUDAGRAPH_CAPTURE_SIZE:-8192}" \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,528,544,560,576,592,608,624,640,656,672,688,704,720,736,752,768,784,800,816,832,848,864,880,896,912,928,944,960,976,992,1008,1024,1536,2048,3072,4096,6144,8192]}' \
    --enable-chunked-prefill --no-enable-prefix-caching \
    --reasoning-parser qwen3 --seed 42 "$@"
}

serve_deepseek() {
  local model=$1
  shift
  local compilation_config=${DEEPSEEK_COMPILATION_CONFIG:-'{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,768,1024,1536,2048]}'}
  export VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_V2_MODEL_RUNNER=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec vllm serve "$model" --host 0.0.0.0 --port 8000 \
    --language-model-only --tokenizer-mode deepseek_v41 \
    --tensor-parallel-size "${DEEPSEEK_TP:-4}" \
    --data-parallel-size "${DEEPSEEK_DP:-1}" \
    --engram-config '{"cpu_offload":true}' \
    --gpu-memory-utilization "${DEEPSEEK_GPU_MEMORY_UTILIZATION:-0.95}" \
    --max-model-len "${DEEPSEEK_MAX_MODEL_LEN:-32768}" \
    --max-num-seqs "${DEEPSEEK_MAX_NUM_SEQS:-512}" \
    --max-num-batched-tokens "${DEEPSEEK_MAX_NUM_BATCHED_TOKENS:-2048}" \
    --max-cudagraph-capture-size "${DEEPSEEK_MAX_CUDAGRAPH_CAPTURE_SIZE:-2048}" \
    --compilation-config "$compilation_config" \
    --enable-chunked-prefill --no-enable-prefix-caching \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
    --reasoning-parser deepseek_v41 "$@"
}

case "$MODEL_KEY" in
  qwen)
    serve_qwen nvidia/Qwen3.8-27B-NVFP4 --load-format fastsafetensors "$@"
    ;;
  deepseek)
    export DEEPSEEK_DP=2 DEEPSEEK_TP=2
    export DEEPSEEK_MAX_NUM_SEQS=2048
    export DEEPSEEK_MAX_NUM_BATCHED_TOKENS=4096
    export DEEPSEEK_MAX_CUDAGRAPH_CAPTURE_SIZE=4096
    export DEEPSEEK_COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,768,1024,1280,1536,1792,2048,2560,3072,3584,4096]}'
    serve_deepseek deepseek-ai/DeepSeek-V4.1-Flash --enable-expert-parallel "$@"
    ;;
  *) echo "MODEL_KEY must be qwen or deepseek" >&2; exit 2 ;;
esac
