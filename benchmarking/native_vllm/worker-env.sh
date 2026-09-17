#!/bin/bash
# Source inside the selected worker environment/container.
: "${WORKTREE:?}" "${TASK_ROOT:?}" "${HF_CACHE:?}" "${TOOLS_CACHE:?}"
export HF_HOME=$HF_CACHE HF_HUB_CACHE=$HF_CACHE/hub
export CUDA_CACHE_PATH=$TOOLS_CACHE/native-vllm/cuda
export TRITON_CACHE_DIR=$TOOLS_CACHE/native-vllm/triton
export VLLM_CACHE_ROOT=$TOOLS_CACHE/native-vllm/vllm
export TMPDIR=$TASK_ROOT/runtime/tmp
# Keep Unix-domain sockets below the platform's path-length limit.
export RAY_TMPDIR=/tmp/nvs-${SLURM_JOB_ID:?}
export VLLM_RPC_BASE_PATH=$RAY_TMPDIR/vllm
export VLLM_USE_RUST_FRONTEND=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$WORKTREE${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR" "$VLLM_RPC_BASE_PATH"
cd "$WORKTREE"
