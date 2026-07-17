#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURATOR_DIR="${CURATOR_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PIPELINE_CONFIG="${SCRIPT_DIR}/fuzzy-deduped-qwen3-0p6b.yaml"
PATHS_CONFIG="${SCRIPT_DIR}/paths.yaml"

source "${CURATOR_DIR}/.venv/bin/activate"
PYTHON_PATH="$(command -v python)"
if [[ "${PYTHON_PATH}" != "${CURATOR_DIR}/.venv/bin/python" ]]; then
    echo "Unexpected Python interpreter: ${PYTHON_PATH}" >&2
    exit 2
fi

SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export CUDA_HOME="${SITE_PACKAGES}/nvidia/cu13"
export PATH="${CURATOR_DIR}/.venv/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="${SITE_PACKAGES}/nvidia/cublas/include:${SITE_PACKAGES}/nvidia/cuda_runtime/include:${CUDA_HOME}/include"
export LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${CUDA_HOME}/lib"
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_CACHE_ROOT="${RUNTIME_ROOT}/cache/vllm"
export TRITON_CACHE_DIR="${RUNTIME_ROOT}/cache/triton"
export CUDA_CACHE_PATH="${RUNTIME_ROOT}/cache/cuda"
mkdir -p "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}"

cd "${CURATOR_DIR}"
exec python benchmarking/run.py \
    --config "${PIPELINE_CONFIG}" \
    --config "${PATHS_CONFIG}" \
    --session-name "${SESSION_NAME:-embedding-try}" \
    --strict-config-check
