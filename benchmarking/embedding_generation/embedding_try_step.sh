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
CUDA_RUNTIME_LIB="${SITE_PACKAGES}/nvidia/cu13/lib"
if [[ -d "${CUDA_RUNTIME_LIB}" ]]; then
    export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}:${LD_LIBRARY_PATH:-}"
fi

cd "${CURATOR_DIR}"
exec python benchmarking/run.py \
    --config "${PIPELINE_CONFIG}" \
    --config "${PATHS_CONFIG}" \
    --session-name "${SESSION_NAME:-embedding-try}" \
    --strict-config-check
