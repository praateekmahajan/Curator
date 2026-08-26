#!/usr/bin/env bash
set -u

echo "host=$(hostname)"
echo "PATH=$PATH"
echo "CUDA_HOME=${CUDA_HOME-UNSET}"
echo "CUDA_PATH=${CUDA_PATH-UNSET}"
command -v nvcc || true
nvcc --version 2>&1 || true
command -v nvidia-smi || true
nvidia-smi -L 2>&1 || true
command -v module || true
module list 2>&1 || true
module avail cuda 2>&1 | head -80 || true
for candidate in /usr/local/cuda/include/cuda.h /usr/local/cuda-*/include/cuda.h /cm/shared/apps/cuda/*/include/cuda.h /cm/local/apps/cuda/*/include/cuda.h; do
  if [ -f "$candidate" ]; then
    echo "cuda_header=$candidate"
  fi
done
ldconfig -p 2>/dev/null | grep -E 'libcuda|libcudart' || true
