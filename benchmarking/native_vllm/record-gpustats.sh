#!/bin/bash
# Run one task on each serving node for the lifetime of the benchmarks.
set -euo pipefail
output=${TELEMETRY_DIR:?}/$(hostname -s)/gpustats.csv
mkdir -p "$(dirname "$output")"
test ! -e "$output"  # Preserve previous recordings on retries.
export TZ=UTC
exec nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
  --format=csv --loop=1 --filename="$output"
