#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

mkdir -p "/tmp/curator/results/${BRANCH_NAME}"

if [[ "${ENTRY_NAME}" == video_* ]]; then
  bash /opt/Curator/docker/common/install_h264_support.sh --with-libopenh264
  ffmpeg -hide_banner -encoders 2>&1 | grep -w libopenh264
fi

cd /opt/Curator
uv pip install GitPython pynvml pyyaml rich

if [ -n "${NEMO_CI_SESSION_NAME:-}" ]; then
  SESSION_NAME="${NEMO_CI_SESSION_NAME}"
else
  SESSION_NAME="benchmark_run_${CI_PIPELINE_ID}"
fi

python benchmarking/run.py \
  --config /opt/Curator/benchmarking/nightly-benchmark.yaml \
  --config /opt/Curator/benchmarking/test-paths.yaml \
  --session-name "${SESSION_NAME}" \
  --entries "${ENTRY_NAME}"
