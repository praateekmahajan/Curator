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

"""Run one benchmark entry in a fresh multi-node Ray cluster."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner.gpu_stats_recorder import GPUStatsRecorder

from nemo_curator.core.client import SlurmRayClient


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--entry-name", required=True)
    parser.add_argument("--entry-index", type=int, required=True)
    parser.add_argument("--reason", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    curator_dir = Path(os.environ["CURATOR_DIR"])
    node_id = int(os.environ["SLURM_NODEID"])
    entry_results_dir = Path(os.environ["BENCHMARK_SESSION_ROOT"]) / args.entry_name
    logs_dir = Path(os.environ["SESSION_LOG_ROOT"]) / args.entry_name
    ray_temp_dir = Path(os.environ["RAY_TMPDIR"]) / f"entry-{args.entry_index}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    entry_results_dir.mkdir(parents=True, exist_ok=True)
    ray_temp_dir.mkdir(parents=True, exist_ok=True)

    client = SlurmRayClient(
        ray_temp_dir=str(ray_temp_dir),
        include_dashboard=False,
        num_cpus=int(os.environ["SLURM_CPUS_ON_NODE"]),
        num_gpus=int(os.environ["SLURM_GPUS_ON_NODE"]),
        object_store_memory=int(os.environ["RAY_OBJECT_STORE_BYTES"]),
        ray_stdouterr_capture_file=str(logs_dir / f"ray-node-{node_id}.log"),
    )

    return_code = 1
    try:
        with GPUStatsRecorder(entry_results_dir / f"gpustats-node-{node_id}.csv"):
            client.start()
            command = [sys.executable, str(curator_dir / "benchmarking/run.py")]
            for config_path in args.config:
                command.extend(["--config", str(config_path)])
            command.extend(
                [
                    "--session-name",
                    args.session_name,
                    "--entries-exact",
                    args.entry_name,
                    "--reason",
                    args.reason,
                    "--strict-config-check",
                ]
            )
            return_code = subprocess.run(command, cwd=curator_dir, check=False).returncode  # noqa: S603
    finally:
        if node_id == 0:
            status_path = Path(os.environ["ENTRY_STATUS_PATH"])
            status_tmp = status_path.with_suffix(".tmp")
            status_tmp.write_text(str(return_code))
            status_tmp.replace(status_path)
        client.stop()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
