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

"""Run benchmark entries sequentially, with a fresh Slurm Ray cluster each."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner.session import Session
from runner.utils import (
    assert_valid_config_dict,
    merge_config_files,
    remove_disabled_blocks,
    resolve_env_vars,
)

STATUS_TIMEOUT_S = 60


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--reason", required=True)
    return parser


def _entry_names(config_paths: list[Path]) -> list[str]:
    config = merge_config_files(config_paths)
    assert_valid_config_dict(config)
    config = remove_disabled_blocks(config)
    config = resolve_env_vars(config, strict=True)
    return [entry.name for entry in Session.from_dict(config).entries]


def _wait_for_status(status_path: Path) -> int:
    deadline = time.monotonic() + STATUS_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            return int(status_path.read_text())
        except (FileNotFoundError, ValueError):
            time.sleep(1)
    msg = f"Timed out waiting for entry status at {status_path}"
    raise TimeoutError(msg)


def main() -> int:
    args = _parser().parse_args()
    script_dir = Path(__file__).parent
    node_id = int(os.environ["SLURM_NODEID"])
    entries = _entry_names(args.config)
    if not entries:
        print("No enabled benchmark entries", file=sys.stderr)
        return 2

    status_root = Path(os.environ["RAY_PORT_BROADCAST_DIR"])
    for entry_index, entry_name in enumerate(entries):
        status_path = status_root / f"entry-{entry_index}.status"
        if node_id == 0 and status_path.exists():
            print(f"Refusing to reuse stale status file {status_path}", file=sys.stderr)
            return 2

        env = {**os.environ, "ENTRY_STATUS_PATH": str(status_path)}
        command = [sys.executable, str(script_dir / "slurm_entrypoint.py")]
        for config_path in args.config:
            command.extend(["--config", str(config_path)])
        command.extend(
            [
                "--session-name",
                args.session_name,
                "--entry-name",
                entry_name,
                "--entry-index",
                str(entry_index),
                "--reason",
                args.reason,
            ]
        )
        subprocess.run(command, env=env, check=False)  # noqa: S603
        return_code = _wait_for_status(status_path)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
