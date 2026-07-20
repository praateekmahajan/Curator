# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Run configured entries sequentially with a fresh multi-node Ray cluster."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner.session import Session
from runner.slurm_launcher import allocated_slurm_num_nodes
from runner.utils import merge_config_files, remove_disabled_blocks, resolve_env_vars

STATUS_TIMEOUT_S = 120
TEARDOWN_TIMEOUT_S = 180


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--entries")
    parser.add_argument("--entries-exact")
    parser.add_argument("--strict-config-check", action="store_true")
    return parser.parse_known_args(argv)[0]


def _config(args: argparse.Namespace) -> dict:
    return resolve_env_vars(
        remove_disabled_blocks(merge_config_files(args.config)),
        strict=args.strict_config_check,
    )


def _session(args: argparse.Namespace, config: dict) -> Session:
    exact = None
    if args.entries_exact:
        exact = [name.strip() for name in args.entries_exact.split(",") if name.strip()]
    return Session.from_dict(config, entry_filter_expr=args.entries, entries_exact=exact)


def _wait_for_status(status_path: Path) -> int:
    deadline = time.monotonic() + STATUS_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            return int(status_path.read_text())
        except (FileNotFoundError, ValueError):
            time.sleep(1)
    msg = f"Timed out waiting for entry status at {status_path}"
    raise TimeoutError(msg)


def _wait_for_teardown(logs_dir: Path, num_nodes: int) -> None:
    deadline = time.monotonic() + TEARDOWN_TIMEOUT_S
    while time.monotonic() < deadline:
        if len(list(logs_dir.glob("teardown-node-*.done"))) == num_nodes:
            return
        time.sleep(1)
    msg = f"Timed out waiting for {num_nodes} node teardown markers under {logs_dir}"
    raise TimeoutError(msg)


def main() -> int:
    raw_args = sys.argv[1:]
    args = _args(raw_args)
    config = _config(args)
    session = _session(args, config)
    gpu_stats_interval_s = float(config.get("gpu_stats_recorder", {}).get("interval_s", 1.0))
    node_id = int(os.environ["SLURM_NODEID"])
    num_nodes = allocated_slurm_num_nodes()
    script = Path(__file__).with_name("slurm_entrypoint.py")

    for entry in session.entries:
        session_entry = session.results_path / args.session_name / entry.name
        logs_dir = session_entry / "logs"
        status_path = logs_dir / "entry.status"
        logs_dir.mkdir(parents=True, exist_ok=True)
        if node_id == 0 and status_path.exists():
            print(f"Refusing to reuse stale status file {status_path}", file=sys.stderr)
            return 2

        object_store_memory = None if entry.object_store_size == "default" else entry.object_store_size
        command = [
            sys.executable,
            str(script),
            "--entry-name",
            entry.name,
            "--session-entry-path",
            str(session_entry),
            "--logs-dir",
            str(logs_dir),
            "--status-path",
            str(status_path),
            "--num-cpus",
            str(entry.ray.get("num_cpus", 1)),
            "--num-gpus",
            str(entry.ray.get("num_gpus", 0)),
            "--object-store-memory",
            json.dumps(object_store_memory),
            "--gpu-stats-interval-s",
            str(gpu_stats_interval_s),
            "--",
            *raw_args,
        ]
        with (logs_dir / f"launcher-node-{node_id}.log").open("a") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)  # noqa: S603
        return_code = _wait_for_status(status_path)
        if node_id == 0:
            _wait_for_teardown(logs_dir, num_nodes)
            if entry.delete_scratch:
                shutil.rmtree(session_entry / "scratch", ignore_errors=True)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
