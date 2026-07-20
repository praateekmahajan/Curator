# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Launch the generic benchmark runner once per node in a Slurm allocation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from runner.session import Session
from runner.utils import merge_config_files, remove_disabled_blocks, resolve_env_vars

from nemo_curator.core.client import _parse_slurm_nodelist


def allocated_slurm_num_nodes() -> int:
    """Return the parent allocation size, even when called inside a smaller Slurm step."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if nodelist:
        return len(_parse_slurm_nodelist(nodelist))
    return int(os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES", "1")))


def should_launch_multi_node_slurm() -> bool:
    return (
        allocated_slurm_num_nodes() > 1
        and bool(os.environ.get("SLURM_JOB_ID"))
        and not os.environ.get("RAY_ADDRESS")
        and not os.environ.get("NEMO_BENCHMARK_SLURM_STEP")
    )


def _launch_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--session-name")
    parser.add_argument("--entries")
    parser.add_argument("--entries-exact")
    parser.add_argument("--strict-config-check", action="store_true")
    return parser.parse_known_args(argv)


def _already_running_on_each_node(num_nodes: int) -> bool:
    step_nodes = int(os.environ.get("SLURM_STEP_NUM_NODES", "0"))
    step_tasks = int(os.environ.get("SLURM_STEP_NUM_TASKS", os.environ.get("SLURM_NTASKS", "0")))
    return step_nodes == num_nodes and step_tasks == num_nodes


def launch_multi_node_slurm(argv: list[str]) -> int:
    args, _ = _launch_args(argv)
    if args.session_name is None:
        args.session_name = time.strftime("benchmark-run__%Y-%m-%d_%H-%M-%S_UTC")
        argv = [*argv, "--session-name", args.session_name]

    config = resolve_env_vars(
        remove_disabled_blocks(merge_config_files(args.config)),
        strict=args.strict_config_check,
    )
    exact = None
    if args.entries_exact:
        exact = [name.strip() for name in args.entries_exact.split(",") if name.strip()]
    session = Session.from_dict(config, entry_filter_expr=args.entries, entries_exact=exact)
    if not session.entries:
        print("No enabled benchmark entries", file=sys.stderr)
        return 2

    num_nodes = allocated_slurm_num_nodes()
    num_cpus = max(int(entry.ray.get("num_cpus", 1)) for entry in session.entries)
    num_gpus = max(int(entry.ray.get("num_gpus", 0)) for entry in session.entries)
    script = Path(__file__).with_name("slurm_run.py")
    if _already_running_on_each_node(num_nodes):
        environment = {**os.environ, "NEMO_BENCHMARK_SLURM_STEP": "1"}
        return subprocess.run([sys.executable, str(script), *argv], env=environment, check=False).returncode  # noqa: S603

    command = [
        "srun",
        f"--jobid={os.environ['SLURM_JOB_ID']}",
        "--overlap",
        f"--nodes={num_nodes}",
        f"--ntasks={num_nodes}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={num_cpus}",
    ]
    if num_gpus:
        command.append(f"--gpus-per-node={num_gpus}")
    command.extend(
        [
            "/usr/bin/env",
            "NEMO_BENCHMARK_SLURM_STEP=1",
            sys.executable,
            str(script),
            *argv,
        ]
    )
    return subprocess.run(command, check=False).returncode  # noqa: S603
