# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Run one benchmark entry in a fresh multi-node Ray cluster."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runner.gpu_stats_recorder import GPUStatsRecorder

from nemo_curator.core.client import SlurmRayClient


def _args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-name", required=True)
    parser.add_argument("--session-entry-path", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--num-cpus", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--object-store-memory", required=True)
    parser.add_argument("--gpu-stats-interval-s", type=float, required=True)
    return parser.parse_known_args()


def _without_entry_filters(argv: list[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--entries", "--entries-exact"}:
            skip_next = True
            continue
        if arg.startswith(("--entries=", "--entries-exact=")):
            continue
        filtered.append(arg)
    return filtered


def _ensure_short_symlink(link_path: Path, target_path: Path) -> None:
    target_path.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(link_path):
        if not link_path.is_symlink() or os.readlink(link_path) != str(target_path):
            msg = f"Refusing to replace unexpected runtime path {link_path}"
            raise RuntimeError(msg)
    else:
        link_path.symlink_to(target_path)


def main() -> int:
    args, run_args = _args()
    if run_args and run_args[0] == "--":
        run_args = run_args[1:]
    node_id = int(os.environ["SLURM_NODEID"])
    job_id = os.environ["SLURM_JOB_ID"]
    entry_token = hashlib.sha256(args.entry_name.encode()).hexdigest()[:6]
    node_runtime_dir = args.session_entry_path / "scratch/slurm" / f"node-{node_id}"
    ray_target_dir = args.session_entry_path / "ray_cluster" / f"node-{node_id}"
    short_runtime = Path(f"/tmp/b{job_id}_{node_id}_{entry_token}")  # noqa: S108
    short_ray_temp = Path(f"/tmp/r{job_id}_{node_id}_{entry_token}")  # noqa: S108
    _ensure_short_symlink(short_runtime, node_runtime_dir)
    _ensure_short_symlink(short_ray_temp, ray_target_dir)
    for directory in [node_runtime_dir / "tmp", node_runtime_dir / "cache"]:
        directory.mkdir(parents=True, exist_ok=True)
    ray_port_dir = args.session_entry_path / "scratch/slurm/ray-port"
    ray_port_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "TMPDIR": str(short_runtime / "tmp"),
            "XDG_CACHE_HOME": str(short_runtime / "cache"),
            "RAY_PORT_BROADCAST_DIR": str(ray_port_dir),
            "NEMO_BENCHMARK_EXTERNAL_RAY": "1",
            "NEMO_BENCHMARK_PER_NODE_GPU_STATS": "1",
        }
    )
    object_store_memory = json.loads(args.object_store_memory)
    client = SlurmRayClient(
        ray_temp_dir=str(short_ray_temp),
        include_dashboard=False,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        object_store_memory=object_store_memory,
        ray_stdouterr_capture_file=str(args.logs_dir / f"ray-node-{node_id}.log"),
    )

    return_code = 1
    try:
        recorder = (
            GPUStatsRecorder(
                args.session_entry_path / f"gpustats-node-{node_id}.csv",
                interval_s=args.gpu_stats_interval_s,
            )
            if args.gpu_stats_interval_s > 0
            else contextlib.nullcontext()
        )
        with recorder:
            client.start()
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "run.py"),
                *_without_entry_filters(run_args),
                "--entries-exact",
                args.entry_name,
                "--strict-config-check",
            ]
            return_code = subprocess.run(command, check=False).returncode  # noqa: S603
    finally:
        if node_id == 0:
            args.status_path.parent.mkdir(parents=True, exist_ok=True)
            status_tmp = args.status_path.with_suffix(".tmp")
            status_tmp.write_text(str(return_code))
            status_tmp.replace(args.status_path)
        client.stop()
        for link_path, target_path in [
            (short_runtime, node_runtime_dir),
            (short_ray_temp, ray_target_dir),
        ]:
            try:
                if link_path.is_symlink() and os.readlink(link_path) == str(target_path):
                    link_path.unlink()
            except FileNotFoundError:
                pass
        (args.logs_dir / f"teardown-node-{node_id}.done").touch()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
