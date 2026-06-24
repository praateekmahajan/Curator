# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import ray
from loguru import logger
from runner.utils import get_shm_usage

from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.core.utils import check_ray_responsive

ray_client_start_timeout_s = 30
ray_client_start_poll_interval_s = 0.5


_RAY_CLEANUP_WAIT_S = 10


def _slurm_num_nodes() -> int:
    """Return the SLURM allocation node count, or 1 outside SLURM."""
    for env_name in ("SLURM_JOB_NUM_NODES", "SLURM_NNODES", "SLURM_STEP_NUM_NODES"):
        value = os.environ.get(env_name)
        if value:
            return int(value)

    nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if nodelist:
        try:
            scontrol = shutil.which("scontrol")
            if scontrol:
                result = subprocess.run(  # noqa: S603
                    [scontrol, "show", "hostnames", nodelist],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                nodes = [line for line in result.stdout.strip().splitlines() if line]
                if nodes:
                    return len(nodes)
        except Exception:
            logger.debug("Failed to derive SLURM node count from node list", exc_info=True)

    return 1


def _use_slurm_ray_client() -> bool:
    """Return whether this process should bootstrap Ray through SlurmRayClient."""
    return bool(os.environ.get("SLURM_JOB_ID")) and _slurm_num_nodes() > 1


def _wait_for_ray_cleanup() -> None:
    """Wait for Ray child processes to exit and /dev/shm segments to release after stopping a cluster."""
    logger.info(f"Waiting {_RAY_CLEANUP_WAIT_S}s for Ray to clean up child processes and release /dev/shm...")
    time.sleep(_RAY_CLEANUP_WAIT_S)

    shm = get_shm_usage()
    if shm["summary"]:
        logger.info(f"SHM usage after cleanup wait: {shm['summary']}")


def setup_ray_cluster_and_env(  # noqa: PLR0913
    num_cpus: int,
    num_gpus: int,
    enable_object_spilling: bool,
    ray_log_path: Path,
    object_store_size: int | None = None,
    include_dashboard: bool = True,
    metrics_dir: Path | None = None,
) -> tuple[RayClient, Path]:
    """Setup a Ray cluster and set the RAY_ADDRESS environment variable and return the Ray client and temp dir."""
    # Create a short temp dir to avoid Unix socket path length limits
    short_temp_path = Path(f"/tmp/ray_{uuid.uuid4().hex[:8]}")  # noqa: S108
    short_temp_path.mkdir(parents=True, exist_ok=True)

    # Capture stdout/stderr to a file if provided, otherwise suppress it
    ray_stdouterr_capture_file = str(ray_log_path) if ray_log_path else os.devnull

    if _use_slurm_ray_client() and ray_log_path:
        # SlurmRayClient broadcasts the head port through a shared file.  /tmp is
        # usually node-local on SLURM clusters, so put this beside benchmark logs.
        port_broadcast_dir = ray_log_path.parent / "ray_ports"
        port_broadcast_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RAY_PORT_BROADCAST_DIR"] = str(port_broadcast_dir)
        logger.info(f"Using SlurmRayClient port broadcast dir: {port_broadcast_dir}")

    # Check environment variables that might interfere
    ray_address_env = os.environ.get("RAY_ADDRESS")
    if ray_address_env:
        logger.warning(f"RAY_ADDRESS already set in environment: {ray_address_env}")

    shm = get_shm_usage()
    if shm["summary"]:
        logger.info(f"SHM usage before Ray cluster setup: {shm['summary']}")

    responsive = False
    retries = 0
    max_retries = 5
    client = None
    while not responsive and retries < max_retries:
        logger.info(f"Starting Ray cluster (attempt {retries + 1} of {max_retries})...")

        # Capture the ray cluster output for each attempt to start it using a unique file name
        if ray_log_path and retries > 0:
            ray_stdouterr_capture_file = f"{ray_log_path!s}-{retries + 1}"

        # Create and start the Ray client.  In multi-node SLURM, worker
        # processes block inside SlurmRayClient.start(); only the head returns.
        client_cls = SlurmRayClient if _use_slurm_ray_client() else RayClient
        capture_file = ray_stdouterr_capture_file
        if _use_slurm_ray_client() and ray_log_path:
            node_id = int(os.environ.get("SLURM_NODEID", "0"))
            if node_id != 0:
                capture_file = str(ray_log_path.with_name(f"{ray_log_path.stem}.rank{node_id}{ray_log_path.suffix}"))
        client = client_cls(
            ray_temp_dir=str(short_temp_path),
            include_dashboard=include_dashboard,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            enable_object_spilling=enable_object_spilling,
            ray_dashboard_host="0.0.0.0",  # noqa: S104
            ray_stdouterr_capture_file=capture_file,
            object_store_memory=object_store_size,
            metrics_dir=str(metrics_dir) if metrics_dir is not None else None,
        )

        try:
            client.start()
            _ensure_ray_client_process_started(client, ray_client_start_timeout_s, ray_client_start_poll_interval_s)
            responsive = True
        except Exception:
            logger.exception(f"Ray cluster start failed on attempt {retries + 1}")
            responsive = False

        if not responsive:
            logger.info("Ray cluster did not become responsive, cleaning up before retry...")
            try:
                client.stop()
            except Exception:
                logger.exception("Failed to stop client during retry cleanup")
            os.environ.pop("RAY_ADDRESS", None)
            _wait_for_ray_cleanup()
            retries += 1

    if not responsive:
        msg = f"Failed to start Ray cluster after {max_retries} attempts"
        raise RuntimeError(msg)

    pid = client.ray_process.pid if client.ray_process else None
    logger.info(f"{client.__class__.__name__} started successfully: pid={pid}, port={client.ray_port}")
    return client, short_temp_path


def teardown_ray_cluster_and_env(
    ray_client: RayClient,
    ray_temp_path: Path,
    ray_cluster_path: Path,
) -> None:
    """Teardown Ray cluster and environment variables."""
    if ray_client is not None:
        # This also removes the RAY_ADDRESS environment variable if the client also started the Ray cluster
        try:
            # Stop the Ray client
            # This also removes the RAY_ADDRESS environment variable if the client also started the Ray cluster
            ray_client.stop()
        except Exception:
            logger.exception("Failed to stop Ray client")

        # Wait for Ray child processes to exit and /dev/shm to release
        _wait_for_ray_cleanup()

        # Copy debugging artifacts and clean up temp directory
        try:
            _copy_ray_debug_artifacts(ray_temp_path, ray_cluster_path)
            shutil.rmtree(ray_temp_path, ignore_errors=True)
        except Exception:
            logger.exception("Failed to copy/remove Ray temp dir")


def get_ray_cluster_data() -> dict[str, Any]:
    """Get resource data from the Ray cluster.

    If the cluster is not responsive (e.g. crashed due to OOM), returns an empty dict
    instead of connecting — ray.init() on a dead cluster fatally terminates the process
    via Ray's C++ core worker.
    """
    if not check_ray_responsive():
        logger.warning("Ray cluster is not responsive, skipping cluster data collection")
        return {}
    with ray.init(ignore_reinit_error=True):
        time.sleep(0.2)  # ray.available_resources() returns might have a lag
        return ray.cluster_resources()


def _ensure_ray_client_process_started(client: RayClient, timeout_s: int, poll_interval_s: float) -> None:
    """Ensure the Ray client process has been started, no longer than timeout."""
    elapsed_s = 0
    while client.ray_process is None and elapsed_s < timeout_s:
        time.sleep(poll_interval_s)
        elapsed_s += poll_interval_s
    if client.ray_process is None:
        msg = f"Ray client process failed to start in {timeout_s} seconds"
        raise RuntimeError(msg)


def _copy_item_safely(src_path: Path, dst_path: Path) -> None:
    """Copy a single file or directory, logging warnings on failure."""
    try:
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
    except Exception as e:
        logger.warning(f"Failed to copy {src_path.name}: {e}")


def _copy_session_contents(session_src: Path, session_dst: Path) -> None:
    """Copy session directory contents, excluding sockets and runtime_env packages.

    ``runtime_resources/`` holds Ray's runtime_env-resolved venvs (uv/pip/conda)
    which can be many GB per actor — copying them into every benchmark artifact
    archive bloats the result without aiding debugging.
    """
    session_dst.mkdir(parents=True, exist_ok=True)

    skip_names = {"sockets", "runtime_resources"}
    for item in session_src.iterdir():
        if item.name in skip_names:
            logger.debug(f"Skipping {item.name} directory")
            continue

        dst_item = session_dst / item.name
        _copy_item_safely(item, dst_item)


def _copy_ray_debug_artifacts(short_temp_path: Path, ray_destination_path: Path) -> None:
    """Copy Ray debugging artifacts to the specified ray destination directory."""

    if not short_temp_path.exists():
        return

    # Use the provided ray destination directory directly
    ray_destination_path.mkdir(parents=True, exist_ok=True)

    # Copy log files from Ray temp dir
    logs_src = short_temp_path / "logs"
    if logs_src.exists():
        logs_dst = ray_destination_path / "logs"
        shutil.copytree(logs_src, logs_dst, dirs_exist_ok=True, ignore_errors=True)

    # Copy session info but skip sockets directory
    session_src = short_temp_path / "session_latest"
    if session_src.exists():
        session_dst = ray_destination_path / "session_latest"
        _copy_session_contents(session_src, session_dst)
