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

from __future__ import annotations

import contextlib
import http
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.core.serve.internal.base import InferenceBackend
from nemo_curator.core.serve.internal.constants import DEFAULT_ETCD_PORT, DEFAULT_NATS_PORT

if TYPE_CHECKING:
    from nemo_curator.core.serve.server import InferenceModelConfig, InferenceServer

# ---------------------------------------------------------------------------
# GPU placement planner
# ---------------------------------------------------------------------------


@dataclass
class WorkerPlacement:
    """Describes where a single Dynamo vLLM worker should run."""

    node_id: str
    node_ip: str
    gpu_ids: list[int]
    worker_index: int


def _get_gpu_inventory(head_node_id: str | None = None) -> list[dict[str, Any]]:
    """Return per-node GPU information from the Ray cluster.

    Each entry: ``{"node_id": str, "node_ip": str, "num_gpus": int, "is_head": bool}``.
    Must be called from within an active Ray context.
    """
    import ray

    nodes = ray.nodes()
    inventory = []
    for node in nodes:
        if not node.get("Alive", False):
            continue
        resources = node.get("Resources", {})
        num_gpus = int(resources.get("GPU", 0))
        if num_gpus == 0:
            continue
        node_id = node["NodeID"]
        inventory.append(
            {
                "node_id": node_id,
                "node_ip": node["NodeManagerAddress"],
                "num_gpus": num_gpus,
                "is_head": node_id == head_node_id,
            }
        )
    return inventory


def build_gpu_placement(
    num_replicas: int,
    gpus_per_replica: int,
    head_node_id: str | None = None,
) -> list[WorkerPlacement]:
    """Assign GPU slots to workers across the Ray cluster.

    Greedily fills nodes in inventory order. Raises if insufficient GPUs.
    Must be called from within an active Ray context.
    """
    inventory = _get_gpu_inventory(head_node_id)
    if not inventory:
        msg = "No GPU nodes found in the Ray cluster."
        raise RuntimeError(msg)

    total_gpus = sum(n["num_gpus"] for n in inventory)
    needed = num_replicas * gpus_per_replica
    if needed > total_gpus:
        msg = (
            f"Need {needed} GPUs ({num_replicas} replicas x {gpus_per_replica} GPUs/replica) "
            f"but only {total_gpus} available across {len(inventory)} node(s)."
        )
        raise RuntimeError(msg)

    placements: list[WorkerPlacement] = []
    worker_idx = 0
    node_next_gpu: dict[str, int] = {n["node_id"]: 0 for n in inventory}

    for _ in range(num_replicas):
        placed = False
        for node in inventory:
            nid = node["node_id"]
            start = node_next_gpu[nid]
            if start + gpus_per_replica <= node["num_gpus"]:
                gpu_ids = list(range(start, start + gpus_per_replica))
                placements.append(
                    WorkerPlacement(
                        node_id=nid,
                        node_ip=node["node_ip"],
                        gpu_ids=gpu_ids,
                        worker_index=worker_idx,
                    )
                )
                node_next_gpu[nid] = start + gpus_per_replica
                worker_idx += 1
                placed = True
                break
        if not placed:
            msg = (
                f"Cannot place worker {worker_idx}: no node has {gpus_per_replica} "
                f"contiguous free GPUs. Allocated {worker_idx}/{num_replicas} so far."
            )
            raise RuntimeError(msg)

    return placements


# ---------------------------------------------------------------------------
# Ray actors for subprocess management
# ---------------------------------------------------------------------------

_SIGTERM_WAIT_S = 10
_SIGKILL_WAIT_S = 5


def _stop_subprocess(proc: Any, sigterm_wait: float = _SIGTERM_WAIT_S) -> int | None:  # noqa: ANN401
    """SIGTERM → wait → SIGKILL a subprocess.  Used inside Ray actors."""
    if proc.poll() is not None:
        return proc.returncode
    proc.terminate()
    try:
        proc.wait(timeout=sigterm_wait)
    except Exception:  # noqa: BLE001
        proc.kill()
        proc.wait(timeout=_SIGKILL_WAIT_S)
    return proc.returncode


def _define_infra_actor() -> type:
    """Return the ``_InfraActor`` Ray remote class."""
    import ray

    @ray.remote(num_cpus=1, num_gpus=0)
    class _InfraActor:
        """Manages an etcd, nats-server, or frontend subprocess on a Ray node."""

        def __init__(self, command: list[str], env: dict[str, str], label: str, log_file: str | None = None) -> None:
            import subprocess as _sp

            self._label = label
            if log_file:
                self._log_fh = open(log_file, "w")  # noqa: SIM115
                self._proc = _sp.Popen(command, env=env, stdout=self._log_fh, stderr=_sp.STDOUT)  # noqa: S603
            else:
                self._proc = _sp.Popen(command, env=env, stdout=_sp.DEVNULL, stderr=_sp.STDOUT)  # noqa: S603
            self._log_file = log_file

        def pid(self) -> int:
            return self._proc.pid

        def is_alive(self) -> bool:
            return self._proc.poll() is None

        def log_file(self) -> str | None:
            return self._log_file

        def stop(self, sigterm_wait: float = _SIGTERM_WAIT_S) -> int | None:
            rc = _stop_subprocess(self._proc, sigterm_wait)
            if hasattr(self, "_log_fh"):
                self._log_fh.close()
            return rc

    return _InfraActor


def _define_worker_actor() -> type:
    """Return the ``_WorkerActor`` Ray remote class."""
    import ray

    @ray.remote(num_cpus=1, num_gpus=0)
    class _WorkerActor:
        """Manages a ``python -m dynamo.vllm`` subprocess on a Ray node."""

        def __init__(
            self, command: list[str], env: dict[str, str], worker_index: int, log_file: str | None = None
        ) -> None:
            import subprocess as _sp

            self._worker_index = worker_index
            if log_file:
                self._log_fh = open(log_file, "w")  # noqa: SIM115
                self._proc = _sp.Popen(command, env=env, stdout=self._log_fh, stderr=_sp.STDOUT)  # noqa: S603
            else:
                self._proc = _sp.Popen(command, env=env, stdout=_sp.DEVNULL, stderr=_sp.STDOUT)  # noqa: S603
            self._log_file = log_file

        def pid(self) -> int:
            return self._proc.pid

        def worker_index(self) -> int:
            return self._worker_index

        def is_alive(self) -> bool:
            return self._proc.poll() is None

        def log_file(self) -> str | None:
            return self._log_file

        def stop(self, sigterm_wait: float = _SIGTERM_WAIT_S) -> int | None:
            rc = _stop_subprocess(self._proc, sigterm_wait)
            if hasattr(self, "_log_fh"):
                self._log_fh.close()
            return rc

    return _WorkerActor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_binary(name: str) -> None:
    """Raise if *name* is not found on ``$PATH``."""
    import shutil as _shutil

    if _shutil.which(name) is None:
        msg = f"Required binary '{name}' not found on $PATH. Install etcd and nats-server, then try again."
        raise FileNotFoundError(msg)


def _engine_kwargs_to_cli_flags(engine_kwargs: dict[str, Any]) -> list[str]:
    """Convert engine_kwargs dict to vLLM CLI flags.

    Example: ``{"tensor_parallel_size": 4, "enforce_eager": True}``
    becomes ``["--tensor-parallel-size", "4", "--enforce-eager"]``
    """
    flags: list[str] = []
    for key, value in engine_kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        else:
            flags.extend([flag, str(value)])
    return flags


def _wait_for_port(host: str, port: int, timeout_s: float = 30, label: str = "") -> None:
    """Block until a TCP connection to *host:port* succeeds."""
    import socket as _socket

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError), _socket.create_connection((host, port), timeout=2):
            return
        time.sleep(0.5)
    tag = f" ({label})" if label else ""
    msg = f"Port {port}{tag} did not become reachable within {timeout_s}s"
    raise TimeoutError(msg)


def _kill_actor(ray_mod: Any, label: str, actor: Any) -> None:  # noqa: ANN401
    """Best-effort stop + kill of a single detached actor."""
    try:
        ray_mod.get(actor.stop.remote(), timeout=_SIGTERM_WAIT_S + _SIGKILL_WAIT_S + 5)
    except Exception:  # noqa: BLE001
        logger.debug(f"Dynamo {label} actor stop() failed, force-killing")
    with contextlib.suppress(Exception):
        ray_mod.kill(actor, no_restart=True)


# ---------------------------------------------------------------------------
# DynamoBackend
# ---------------------------------------------------------------------------


class DynamoBackend(InferenceBackend):
    """NVIDIA Dynamo inference backend.

    Launches Dynamo infrastructure (etcd + NATS), vLLM workers, and an
    OpenAI-compatible frontend as subprocesses managed by detached Ray actors.
    Workers are placed on cluster nodes via ``NodeAffinitySchedulingStrategy``.

    This backend does NOT participate in Ray's GPU scheduling — pipelines
    with GPU stages will fail-fast with a ``RuntimeError`` (enforced by
    ``Pipeline.run()``).
    """

    def __init__(self, server: InferenceServer) -> None:
        self._server = server

        # Populated during start()
        self._runtime_dir: str | None = None
        self._etcd_actor: Any = None
        self._nats_actor: Any = None
        self._worker_actors: list[Any] = []
        self._frontend_actor: Any = None
        self._etcd_port: int | None = None
        self._nats_port: int | None = None
        self._actor_name_prefix: str = ""

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self) -> None:
        import ray

        from nemo_curator.core.utils import get_free_port

        server = self._server
        if not server.models:
            msg = "At least one InferenceModelConfig is required."
            raise ValueError(msg)

        # Validate required binaries before doing anything else
        if not server.etcd_endpoint:
            _check_binary("etcd")
        if not server.nats_url:
            _check_binary("nats-server")

        model_config = server.models[0]
        if len(server.models) > 1:
            logger.warning("Dynamo backend currently supports a single model; using the first one.")

        tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
        dynamo_cfg = model_config.dynamo_config
        num_replicas = self._resolve_num_replicas(model_config)

        short_id = uuid.uuid4().hex[:8]
        self._actor_name_prefix = f"dynamo_{server.name}_{short_id}"
        self._runtime_dir = tempfile.mkdtemp(prefix=f"nemo_curator_dynamo_{short_id}_")
        logger.info(f"Dynamo runtime dir: {self._runtime_dir}")

        # Detached actors survive the ``with ray.init()`` disconnect.
        with ray.init(ignore_reinit_error=True):
            head_node_id = ray.get_runtime_context().get_node_id()

            self._etcd_port = (
                int(server.etcd_endpoint.rsplit(":", 1)[-1])
                if server.etcd_endpoint
                else get_free_port(DEFAULT_ETCD_PORT)
            )
            self._nats_port = (
                int(server.nats_url.rsplit(":", 1)[-1]) if server.nats_url else get_free_port(DEFAULT_NATS_PORT)
            )
            server.port = get_free_port(server.port)

            infra_cls = _define_infra_actor()
            worker_cls = _define_worker_actor()

            # Infrastructure
            if not server.etcd_endpoint:
                self._etcd_actor = self._start_etcd(infra_cls, head_node_id, self._etcd_port)
            if not server.nats_url:
                self._nats_actor = self._start_nats(infra_cls, head_node_id, self._nats_port)

            etcd_endpoint = server.etcd_endpoint or f"http://localhost:{self._etcd_port}"
            nats_url = server.nats_url or f"nats://localhost:{self._nats_port}"

            # Workers — let Ray handle GPU placement via num_gpus on the actor
            namespace = dynamo_cfg.get("namespace", "curator")
            request_plane = dynamo_cfg.get("request_plane", "nats")
            event_plane = dynamo_cfg.get("event_plane", "nats")
            worker_env = self._build_env(etcd_endpoint, nats_url)
            logger.info(f"Launching {num_replicas} Dynamo worker(s) with TP={tp_size}")

            for i in range(num_replicas):
                actor = self._launch_worker(
                    worker_cls,
                    worker_index=i,
                    model_config=model_config,
                    base_env=worker_env,
                    gpus_per_worker=tp_size,
                    namespace=namespace,
                    request_plane=request_plane,
                    event_plane=event_plane,
                )
                self._worker_actors.append(actor)

            # Frontend
            model_name = model_config.model_name or model_config.model_identifier
            self._frontend_actor = self._launch_frontend(
                infra_cls,
                head_node_id,
                server.port,
                etcd_endpoint,
                nats_url,
                namespace=namespace,
                model_name=model_name,
                request_plane=request_plane,
                event_plane=event_plane,
            )

        # Health check — outside ray.init context, pure HTTP.
        # For Dynamo the frontend starts before workers register, so we need
        # to wait for both the HTTP endpoint AND the model to appear.
        expected_model = model_config.model_name or model_config.model_identifier
        self._wait_for_model(server, expected_model)

    def _wait_for_model(self, server: InferenceServer, model_name: str) -> None:
        """Poll ``/v1/models`` until *model_name* appears in the response.

        Also checks that worker/frontend subprocesses are still alive. If any
        crash, reads their log file and raises immediately with the output.
        """
        import json
        import urllib.request

        models_url = f"{server.endpoint}/models"
        deadline = time.monotonic() + server.health_check_timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1

            # Check subprocess liveness every few attempts
            if attempt % 3 == 0:
                self._check_subprocess_health()

            try:
                resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
                if resp.status == http.HTTPStatus.OK:
                    body = json.loads(resp.read())
                    ids = [m["id"] for m in body.get("data", [])]
                    if model_name in ids:
                        logger.info(f"Dynamo model '{model_name}' registered after {attempt} health check(s)")
                        return
                    if server.verbose:
                        logger.debug(f"Models so far: {ids}, waiting for '{model_name}'...")
            except Exception:  # noqa: BLE001
                if server.verbose:
                    logger.debug(f"Health check attempt {attempt} failed, retrying...")
            time.sleep(2)

        # Final liveness check before giving up — surface crash info if available
        self._check_subprocess_health()
        msg = f"Model '{model_name}' did not appear at {models_url} within {server.health_check_timeout_s}s"
        raise TimeoutError(msg)

    def _check_subprocess_health(self) -> None:
        """Check log files for signs of subprocess crashes.

        Scans all log files in the runtime directory for Python tracebacks
        or fatal-level log lines. Raises immediately with the log tail if found.
        """
        if not self._runtime_dir:
            return

        crash_patterns = ("Traceback (most recent call last)", '"level":"fatal"')
        for label in self._log_labels():
            log_path = os.path.join(self._runtime_dir, f"{label}.log")
            try:
                with open(log_path) as f:
                    content = f.read()
            except Exception:  # noqa: BLE001, S112
                continue
            if not content:
                continue
            if any(pat in content for pat in crash_patterns):
                tail = "\n".join(content.splitlines()[-50:])
                msg = (
                    f"Dynamo {label} subprocess crashed during startup.\n\n--- {label} log (last 50 lines) ---\n{tail}"
                )
                raise RuntimeError(msg)

    def _log_labels(self) -> list[str]:
        """Return labels for all managed subprocess log files."""
        labels = []
        if self._frontend_actor is not None:
            labels.append("frontend")
        for i in range(len(self._worker_actors)):
            labels.append(f"worker_{i}")
        return labels

    def _read_log_tail(self, label: str, lines: int = 50) -> str:
        """Read the last *lines* from a subprocess log file."""
        if not self._runtime_dir:
            return ""
        log_path = os.path.join(self._runtime_dir, f"{label}.log")
        try:
            with open(log_path) as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _resolve_num_replicas(model_config: InferenceModelConfig) -> int:
        num = model_config.deployment_config.get("num_replicas", 0)
        if num and num > 0:
            return num
        autoscaling = model_config.deployment_config.get("autoscaling_config", {})
        num = autoscaling.get("min_replicas", 1)
        return max(num, 1)

    # ------------------------------------------------------------------
    # Infrastructure actors
    # ------------------------------------------------------------------

    def _start_etcd(self, actor_cls: type, head_node_id: str, port: int) -> Any:  # noqa: ANN401
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from nemo_curator.core.utils import get_free_port

        data_dir = os.path.join(self._runtime_dir, "etcd_data")
        os.makedirs(data_dir, exist_ok=True)

        # etcd also needs a peer port — default 2380 may conflict with other instances.
        # Use 127.0.0.1 (not localhost) to avoid DNS resolution mismatches between
        # --initial-advertise-peer-urls and --initial-cluster.
        peer_port = get_free_port(2380)
        command = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{port}",
            "--advertise-client-urls",
            f"http://0.0.0.0:{port}",
            "--listen-peer-urls",
            f"http://127.0.0.1:{peer_port}",
            "--initial-advertise-peer-urls",
            f"http://127.0.0.1:{peer_port}",
            "--initial-cluster",
            f"default=http://127.0.0.1:{peer_port}",
            "--data-dir",
            data_dir,
        ]
        env = {**os.environ, "ALLOW_NONE_AUTHENTICATION": "yes"}

        log_file = os.path.join(self._runtime_dir, "etcd.log") if self._runtime_dir else None
        actor_name = f"{self._actor_name_prefix}_etcd"
        actor = actor_cls.options(
            name=actor_name,
            lifetime="detached",
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
        ).remote(command=command, env=env, label="etcd", log_file=log_file)

        logger.info(f"Starting etcd on port {port} (actor: {actor_name})")
        _wait_for_port("localhost", port, timeout_s=30, label="etcd")
        logger.info("etcd is ready")
        return actor

    def _start_nats(self, actor_cls: type, head_node_id: str, port: int) -> Any:  # noqa: ANN401
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        store_dir = os.path.join(self._runtime_dir, "nats_data")
        os.makedirs(store_dir, exist_ok=True)

        command = ["nats-server", "-p", str(port), "-js", "--store_dir", store_dir]

        log_file = os.path.join(self._runtime_dir, "nats.log") if self._runtime_dir else None
        actor_name = f"{self._actor_name_prefix}_nats"
        actor = actor_cls.options(
            name=actor_name,
            lifetime="detached",
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
        ).remote(command=command, env=dict(os.environ), label="nats", log_file=log_file)

        logger.info(f"Starting NATS on port {port} (actor: {actor_name})")
        _wait_for_port("localhost", port, timeout_s=30, label="nats")
        logger.info("NATS is ready")
        return actor

    # ------------------------------------------------------------------
    # Worker / frontend actors
    # ------------------------------------------------------------------

    def _launch_worker(  # noqa: PLR0913
        self,
        actor_cls: type,
        worker_index: int,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        *,
        gpus_per_worker: int,
        namespace: str,
        request_plane: str,
        event_plane: str,
    ) -> Any:  # noqa: ANN401
        model_name = model_config.model_name or model_config.model_identifier
        command = [
            sys.executable,
            "-m",
            "dynamo.vllm",
            "--model",
            model_config.model_identifier,
            "--served-model-name",
            model_name,
            "--namespace",
            namespace,
            "--discovery-backend",
            "etcd",
            "--request-plane",
            request_plane,
            "--event-plane",
            event_plane,
            *_engine_kwargs_to_cli_flags(model_config.engine_kwargs),
        ]

        log_file = os.path.join(self._runtime_dir, f"worker_{worker_index}.log") if self._runtime_dir else None
        actor_name = f"{self._actor_name_prefix}_worker_{worker_index}"
        # Let Ray handle GPU placement — the actor gets num_gpus GPUs, and the
        # subprocess inherits the actor's CUDA_VISIBLE_DEVICES automatically.
        actor = actor_cls.options(
            name=actor_name,
            lifetime="detached",
            num_gpus=gpus_per_worker,
        ).remote(command=command, env=base_env, worker_index=worker_index, log_file=log_file)

        logger.info(f"Starting Dynamo worker {worker_index} (actor: {actor_name}, gpus={gpus_per_worker})")
        return actor

    def _launch_frontend(  # noqa: PLR0913
        self,
        actor_cls: type,
        head_node_id: str,
        port: int,
        etcd_endpoint: str,
        nats_url: str,
        *,
        namespace: str,
        model_name: str,
        request_plane: str,
        event_plane: str,
    ) -> Any:  # noqa: ANN401
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        env = self._build_env(etcd_endpoint, nats_url)
        command = [
            sys.executable,
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(port),
            "--namespace",
            namespace,
            "--model-name",
            model_name,
            "--discovery-backend",
            "etcd",
            "--request-plane",
            request_plane,
            "--event-plane",
            event_plane,
        ]

        log_file = os.path.join(self._runtime_dir, "frontend.log") if self._runtime_dir else None
        actor_name = f"{self._actor_name_prefix}_frontend"
        actor = actor_cls.options(
            name=actor_name,
            lifetime="detached",
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=head_node_id, soft=False),
        ).remote(command=command, env=env, label="frontend", log_file=log_file)

        logger.info(f"Starting Dynamo frontend on port {port} (actor: {actor_name})")
        return actor

    @staticmethod
    def _build_env(etcd_endpoint: str, nats_url: str) -> dict[str, str]:
        """Base environment dict for all Dynamo subprocesses."""
        env = dict(os.environ)
        env["ETCD_ENDPOINTS"] = etcd_endpoint
        env["NATS_SERVER"] = nats_url
        return env

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> None:
        # Collect actors in reverse-start order: frontend → workers → nats → etcd
        actors_to_kill = self._collect_actors_for_shutdown()

        if actors_to_kill:
            self._kill_all_actors(actors_to_kill)

        self._cleanup_runtime_dir()
        logger.info("Dynamo backend stopped")

    def _collect_actors_for_shutdown(self) -> list[tuple[str, Any]]:
        actors: list[tuple[str, Any]] = []

        if self._frontend_actor is not None:
            actors.append(("frontend", self._frontend_actor))
            self._frontend_actor = None

        for actor in self._worker_actors:
            actors.append(("worker", actor))
        self._worker_actors.clear()

        if self._nats_actor is not None:
            actors.append(("nats", self._nats_actor))
            self._nats_actor = None

        if self._etcd_actor is not None:
            actors.append(("etcd", self._etcd_actor))
            self._etcd_actor = None

        return actors

    @staticmethod
    def _kill_all_actors(actors: list[tuple[str, Any]]) -> None:
        import ray

        try:
            with ray.init(ignore_reinit_error=True):
                for label, actor in actors:
                    _kill_actor(ray, label, actor)
        except Exception:  # noqa: BLE001
            logger.debug("Could not connect to Ray during Dynamo shutdown (cluster may be gone)")

    def _cleanup_runtime_dir(self) -> None:
        if self._runtime_dir and os.path.isdir(self._runtime_dir):
            try:
                shutil.rmtree(self._runtime_dir)
                logger.debug(f"Cleaned up runtime dir: {self._runtime_dir}")
            except Exception:  # noqa: BLE001
                logger.debug(f"Failed to clean up runtime dir: {self._runtime_dir}")
            self._runtime_dir = None
