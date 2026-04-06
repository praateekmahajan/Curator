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
class NodeAllocation:
    """A single node's contribution to a replica's TP group.

    Attributes:
        node_id: Ray node ID.
        node_ip: Routable IP address of the node.
        num_gpus: Number of GPUs allocated on this node.
        node_rank: Rank within the TP group (0 = leader, 1+ = headless).
    """

    node_id: str
    node_ip: str
    num_gpus: int
    node_rank: int


@dataclass
class ReplicaPlan:
    """Placement plan for one model replica (may span multiple nodes).

    For single-node TP, ``ranks`` contains a single entry.  For multi-node
    TP (TP size > GPUs on any single node), ``ranks`` contains one entry
    per participating node.  Rank 0 is always the leader that runs the full
    Dynamo worker; rank 1+ run headless vLLM workers coordinated via
    ``torch.distributed``.
    """

    replica_index: int
    ranks: list[NodeAllocation]

    @property
    def is_multi_node(self) -> bool:
        return len(self.ranks) > 1

    @property
    def nnodes(self) -> int:
        return len(self.ranks)

    @property
    def master_addr(self) -> str:
        """IP of the rank-0 (leader) node."""
        return self.ranks[0].node_ip

    @property
    def total_gpus(self) -> int:
        return sum(r.num_gpus for r in self.ranks)


def _get_gpu_inventory(
    head_node_id: str | None = None,
    nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return per-node GPU information from the Ray cluster.

    Each entry: ``{"node_id": str, "node_ip": str, "num_gpus": int, "is_head": bool}``.

    Args:
        head_node_id: Optional node ID to tag as head in inventory.
        nodes: Pre-fetched ``ray.nodes()`` result to avoid a redundant API call.
    """
    import ray

    inventory = []
    for node in nodes or ray.nodes():
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


def _resolve_node_ip(node_id: str, nodes: list[dict[str, Any]] | None = None) -> str:
    """Get the routable IP address for a Ray node by its ID.

    Args:
        node_id: The Ray node ID to resolve.
        nodes: Pre-fetched ``ray.nodes()`` result to avoid a redundant API call.
    """
    import ray

    for node in nodes or ray.nodes():
        if node["NodeID"] == node_id and node.get("Alive", False):
            return node["NodeManagerAddress"]
    msg = f"Could not resolve IP for Ray node {node_id}"
    raise RuntimeError(msg)


def plan_replica_placement(
    num_replicas: int,
    tp_size: int,
    head_node_id: str | None = None,
    _inventory: list[dict[str, Any]] | None = None,
    _nodes: list[dict[str, Any]] | None = None,
) -> list[ReplicaPlan]:
    """Plan GPU allocation for replicas, supporting multi-node TP.

    For each replica, GPUs are allocated greedily across nodes:
    - If ``tp_size`` fits on a single node, the replica gets one rank (rank 0).
    - If ``tp_size`` exceeds any single node, the replica spans multiple nodes.
      Rank 0 runs the full Dynamo worker; rank 1+ run headless vLLM workers.

    Args:
        num_replicas: Number of independent model replicas.
        tp_size: Tensor parallel degree (GPUs per replica).
        head_node_id: Optional node ID to tag as head in inventory.
        _inventory: Pre-built inventory for testing (bypasses Ray call).
        _nodes: Pre-fetched ``ray.nodes()`` result passed to ``_get_gpu_inventory``.

    Returns:
        List of ``ReplicaPlan``, one per replica.
    """
    inventory = _inventory if _inventory is not None else _get_gpu_inventory(head_node_id, nodes=_nodes)
    if not inventory:
        msg = "No GPU nodes found in the Ray cluster."
        raise RuntimeError(msg)

    total_gpus = sum(n["num_gpus"] for n in inventory)
    needed = num_replicas * tp_size
    if needed > total_gpus:
        msg = (
            f"Need {needed} GPUs ({num_replicas} replicas x {tp_size} TP) "
            f"but only {total_gpus} available across {len(inventory)} node(s)."
        )
        raise RuntimeError(msg)

    available = {n["node_id"]: n["num_gpus"] for n in inventory}

    plans: list[ReplicaPlan] = []
    for replica_idx in range(num_replicas):
        ranks: list[NodeAllocation] = []
        remaining = tp_size
        node_rank = 0

        for node in inventory:
            nid = node["node_id"]
            avail = available[nid]
            if avail <= 0:
                continue

            take = min(avail, remaining)
            ranks.append(
                NodeAllocation(
                    node_id=nid,
                    node_ip=node["node_ip"],
                    num_gpus=take,
                    node_rank=node_rank,
                )
            )
            available[nid] -= take
            remaining -= take
            node_rank += 1

            if remaining == 0:
                break

        if remaining > 0:
            placed = tp_size - remaining
            msg = (
                f"Cannot place replica {replica_idx}: need {tp_size} GPUs but only "
                f"{placed} available. Allocated {replica_idx}/{num_replicas} replicas."
            )
            raise RuntimeError(msg)

        plans.append(ReplicaPlan(replica_index=replica_idx, ranks=ranks))

    return plans


# ---------------------------------------------------------------------------
# Ray actors for subprocess management
# ---------------------------------------------------------------------------

_SIGTERM_WAIT_S = 10
_SIGKILL_WAIT_S = 5


def _stop_subprocess(proc: Any, sigterm_wait: float = _SIGTERM_WAIT_S) -> int | None:  # noqa: ANN401
    """SIGTERM -> wait -> SIGKILL a subprocess.  Used inside Ray actors."""
    if proc.poll() is not None:
        return proc.returncode
    proc.terminate()
    try:
        proc.wait(timeout=sigterm_wait)
    except Exception:  # noqa: BLE001
        proc.kill()
        proc.wait(timeout=_SIGKILL_WAIT_S)
    return proc.returncode


def _define_subprocess_actor() -> type:  # noqa: C901
    """Return the ``_SubprocessActor`` Ray remote class.

    A single actor class used for all Dynamo subprocesses (etcd, NATS,
    frontend, vLLM workers).  GPU resources are configured per-instance
    via ``.options(num_gpus=...)``.
    """
    import ray

    @ray.remote(num_cpus=1, num_gpus=0)
    class _SubprocessActor:
        """Manages a subprocess on a Ray node with optional file-based logging."""

        def __init__(self, command: list[str], env: dict[str, str], log_file: str | None = None) -> None:
            import subprocess as _sp

            # Start from the actor's own os.environ (which has Ray-assigned
            # CUDA_VISIBLE_DEVICES), then overlay the caller-provided env vars.
            # The caller's env originates from the *driver* process and must not
            # replace the actor's CUDA_VISIBLE_DEVICES.
            merged_env = dict(os.environ)
            merged_env.update(env)
            env = merged_env

            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
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

        def read_log_tail(self, num_bytes: int = 8192) -> str:
            """Read the last *num_bytes* of the log file.

            Flushes the write handle first so buffered output is visible.
            Called remotely by the driver for crash diagnosis.
            """
            if not self._log_file:
                return ""
            try:
                if hasattr(self, "_log_fh") and not self._log_fh.closed:
                    self._log_fh.flush()
                with open(self._log_file, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - num_bytes))
                    return f.read().decode(errors="replace")
            except Exception:  # noqa: BLE001
                return ""

        def stop(self, sigterm_wait: float = _SIGTERM_WAIT_S) -> int | None:
            rc = _stop_subprocess(self._proc, sigterm_wait)
            if hasattr(self, "_log_fh"):
                self._log_fh.close()
            return rc

    return _SubprocessActor


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
    import json

    flags: list[str] = []
    for key, value in engine_kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        elif isinstance(value, (dict, list)):
            flags.extend([flag, json.dumps(value)])
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

    Supports both single-node and multi-node tensor parallelism:

    - **Single-node TP**: Each replica runs on one node.  One Ray actor per
      replica with ``num_gpus=tp_size``.
    - **Multi-node TP** (TP > GPUs on any single node): Each replica spans
      multiple nodes.  Rank 0 runs the full Dynamo worker (model registration
      in etcd); rank 1+ run headless vLLM workers coordinated via
      ``torch.distributed``.  Each rank is pinned to its planned node via
      ``NodeAffinitySchedulingStrategy``.

    This backend does NOT participate in Ray's GPU scheduling -- pipelines
    with GPU stages will fail-fast with a ``RuntimeError`` (enforced by
    ``Pipeline.run()``).
    """

    def __init__(self, server: InferenceServer) -> None:
        self._server = server
        self._runtime_dir: str | None = None
        self._head_ip: str | None = None
        self._etcd_actor: Any = None
        self._nats_actor: Any = None
        self._worker_actors: list[tuple[str, Any]] = []
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

        with ray.init(ignore_reinit_error=True):
            head_node_id = ray.get_runtime_context().get_node_id()
            cluster_nodes = ray.nodes()
            self._head_ip = _resolve_node_ip(head_node_id, nodes=cluster_nodes)
            server._host = self._head_ip

            # Kill any orphaned actors from a previous run with the same server name
            # (e.g. after a Jupyter kernel restart that bypassed atexit cleanup).
            self._kill_orphaned_actors(ray, server.name)

            self._etcd_port = (
                int(server.etcd_endpoint.rsplit(":", 1)[-1])
                if server.etcd_endpoint
                else get_free_port(DEFAULT_ETCD_PORT)
            )
            self._nats_port = (
                int(server.nats_url.rsplit(":", 1)[-1]) if server.nats_url else get_free_port(DEFAULT_NATS_PORT)
            )
            server.port = get_free_port(server.port)

            actor_cls = _define_subprocess_actor()

            try:
                # Infrastructure -- always on head node
                if not server.etcd_endpoint:
                    self._etcd_actor = self._start_etcd(actor_cls, head_node_id, self._etcd_port)
                if not server.nats_url:
                    self._nats_actor = self._start_nats(actor_cls, head_node_id, self._nats_port)

                etcd_endpoint = server.etcd_endpoint or f"http://{self._head_ip}:{self._etcd_port}"
                nats_url = server.nats_url or f"nats://{self._head_ip}:{self._nats_port}"

                # Plan GPU placement across cluster
                replica_plans = plan_replica_placement(num_replicas, tp_size, head_node_id, _nodes=cluster_nodes)

                namespace = dynamo_cfg.get("namespace", "curator")
                request_plane = dynamo_cfg.get("request_plane", "nats")
                event_plane = dynamo_cfg.get("event_plane", "nats")
                base_env = self._build_env(etcd_endpoint, nats_url)

                self._launch_replicas(
                    actor_cls,
                    replica_plans,
                    model_config,
                    base_env,
                    namespace=namespace,
                    request_plane=request_plane,
                    event_plane=event_plane,
                )

                # Frontend -- always on head node
                model_name = model_config.model_name or model_config.model_identifier
                self._frontend_actor = self._launch_frontend(
                    actor_cls,
                    head_node_id,
                    server.port,
                    etcd_endpoint,
                    nats_url,
                    namespace=namespace,
                    model_name=model_name,
                    request_plane=request_plane,
                    event_plane=event_plane,
                )

                # Health check — must stay inside ray.init context so actor handles
                # (is_alive, read_log_tail) remain valid for subprocess liveness checks.
                expected_model = model_config.model_name or model_config.model_identifier
                self._wait_for_model(server, expected_model)
            except Exception:
                # Still connected to Ray here — kill directly without opening another context.
                for label, actor in self._collect_actors_for_shutdown():
                    _kill_actor(ray, label, actor)
                self._cleanup_runtime_dir()
                raise

    def _wait_for_model(self, server: InferenceServer, model_name: str) -> None:
        """Poll ``/v1/models`` until *model_name* appears in the response.

        Also checks subprocess liveness via Ray actors periodically.  If any
        crash, reads their log tail and raises immediately with the output.
        """
        import json
        import urllib.request

        models_url = f"{server.endpoint}/models"
        deadline = time.monotonic() + server.health_check_timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1

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

        # Final liveness check before giving up -- surface crash info if available
        self._check_subprocess_health()
        msg = f"Model '{model_name}' did not appear at {models_url} within {server.health_check_timeout_s}s"
        raise TimeoutError(msg)

    def _check_subprocess_health(self) -> None:
        """Check subprocess liveness via Ray actors.

        Calls ``is_alive()`` on each managed actor.  If a subprocess has
        exited, reads its log tail via ``read_log_tail()`` and raises with
        the output.  Works for both local and remote (multi-node) actors.

        Note: does NOT use ``with ray.init()`` — the driver must already be
        connected.  Using a context manager here would disconnect Ray on
        exit, invalidating actor handles and causing silent failures on the
        next health check.
        """
        import ray

        actors_to_check = self._get_actors_for_health_check()
        if not actors_to_check:
            return

        for label, actor in actors_to_check:
            try:
                alive = ray.get(actor.is_alive.remote(), timeout=10)
            except Exception:  # noqa: BLE001
                # Actor unreachable — try reading its log for crash info
                log_tail = self._read_actor_log_tail(actor)
                if log_tail:
                    self._raise_subprocess_error(
                        label, log_tail, reason="actor unreachable and has log output — likely crashed"
                    )
                continue
            if not alive:
                log_tail = self._read_actor_log_tail(actor)
                self._raise_subprocess_error(label, log_tail, reason="subprocess crashed during startup")

    def _get_actors_for_health_check(self) -> list[tuple[str, Any]]:
        """Return (label, actor) pairs for all subprocess actors to health-check."""
        actors: list[tuple[str, Any]] = []
        if self._frontend_actor is not None:
            actors.append(("frontend", self._frontend_actor))
        actors.extend(self._worker_actors)
        return actors

    @staticmethod
    def _read_actor_log_tail(actor: Any) -> str:  # noqa: ANN401
        """Read log tail from a subprocess actor, returning empty string on failure."""
        import ray

        with contextlib.suppress(Exception):
            return ray.get(actor.read_log_tail.remote(), timeout=5)
        return ""

    @staticmethod
    def _raise_subprocess_error(label: str, log_tail: str, *, reason: str) -> None:
        """Raise ``RuntimeError`` with formatted subprocess crash info."""
        tail = "\n".join(log_tail.splitlines()[-50:]) if log_tail else "(no log output)"
        msg = f"Dynamo {label} {reason}.\n\n--- {label} log (last 50 lines) ---\n{tail}"
        raise RuntimeError(msg)

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

        peer_port = get_free_port(port + 1)
        command = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{port}",
            "--advertise-client-urls",
            f"http://{self._head_ip}:{port}",
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
        ).remote(command=command, env=env, log_file=log_file)

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
        ).remote(command=command, env=dict(os.environ), log_file=log_file)

        logger.info(f"Starting NATS on port {port} (actor: {actor_name})")
        _wait_for_port("localhost", port, timeout_s=30, label="nats")
        logger.info("NATS is ready")
        return actor

    # ------------------------------------------------------------------
    # Worker / frontend actors
    # ------------------------------------------------------------------

    def _launch_replicas(  # noqa: PLR0913
        self,
        actor_cls: type,
        replica_plans: list[ReplicaPlan],
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        *,
        namespace: str,
        request_plane: str,
        event_plane: str,
    ) -> None:
        """Launch all worker actors for the given replica plans."""
        for plan in replica_plans:
            if plan.is_multi_node:
                logger.info(
                    f"Replica {plan.replica_index}: multi-node TP across {plan.nnodes} nodes "
                    f"(total {plan.total_gpus} GPUs, master={plan.master_addr})"
                )
            else:
                logger.info(f"Replica {plan.replica_index}: single-node, {plan.total_gpus} GPU(s)")

            for rank in plan.ranks:
                if rank.node_rank == 0:
                    actor = self._launch_worker(
                        actor_cls,
                        replica_index=plan.replica_index,
                        model_config=model_config,
                        base_env=base_env,
                        node_alloc=rank,
                        namespace=namespace,
                        request_plane=request_plane,
                        event_plane=event_plane,
                        multi_node_plan=plan if plan.is_multi_node else None,
                    )
                else:
                    actor = self._launch_headless_worker(
                        actor_cls,
                        replica_index=plan.replica_index,
                        model_config=model_config,
                        base_env=base_env,
                        node_alloc=rank,
                        plan=plan,
                    )
                label = f"replica_{plan.replica_index}_rank_{rank.node_rank}"
                self._worker_actors.append((label, actor))

    def _spawn_actor(
        self,
        actor_cls: type,
        label: str,
        command: list[str],
        env: dict[str, str],
        node_alloc: NodeAllocation,
    ) -> Any:  # noqa: ANN401
        """Create a detached Ray actor that runs *command* as a subprocess.

        Handles log file setup, actor naming, GPU reservation, and node
        pinning via ``NodeAffinitySchedulingStrategy``.
        """
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        log_file = os.path.join(self._runtime_dir, f"{label}.log") if self._runtime_dir else None
        actor_name = f"{self._actor_name_prefix}_{label}"
        actor = actor_cls.options(
            name=actor_name,
            lifetime="detached",
            num_gpus=node_alloc.num_gpus,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_alloc.node_id, soft=False),
        ).remote(command=command, env=env, log_file=log_file)

        logger.info(f"Launching {label} on {node_alloc.node_ip} (actor: {actor_name}, gpus={node_alloc.num_gpus})")
        return actor

    def _launch_worker(  # noqa: PLR0913
        self,
        actor_cls: type,
        *,
        replica_index: int,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        node_alloc: NodeAllocation,
        namespace: str,
        request_plane: str,
        event_plane: str,
        multi_node_plan: ReplicaPlan | None = None,
    ) -> Any:  # noqa: ANN401
        """Launch a full Dynamo vLLM worker (rank 0).

        This worker creates a ``DistributedRuntime``, registers the model in
        etcd, and serves inference requests.  For multi-node TP, it also
        coordinates with headless workers via ``torch.distributed``.
        """
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

        if multi_node_plan is not None:
            command.extend(
                [
                    "--nnodes",
                    str(multi_node_plan.nnodes),
                    "--node-rank",
                    "0",
                    "--master-addr",
                    multi_node_plan.master_addr,
                ]
            )

        label = f"replica_{replica_index}_rank_0"
        return self._spawn_actor(actor_cls, label, command, base_env, node_alloc)

    def _launch_headless_worker(  # noqa: PLR0913
        self,
        actor_cls: type,
        *,
        replica_index: int,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        node_alloc: NodeAllocation,
        plan: ReplicaPlan,
    ) -> Any:  # noqa: ANN401
        """Launch a headless vLLM worker (rank > 0, multi-node TP).

        Headless workers bypass Dynamo's ``DistributedRuntime`` entirely --
        they run only vLLM workers coordinated with rank 0 via
        ``torch.distributed`` (NCCL).  No model registration, no etcd/NATS.
        """
        command = [
            sys.executable,
            "-m",
            "dynamo.vllm",
            "--model",
            model_config.model_identifier,
            "--headless",
            "--nnodes",
            str(plan.nnodes),
            "--node-rank",
            str(node_alloc.node_rank),
            "--master-addr",
            plan.master_addr,
            *_engine_kwargs_to_cli_flags(model_config.engine_kwargs),
        ]

        label = f"replica_{replica_index}_rank_{node_alloc.node_rank}"
        return self._spawn_actor(actor_cls, label, command, base_env, node_alloc)

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
        ).remote(command=command, env=env, log_file=log_file)

        logger.info(f"Starting Dynamo frontend on port {port} (actor: {actor_name})")
        return actor

    @staticmethod
    def _build_env(etcd_endpoint: str, nats_url: str) -> dict[str, str]:
        """Extra environment vars for Dynamo subprocesses.

        Returns only the Dynamo-specific vars.  The actor merges these on top
        of its own ``os.environ`` (which carries Ray-assigned
        ``CUDA_VISIBLE_DEVICES``) so GPU isolation is preserved.
        """
        return {
            "ETCD_ENDPOINTS": etcd_endpoint,
            "NATS_SERVER": nats_url,
        }

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def stop(self) -> None:
        import ray

        actors_to_kill = self._collect_actors_for_shutdown()

        try:
            with ray.init(ignore_reinit_error=True):
                for label, actor in actors_to_kill:
                    _kill_actor(ray, label, actor)
                # atexit is unreliable in Jupyter — sweep by name to catch actors
                # orphaned when the kernel was restarted without calling stop().
                self._kill_orphaned_actors(ray, self._server.name)
        except Exception:  # noqa: BLE001
            logger.debug("Could not connect to Ray during Dynamo shutdown (cluster may be gone)")

        self._cleanup_runtime_dir()
        self._server._host = "localhost"
        logger.info("Dynamo backend stopped")

    def _collect_actors_for_shutdown(self) -> list[tuple[str, Any]]:
        """Collect all actors in reverse-start order: frontend -> workers -> nats -> etcd."""
        actors: list[tuple[str, Any]] = []

        if self._frontend_actor is not None:
            actors.append(("frontend", self._frontend_actor))
            self._frontend_actor = None

        actors.extend(self._worker_actors)
        self._worker_actors.clear()

        if self._nats_actor is not None:
            actors.append(("nats", self._nats_actor))
            self._nats_actor = None

        if self._etcd_actor is not None:
            actors.append(("etcd", self._etcd_actor))
            self._etcd_actor = None

        return actors

    @staticmethod
    def _kill_orphaned_actors(ray_mod: Any, server_name: str) -> None:  # noqa: ANN401
        """Kill any live actors whose name starts with ``dynamo_{server_name}_``.

        Searches across all Ray namespaces so it finds actors created in
        anonymous namespaces (e.g. after a Jupyter kernel restart).
        """
        from ray.util.state import list_actors

        prefix = f"dynamo_{server_name}_"
        try:
            all_actors = list_actors(filters=[("state", "=", "ALIVE")], limit=500, timeout=10)
        except Exception:  # noqa: BLE001
            logger.debug("Could not list Ray actors for orphan cleanup")
            return

        orphans = [a for a in all_actors if a.get("name", "").startswith(prefix)]
        if not orphans:
            return

        logger.info(f"Found {len(orphans)} orphaned Dynamo actor(s) — cleaning up")
        for actor_info in orphans:
            name = actor_info["name"]
            ns = actor_info.get("ray_namespace")
            with contextlib.suppress(Exception):
                handle = ray_mod.get_actor(name, namespace=ns)
                _kill_actor(ray_mod, name, handle)
                logger.debug(f"Killed orphaned actor: {name}")

    def _cleanup_runtime_dir(self) -> None:
        if self._runtime_dir:
            try:
                shutil.rmtree(self._runtime_dir)
                logger.debug(f"Cleaned up runtime dir: {self._runtime_dir}")
            except FileNotFoundError:
                pass
            except Exception:  # noqa: BLE001
                logger.debug(f"Failed to clean up runtime dir: {self._runtime_dir}")
            self._runtime_dir = None
