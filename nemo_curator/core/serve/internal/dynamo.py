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
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.core.serve.internal.base import InferenceBackend
from nemo_curator.core.serve.internal.constants import DEFAULT_ETCD_PORT, DEFAULT_NATS_PORT
from nemo_curator.core.serve.internal.subprocess_mgr import (
    ManagedSubprocess,
    NodeAllocation,
    ReplicaPlan,
    _check_binary,
    _define_subprocess_actor,
    _engine_kwargs_to_cli_flags,
    _get_gpu_inventory,
    _ignore_head_node,
    _kill_actor,
    _resolve_node_ip,
    _wait_for_port,
    get_free_port_on_node,
    kill_orphaned_actors,
    plan_replica_placement,
    spawn_actor,
)

if TYPE_CHECKING:
    from nemo_curator.core.serve.server import InferenceModelConfig, InferenceServer


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
        self._infra_node_id: str | None = None
        self._infra_ip: str | None = None
        self._etcd_actor: ManagedSubprocess | None = None
        self._nats_actor: ManagedSubprocess | None = None
        self._worker_actors: list[ManagedSubprocess] = []
        self._frontend_actor: ManagedSubprocess | None = None
        self._etcd_port: int | None = None
        self._nats_port: int | None = None
        self._actor_name_prefix: str = ""

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self) -> None:
        import ray

        server = self._server
        if not server.models:
            msg = "At least one InferenceModelConfig is required."
            raise ValueError(msg)

        if not server.etcd_endpoint:
            _check_binary("etcd")
        if not server.nats_url:
            _check_binary("nats-server")

        short_id = uuid.uuid4().hex[:8]
        self._actor_name_prefix = f"dynamo_{server.name}_{short_id}"
        self._runtime_dir = tempfile.mkdtemp(prefix=f"nemo_curator_dynamo_{short_id}_")
        logger.info(f"Dynamo runtime dir: {self._runtime_dir}")

        with ray.init(ignore_reinit_error=True):
            head_node_id = ray.get_runtime_context().get_node_id()
            cluster_nodes = ray.nodes()
            self._head_ip = _resolve_node_ip(head_node_id, nodes=cluster_nodes)

            self._infra_node_id, self._infra_ip = self._resolve_infra_node(cluster_nodes, head_node_id)
            server._host = self._infra_ip

            # Kill any orphaned actors from a previous run with the same server name
            # (e.g. after a Jupyter kernel restart that bypassed atexit cleanup).
            kill_orphaned_actors(ray, f"dynamo_{server.name}_")

            # Resolve ports on the infra node (not the driver) so that
            # port availability is checked where services actually bind.
            infra_node = self._infra_node_id
            self._etcd_port = (
                int(server.etcd_endpoint.rsplit(":", 1)[-1])
                if server.etcd_endpoint
                else get_free_port_on_node(infra_node, DEFAULT_ETCD_PORT)
            )
            self._nats_port = (
                int(server.nats_url.rsplit(":", 1)[-1])
                if server.nats_url
                else get_free_port_on_node(infra_node, DEFAULT_NATS_PORT)
            )
            server.port = get_free_port_on_node(infra_node, server.port)

            actor_cls = _define_subprocess_actor()

            try:
                self._deploy_and_healthcheck(
                    server,
                    actor_cls,
                    head_node_id=head_node_id,
                    cluster_nodes=cluster_nodes,
                )
            except Exception:
                # Still connected to Ray here — kill directly without opening another context.
                for label, actor in self._collect_actors_for_shutdown():
                    _kill_actor(ray, label, actor)
                self._cleanup_runtime_dir()
                raise

    def _deploy_and_healthcheck(
        self,
        server: InferenceServer,
        actor_cls: type,
        *,
        head_node_id: str,
        cluster_nodes: list[dict[str, Any]],
    ) -> None:
        """Launch infra, workers for all models, frontend and wait for health."""
        infra_node_id = self._infra_node_id

        # Infrastructure -- on infra node (head or first non-head when excluded)
        if not server.etcd_endpoint:
            self._etcd_actor = self._start_etcd(actor_cls, infra_node_id, self._etcd_port)
        if not server.nats_url:
            self._nats_actor = self._start_nats(actor_cls, infra_node_id, self._nats_port)

        etcd_endpoint = server.etcd_endpoint or f"http://{self._infra_ip}:{self._etcd_port}"
        nats_url = server.nats_url or f"nats://{self._infra_ip}:{self._nats_port}"
        base_env = {"ETCD_ENDPOINTS": etcd_endpoint, "NATS_SERVER": nats_url}

        # Build GPU inventory once; each model's placement shrinks it so
        # subsequent models are assigned to different GPUs.
        inventory = _get_gpu_inventory(head_node_id, nodes=cluster_nodes)
        expected_models: set[str] = set()
        has_disagg = False

        for model_config in server.models:
            dynamo_cfg = model_config.dynamo_config
            tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
            num_replicas = self._resolve_num_replicas(model_config)
            namespace = dynamo_cfg.get("namespace", "curator")
            request_plane = dynamo_cfg.get("request_plane", "nats")
            event_plane = dynamo_cfg.get("event_plane", "nats")
            is_disagg = dynamo_cfg.get("mode") == "disagg"
            if is_disagg:
                has_disagg = True

            model_name = model_config.model_name or model_config.model_identifier
            expected_models.add(model_name)
            logger.info(f"Deploying model '{model_name}' (TP={tp_size}, replicas={num_replicas})")

            if is_disagg:
                self._launch_disagg_workers(
                    actor_cls,
                    model_config,
                    base_env,
                    head_node_id=head_node_id,
                    cluster_nodes=cluster_nodes,
                    namespace=namespace,
                    request_plane=request_plane,
                    event_plane=event_plane,
                )
            else:
                plans = plan_replica_placement(num_replicas, tp_size, _inventory=inventory)
                self._launch_replicas(
                    actor_cls,
                    plans,
                    model_config,
                    base_env,
                    namespace=namespace,
                    request_plane=request_plane,
                    event_plane=event_plane,
                )
                # Shrink inventory so the next model gets different GPUs.
                used: dict[str, int] = {}
                for plan in plans:
                    for rank in plan.ranks:
                        used[rank.node_id] = used.get(rank.node_id, 0) + rank.num_gpus
                inventory = [
                    {**n, "num_gpus": n["num_gpus"] - used.get(n["node_id"], 0)}
                    for n in inventory
                    if n["num_gpus"] - used.get(n["node_id"], 0) > 0
                ]

        # Frontend -- co-located with infra, auto-discovers all registered models
        first_cfg = server.models[0].dynamo_config
        namespace = first_cfg.get("namespace", "curator")
        request_plane = first_cfg.get("request_plane", "nats")
        event_plane = first_cfg.get("event_plane", "nats")
        router_mode = first_cfg.get("router_mode", "kv") if has_disagg else None
        frontend_runtime_env = self._merge_model_runtime_envs(server.models)
        self._frontend_actor = self._launch_frontend(
            actor_cls,
            infra_node_id,
            server.port,
            base_env,
            namespace=namespace,
            request_plane=request_plane,
            event_plane=event_plane,
            router_mode=router_mode,
            runtime_env=frontend_runtime_env,
        )

        # Health check — must stay inside ray.init context so actor handles
        # (is_alive, read_log_tail) remain valid for subprocess liveness checks.
        self._wait_for_models(server, expected_models)

    def _wait_for_models(self, server: InferenceServer, expected_models: set[str]) -> None:
        """Poll ``/v1/models`` until all *expected_models* appear in the response.

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
                    ids = {m["id"] for m in body.get("data", [])}
                    if expected_models.issubset(ids):
                        logger.info(
                            f"All Dynamo models registered after {attempt} health check(s): {sorted(expected_models)}"
                        )
                        return
                    if server.verbose:
                        missing = sorted(expected_models - ids)
                        logger.debug(f"Models so far: {sorted(ids)}, waiting for: {missing}")
            except Exception:  # noqa: BLE001
                if server.verbose:
                    logger.debug(f"Health check attempt {attempt} failed, retrying...")
            time.sleep(2)

        # Final liveness check before giving up -- surface crash info if available
        self._check_subprocess_health()
        msg = (
            f"Models {sorted(expected_models)} did not all appear at {models_url} "
            f"within {server.health_check_timeout_s}s"
        )
        raise TimeoutError(msg)

    def _check_subprocess_health(self) -> None:
        """Check subprocess liveness via run refs and ``is_alive()`` fallback.

        First checks ``ray.wait()`` on all run refs (fast, no per-actor RPC).
        If any ref has resolved, the subprocess exited — reads the log tail
        and raises.  Then falls back to ``is_alive()`` for actors without
        run refs.

        Note: does NOT use ``with ray.init()`` — the driver must already be
        connected.
        """
        self._check_liveness_via_refs()
        self._check_liveness_via_polling()

    def _get_all_managed_processes(self) -> list[ManagedSubprocess]:
        """Collect all managed processes for health checking."""
        procs: list[ManagedSubprocess] = []
        if self._frontend_actor is not None:
            procs.append(self._frontend_actor)
        procs.extend(self._worker_actors)
        if self._etcd_actor is not None:
            procs.append(self._etcd_actor)
        if self._nats_actor is not None:
            procs.append(self._nats_actor)
        return procs

    def _check_liveness_via_refs(self) -> None:
        """Detect subprocess exits via ``ray.wait()`` on run refs.

        Run refs resolve when the subprocess exits (or the actor dies).
        ``ray.wait()`` with ``timeout=0`` is non-blocking and requires no
        per-actor RPC, making it much faster than ``is_alive()`` polling.
        """
        import ray

        procs = self._get_all_managed_processes()
        ref_to_proc = {p.run_ref: p for p in procs if p.run_ref is not None}
        if not ref_to_proc:
            return

        ready, _ = ray.wait(list(ref_to_proc.keys()), num_returns=len(ref_to_proc), timeout=0)
        for ref in ready:
            proc = ref_to_proc[ref]
            log_tail = self._read_actor_log_tail(proc.actor)
            self._raise_subprocess_error(proc.label, log_tail, reason="subprocess exited unexpectedly")

    def _check_liveness_via_polling(self) -> None:
        """Fallback liveness check via ``is_alive()`` RPC for processes without run refs."""
        import ray

        procs = [p for p in self._get_all_managed_processes() if p.run_ref is None]
        for proc in procs:
            try:
                alive = ray.get(proc.actor.is_alive.remote(), timeout=10)
            except Exception:  # noqa: BLE001
                log_tail = self._read_actor_log_tail(proc.actor)
                if log_tail:
                    self._raise_subprocess_error(
                        proc.label, log_tail, reason="actor unreachable and has log output — likely crashed"
                    )
                continue
            if not alive:
                log_tail = self._read_actor_log_tail(proc.actor)
                self._raise_subprocess_error(proc.label, log_tail, reason="subprocess crashed during startup")

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

    def _resolve_infra_node(self, cluster_nodes: list[dict[str, Any]], head_node_id: str) -> tuple[str, str]:
        """Pick the node that should host etcd, NATS, and the frontend.

        Returns ``(node_id, node_ip)``.  When ``CURATOR_IGNORE_RAY_HEAD_NODE``
        is set, selects the first non-head node (sorted by CPU count, then
        node ID for stability).  Otherwise returns the head node.
        """
        if not _ignore_head_node():
            return head_node_id, self._head_ip

        non_head = [n for n in cluster_nodes if n["NodeID"] != head_node_id and n.get("Alive", False)]
        if not non_head:
            msg = "CURATOR_IGNORE_RAY_HEAD_NODE is set but no non-head nodes are available."
            raise RuntimeError(msg)
        non_head.sort(key=lambda n: (-n.get("Resources", {}).get("CPU", 0), n["NodeID"]))
        infra = non_head[0]
        logger.info(
            f"Head-node exclusion active: infra will run on {infra['NodeManagerAddress']} (node {infra['NodeID'][:8]})"
        )
        return infra["NodeID"], infra["NodeManagerAddress"]

    @staticmethod
    def _merge_model_runtime_envs(models: list[InferenceModelConfig]) -> dict[str, Any] | None:
        """Merge ``runtime_env`` dicts from all model configs for the frontend.

        Returns ``None`` when no model specifies a ``runtime_env``.
        """
        merged: dict[str, Any] = {}
        for m in models:
            if not m.runtime_env:
                continue
            env_vars = {**merged.get("env_vars", {}), **m.runtime_env.get("env_vars", {})}
            merged.update(m.runtime_env)
            if env_vars:
                merged["env_vars"] = env_vars
        return merged or None

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

    def _start_etcd(self, actor_cls: type, infra_node_id: str, port: int) -> ManagedSubprocess:
        data_dir = os.path.join(self._runtime_dir, "etcd_data")
        os.makedirs(data_dir, exist_ok=True)

        peer_port = get_free_port_on_node(infra_node_id, port + 1)
        command = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{port}",
            "--advertise-client-urls",
            f"http://{self._infra_ip}:{port}",
            "--listen-peer-urls",
            f"http://127.0.0.1:{peer_port}",
            "--initial-advertise-peer-urls",
            f"http://127.0.0.1:{peer_port}",
            "--initial-cluster",
            f"default=http://127.0.0.1:{peer_port}",
            "--data-dir",
            data_dir,
        ]
        node_alloc = NodeAllocation(node_id=infra_node_id, node_ip=self._infra_ip, num_gpus=0, node_rank=0)
        proc = spawn_actor(
            actor_cls,
            "etcd",
            command,
            node_alloc,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env={"ALLOW_NONE_AUTHENTICATION": "yes"},
        )

        logger.info(f"Starting etcd on port {port} via {self._infra_ip}")
        _wait_for_port(self._infra_ip, port, timeout_s=30, label="etcd")
        logger.info("etcd is ready")
        return proc

    def _start_nats(self, actor_cls: type, infra_node_id: str, port: int) -> ManagedSubprocess:
        store_dir = os.path.join(self._runtime_dir, "nats_data")
        os.makedirs(store_dir, exist_ok=True)

        command = ["nats-server", "-p", str(port), "-js", "--store_dir", store_dir]
        node_alloc = NodeAllocation(node_id=infra_node_id, node_ip=self._infra_ip, num_gpus=0, node_rank=0)
        proc = spawn_actor(
            actor_cls,
            "nats",
            command,
            node_alloc,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
        )

        logger.info(f"Starting NATS on port {port} via {self._infra_ip}")
        _wait_for_port(self._infra_ip, port, timeout_s=30, label="nats")
        logger.info("NATS is ready")
        return proc

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
                    proc = self._launch_worker(
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
                    proc = self._launch_headless_worker(
                        actor_cls,
                        replica_index=plan.replica_index,
                        model_config=model_config,
                        base_env=base_env,
                        node_alloc=rank,
                        plan=plan,
                    )
                self._worker_actors.append(proc)

    def _launch_disagg_workers(  # noqa: PLR0913
        self,
        actor_cls: type,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        *,
        head_node_id: str,
        cluster_nodes: list[dict[str, Any]],
        namespace: str,
        request_plane: str,
        event_plane: str,
    ) -> None:
        """Launch separate prefill and decode worker pools for disaggregated serving.

        Each worker gets ``tp_size`` GPUs on a single node.  Workers are placed
        greedily on available GPUs.  Prefill workers additionally get
        ``--kv-events-config`` for ZMQ event publishing.

        Multi-node tensor parallelism *within* a single disagg worker is not
        yet supported — each worker's TP group must fit on one node.

        The ``dynamo_config`` dict on *model_config* controls pool sizes:
        - ``prefill_replicas`` (default 1): number of prefill workers
        - ``decode_replicas`` (default 1): number of decode workers
        """
        import json

        dynamo_cfg = model_config.dynamo_config
        num_prefill = dynamo_cfg.get("prefill_replicas", 1)
        num_decode = dynamo_cfg.get("decode_replicas", 1)
        total_workers = num_prefill + num_decode
        tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)

        # Plan placement: one allocation per worker
        plans = plan_replica_placement(total_workers, tp_size, head_node_id, _nodes=cluster_nodes)

        for plan in plans:
            if plan.is_multi_node:
                msg = (
                    f"Disaggregated serving does not yet support multi-node tensor parallelism. "
                    f"Worker {plan.replica_index} requires {plan.total_gpus} GPUs across "
                    f"{plan.nnodes} nodes. Reduce tensor_parallel_size to fit on a single node."
                )
                raise ValueError(msg)

        kv_transfer_config = json.dumps(
            dynamo_cfg.get(
                "kv_transfer_config",
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
            )
        )

        model_name = model_config.model_name or model_config.model_identifier
        worker_index = 0

        # Launch decode workers first (matching Dynamo example convention)
        for i in range(num_decode):
            plan = plans[worker_index]
            rank0 = plan.ranks[0]
            nixl_port = get_free_port_on_node(rank0.node_id, 20097 + worker_index)

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
                "--disaggregation-mode",
                "decode",
                "--kv-transfer-config",
                kv_transfer_config,
                *_engine_kwargs_to_cli_flags(model_config.engine_kwargs),
            ]

            label = f"decode_{i}"
            logger.info(f"Disagg decode worker {i}: {rank0.num_gpus} GPU(s) on {rank0.node_ip}, nixl_port={nixl_port}")
            proc = spawn_actor(
                actor_cls,
                label,
                command,
                rank0,
                runtime_dir=self._runtime_dir,
                actor_name_prefix=self._actor_name_prefix,
                subprocess_env={**base_env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port), "PYTHONHASHSEED": "0"},
                runtime_env=model_config.runtime_env or None,
            )
            self._worker_actors.append(proc)
            worker_index += 1

        # Launch prefill workers
        for i in range(num_prefill):
            plan = plans[worker_index]
            rank0 = plan.ranks[0]
            nixl_port = get_free_port_on_node(rank0.node_id, 20097 + worker_index)
            kv_events_port = get_free_port_on_node(rank0.node_id, 20081 + i)

            kv_events_config = json.dumps(
                dynamo_cfg.get(
                    "kv_events_config",
                    {
                        "publisher": "zmq",
                        "topic": "kv-events",
                        "endpoint": f"tcp://*:{kv_events_port}",
                        "enable_kv_cache_events": True,
                    },
                )
            )

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
                "--disaggregation-mode",
                "prefill",
                "--kv-transfer-config",
                kv_transfer_config,
                "--kv-events-config",
                kv_events_config,
                *_engine_kwargs_to_cli_flags(model_config.engine_kwargs),
            ]

            label = f"prefill_{i}"
            logger.info(
                f"Disagg prefill worker {i}: {rank0.num_gpus} GPU(s) on {rank0.node_ip}, "
                f"nixl_port={nixl_port}, kv_events_port={kv_events_port}"
            )
            proc = spawn_actor(
                actor_cls,
                label,
                command,
                rank0,
                runtime_dir=self._runtime_dir,
                actor_name_prefix=self._actor_name_prefix,
                subprocess_env={**base_env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port), "PYTHONHASHSEED": "0"},
                runtime_env=model_config.runtime_env or None,
            )
            self._worker_actors.append(proc)
            worker_index += 1

        logger.info(
            f"Disaggregated serving: {num_decode} decode + {num_prefill} prefill "
            f"workers launched ({total_workers * tp_size} GPUs total, TP={tp_size})"
        )

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
    ) -> ManagedSubprocess:
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
        return spawn_actor(
            actor_cls,
            label,
            command,
            node_alloc,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=base_env,
            runtime_env=model_config.runtime_env or None,
        )

    def _launch_headless_worker(  # noqa: PLR0913
        self,
        actor_cls: type,
        *,
        replica_index: int,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        node_alloc: NodeAllocation,
        plan: ReplicaPlan,
    ) -> ManagedSubprocess:
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
        return spawn_actor(
            actor_cls,
            label,
            command,
            node_alloc,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=base_env,
            runtime_env=model_config.runtime_env or None,
        )

    def _launch_frontend(  # noqa: PLR0913
        self,
        actor_cls: type,
        infra_node_id: str,
        port: int,
        base_env: dict[str, str],
        *,
        namespace: str,
        request_plane: str,
        event_plane: str,
        router_mode: str | None = None,
        runtime_env: dict[str, Any] | None = None,
    ) -> ManagedSubprocess:
        frontend_env = dict(base_env)
        if router_mode:
            frontend_env["PYTHONHASHSEED"] = "0"
        command = [
            sys.executable,
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(port),
            "--namespace",
            namespace,
            "--discovery-backend",
            "etcd",
            "--request-plane",
            request_plane,
            "--event-plane",
            event_plane,
        ]
        if router_mode:
            command.extend(["--router-mode", router_mode, "--router-reset-states"])

        node_alloc = NodeAllocation(node_id=infra_node_id, node_ip=self._infra_ip, num_gpus=0, node_rank=0)
        logger.info(f"Starting Dynamo frontend on port {port}")
        return spawn_actor(
            actor_cls,
            "frontend",
            command,
            node_alloc,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=frontend_env,
            runtime_env=runtime_env,
        )

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
                kill_orphaned_actors(ray, f"dynamo_{self._server.name}_")
        except Exception:  # noqa: BLE001
            logger.debug("Could not connect to Ray during Dynamo shutdown (cluster may be gone)")

        self._cleanup_runtime_dir()
        self._server._host = "localhost"
        logger.info("Dynamo backend stopped")

    def _collect_actors_for_shutdown(self) -> list[tuple[str, Any]]:
        """Collect all actors in reverse-start order: frontend -> workers -> nats -> etcd."""
        actors: list[tuple[str, Any]] = []

        if self._frontend_actor is not None:
            actors.append((self._frontend_actor.label, self._frontend_actor.actor))
            self._frontend_actor = None

        for proc in self._worker_actors:
            actors.append((proc.label, proc.actor))
        self._worker_actors.clear()

        if self._nats_actor is not None:
            actors.append((self._nats_actor.label, self._nats_actor.actor))
            self._nats_actor = None

        if self._etcd_actor is not None:
            actors.append((self._etcd_actor.label, self._etcd_actor.actor))
            self._etcd_actor = None

        return actors

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
