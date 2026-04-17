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
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

from nemo_curator.core.serve.internal.base import InferenceBackend
from nemo_curator.core.serve.internal.constants import (
    DEFAULT_DYNAMO_EVENT_PLANE,
    DEFAULT_DYNAMO_NAMESPACE,
    DEFAULT_DYNAMO_REQUEST_PLANE,
    DEFAULT_ETCD_PORT,
    DEFAULT_NATS_PORT,
)
from nemo_curator.core.serve.internal.errors import SubprocessError
from nemo_curator.core.serve.internal.subprocess_mgr import (
    NEMO_CURATOR_DYNAMO_NAMESPACE,
    ManagedSubprocess,
    ReplicaBundleSpec,
    _check_binary,
    _engine_kwargs_to_cli_flags,
    _get_gpu_topology,
    _wait_for_port,
    build_infra_pg,
    build_replica_pg,
    build_worker_actor_name,
    check_total_gpu_capacity,
    get_bundle_node_ip,
    get_free_port_in_bundle,
    graceful_stop_actors,
    plan_replica_bundle_shape,
    remove_named_pgs_with_prefix,
    spawn_actor,
)
from nemo_curator.core.serve.server import InferenceModelConfig

if TYPE_CHECKING:
    from nemo_curator.core.serve.server import InferenceServer


_INFRA_ETCD_BUNDLE = 0
_INFRA_NATS_BUNDLE = 1
_INFRA_FRONTEND_BUNDLE = 2
_INFRA_NUM_BUNDLES = 3


def _model_name_to_component(name: str) -> str:
    """Sanitize a model name into a valid Dynamo component name.

    Dynamo endpoints use ``dyn://namespace.component.endpoint`` format where
    dots are delimiters. This replaces all non-alphanumeric characters with
    underscores to produce a safe component name.

    Differs from ``nemo_curator.stages.text.models.utils.format_name_with_suffix``
    which only takes the last path component and does not replace dots --
    both are required here to avoid collisions across HuggingFace orgs and
    to keep dots out of the ``dyn://`` URI.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        msg = f"Model name {name!r} produces an empty component slug after sanitization."
        raise ValueError(msg)
    return slug


def _dynamo_endpoint(namespace: str, component: str, role: str | None = None) -> str:
    """Build a ``dyn://namespace.component.endpoint`` URI for worker registration."""
    suffix = f"_{role}" if role else ""
    return f"dyn://{namespace}.{component}{suffix}.generate"


# Default runtime_env for Dynamo workers -- installs ai-dynamo[vllm] which
# brings the exact vLLM version matching the installed ai-dynamo release.
_DYNAMO_VLLM_RUNTIME_ENV: dict[str, Any] = {
    "uv": ["ai-dynamo[vllm]"],
}


_FRONTEND_ROUTER_KEYS = (
    "router_mode",
    "router_kv_events",
    "router_kv_overlap_score_weight",
    "router_temperature",
    "router_queue_threshold",
    "router_ttl_secs",
    "router_max_tree_size",
    "router_prune_target_ratio",
    "router_reset_states",
)


class DynamoBackend(InferenceBackend):
    """NVIDIA Dynamo inference backend, built on Ray placement groups.

    Each model replica is backed by one placement group whose bundles
    describe its TP topology:

    - **Single-node TP** (``tp_size <= GPUs-per-node``): one bundle of
      ``tp_size`` GPUs with ``STRICT_PACK``. One Ray actor running
      ``python -m dynamo.vllm``.
    - **Multi-node TP** (``tp_size > GPUs-per-node``): ``nnodes`` bundles of
      ``tp_size / nnodes`` GPUs each with ``STRICT_SPREAD``. Rank 0 runs the
      full Dynamo worker (model registration in etcd); rank 1+ run
      ``dynamo.vllm --headless`` workers coordinated via ``torch.distributed``.
      ``--master-addr`` is resolved by querying the rank-0 bundle's node IP
      after ``pg.ready()``.

    Infra services (etcd, NATS, frontend) share a ``STRICT_PACK`` PG so they
    co-locate. When ``CURATOR_IGNORE_RAY_HEAD_NODE=1``, every bundle carries
    ``{"ray.io/node-type": "worker"}`` as a label selector, keeping both
    workers and infra off the head node.

    Teardown: the primary path is ``actor.stop.remote()`` (graceful SIGTERM
    on the subprocess group) followed by ``remove_placement_group(pg)``
    which reaps any remaining actors atomically. A
    ``remove_named_pgs_with_prefix`` sweep catches leftovers from prior
    driver sessions (e.g. after a Jupyter kernel restart).
    """

    def __init__(self, server: InferenceServer) -> None:
        self._server = server
        self._runtime_dir: str | None = None
        self._infra_pg: PlacementGroup | None = None
        self._replica_pgs: list[PlacementGroup] = []
        self._infra_ip: str | None = None
        self._etcd_actor: ManagedSubprocess | None = None
        self._nats_actor: ManagedSubprocess | None = None
        self._worker_actors: list[ManagedSubprocess] = []
        self._frontend_actor: ManagedSubprocess | None = None
        self._etcd_port: int | None = None
        self._nats_port: int | None = None
        self._actor_name_prefix: str = ""
        self._pg_name_prefix: str = ""

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _write_manifest(self, data: dict[str, Any], *, ready: bool) -> None:
        """Write deployment manifest to ``{runtime_dir}/manifest.json`` and log it."""
        manifest = {**data, "ready": ready, "timestamp": time.time()}

        logger.info(f"Deployment manifest (ready={ready}): {json.dumps(manifest, indent=2)}")

        if not self._runtime_dir:
            return
        manifest_path = os.path.join(self._runtime_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_frontend_router_config(server: InferenceServer) -> dict[str, Any]:
        """Resolve a single set of frontend router settings from all models.

        Returns a dict with the effective value for each ``_FRONTEND_ROUTER_KEYS``
        key. Values are taken from the first model that explicitly sets each
        key, falling back to Dynamo defaults. When no model has a
        ``router_mode`` set, disagg models default to ``"kv"``; otherwise
        ``None`` (round-robin).
        """

        def _first_explicit(key: str) -> str | float | bool | int | None:
            return next((m.dynamo_config[key] for m in server.models if key in m.dynamo_config), None)

        any_disagg = any(m.dynamo_config.get("mode") == "disagg" for m in server.models)
        explicit_mode = _first_explicit("router_mode")
        router_mode = explicit_mode if explicit_mode is not None else ("kv" if any_disagg else None)

        explicit_kv_events = _first_explicit("router_kv_events")
        router_kv_events = explicit_kv_events if explicit_kv_events is not None else True

        def _with_default(key: str, default: str | float | bool) -> str | float | bool:
            val = _first_explicit(key)
            return val if val is not None else default

        return {
            "router_mode": router_mode,
            "router_kv_events": router_kv_events,
            "router_kv_overlap_score_weight": _with_default("router_kv_overlap_score_weight", 1.0),
            "router_temperature": _with_default("router_temperature", 0.0),
            "router_queue_threshold": _first_explicit("router_queue_threshold"),
            "router_ttl_secs": _with_default("router_ttl_secs", 120.0),
            "router_max_tree_size": _with_default("router_max_tree_size", 2**20),
            "router_prune_target_ratio": _with_default("router_prune_target_ratio", 0.8),
            "router_reset_states": _with_default("router_reset_states", False),
        }

    @staticmethod
    def _validate_frontend_config(server: InferenceServer) -> None:
        """Reject mismatched frontend-wide settings across models.

        The Dynamo frontend uses a single namespace, plane config, and router
        config. If models specify conflicting values, they would be silently
        ignored -- fail loud instead.
        """
        if len(server.models) <= 1:
            return

        ref_router = DynamoBackend._resolve_frontend_router_config(server)
        first = server.models[0].dynamo_config
        ref_ns = first.get("namespace", DEFAULT_DYNAMO_NAMESPACE)
        ref_rp = first.get("request_plane", DEFAULT_DYNAMO_REQUEST_PLANE)
        ref_ep = first.get("event_plane", DEFAULT_DYNAMO_EVENT_PLANE)

        for i, m in enumerate(server.models[1:], start=1):
            cfg = m.dynamo_config
            mismatches = []

            ns = cfg.get("namespace", DEFAULT_DYNAMO_NAMESPACE)
            rp = cfg.get("request_plane", DEFAULT_DYNAMO_REQUEST_PLANE)
            ep = cfg.get("event_plane", DEFAULT_DYNAMO_EVENT_PLANE)
            if ns != ref_ns:
                mismatches.append(f"namespace: {ns!r} vs {ref_ns!r}")
            if rp != ref_rp:
                mismatches.append(f"request_plane: {rp!r} vs {ref_rp!r}")
            if ep != ref_ep:
                mismatches.append(f"event_plane: {ep!r} vs {ref_ep!r}")

            effective_router = {**ref_router, **{k: cfg[k] for k in _FRONTEND_ROUTER_KEYS if k in cfg}}
            for key, ref_value in ref_router.items():
                if effective_router[key] != ref_value:
                    mismatches.append(f"{key}: {effective_router[key]!r} vs {ref_value!r}")

            if mismatches:
                model_name = m.model_name or m.model_identifier
                msg = (
                    f"Model '{model_name}' (index {i}) has frontend config that differs "
                    f"from model 0: {', '.join(mismatches)}. All models must share the "
                    f"same frontend configuration."
                )
                raise ValueError(msg)

    @staticmethod
    def _validate_unique_model_names(server: InferenceServer) -> None:
        """Reject duplicate model names and component-slug collisions."""
        seen_names: dict[str, int] = {}
        seen_components: dict[str, tuple[int, str]] = {}
        for i, m in enumerate(server.models):
            name = m.model_name or m.model_identifier
            if name in seen_names:
                msg = (
                    f"Duplicate model name {name!r} at index {i} "
                    f"(first seen at index {seen_names[name]}). "
                    f"When deploying the same model_identifier multiple times, "
                    f"each must have a distinct model_name."
                )
                raise ValueError(msg)
            seen_names[name] = i

            comp = _model_name_to_component(name)
            if comp in seen_components:
                prev_idx, prev_name = seen_components[comp]
                msg = (
                    f"Model names {prev_name!r} (index {prev_idx}) and "
                    f"{name!r} (index {i}) both sanitize to component "
                    f"{comp!r}. Use more distinct model_name values."
                )
                raise ValueError(msg)
            seen_components[comp] = (i, name)

    @staticmethod
    def _validate_gpu_requirements(server: InferenceServer, topology: list[dict[str, Any]] | None = None) -> None:
        """Coarse fail-fast: reject configs that obviously exceed cluster capacity.

        Check A: disagg models require TP to fit on a single node.
        Check B: total GPU demand across all models must not exceed supply.

        Ray's per-PG ``STRICT_PACK`` / ``STRICT_SPREAD`` is the authoritative
        admission gate; this is a coarse pre-check for better error messages.
        """
        if topology is None:
            topology = _get_gpu_topology()
        max_gpus_per_node = max((n["num_gpus"] for n in topology), default=0)
        total_needed = 0

        for model_config in server.models:
            dynamo_cfg = model_config.dynamo_config
            is_disagg = dynamo_cfg.get("mode") == "disagg"
            model_name = model_config.model_name or model_config.model_identifier

            if is_disagg:
                (num_prefill, prefill_ek), (num_decode, decode_ek) = DynamoBackend._resolve_disagg_role_config(
                    model_config
                )
                prefill_tp = prefill_ek.get("tensor_parallel_size", 1)
                decode_tp = decode_ek.get("tensor_parallel_size", 1)

                for role, tp in [("prefill", prefill_tp), ("decode", decode_tp)]:
                    if tp > max_gpus_per_node:
                        msg = (
                            f"Model '{model_name}' {role} requests TP={tp} in disaggregated mode, "
                            f"but max GPUs per node is {max_gpus_per_node}. "
                            f"Disaggregated mode does not support multi-node TP."
                        )
                        raise ValueError(msg)

                total_needed += num_prefill * prefill_tp + num_decode * decode_tp
            else:
                tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
                num_replicas = DynamoBackend._resolve_num_replicas(model_config)
                total_needed += num_replicas * tp_size

        check_total_gpu_capacity(total_needed)

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
        self._pg_name_prefix = f"dynamo_{server.name}_"
        self._actor_name_prefix = f"{self._pg_name_prefix}{short_id}"
        self._runtime_dir = tempfile.mkdtemp(prefix=f"nemo_curator_dynamo_{short_id}_")
        logger.info(f"Dynamo runtime dir: {self._runtime_dir}")

        with ray.init(namespace=NEMO_CURATOR_DYNAMO_NAMESPACE, ignore_reinit_error=True):
            # Reap PGs + actors from previous runs with the same server name
            # (e.g. after a Jupyter kernel restart that bypassed atexit).
            remove_named_pgs_with_prefix(self._pg_name_prefix)

            try:
                self._deploy_and_healthcheck(server)
            except Exception:
                # Still connected to Ray -- tear down inside the same context.
                self._teardown_actors_and_pgs(ray)
                self._cleanup_runtime_dir()
                raise

    def _deploy_and_healthcheck(self, server: InferenceServer) -> None:
        """Validate config, create PGs, launch infra/workers/frontend, health check.

        Sequence: fail-fast validations -> infra PG -> etcd/NATS -> per-model
        replica PGs + worker actors -> manifest (ready=False) -> frontend ->
        health check -> manifest (ready=True).
        """
        # Snapshot cluster topology once; all per-replica planning reuses it
        # instead of re-calling ``ray.nodes()`` per model/replica.
        topology = _get_gpu_topology()

        self._validate_gpu_requirements(server, topology)
        self._validate_frontend_config(server)
        self._validate_unique_model_names(server)

        # Infra PG (etcd + NATS + frontend) -- STRICT_PACK + optional
        # worker-label selector for head-node exclusion.
        infra_pg_name = f"{self._actor_name_prefix}_pg_infra"
        self._infra_pg = build_infra_pg(name=infra_pg_name, num_bundles=_INFRA_NUM_BUNDLES)
        self._infra_ip = get_bundle_node_ip(self._infra_pg, _INFRA_ETCD_BUNDLE)
        server._host = self._infra_ip

        self._etcd_port = (
            int(server.etcd_endpoint.rsplit(":", 1)[-1])
            if server.etcd_endpoint
            else get_free_port_in_bundle(self._infra_pg, _INFRA_ETCD_BUNDLE, DEFAULT_ETCD_PORT)
        )
        self._nats_port = (
            int(server.nats_url.rsplit(":", 1)[-1])
            if server.nats_url
            else get_free_port_in_bundle(self._infra_pg, _INFRA_NATS_BUNDLE, DEFAULT_NATS_PORT)
        )
        server.port = get_free_port_in_bundle(self._infra_pg, _INFRA_FRONTEND_BUNDLE, server.port)

        if not server.etcd_endpoint:
            self._etcd_actor = self._start_etcd(self._etcd_port)
        if not server.nats_url:
            self._nats_actor = self._start_nats(self._nats_port)

        etcd_endpoint = server.etcd_endpoint or f"http://{self._infra_ip}:{self._etcd_port}"
        nats_url = server.nats_url or f"nats://{self._infra_ip}:{self._nats_port}"
        base_env = {"ETCD_ENDPOINTS": etcd_endpoint, "NATS_SERVER": nats_url}

        expected_models: set[str] = set()
        placements: list[dict[str, Any]] = []

        for model_config in server.models:
            dynamo_cfg = model_config.dynamo_config
            namespace = dynamo_cfg.get("namespace", DEFAULT_DYNAMO_NAMESPACE)
            request_plane = dynamo_cfg.get("request_plane", DEFAULT_DYNAMO_REQUEST_PLANE)
            event_plane = dynamo_cfg.get("event_plane", DEFAULT_DYNAMO_EVENT_PLANE)
            is_disagg = dynamo_cfg.get("mode") == "disagg"

            model_name = model_config.model_name or model_config.model_identifier
            expected_models.add(model_name)

            if is_disagg:
                entries = self._launch_disagg_workers(
                    model_config,
                    base_env,
                    namespace=namespace,
                    request_plane=request_plane,
                    event_plane=event_plane,
                    topology=topology,
                )
            else:
                tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
                num_replicas = self._resolve_num_replicas(model_config)
                logger.info(f"Deploying model '{model_name}' (TP={tp_size}, replicas={num_replicas})")
                entries = self._launch_replicas(
                    model_config,
                    base_env,
                    num_replicas=num_replicas,
                    namespace=namespace,
                    request_plane=request_plane,
                    event_plane=event_plane,
                    topology=topology,
                )
            placements.extend(entries)

        manifest_data = {
            "models": sorted(expected_models),
            "endpoint": server.endpoint,
            "etcd": etcd_endpoint,
            "nats": nats_url,
            "port": server.port,
            "placements": placements,
        }
        self._write_manifest(manifest_data, ready=False)

        # Frontend -- auto-discovers all registered models.
        first_cfg = server.models[0].dynamo_config
        namespace = first_cfg.get("namespace", DEFAULT_DYNAMO_NAMESPACE)
        request_plane = first_cfg.get("request_plane", DEFAULT_DYNAMO_REQUEST_PLANE)
        event_plane = first_cfg.get("event_plane", DEFAULT_DYNAMO_EVENT_PLANE)
        frontend_router_cfg = self._resolve_frontend_router_config(server)
        frontend_runtime_env = self._merge_model_runtime_envs(server.models)
        self._frontend_actor = self._launch_frontend(
            server.port,
            base_env,
            namespace=namespace,
            request_plane=request_plane,
            event_plane=event_plane,
            frontend_router_cfg=frontend_router_cfg,
            runtime_env=frontend_runtime_env,
        )

        # Health check -- stay inside ray.init context so actor handles remain valid.
        self._wait_for_models(server, expected_models)

        self._write_manifest(manifest_data, ready=True)

    def _wait_for_models(self, server: InferenceServer, expected_models: set[str]) -> None:
        """Poll ``/v1/models`` until all *expected_models* appear."""
        import urllib.request

        models_url = f"{server.endpoint}/models"
        deadline = time.monotonic() + server.health_check_timeout_s
        start_time = time.monotonic()
        attempt = 0
        last_error: str | None = None
        models_found: set[str] = set()
        while time.monotonic() < deadline:
            attempt += 1

            self._check_subprocess_health()

            try:
                resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
                if resp.status == http.HTTPStatus.OK:
                    body = json.loads(resp.read())
                    models_found = {m["id"] for m in body.get("data", [])}
                    if expected_models.issubset(models_found):
                        logger.info(
                            f"All Dynamo models registered after {attempt} health check(s): {sorted(expected_models)}"
                        )
                        return
                    if server.verbose:
                        missing = sorted(expected_models - models_found)
                        logger.debug(f"Models so far: {sorted(models_found)}, waiting for: {missing}")
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if server.verbose:
                    logger.debug(f"Health check attempt {attempt} failed, retrying...")
            time.sleep(2)

        self._check_subprocess_health()
        elapsed_s = round(time.monotonic() - start_time, 1)
        msg = (
            f"Models {sorted(expected_models)} did not all appear at {models_url} "
            f"within {server.health_check_timeout_s}s"
        )
        raise SubprocessError(
            msg,
            debug_context={
                "backend": "dynamo",
                "models_expected": sorted(expected_models),
                "models_found": sorted(models_found),
                "elapsed_s": elapsed_s,
                "last_error": last_error,
            },
        )

    def _check_subprocess_health(self) -> None:
        """Detect subprocess exits via ``ray.wait()`` on run refs."""
        import ray

        procs: list[ManagedSubprocess] = []
        if self._frontend_actor is not None:
            procs.append(self._frontend_actor)
        procs.extend(self._worker_actors)
        if self._etcd_actor is not None:
            procs.append(self._etcd_actor)
        if self._nats_actor is not None:
            procs.append(self._nats_actor)

        ref_to_proc = {p.run_ref: p for p in procs if p.run_ref is not None}
        if not ref_to_proc:
            return

        ready, _ = ray.wait(list(ref_to_proc.keys()), num_returns=len(ref_to_proc), timeout=0)
        for ref in ready:
            proc = ref_to_proc[ref]
            log_tail = self._read_actor_log_tail(proc.actor)
            self._raise_subprocess_error(proc.label, log_tail, reason="subprocess exited unexpectedly")

    @staticmethod
    def _read_actor_log_tail(actor: Any) -> str:  # noqa: ANN401
        """Read log tail from a subprocess actor, returning empty string on failure."""
        import ray

        with contextlib.suppress(Exception):
            return ray.get(actor.read_log_tail.remote(), timeout=5)
        return ""

    @staticmethod
    def _raise_subprocess_error(label: str, log_tail: str, *, reason: str) -> None:
        """Raise ``SubprocessError`` with formatted subprocess crash info."""
        tail = "\n".join(log_tail.splitlines()[-50:]) if log_tail else "(no log output)"
        msg = f"Dynamo {label} {reason}.\n\n--- {label} log (last 50 lines) ---\n{tail}"
        raise SubprocessError(
            msg,
            debug_context={"label": label, "reason": reason, "log_tail": tail},
        )

    @staticmethod
    def _dynamo_runtime_env(model_config: InferenceModelConfig) -> dict[str, Any]:
        """Build the runtime_env for a Dynamo worker/frontend actor."""
        return InferenceModelConfig._merge_runtime_envs(
            _DYNAMO_VLLM_RUNTIME_ENV,
            model_config.runtime_env or None,
        )

    @staticmethod
    def _merge_model_runtime_envs(models: list[InferenceModelConfig]) -> dict[str, Any]:
        """Merge ``runtime_env`` dicts from all model configs for the frontend."""
        from functools import reduce

        envs = [m.runtime_env for m in models if m.runtime_env]
        user_merged = reduce(InferenceModelConfig._merge_runtime_envs, envs) if envs else None
        return InferenceModelConfig._merge_runtime_envs(
            _DYNAMO_VLLM_RUNTIME_ENV,
            user_merged,
        )

    @staticmethod
    def _resolve_disagg_role_config(
        model_config: InferenceModelConfig,
    ) -> tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]]:
        """Resolve per-role replica count and engine_kwargs for disaggregated serving."""
        dynamo_cfg = model_config.dynamo_config
        base_ek = model_config.engine_kwargs

        def _resolve_role(role: str) -> tuple[int, dict[str, Any]]:
            role_cfg = dynamo_cfg.get(role, {})
            num = role_cfg.get("num_replicas", 1)
            return num, {**base_ek, **role_cfg.get("engine_kwargs", {})}

        return _resolve_role("prefill"), _resolve_role("decode")

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

    def _start_infra_service(
        self,
        *,
        label: str,
        bundle_index: int,
        port: int,
        command: list[str],
        subprocess_env: dict[str, str] | None = None,
    ) -> ManagedSubprocess:
        """Spawn an infra subprocess into ``_infra_pg`` and wait for its port."""
        proc = spawn_actor(
            label,
            self._infra_pg,
            bundle_index,
            num_gpus=0,
            command=command,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=subprocess_env,
        )
        short_label = label.rsplit("_", 1)[-1].lower()
        logger.info(f"Starting {short_label} on port {port} via {self._infra_ip}")
        _wait_for_port(self._infra_ip, port, timeout_s=30, label=short_label)
        logger.info(f"{short_label} is ready")
        return proc

    def _start_etcd(self, port: int) -> ManagedSubprocess:
        data_dir = os.path.join(self._runtime_dir, "etcd_data")
        os.makedirs(data_dir, exist_ok=True)
        peer_port = get_free_port_in_bundle(self._infra_pg, _INFRA_ETCD_BUNDLE, port + 1)
        return self._start_infra_service(
            label="Dynamo_ETCD",
            bundle_index=_INFRA_ETCD_BUNDLE,
            port=port,
            command=[
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
            ],
            subprocess_env={"ALLOW_NONE_AUTHENTICATION": "yes"},
        )

    def _start_nats(self, port: int) -> ManagedSubprocess:
        store_dir = os.path.join(self._runtime_dir, "nats_data")
        os.makedirs(store_dir, exist_ok=True)
        return self._start_infra_service(
            label="Dynamo_NATS",
            bundle_index=_INFRA_NATS_BUNDLE,
            port=port,
            command=["nats-server", "-p", str(port), "-js", "--store_dir", store_dir],
        )

    # ------------------------------------------------------------------
    # Worker / frontend actors
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregated_model_uses_exact_kv_events(model_config: InferenceModelConfig) -> bool:
        """Return True if an aggregated model should publish ZMQ KV events."""
        cfg = model_config.dynamo_config
        if cfg.get("mode") == "disagg":
            return False
        if cfg.get("router_mode") != "kv":
            return False
        return cfg.get("router_kv_events", True)

    @staticmethod
    def _build_worker_kv_events_config(
        model_config: InferenceModelConfig,
        *,
        pg: PlacementGroup,
        bundle_index: int,
        port_seed: int,
        enabled: bool,
    ) -> str:
        """Build a ``--kv-events-config`` JSON string for a Dynamo worker.

        When *enabled*, allocates a unique port via ``get_free_port_in_bundle``
        and merges the user's optional ``kv_events_config`` template (but
        always owns ``endpoint`` and ``enable_kv_cache_events``).

        When not enabled, explicitly disables KV events to prevent Dynamo's
        auto-configuration from binding a conflicting default port.
        """
        template = dict(model_config.dynamo_config.get("kv_events_config", {}))

        if not enabled:
            template["enable_kv_cache_events"] = False
            template.pop("endpoint", None)
            return json.dumps(template)

        kv_events_port = get_free_port_in_bundle(pg, bundle_index, port_seed)
        template.update(
            {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{kv_events_port}",
                "enable_kv_cache_events": True,
            }
        )
        return json.dumps(template)

    def _launch_replicas(  # noqa: PLR0913
        self,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        *,
        num_replicas: int,
        namespace: str,
        request_plane: str,
        event_plane: str,
        topology: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Plan PGs and launch all worker actors for a non-disagg model.

        Returns a list of manifest entries (one per replica).
        """
        tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
        model_name = model_config.model_name or model_config.model_identifier
        entries: list[dict[str, Any]] = []

        for replica_index in range(num_replicas):
            spec = plan_replica_bundle_shape(tp_size, _topology=topology)
            pg_name = f"{self._actor_name_prefix}_pg_{_model_name_to_component(model_name)}_DP{replica_index}"
            pg = build_replica_pg(spec, name=pg_name)
            self._replica_pgs.append(pg)

            master_addr = get_bundle_node_ip(pg, 0) if spec.is_multi_node else None
            if spec.is_multi_node:
                logger.info(
                    f"Replica {replica_index}: multi-node TP across {spec.nnodes} nodes "
                    f"(total {spec.total_gpus} GPUs, master={master_addr})"
                )
            else:
                logger.info(f"Replica {replica_index}: single-node, {spec.total_gpus} GPU(s)")

            for bundle_index in range(spec.nnodes):
                if bundle_index == 0:
                    proc = self._launch_worker(
                        replica_index=replica_index,
                        model_config=model_config,
                        base_env=base_env,
                        pg=pg,
                        bundle_index=bundle_index,
                        num_gpus=spec.per_node_gpus,
                        namespace=namespace,
                        request_plane=request_plane,
                        event_plane=event_plane,
                        spec=spec,
                        master_addr=master_addr,
                    )
                else:
                    proc = self._launch_headless_worker(
                        replica_index=replica_index,
                        model_config=model_config,
                        base_env=base_env,
                        pg=pg,
                        bundle_index=bundle_index,
                        num_gpus=spec.per_node_gpus,
                        spec=spec,
                        master_addr=master_addr,
                    )
                self._worker_actors.append(proc)

            entries.append(
                {
                    "model": model_name,
                    "replica": replica_index,
                    "nnodes": spec.nnodes,
                    "gpus_per_node": spec.per_node_gpus,
                    "multi_node": spec.is_multi_node,
                    "master_addr": master_addr,
                }
            )

        return entries

    def _launch_disagg_workers(  # noqa: PLR0913
        self,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        *,
        namespace: str,
        request_plane: str,
        event_plane: str,
        topology: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Launch separate prefill and decode worker pools for disaggregated serving.

        Each role can have its own ``num_replicas`` and ``engine_kwargs``.
        Each worker (prefill or decode) gets its own PG. Multi-node TP
        within a disagg worker is not supported -- each worker's TP group
        must fit on one node.
        """
        dynamo_cfg = model_config.dynamo_config
        (num_prefill, prefill_ek), (num_decode, decode_ek) = self._resolve_disagg_role_config(model_config)
        prefill_tp = prefill_ek.get("tensor_parallel_size", 1)
        decode_tp = decode_ek.get("tensor_parallel_size", 1)

        kv_transfer_config = json.dumps(
            dynamo_cfg.get(
                "kv_transfer_config",
                {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
            )
        )

        model_name = model_config.model_name or model_config.model_identifier
        component = _model_name_to_component(model_name)
        entries: list[dict[str, Any]] = []
        worker_index = 0

        # Decode first (matching Dynamo example convention), then prefill.
        # Prefill alone publishes KV events; decode does not.
        for role, num_workers, engine_kwargs, publishes_kv_events in (
            ("decode", num_decode, decode_ek, False),
            ("prefill", num_prefill, prefill_ek, True),
        ):
            role_entries, worker_index = self._launch_disagg_role(
                model_config,
                base_env,
                role=role,
                num_workers=num_workers,
                engine_kwargs=engine_kwargs,
                publishes_kv_events=publishes_kv_events,
                namespace=namespace,
                request_plane=request_plane,
                event_plane=event_plane,
                component=component,
                kv_transfer_config=kv_transfer_config,
                worker_index_start=worker_index,
                topology=topology,
            )
            entries.extend(role_entries)

        total_gpus = num_decode * decode_tp + num_prefill * prefill_tp
        tp_desc = f"TP={decode_tp}" if decode_tp == prefill_tp else f"prefill_TP={prefill_tp}, decode_TP={decode_tp}"
        logger.info(
            f"Disaggregated serving: {num_decode} decode + {num_prefill} prefill "
            f"workers launched ({total_gpus} GPUs total, {tp_desc})"
        )
        return entries

    def _launch_disagg_role(  # noqa: PLR0913
        self,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        *,
        role: Literal["decode", "prefill"],
        num_workers: int,
        engine_kwargs: dict[str, Any],
        publishes_kv_events: bool,
        namespace: str,
        request_plane: str,
        event_plane: str,
        component: str,
        kv_transfer_config: str,
        worker_index_start: int,
        topology: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Launch the N workers for a single disagg role (decode or prefill)."""
        tp_size = engine_kwargs.get("tensor_parallel_size", 1)
        model_name = model_config.model_name or model_config.model_identifier
        entries: list[dict[str, Any]] = []
        worker_index = worker_index_start

        for i in range(num_workers):
            spec = self._plan_disagg_shape(
                tp_size, role=role, worker_index=i, model_name=model_name, topology=topology
            )
            pg_name = f"{self._actor_name_prefix}_pg_{component}_{role}_{i}"
            pg = build_replica_pg(spec, name=pg_name)
            self._replica_pgs.append(pg)

            nixl_port = get_free_port_in_bundle(pg, 0, 20097 + worker_index)

            role_args: list[str] = []
            if publishes_kv_events:
                kv_events_config = self._build_worker_kv_events_config(
                    model_config, pg=pg, bundle_index=0, port_seed=20081 + i, enabled=True
                )
                role_args.extend(["--kv-events-config", kv_events_config])

            python_args = [
                "-m",
                "dynamo.vllm",
                "--model",
                model_config.model_identifier,
                "--served-model-name",
                model_name,
                "--endpoint",
                _dynamo_endpoint(namespace, component, role=role),
                "--discovery-backend",
                "etcd",
                "--request-plane",
                request_plane,
                "--event-plane",
                event_plane,
                "--disaggregation-mode",
                role,
                "--kv-transfer-config",
                kv_transfer_config,
                *role_args,
                *_engine_kwargs_to_cli_flags(engine_kwargs),
            ]

            label = build_worker_actor_name(model_name, i, 0, tp_size, role=role)
            logger.info(f"Disagg {role} worker {i}: {spec.per_node_gpus} GPU(s), nixl_port={nixl_port}")
            proc = spawn_actor(
                label,
                pg,
                0,
                num_gpus=spec.per_node_gpus,
                python_args=python_args,
                runtime_dir=self._runtime_dir,
                actor_name_prefix=self._actor_name_prefix,
                subprocess_env={**base_env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port), "PYTHONHASHSEED": "0"},
                runtime_env=self._dynamo_runtime_env(model_config),
            )
            self._worker_actors.append(proc)
            entries.append(
                {
                    "model": model_name,
                    "replica": i,
                    "mode": "disagg",
                    "role": role,
                    "gpus_per_node": spec.per_node_gpus,
                }
            )
            worker_index += 1

        return entries, worker_index

    @staticmethod
    def _plan_disagg_shape(
        tp_size: int,
        *,
        role: str,
        worker_index: int,
        model_name: str,
        topology: list[dict[str, Any]] | None = None,
    ) -> ReplicaBundleSpec:
        """Disagg workers don't support multi-node TP -- fail loudly if the
        planner returns a multi-bundle spec.
        """
        spec = plan_replica_bundle_shape(tp_size, _topology=topology)
        if spec.is_multi_node:
            msg = (
                f"Disaggregated serving does not support multi-node TP. "
                f"Model '{model_name}' {role} worker {worker_index} requires TP={tp_size} "
                f"which cannot fit on a single node. Reduce tensor_parallel_size for this role."
            )
            raise ValueError(msg)
        return spec

    def _launch_worker(  # noqa: PLR0913
        self,
        *,
        replica_index: int,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        pg: PlacementGroup,
        bundle_index: int,
        num_gpus: int,
        namespace: str,
        request_plane: str,
        event_plane: str,
        spec: ReplicaBundleSpec,
        master_addr: str | None,
    ) -> ManagedSubprocess:
        model_name = model_config.model_name or model_config.model_identifier
        component = _model_name_to_component(model_name)

        # Always pass explicit --kv-events-config. Without this, Dynamo's
        # args.py auto-creates KVEventsConfig binding tcp://*:20080 when
        # prefix_caching is enabled (default in vLLM >=0.16), causing all
        # workers on the same node to fight over the same port.
        kv_events_enabled = self._aggregated_model_uses_exact_kv_events(model_config)
        kv_events_config = self._build_worker_kv_events_config(
            model_config,
            pg=pg,
            bundle_index=bundle_index,
            port_seed=20080 + replica_index,
            enabled=kv_events_enabled,
        )

        python_args = [
            "-m",
            "dynamo.vllm",
            "--model",
            model_config.model_identifier,
            "--served-model-name",
            model_name,
            "--endpoint",
            _dynamo_endpoint(namespace, component),
            "--discovery-backend",
            "etcd",
            "--request-plane",
            request_plane,
            "--event-plane",
            event_plane,
            "--kv-events-config",
            kv_events_config,
            *_engine_kwargs_to_cli_flags(model_config.engine_kwargs),
        ]

        if spec.is_multi_node:
            python_args.extend(
                [
                    "--nnodes",
                    str(spec.nnodes),
                    "--node-rank",
                    "0",
                    "--master-addr",
                    master_addr,
                ]
            )

        tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
        label = build_worker_actor_name(model_name, replica_index, 0, tp_size)
        return spawn_actor(
            label,
            pg,
            bundle_index,
            num_gpus=num_gpus,
            python_args=python_args,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=base_env,
            runtime_env=self._dynamo_runtime_env(model_config),
        )

    def _launch_headless_worker(  # noqa: PLR0913
        self,
        *,
        replica_index: int,
        model_config: InferenceModelConfig,
        base_env: dict[str, str],
        pg: PlacementGroup,
        bundle_index: int,
        num_gpus: int,
        spec: ReplicaBundleSpec,
        master_addr: str,
    ) -> ManagedSubprocess:
        # Headless workers don't run the scheduler so they won't bind a KV
        # events port. Pass explicit disable defensively so that future
        # Dynamo/vLLM versions can't auto-configure one.
        kv_events_config = self._build_worker_kv_events_config(
            model_config,
            pg=pg,
            bundle_index=bundle_index,
            port_seed=20080 + replica_index + bundle_index,
            enabled=False,
        )

        python_args = [
            "-m",
            "dynamo.vllm",
            "--model",
            model_config.model_identifier,
            "--headless",
            "--kv-events-config",
            kv_events_config,
            "--nnodes",
            str(spec.nnodes),
            "--node-rank",
            str(bundle_index),
            "--master-addr",
            master_addr,
            *_engine_kwargs_to_cli_flags(model_config.engine_kwargs),
        ]

        tp_size = model_config.engine_kwargs.get("tensor_parallel_size", 1)
        model_name = model_config.model_name or model_config.model_identifier
        label = build_worker_actor_name(model_name, replica_index, bundle_index, tp_size)
        return spawn_actor(
            label,
            pg,
            bundle_index,
            num_gpus=num_gpus,
            python_args=python_args,
            runtime_dir=self._runtime_dir,
            actor_name_prefix=self._actor_name_prefix,
            subprocess_env=base_env,
            runtime_env=self._dynamo_runtime_env(model_config),
        )

    def _launch_frontend(  # noqa: PLR0913, C901
        self,
        port: int,
        base_env: dict[str, str],
        *,
        namespace: str,
        request_plane: str,
        event_plane: str,
        frontend_router_cfg: dict[str, Any],
        runtime_env: dict[str, Any] | None = None,
    ) -> ManagedSubprocess:
        frontend_env = dict(base_env)
        router_mode = frontend_router_cfg.get("router_mode")
        if router_mode:
            frontend_env["PYTHONHASHSEED"] = "0"

        python_args = [
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
            python_args.extend(["--router-mode", router_mode])

            if not frontend_router_cfg.get("router_kv_events", True):
                python_args.append("--no-router-kv-events")

            if frontend_router_cfg.get("router_kv_overlap_score_weight") is not None:
                python_args.extend(
                    [
                        "--router-kv-overlap-score-weight",
                        str(frontend_router_cfg["router_kv_overlap_score_weight"]),
                    ]
                )
            if frontend_router_cfg.get("router_temperature") is not None:
                python_args.extend(["--router-temperature", str(frontend_router_cfg["router_temperature"])])
            if frontend_router_cfg.get("router_queue_threshold") is not None:
                python_args.extend(["--router-queue-threshold", str(frontend_router_cfg["router_queue_threshold"])])

            # Approximate-mode knobs only when KV events are disabled.
            if not frontend_router_cfg.get("router_kv_events", True):
                if frontend_router_cfg.get("router_ttl_secs") is not None:
                    python_args.extend(["--router-ttl-secs", str(frontend_router_cfg["router_ttl_secs"])])
                if frontend_router_cfg.get("router_max_tree_size") is not None:
                    python_args.extend(["--router-max-tree-size", str(frontend_router_cfg["router_max_tree_size"])])
                if frontend_router_cfg.get("router_prune_target_ratio") is not None:
                    python_args.extend(
                        [
                            "--router-prune-target-ratio",
                            str(frontend_router_cfg["router_prune_target_ratio"]),
                        ]
                    )

            if frontend_router_cfg.get("router_reset_states"):
                python_args.append("--router-reset-states")

        logger.info(f"Starting Dynamo frontend on port {port}")
        return spawn_actor(
            "Dynamo_Frontend",
            self._infra_pg,
            _INFRA_FRONTEND_BUNDLE,
            num_gpus=0,
            python_args=python_args,
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

        try:
            with ray.init(namespace=NEMO_CURATOR_DYNAMO_NAMESPACE, ignore_reinit_error=True):
                self._teardown_actors_and_pgs(ray)
                # Safety net: reap anything still tagged with our prefix (e.g.
                # from an atexit-bypassed Jupyter kernel restart).
                remove_named_pgs_with_prefix(self._pg_name_prefix)
        except Exception:  # noqa: BLE001
            logger.debug("Could not connect to Ray during Dynamo shutdown (cluster may be gone)")

        self._cleanup_runtime_dir()
        self._server._host = "localhost"
        logger.info("Dynamo backend stopped")

    def _teardown_actors_and_pgs(self, ray_mod: Any) -> None:  # noqa: ANN401
        """Stop all actors in parallel, then release their placement groups.

        Subprocess teardown (``actor.stop.remote()``) runs concurrently so
        shutdown is bounded by a single SIGTERM→SIGKILL window instead of
        summing across N actors. ``remove_placement_group`` then releases
        the reservations.
        """
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

        graceful_stop_actors(ray_mod, actors)

        for pg in self._replica_pgs:
            with contextlib.suppress(Exception):
                ray_mod.util.remove_placement_group(pg)
        self._replica_pgs.clear()

        if self._infra_pg is not None:
            with contextlib.suppress(Exception):
                ray_mod.util.remove_placement_group(self._infra_pg)
            self._infra_pg = None

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
