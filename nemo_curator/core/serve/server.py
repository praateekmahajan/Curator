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

import atexit
import http
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ray.serve.llm import LLMConfig

    from nemo_curator.core.serve.internal.base import InferenceBackend

from loguru import logger

from nemo_curator.core.serve.internal.constants import DEFAULT_SERVE_HEALTH_TIMEOUT_S, DEFAULT_SERVE_PORT

# Track which application names are currently managed by an InferenceServer in
# this process.  ``is_inference_server_active()`` checks this dict so that other
# parts of the codebase (e.g. Pipeline.run()) can detect potential GPU
# resource contention.
_active_servers: dict[str, str] = {}  # name -> backend type


def is_inference_server_active() -> bool:
    """Check whether any InferenceServer is currently running in this process."""
    return bool(_active_servers)


def is_ray_serve_active() -> bool:
    """Check whether a Ray Serve InferenceServer is currently running."""
    return any(v == "ray_serve" for v in _active_servers.values())


def get_active_backend() -> str | None:
    """Return the backend type of the active InferenceServer, or None."""
    if _active_servers:
        return next(iter(_active_servers.values()))
    return None


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


@dataclass
class InferenceModelConfig:
    """Configuration for a single model to be served.

    Args:
        model_identifier: HuggingFace model ID or local path.
        model_name: API-facing model name clients use in requests. Defaults to model_identifier.
        deployment_config: Ray Serve deployment configuration (autoscaling, replicas, etc.).
            Only used when ``backend="ray_serve"``.
        engine_kwargs: vLLM engine keyword arguments (tensor_parallel_size, etc.).
            Common to both Ray Serve and Dynamo backends.
        runtime_env: Ray runtime environment configuration.
            Only used when ``backend="ray_serve"``.
        dynamo_config: Dynamo-specific configuration.
            Only used when ``backend="dynamo"``. Keys: ``namespace`` (default "curator"),
            ``component`` (default "backend"), ``endpoint`` (default "generate").
    """

    model_identifier: str
    model_name: str | None = None
    deployment_config: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    runtime_env: dict[str, Any] = field(default_factory=dict)
    dynamo_config: dict[str, Any] = field(default_factory=dict)

    def to_llm_config(self, quiet_runtime_env: dict[str, Any] | None = None) -> LLMConfig:
        """Convert to a Ray Serve LLMConfig.

        Args:
            quiet_runtime_env: Optional runtime environment with quiet/logging
                overrides.  Merged on top of ``self.runtime_env`` so that
                quiet env vars take precedence while preserving user-provided
                keys (e.g. ``pip``, ``working_dir``).
        """
        from ray.serve.llm import LLMConfig

        merged_env = self._merge_runtime_envs(self.runtime_env, quiet_runtime_env)

        return LLMConfig(
            model_loading_config={
                "model_id": self.model_name or self.model_identifier,
                "model_source": self.model_identifier,
            },
            deployment_config=self.deployment_config,
            engine_kwargs=self.engine_kwargs,
            runtime_env=merged_env or None,
        )

    @staticmethod
    def _merge_runtime_envs(
        base: dict[str, Any],
        override: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge two runtime_env dicts, with special handling for ``env_vars``."""
        if not base and not override:
            return {}
        if not override:
            return {**base}
        if not base:
            return {**override}

        merged = {**base, **override}
        base_env_vars = base.get("env_vars", {})
        override_env_vars = override.get("env_vars", {})
        if base_env_vars or override_env_vars:
            merged["env_vars"] = {**base_env_vars, **override_env_vars}
        return merged


# ---------------------------------------------------------------------------
# InferenceServer
# ---------------------------------------------------------------------------


@dataclass
class InferenceServer:
    """Serve one or more models via an OpenAI-compatible endpoint.

    Supports multiple backends:
    - ``"ray_serve"`` (default): Ray Serve + vLLM. Unified GPU scheduling with
      pipeline stages. Requires ``ray[serve,llm]``.
    - ``"dynamo"`` : NVIDIA Dynamo. Subprocess-based workers with KV-cache-aware
      routing. Runs outside Ray — no unified GPU scheduling with pipeline stages.
      Requires ``ai-dynamo``.

    Args:
        models: List of InferenceModelConfig instances to deploy.
        name: Application name (default ``"default"``).
        port: HTTP port for the OpenAI-compatible endpoint.
        health_check_timeout_s: Seconds to wait for models to become healthy.
        verbose: If True, keep logging at default levels.
        backend: Backend to use — ``"ray_serve"`` or ``"dynamo"``.
        etcd_endpoint: Pre-existing etcd endpoint for Dynamo (skips managed infra).
        nats_url: Pre-existing NATS URL for Dynamo (skips managed infra).

    Example::

        from nemo_curator.core.serve import InferenceModelConfig, InferenceServer

        config = InferenceModelConfig(
            model_identifier="google/gemma-3-27b-it",
            engine_kwargs={"tensor_parallel_size": 4},
            deployment_config={
                "autoscaling_config": {
                    "min_replicas": 1,
                    "max_replicas": 1,
                },
            },
        )

        with InferenceServer(models=[config]) as server:
            print(server.endpoint)  # http://localhost:8000/v1
    """

    models: list[InferenceModelConfig]
    name: str = "default"
    port: int = DEFAULT_SERVE_PORT
    health_check_timeout_s: int = DEFAULT_SERVE_HEALTH_TIMEOUT_S
    verbose: bool = False
    backend: str = "ray_serve"

    # Dynamo infra options (ignored when backend="ray_serve")
    etcd_endpoint: str | None = None
    nats_url: str | None = None

    _started: bool = field(init=False, default=False, repr=False)
    _backend_impl: InferenceBackend | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.verbose:
            logging.getLogger("ray.serve").setLevel(logging.WARNING)

    def start(self) -> None:
        """Deploy all models and wait for them to become healthy.

        Raises:
            RuntimeError: If another InferenceServer is already active.
            ValueError: If an unknown backend is specified.
        """
        if _active_servers:
            running = ", ".join(sorted(_active_servers))
            msg = (
                f"Cannot start InferenceServer '{self.name}': another InferenceServer is "
                f"already active (running: {running}). Stop the existing server first."
            )
            raise RuntimeError(msg)

        atexit.register(self.stop)

        self._backend_impl = self._create_backend()
        self._backend_impl.start()

        _active_servers[self.name] = self.backend
        self._started = True
        logger.info(f"Inference server ({self.backend}) is ready at {self.endpoint}")

    def _create_backend(self) -> InferenceBackend:
        if self.backend == "ray_serve":
            from nemo_curator.core.serve.internal.ray_serve import RayServeBackend

            return RayServeBackend(self)
        if self.backend == "dynamo":
            from nemo_curator.core.serve.internal.dynamo import DynamoBackend

            return DynamoBackend(self)
        msg = f"Unknown backend: {self.backend!r}. Choose 'ray_serve' or 'dynamo'."
        raise ValueError(msg)

    def stop(self) -> None:
        """Shut down the inference server and release resources."""
        if not self._started:
            return

        if self._backend_impl is not None:
            self._backend_impl.stop()
            self._backend_impl = None

        _active_servers.pop(self.name, None)
        self._started = False
        atexit.unregister(self.stop)

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible base URL for the served models."""
        return f"http://localhost:{self.port}/v1"

    def _wait_for_healthy(self) -> None:
        """Poll the /v1/models endpoint until all models are ready."""
        models_url = f"{self.endpoint}/models"
        deadline = time.monotonic() + self.health_check_timeout_s
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
                if resp.status == http.HTTPStatus.OK:
                    logger.info(f"Model server ready after {attempt} health check(s)")
                    return
            except Exception:  # noqa: BLE001
                if self.verbose:
                    logger.debug(f"Health check attempt {attempt} failed, retrying...")
            time.sleep(1)
        msg = f"Model server did not become ready within {self.health_check_timeout_s}s"
        raise TimeoutError(msg)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
