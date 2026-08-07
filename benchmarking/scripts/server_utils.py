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

# ruff: noqa: ANN401

"""Reusable benchmark helpers for local InferenceServer lifecycle and requests."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

ServerBackend = Literal["dynamo", "ray_serve"]


@dataclass
class StartedInferenceServer:
    server: Any
    startup_s: float


def parse_json_arg(value: str | None, *, arg_name: str) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        msg = f"{arg_name} must be valid JSON: {e}"
        raise ValueError(msg) from e
    if not isinstance(parsed, dict):
        msg = f"{arg_name} must decode to a JSON object"
        raise TypeError(msg)
    return parsed


def static_autoscaling_config(num_replicas: int) -> dict[str, int]:
    num_replicas = max(1, int(num_replicas))
    return {"min_replicas": num_replicas, "max_replicas": num_replicas}


def static_num_replicas(autoscaling_config: dict[str, Any] | None) -> int:
    if not autoscaling_config:
        return 1
    min_replicas = int(autoscaling_config.get("min_replicas", 1))
    max_replicas = int(autoscaling_config.get("max_replicas", min_replicas))
    if min_replicas != max_replicas:
        msg = (
            "This benchmark uses fixed replica counts and does not support autoscaling; "
            f"min_replicas ({min_replicas}) must equal max_replicas ({max_replicas})."
        )
        raise ValueError(msg)
    return max(1, min_replicas)


def _engine_gpu_count(engine_kwargs: dict[str, Any] | None) -> int:
    engine_kwargs = engine_kwargs or {}
    tensor_parallel_size = int(engine_kwargs.get("tensor_parallel_size", 1))
    pipeline_parallel_size = int(engine_kwargs.get("pipeline_parallel_size", 1))
    return tensor_parallel_size * pipeline_parallel_size


def _dynamo_disagg_role_config(
    dynamo_disagg_config: dict[str, Any], role: Literal["prefill", "decode"]
) -> dict[str, Any]:
    role_config = dynamo_disagg_config.get(role)
    if not isinstance(role_config, dict):
        msg = f"--dynamo-disagg-config requires a JSON object for {role!r}"
        raise TypeError(msg)
    return role_config


def _dynamo_disagg_role_num_replicas(dynamo_disagg_config: dict[str, Any], role: Literal["prefill", "decode"]) -> int:
    role_config = _dynamo_disagg_role_config(dynamo_disagg_config, role)
    num_replicas = int(role_config.get("num_replicas", 1))
    if num_replicas < 1:
        msg = f"--dynamo-disagg-config {role}.num_replicas must be >= 1, got {num_replicas}"
        raise ValueError(msg)
    return num_replicas


def _dynamo_disagg_role_engine_kwargs(
    dynamo_disagg_config: dict[str, Any], role: Literal["prefill", "decode"]
) -> dict[str, Any]:
    role_config = _dynamo_disagg_role_config(dynamo_disagg_config, role)
    role_engine_kwargs = role_config.get("engine_kwargs") or {}
    if not isinstance(role_engine_kwargs, dict):
        msg = f"--dynamo-disagg-config {role}.engine_kwargs must be a JSON object"
        raise TypeError(msg)
    return role_engine_kwargs


def _dynamo_disagg_gpu_count(engine_kwargs: dict[str, Any] | None, dynamo_disagg_config: dict[str, Any]) -> int:
    total = 0
    for role in ("prefill", "decode"):
        role_engine_kwargs = {
            **(engine_kwargs or {}),
            **_dynamo_disagg_role_engine_kwargs(dynamo_disagg_config, role),
        }
        total += _dynamo_disagg_role_num_replicas(dynamo_disagg_config, role) * _engine_gpu_count(role_engine_kwargs)
    return total


def _dynamo_encoder_disagg_encode_config(dynamo_encoder_disagg_config: dict[str, Any]) -> dict[str, Any]:
    role_config = dynamo_encoder_disagg_config.get("encode") or {}
    if not isinstance(role_config, dict):
        msg = "--dynamo-encoder-disagg-config encode must be a JSON object"
        raise TypeError(msg)
    return role_config


def _dynamo_encoder_disagg_encode_num_replicas(dynamo_encoder_disagg_config: dict[str, Any]) -> int:
    role_config = _dynamo_encoder_disagg_encode_config(dynamo_encoder_disagg_config)
    num_replicas = int(role_config.get("num_replicas", 1))
    if num_replicas < 1:
        msg = f"--dynamo-encoder-disagg-config encode.num_replicas must be >= 1, got {num_replicas}"
        raise ValueError(msg)
    return num_replicas


def _dynamo_encoder_disagg_encode_engine_kwargs(dynamo_encoder_disagg_config: dict[str, Any]) -> dict[str, Any]:
    role_config = _dynamo_encoder_disagg_encode_config(dynamo_encoder_disagg_config)
    role_engine_kwargs = role_config.get("engine_kwargs") or {}
    if not isinstance(role_engine_kwargs, dict):
        msg = "--dynamo-encoder-disagg-config encode.engine_kwargs must be a JSON object"
        raise TypeError(msg)
    return role_engine_kwargs


def _dynamo_encoder_disagg_gpu_count(
    engine_kwargs: dict[str, Any] | None,
    autoscaling_config: dict[str, Any] | None,
    dynamo_encoder_disagg_config: dict[str, Any],
) -> int:
    backend_total = static_num_replicas(autoscaling_config) * _engine_gpu_count(engine_kwargs)
    encode_engine_kwargs = {
        **(engine_kwargs or {}),
        **_dynamo_encoder_disagg_encode_engine_kwargs(dynamo_encoder_disagg_config),
    }
    return backend_total + (
        _dynamo_encoder_disagg_encode_num_replicas(dynamo_encoder_disagg_config)
        * _engine_gpu_count(encode_engine_kwargs)
    )


def server_gpu_count(
    engine_kwargs: dict[str, Any] | None,
    autoscaling_config: dict[str, Any] | None,
    dynamo_disagg_config: dict[str, Any] | None = None,
    dynamo_encoder_disagg_config: dict[str, Any] | None = None,
) -> int:
    if dynamo_disagg_config and dynamo_encoder_disagg_config:
        msg = "--dynamo-disagg-config and --dynamo-encoder-disagg-config are mutually exclusive"
        raise ValueError(msg)
    if dynamo_disagg_config:
        return _dynamo_disagg_gpu_count(engine_kwargs, dynamo_disagg_config)
    if dynamo_encoder_disagg_config:
        return _dynamo_encoder_disagg_gpu_count(engine_kwargs, autoscaling_config, dynamo_encoder_disagg_config)
    return static_num_replicas(autoscaling_config) * _engine_gpu_count(engine_kwargs)


def default_nemotron_parse_engine_kwargs(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    engine_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "limit_mm_per_prompt": {"image": 1},
        # Nemotron-Parse is encoder-decoder multimodal; vLLM 0.19 can leave
        # prefix caching enabled and crash in CrossAttentionManager.
        "enable_prefix_caching": False,
    }
    engine_kwargs.update(overrides or {})
    return engine_kwargs


def default_dynamo_frontend_kwargs(
    *,
    use_vllm_chat_processor: bool,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frontend_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if use_vllm_chat_processor:
        frontend_kwargs["dyn_chat_processor"] = "vllm"
        frontend_kwargs["chat_template_content_format"] = "string"
    frontend_kwargs.update(overrides or {})
    return frontend_kwargs


def build_multimodal_chat_messages(task_prompt: str, image_ref: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": task_prompt},
                {"type": "image_url", "image_url": {"url": image_ref}},
            ],
        }
    ]


def build_inference_server(  # noqa: PLR0913
    *,
    backend: ServerBackend,
    model_id: str,
    model_path: str | None,
    engine_kwargs: dict[str, Any] | None,
    autoscaling_config: dict[str, Any] | None,
    frontend_kwargs: dict[str, Any] | None,
    request_timeout_s: int,
    dynamo_disagg_config: dict[str, Any] | None = None,
    dynamo_encoder_disagg_config: dict[str, Any] | None = None,
    dynamo_model_runtime_env: dict[str, Any] | None = None,
    health_check_timeout_s: int = 900,
) -> Any:
    """Build, but do not start, an InferenceServer for a benchmark entry."""
    from nemo_curator.core.serve import (
        DynamoRoleConfig,
        DynamoRouterConfig,
        DynamoServerConfig,
        DynamoVLLMModelConfig,
        InferenceServer,
        RayServeModelConfig,
        RayServeServerConfig,
    )

    if backend == "dynamo":
        if dynamo_disagg_config and dynamo_encoder_disagg_config:
            msg = "--dynamo-disagg-config and --dynamo-encoder-disagg-config are mutually exclusive"
            raise ValueError(msg)
        model_kwargs: dict[str, Any] = {
            "model_identifier": model_path or model_id,
            "model_name": model_id if model_path else None,
            "engine_kwargs": engine_kwargs or {},
            "dynamo_kwargs": {"enable_multimodal": True},
        }
        if dynamo_model_runtime_env:
            model_kwargs["runtime_env"] = dynamo_model_runtime_env
        if dynamo_disagg_config:
            model_kwargs.update(
                {
                    "num_replicas": 1,
                    "mode": "disagg",
                    "prefill": DynamoRoleConfig(
                        num_replicas=_dynamo_disagg_role_num_replicas(dynamo_disagg_config, "prefill"),
                        engine_kwargs=_dynamo_disagg_role_engine_kwargs(dynamo_disagg_config, "prefill"),
                    ),
                    "decode": DynamoRoleConfig(
                        num_replicas=_dynamo_disagg_role_num_replicas(dynamo_disagg_config, "decode"),
                        engine_kwargs=_dynamo_disagg_role_engine_kwargs(dynamo_disagg_config, "decode"),
                    ),
                }
            )
        elif dynamo_encoder_disagg_config:
            model_kwargs.update(
                {
                    "num_replicas": static_num_replicas(autoscaling_config),
                    "mode": "encoder_disagg",
                    "encode": DynamoRoleConfig(
                        num_replicas=_dynamo_encoder_disagg_encode_num_replicas(dynamo_encoder_disagg_config),
                        engine_kwargs=_dynamo_encoder_disagg_encode_engine_kwargs(dynamo_encoder_disagg_config),
                    ),
                }
            )
        else:
            model_kwargs["num_replicas"] = static_num_replicas(autoscaling_config)

        model_config = DynamoVLLMModelConfig(
            **model_kwargs,
        )
        return InferenceServer(
            models=[model_config],
            backend=DynamoServerConfig(
                request_plane="tcp",
                router=DynamoRouterConfig(router_kwargs=frontend_kwargs or {}),
                subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": str(int(request_timeout_s))},
            ),
            health_check_timeout_s=health_check_timeout_s,
        )

    if backend == "ray_serve":
        model_config = RayServeModelConfig(
            model_identifier=model_path or model_id,
            model_name=model_id if model_path else None,
            deployment_config={"autoscaling_config": autoscaling_config or static_autoscaling_config(1)},
            engine_kwargs=engine_kwargs or {},
        )
        return InferenceServer(
            models=[model_config],
            backend=RayServeServerConfig(),
            health_check_timeout_s=health_check_timeout_s,
        )

    msg = f"Unsupported server backend: {backend}"
    raise ValueError(msg)


def start_inference_server(**kwargs: Any) -> StartedInferenceServer:
    server = build_inference_server(**kwargs)
    start_time = time.perf_counter()
    server.start()
    return StartedInferenceServer(server=server, startup_s=time.perf_counter() - start_time)
