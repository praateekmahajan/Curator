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

"""Seeded DataDesigner pipeline for SLURM array runs.

The script is intentionally narrow: it covers seed-data generation from JSONL
or Parquet files. Pure non-seed generation that needs record-range partitioning
is left out of this tutorial prototype.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from slurm_array_support import (
    SlurmArrayContext,
    SlurmArrayFilePartitioningStage,
    SlurmArrayJsonlWriter,
    SlurmArrayParquetWriter,
    SlurmArrayPartitionPlan,
    SlurmArrayPipeline,
    data_output_path,
    make_slurm_array_jsonl_reader,
    make_slurm_array_parquet_reader,
    success_marker_path,
    write_success_marker,
)

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.utils import get_available_cpu_gpu_resources
from nemo_curator.core.client import RayClient, SlurmRayClient
from nemo_curator.core.serve import (
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
    RayServeModelConfig,
    RayServeServerConfig,
)
from nemo_curator.stages.synthetic.nemo_data_designer.data_designer import DataDesignerStage
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS

if TYPE_CHECKING:
    from nemo_curator.stages.base import CompositeStage


@dataclass(frozen=True)
class ModelServeSpec:
    model_identifier: str
    model_name: str
    tensor_parallel_size: int
    num_replicas: int | str
    engine_kwargs: dict[str, Any]
    data_designer_aliases: tuple[str, ...] = ()


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def _default_ray_tmpdir() -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    return f"/tmp/ray_curator_{job_id}_{array_task_id}"  # noqa: S108


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Curator DataDesigner pipeline as one SLURM array shard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-path", default=os.environ.get("INPUT_PATH"))
    parser.add_argument("--output-path", default=os.environ.get("OUTPUT_PATH"))
    parser.add_argument(
        "--data-designer-config-file",
        default=os.environ.get("DD_CONFIG") or os.environ.get("DATA_DESIGNER_CONFIG_FILE"),
    )
    parser.add_argument(
        "--models-json",
        default=os.environ.get("MODELS_JSON"),
        help=(
            "JSON list of model specs. Each spec supports model/model_identifier, "
            "served_model_name/model_name, tensor_parallel_size/tp, num_replicas/replicas, "
            "max_model_len, engine_kwargs, and alias/model_alias/aliases for DataDesigner retargeting."
        ),
    )
    parser.add_argument("--models-json-file", default=os.environ.get("MODELS_JSON_FILE"))
    parser.add_argument("--input-format", choices=["jsonl", "parquet"], default=os.environ.get("INPUT_FORMAT", "jsonl"))
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "parquet"],
        default=os.environ.get("OUTPUT_FORMAT", "jsonl"),
    )
    parser.add_argument(
        "--output-layout",
        choices=["flat", "by_shard"],
        default=os.environ.get("OUTPUT_LAYOUT", "flat"),
        help="Use 'flat' for a shared data directory or 'by_shard' for data/shard_00000 directories.",
    )
    parser.add_argument("--fields", default=os.environ.get("FIELDS"), help="Comma-separated seed columns to read.")
    parser.add_argument("--files-per-partition", type=int, default=_env_int("FILES_PER_PARTITION"))
    parser.add_argument("--blocksize", default=_env_str("BLOCKSIZE"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME"))
    parser.add_argument("--provider-name", default=os.environ.get("PROVIDER_NAME", "local"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "unused"))  # pragma: allowlist secret
    parser.add_argument("--tensor-parallel-size", type=int, default=_env_int("TP", 1))
    parser.add_argument("--num-replicas", default=os.environ.get("REPLICAS", "auto"))
    parser.add_argument(
        "--server-backend",
        choices=["dynamo", "ray-serve"],
        default=os.environ.get("SERVER_BACKEND", "dynamo"),
    )
    parser.add_argument("--server-port", type=int, default=_env_int("SERVE_PORT", 8000))
    parser.add_argument("--health-check-timeout-s", type=int, default=_env_int("HEALTH_CHECK_TIMEOUT_S", 600))
    parser.add_argument("--max-model-len", type=int, default=_env_int("MAX_MODEL_LEN"))
    parser.add_argument(
        "--vllm-engine-kwargs-json",
        default=os.environ.get("VLLM_ENGINE_KWARGS_JSON"),
        help="JSON object merged into vLLM engine_kwargs before tensor_parallel_size.",
    )
    parser.add_argument("--ray-temp-dir", default=os.environ.get("RAY_TMPDIR", _default_ray_tmpdir()))
    parser.add_argument(
        "--ray-worker-connect-timeout-s",
        type=int,
        default=_env_int("RAY_WORKER_CONNECT_TIMEOUT_S", 600),
    )
    parser.add_argument(
        "--ignore-head-node-for-replicas",
        action="store_true",
        default=_env_bool("IGNORE_HEAD_NODE_FOR_REPLICAS", False),
    )
    parser.add_argument(
        "--no-retarget-data-designer-config",
        action="store_true",
        help="Do not rewrite DataDesigner model configs to the local provider/model.",
    )
    parser.add_argument("--verbose", action="store_true", default=_env_bool("VERBOSE", False))
    args = parser.parse_args()

    missing = [
        name
        for name, value in {
            "--input-path or INPUT_PATH": args.input_path,
            "--output-path or OUTPUT_PATH": args.output_path,
            "--data-designer-config-file or DD_CONFIG": args.data_designer_config_file,
        }.items()
        if not value
    ]
    if not args.model and not args.models_json and not args.models_json_file:
        missing.append("--model/MODEL or --models-json/MODELS_JSON or --models-json-file/MODELS_JSON_FILE")
    if missing:
        parser.error("Missing required value(s): " + ", ".join(missing))
    if args.files_per_partition is not None and args.blocksize is not None:
        parser.error("--files-per-partition and --blocksize are mutually exclusive")
    return args


def _parse_fields(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [field.strip() for field in value.split(",") if field.strip()]


def _json_object(value: str | None, *, field_name: str) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        msg = f"{field_name} must decode to a JSON object"
        raise TypeError(msg)
    return loaded


def _base_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.vllm_engine_kwargs_json:
        kwargs.update(_json_object(args.vllm_engine_kwargs_json, field_name="--vllm-engine-kwargs-json"))
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    return kwargs


def _normalise_replicas(value: object) -> int | str:
    if value is None:
        return "auto"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    replicas = int(value)
    if replicas < 1:
        msg = f"num_replicas must be >= 1 or 'auto', got {value}"
        raise ValueError(msg)
    return replicas


def _aliases_from_model_spec(raw_spec: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("alias", "model_alias", "data_designer_alias"):
        value = raw_spec.get(key)
        if value:
            values.append(str(value))
    aliases = raw_spec.get("aliases")
    if aliases:
        if isinstance(aliases, str):
            values.append(aliases)
        elif isinstance(aliases, list):
            values.extend(str(alias) for alias in aliases)
        else:
            msg = "Model spec field 'aliases' must be a string or list of strings"
            raise TypeError(msg)
    return tuple(dict.fromkeys(values))


def _model_spec_from_raw(
    raw_spec: dict[str, Any],
    *,
    args: argparse.Namespace,
    base_engine_kwargs: dict[str, Any],
) -> ModelServeSpec:
    model_identifier = raw_spec.get("model_identifier") or raw_spec.get("model")
    if not model_identifier:
        msg = "Each model spec needs 'model' or 'model_identifier'"
        raise ValueError(msg)
    model_identifier = str(model_identifier)
    model_name = str(raw_spec.get("served_model_name") or raw_spec.get("model_name") or model_identifier)
    tensor_parallel_size = int(raw_spec.get("tensor_parallel_size", raw_spec.get("tp", args.tensor_parallel_size)))
    if tensor_parallel_size < 1:
        msg = f"tensor_parallel_size must be >= 1 for {model_identifier}"
        raise ValueError(msg)

    engine_kwargs = dict(base_engine_kwargs)
    engine_kwargs.update(_json_object(raw_spec.get("engine_kwargs_json"), field_name="engine_kwargs_json"))
    raw_engine_kwargs = raw_spec.get("engine_kwargs", {})
    if not isinstance(raw_engine_kwargs, dict):
        msg = f"engine_kwargs must be a JSON object for {model_identifier}"
        raise TypeError(msg)
    engine_kwargs.update(raw_engine_kwargs)
    if raw_spec.get("max_model_len") is not None:
        engine_kwargs["max_model_len"] = int(raw_spec["max_model_len"])
    engine_kwargs["tensor_parallel_size"] = tensor_parallel_size

    return ModelServeSpec(
        model_identifier=model_identifier,
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        num_replicas=_normalise_replicas(raw_spec.get("num_replicas", raw_spec.get("replicas", "auto"))),
        engine_kwargs=engine_kwargs,
        data_designer_aliases=_aliases_from_model_spec(raw_spec),
    )


def _model_specs_from_args(args: argparse.Namespace) -> list[ModelServeSpec]:
    base_engine_kwargs = _base_engine_kwargs(args)

    models_json = args.models_json
    if not models_json and args.models_json_file:
        models_json = Path(args.models_json_file).read_text(encoding="utf-8")

    if not models_json:
        model_name = args.served_model_name or args.model
        engine_kwargs = {**base_engine_kwargs, "tensor_parallel_size": args.tensor_parallel_size}
        return [
            ModelServeSpec(
                model_identifier=args.model,
                model_name=model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                num_replicas=_normalise_replicas(args.num_replicas),
                engine_kwargs=engine_kwargs,
            )
        ]

    raw_specs = json.loads(models_json)
    if not isinstance(raw_specs, list) or not raw_specs:
        msg = "--models-json must decode to a non-empty JSON list"
        raise TypeError(msg)

    specs: list[ModelServeSpec] = []
    for raw in raw_specs:
        if not isinstance(raw, dict):
            msg = "Each --models-json entry must be a JSON object"
            raise TypeError(msg)
        specs.append(_model_spec_from_raw(raw, args=args, base_engine_kwargs=base_engine_kwargs))

    served_names = [spec.model_name for spec in specs]
    if len(served_names) != len(set(served_names)):
        msg = f"Served model names must be unique, got {served_names}"
        raise ValueError(msg)
    return specs


def _slurm_gpu_count_for_replicas(*, ignore_head_node: bool) -> int | None:
    node_count = os.environ.get("SLURM_JOB_NUM_NODES") or os.environ.get("SLURM_NNODES")
    gpus_on_node = os.environ.get("SLURM_GPUS_ON_NODE")
    if not node_count or not gpus_on_node:
        return None
    try:
        nodes = int(node_count)
        gpus_per_node = int(gpus_on_node)
    except ValueError:
        return None
    usable_nodes = max(0, nodes - 1) if ignore_head_node else nodes
    return usable_nodes * gpus_per_node


def _available_gpu_count_for_replicas(*, ignore_head_node: bool) -> int:
    slurm_gpu_count = _slurm_gpu_count_for_replicas(ignore_head_node=ignore_head_node)
    if slurm_gpu_count is not None:
        logger.info(f"Using SLURM GPU count for replica sizing: {slurm_gpu_count}")
        return slurm_gpu_count
    _, available_gpus = get_available_cpu_gpu_resources(
        init_and_shutdown=True,
        ignore_head_node=ignore_head_node,
    )
    return int(available_gpus)


def _resolve_model_replicas(
    model_specs: list[ModelServeSpec],
    *,
    ignore_head_node: bool,
) -> list[int]:
    resolved: list[int | None] = []
    explicit_gpu_count = 0
    auto_tp_sum = 0

    for spec in model_specs:
        if spec.num_replicas == "auto":
            resolved.append(None)
            auto_tp_sum += spec.tensor_parallel_size
        else:
            replicas = int(spec.num_replicas)
            if replicas < 1:
                msg = f"num_replicas must be >= 1 for {spec.model_name}"
                raise ValueError(msg)
            resolved.append(replicas)
            explicit_gpu_count += replicas * spec.tensor_parallel_size

    if auto_tp_sum == 0:
        return [int(value) for value in resolved if value is not None]

    available_gpus = _available_gpu_count_for_replicas(ignore_head_node=ignore_head_node)
    remaining_gpus = available_gpus - explicit_gpu_count
    if remaining_gpus < auto_tp_sum:
        msg = (
            f"Not enough GPUs for one replica of each auto-sized model: "
            f"available={available_gpus}, explicit_gpu_count={explicit_gpu_count}, "
            f"auto_tp_sum={auto_tp_sum}"
        )
        raise RuntimeError(msg)

    auto_replicas = max(1, remaining_gpus // auto_tp_sum)
    logger.info(
        f"Auto-selected {auto_replicas} replica(s) for each auto-sized model from "
        f"{available_gpus} visible GPU(s), {explicit_gpu_count} explicit GPU(s), "
        f"and auto tensor-parallel sum {auto_tp_sum}"
    )
    return [auto_replicas if value is None else int(value) for value in resolved]


def _start_inference_server(args: argparse.Namespace, model_specs: list[ModelServeSpec]) -> InferenceServer:
    replicas = _resolve_model_replicas(
        model_specs,
        ignore_head_node=args.ignore_head_node_for_replicas,
    )

    if args.server_backend == "dynamo":
        model_configs = [
            DynamoVLLMModelConfig(
                model_identifier=spec.model_identifier,
                model_name=spec.model_name,
                num_replicas=replicas[idx],
                engine_kwargs=spec.engine_kwargs,
            )
            for idx, spec in enumerate(model_specs)
        ]
        server = InferenceServer(
            models=model_configs,
            backend=DynamoServerConfig(),
            port=args.server_port,
            health_check_timeout_s=args.health_check_timeout_s,
            verbose=args.verbose,
        )
    else:
        model_configs = [
            RayServeModelConfig(
                model_identifier=spec.model_identifier,
                model_name=spec.model_name,
                deployment_config={
                    "autoscaling_config": {
                        "min_replicas": replicas[idx],
                        "max_replicas": replicas[idx],
                    }
                },
                engine_kwargs=spec.engine_kwargs,
            )
            for idx, spec in enumerate(model_specs)
        ]
        server = InferenceServer(
            models=model_configs,
            backend=RayServeServerConfig(),
            port=args.server_port,
            health_check_timeout_s=args.health_check_timeout_s,
            verbose=args.verbose,
        )

    server.start()
    served_models = ", ".join(
        f"{spec.model_name}({replicas[idx]}x TP={spec.tensor_parallel_size})"
        for idx, spec in enumerate(model_specs)
    )
    logger.info(f"Local inference server ready at {server.endpoint}; models: {served_models}")
    return server


def _model_configs_from_builder(config_builder: object) -> list[object]:
    for attr in ("model_configs", "_model_configs"):
        value = getattr(config_builder, attr, None)
        if value is not None:
            return list(value)
    config = getattr(config_builder, "config", None)
    if config is not None:
        for attr in ("model_configs", "_model_configs"):
            value = getattr(config, attr, None)
            if value is not None:
                return list(value)
    return []


def _retarget_data_designer_config(
    config_builder: object,
    *,
    provider_name: str,
    model_specs: list[ModelServeSpec],
) -> None:
    """Point existing DataDesigner model configs at the local inference server."""

    model_configs = _model_configs_from_builder(config_builder)
    if not model_configs:
        logger.warning(
            "Could not find model_configs on the DataDesigner config builder. "
            "The config must already point at the desired local provider/model."
        )
        return

    model_by_alias = {
        alias: spec.model_name
        for spec in model_specs
        for alias in spec.data_designer_aliases
    }
    served_model_names = {spec.model_name for spec in model_specs}

    for model_config in model_configs:
        updates: dict[str, object] = {
            "provider": provider_name,
            "skip_health_check": True,
        }
        data_designer_alias = getattr(model_config, "alias", None)
        current_model = getattr(model_config, "model", None)
        if len(model_specs) == 1:
            updates["model"] = model_specs[0].model_name
        elif data_designer_alias in model_by_alias:
            updates["model"] = model_by_alias[data_designer_alias]
        elif current_model not in served_model_names:
            logger.warning(
                f"DataDesigner model config alias={data_designer_alias!r} model={current_model!r} "
                "was not retargeted to a served model. Add alias/model_alias/aliases to MODELS_JSON "
                "or set the DataDesigner model field to one of the served_model_name values."
            )

        for attr, value in updates.items():
            try:
                setattr(model_config, attr, value)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not set DataDesigner model config {attr}={value!r}: {exc}")


def _reader_for_args(args: argparse.Namespace, fields: list[str] | None) -> CompositeStage:
    reader_kwargs = {
        "file_paths": args.input_path,
        "output_path": args.output_path,
        "output_file_extension": args.output_format,
        "output_layout": args.output_layout,
        "files_per_partition": args.files_per_partition,
        "blocksize": args.blocksize,
        "fields": fields,
    }
    if args.input_format == "jsonl":
        return make_slurm_array_jsonl_reader(**reader_kwargs)
    return make_slurm_array_parquet_reader(**reader_kwargs)


def _partition_stage_for_args(args: argparse.Namespace) -> SlurmArrayFilePartitioningStage:
    return SlurmArrayFilePartitioningStage(
        file_paths=args.input_path,
        files_per_partition=args.files_per_partition,
        blocksize=args.blocksize,
        file_extensions=FILETYPE_TO_DEFAULT_EXTENSIONS[args.input_format],
        output_path=args.output_path,
        output_file_extension=args.output_format,
        output_layout=args.output_layout,
    )


def _write_empty_or_complete_success_marker(
    args: argparse.Namespace,
    context: SlurmArrayContext,
    plan: SlurmArrayPartitionPlan,
    *,
    preflight_s: float,
    started_at: float,
) -> None:
    if plan.already_successful:
        logger.info(
            f"Shard {context.shard_index}/{context.num_shards} already has "
            f"{success_marker_path(args.output_path, context.shard_index)}; exiting before Ray/model startup"
        )
        return

    marker_path = write_success_marker(
        args.output_path,
        context,
        payload={
            "pipeline_name": "slurm_array_ndd_seeded_sdg",
            "num_output_tasks": 0,
            "output_files": [],
            "partition_plan": plan.marker_payload(),
            "empty_shard": plan.total_file_groups == plan.other_shard_file_groups,
            "all_outputs_already_complete": plan.completed_outputs > 0,
            "metrics": {
                "preflight_s": preflight_s,
                "total_elapsed_s": time.perf_counter() - started_at,
                "num_output_tasks": 0,
                "num_output_files": 0,
                "output_bytes": 0,
            },
        },
    )
    logger.info(f"Wrote success marker {marker_path} without starting Ray/model server")


def _writer_for_args(args: argparse.Namespace, writer_path: str) -> SlurmArrayJsonlWriter | SlurmArrayParquetWriter:
    if args.output_format == "jsonl":
        return SlurmArrayJsonlWriter(path=writer_path)
    return SlurmArrayParquetWriter(path=writer_path)


def _ray_client_for_env(args: argparse.Namespace) -> RayClient | SlurmRayClient:
    if os.environ.get("SLURM_JOB_ID"):
        return SlurmRayClient(
            include_dashboard=False,
            ray_temp_dir=args.ray_temp_dir,
            worker_connect_timeout_s=args.ray_worker_connect_timeout_s,
        )
    return RayClient(include_dashboard=False, ray_temp_dir=args.ray_temp_dir)


def main() -> None:
    args = parse_args()
    try:
        import data_designer.config as dd
    except ModuleNotFoundError as exc:
        msg = (
            "data_designer is required to run this tutorial pipeline. "
            "Use a Curator container or environment that includes NeMo Data Designer."
        )
        raise ModuleNotFoundError(msg) from exc

    context = SlurmArrayContext.from_env()
    writer_path = data_output_path(args.output_path, args.output_layout, context.shard_index)
    fields = _parse_fields(args.fields)
    started_at = time.perf_counter()

    logger.info(
        f"Starting shard {context.shard_index}/{context.num_shards}: "
        f"input={args.input_path}, output_root={args.output_path}, writer_path={writer_path}"
    )

    preflight_started_at = time.perf_counter()
    partition_plan = _partition_stage_for_args(args).plan()
    preflight_s = time.perf_counter() - preflight_started_at
    logger.info(f"Preflight partition plan: {partition_plan.marker_payload()}")
    if partition_plan.already_successful or partition_plan.pending_file_groups == 0:
        _write_empty_or_complete_success_marker(
            args,
            context,
            partition_plan,
            preflight_s=preflight_s,
            started_at=started_at,
        )
        return

    ray_client = _ray_client_for_env(args)
    inference_server: InferenceServer | None = None
    ray_start = time.perf_counter()
    ray_client.start()
    ray_start_s = time.perf_counter() - ray_start

    try:
        model_specs = _model_specs_from_args(args)
        config_start = time.perf_counter()
        config_builder = dd.DataDesignerConfigBuilder.from_config(args.data_designer_config_file)
        if not args.no_retarget_data_designer_config:
            _retarget_data_designer_config(
                config_builder,
                provider_name=args.provider_name,
                model_specs=model_specs,
            )
        config_load_s = time.perf_counter() - config_start
        serve_start = time.perf_counter()
        inference_server = _start_inference_server(args, model_specs)
        serve_startup_s = time.perf_counter() - serve_start

        model_providers = [
            dd.ModelProvider(
                name=args.provider_name,
                endpoint=inference_server.endpoint,
                provider_type="openai",
                api_key=args.api_key,
            )
        ]

        pipeline = SlurmArrayPipeline(
            name="slurm_array_ndd_seeded_sdg",
            description="Seeded DataDesigner generation over one SLURM array shard",
            output_path=args.output_path,
            success_payload={
                "partition_plan": partition_plan.marker_payload(),
                "served_models": [
                    {
                        "model_name": spec.model_name,
                        "model_identifier": spec.model_identifier,
                        "tensor_parallel_size": spec.tensor_parallel_size,
                    }
                    for spec in model_specs
                ],
                "metrics": {
                    "preflight_s": preflight_s,
                    "ray_start_s": ray_start_s,
                    "config_load_s": config_load_s,
                    "serve_startup_s": serve_startup_s,
                },
            },
        )
        pipeline.add_stage(_reader_for_args(args, fields))
        pipeline.add_stage(
            DataDesignerStage(
                config_builder=config_builder,
                model_providers=model_providers,
                verbose=args.verbose,
            )
        )
        pipeline.add_stage(_writer_for_args(args, writer_path))

        logger.info("\n" + pipeline.describe())
        results = pipeline.run(executor=RayDataExecutor())
        logger.info(f"Shard {context.shard_index} completed with {len(results or [])} output task(s)")
    finally:
        if inference_server is not None:
            inference_server.stop()
        ray_client.stop()
        elapsed_s = time.perf_counter() - started_at
        logger.info(f"Shard {context.shard_index} elapsed time: {elapsed_s:.2f}s")


if __name__ == "__main__":
    main()
