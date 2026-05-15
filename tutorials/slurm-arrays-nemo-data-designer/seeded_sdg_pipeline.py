# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Seeded DataDesigner pipeline for SLURM array runs.

The script covers seed-data generation from JSONL or Parquet files. Pure
non-seed generation (``--num-records N`` with no seed) needs record-range
partitioning and is left out of this prototype.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

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
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter

from slurm_array_support import Shard, SlurmArrayPipeline, enable_slurm_array_partitioning


# ---------------------------------------------------------------------------
# Model serve spec — one per served endpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelServeSpec:
    """One served model endpoint. Loaded from MODELS_JSON list entries."""

    model_identifier: str
    model_name: str
    tensor_parallel_size: int
    num_replicas: int | str  # int or "auto"
    engine_kwargs: dict[str, Any]

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, base_engine_kwargs: dict[str, Any]) -> "ModelServeSpec":
        model_id = raw.get("model") or raw.get("model_identifier")
        if not model_id:
            msg = f"Each model spec needs 'model' (got: {raw})"
            raise ValueError(msg)
        tp = raw.get("tp") or raw.get("tensor_parallel_size")
        if not tp:
            msg = f"Each model spec needs 'tp'/'tensor_parallel_size' (got: {raw})"
            raise ValueError(msg)
        engine_kwargs = {**base_engine_kwargs, **(raw.get("engine_kwargs") or {}), "tensor_parallel_size": int(tp)}
        if raw.get("max_model_len") is not None:
            engine_kwargs["max_model_len"] = int(raw["max_model_len"])
        replicas = raw.get("replicas", raw.get("num_replicas", "auto"))
        return cls(
            model_identifier=str(model_id),
            model_name=str(raw.get("served_model_name") or raw.get("model_name") or model_id),
            tensor_parallel_size=int(tp),
            num_replicas="auto" if str(replicas).lower() == "auto" else int(replicas),
            engine_kwargs=engine_kwargs,
        )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a Curator DataDesigner pipeline as one SLURM array shard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    e = os.environ.get
    p.add_argument("--input-path", default=e("INPUT_PATH"))
    p.add_argument("--output-path", default=e("OUTPUT_PATH"))
    p.add_argument("--data-designer-config-file", default=e("DD_CONFIG") or e("DATA_DESIGNER_CONFIG_FILE"))
    p.add_argument("--models-json", default=e("MODELS_JSON"))
    p.add_argument("--models-json-file", default=e("MODELS_JSON_FILE"))
    p.add_argument("--input-format", choices=["jsonl", "parquet"], default=e("INPUT_FORMAT", "jsonl"))
    p.add_argument("--output-format", choices=["jsonl", "parquet"], default=e("OUTPUT_FORMAT", "jsonl"))
    p.add_argument("--output-layout", choices=["flat", "by_shard"], default=e("OUTPUT_LAYOUT", "flat"))
    p.add_argument("--fields", default=e("FIELDS"), help="Comma-separated seed columns to read.")
    p.add_argument("--files-per-partition", type=int, default=int(e("FILES_PER_PARTITION") or 0) or None)
    p.add_argument("--blocksize", default=e("BLOCKSIZE"))
    p.add_argument("--provider-name", default=e("PROVIDER_NAME", "local"))
    p.add_argument("--api-key", default=e("OPENAI_API_KEY", "unused"))  # pragma: allowlist secret
    p.add_argument("--server-backend", choices=["dynamo", "ray-serve"], default=e("SERVER_BACKEND", "dynamo"))
    p.add_argument("--server-port", type=int, default=int(e("SERVE_PORT") or 8000))
    p.add_argument("--health-check-timeout-s", type=int, default=int(e("HEALTH_CHECK_TIMEOUT_S") or 600))
    p.add_argument("--max-model-len", type=int, default=int(e("MAX_MODEL_LEN") or 0) or None)
    p.add_argument("--vllm-engine-kwargs-json", default=e("VLLM_ENGINE_KWARGS_JSON"))
    p.add_argument("--ray-temp-dir", default=e("RAY_TMPDIR", "/tmp/ray_curator"))  # noqa: S108
    p.add_argument("--ray-worker-connect-timeout-s", type=int, default=int(e("RAY_WORKER_CONNECT_TIMEOUT_S") or 600))
    p.add_argument("--ignore-head-node-for-replicas", action="store_true",
                   default=e("IGNORE_HEAD_NODE_FOR_REPLICAS", "0") not in ("0", "", "false", "False"))
    p.add_argument("--verbose", action="store_true", default=e("VERBOSE", "0") not in ("0", "", "false", "False"))
    args = p.parse_args()

    missing = [n for n, v in {
        "--input-path or INPUT_PATH": args.input_path,
        "--output-path or OUTPUT_PATH": args.output_path,
        "--data-designer-config-file or DD_CONFIG": args.data_designer_config_file,
    }.items() if not v]
    if not (args.models_json or args.models_json_file):
        missing.append("--models-json/MODELS_JSON or --models-json-file/MODELS_JSON_FILE")
    if missing:
        p.error("Missing required value(s): " + ", ".join(missing))
    if args.files_per_partition is not None and args.blocksize is not None:
        p.error("--files-per-partition and --blocksize are mutually exclusive")
    return args


# ---------------------------------------------------------------------------
# Model specs + inference server
# ---------------------------------------------------------------------------


def _model_specs(args: argparse.Namespace) -> list[ModelServeSpec]:
    """Load model specs from MODELS_JSON (inline) or MODELS_JSON_FILE.

    Always a JSON list — even a single-model run goes through the list form
    so there's one parsing path. See configs/cached_hf_gretel_models.json
    for an example.
    """
    base_engine_kwargs: dict[str, Any] = {}
    if args.vllm_engine_kwargs_json:
        base_engine_kwargs.update(json.loads(args.vllm_engine_kwargs_json))
    if args.max_model_len is not None:
        base_engine_kwargs["max_model_len"] = args.max_model_len

    raw = args.models_json or Path(args.models_json_file).read_text(encoding="utf-8")
    entries = json.loads(raw)
    if not isinstance(entries, list) or not entries:
        msg = "MODELS_JSON must decode to a non-empty JSON list"
        raise TypeError(msg)
    specs = [ModelServeSpec.from_json(e, base_engine_kwargs=base_engine_kwargs) for e in entries]
    names = [s.model_name for s in specs]
    if len(names) != len(set(names)):
        msg = f"served_model_name values must be unique, got {names}"
        raise ValueError(msg)
    return specs


def _resolve_replicas(specs: list[ModelServeSpec], *, ignore_head_node: bool) -> list[int]:
    """Resolve ``'auto'`` replicas by using as many GPUs as possible across auto-sized models.

    ``init_and_shutdown=True`` so we re-attach this client to the already-running
    Ray cluster via ``RAY_ADDRESS``. (``SlurmRayClient._wait_for_workers`` calls
    ``ray.shutdown()`` in its finally block, detaching this process from Ray;
    the cluster itself keeps running as a subprocess.)
    """
    explicit_gpus = sum(int(s.num_replicas) * s.tensor_parallel_size for s in specs if s.num_replicas != "auto")
    auto_tp_sum = sum(s.tensor_parallel_size for s in specs if s.num_replicas == "auto")
    if auto_tp_sum == 0:
        return [int(s.num_replicas) for s in specs]

    _, available = get_available_cpu_gpu_resources(init_and_shutdown=True, ignore_head_node=ignore_head_node)
    remaining = int(available) - explicit_gpus
    if remaining < auto_tp_sum:
        msg = (f"Not enough GPUs for one replica of each auto-sized model: available={int(available)}, "
               f"explicit_gpus={explicit_gpus}, auto_tp_sum={auto_tp_sum}")
        raise RuntimeError(msg)
    auto = remaining // auto_tp_sum
    logger.info(f"Auto-selected {auto} replica(s) per auto-sized model from {int(available)} GPU(s)")
    return [auto if s.num_replicas == "auto" else int(s.num_replicas) for s in specs]


def _start_inference_server(args: argparse.Namespace, specs: list[ModelServeSpec]) -> InferenceServer:
    replicas = _resolve_replicas(specs, ignore_head_node=args.ignore_head_node_for_replicas)
    if args.server_backend == "dynamo":
        models = [
            DynamoVLLMModelConfig(
                model_identifier=s.model_identifier, model_name=s.model_name,
                num_replicas=replicas[i], engine_kwargs=s.engine_kwargs,
            ) for i, s in enumerate(specs)
        ]
        server = InferenceServer(models=models, backend=DynamoServerConfig(),
                                 port=args.server_port, health_check_timeout_s=args.health_check_timeout_s,
                                 verbose=args.verbose)
    else:
        models = [
            RayServeModelConfig(
                model_identifier=s.model_identifier, model_name=s.model_name,
                deployment_config={"autoscaling_config": {"min_replicas": replicas[i], "max_replicas": replicas[i]}},
                engine_kwargs=s.engine_kwargs,
            ) for i, s in enumerate(specs)
        ]
        server = InferenceServer(models=models, backend=RayServeServerConfig(),
                                 port=args.server_port, health_check_timeout_s=args.health_check_timeout_s,
                                 verbose=args.verbose)
    server.start()
    served = ", ".join(f"{s.model_name}({replicas[i]}x TP={s.tensor_parallel_size})" for i, s in enumerate(specs))
    logger.info(f"Inference server at {server.endpoint}; models: {served}")
    return server


# ---------------------------------------------------------------------------
# Stage builders
# ---------------------------------------------------------------------------


def _reader(args: argparse.Namespace, fields: list[str] | None) -> Any:
    cls = JsonlReader if args.input_format == "jsonl" else ParquetReader
    reader = cls(
        file_paths=args.input_path,
        files_per_partition=args.files_per_partition,
        blocksize=args.blocksize,
        fields=fields,
    )
    return enable_slurm_array_partitioning(
        reader,
        output_path=args.output_path,
        output_file_extension=args.output_format,
        output_layout=args.output_layout,
    )


def _writer(args: argparse.Namespace, path: str) -> Any:
    return (JsonlWriter if args.output_format == "jsonl" else ParquetWriter)(path=path)


def _ray_client(args: argparse.Namespace) -> RayClient | SlurmRayClient:
    if os.environ.get("SLURM_JOB_ID"):
        return SlurmRayClient(
            include_dashboard=False,
            ray_temp_dir=args.ray_temp_dir,
            worker_connect_timeout_s=args.ray_worker_connect_timeout_s,
        )
    return RayClient(include_dashboard=False, ray_temp_dir=args.ray_temp_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    try:
        import data_designer.config as dd
    except ModuleNotFoundError as exc:
        msg = "data_designer is required; use a Curator container that includes NeMo Data Designer."
        raise ModuleNotFoundError(msg) from exc

    shard_idx, num_shards = Shard.env()
    writer_path = Shard.data_path(args.output_path, args.output_layout, shard_idx)
    fields = [f.strip() for f in (args.fields or "").split(",") if f.strip()] or None
    t0 = time.perf_counter()
    logger.info(f"Shard {shard_idx}/{num_shards}: input={args.input_path}, writer_path={writer_path}")

    # 1. Cheap driver-side short-circuit: if this shard already has a marker, exit.
    #    Real partitioning + resume logic runs as a stage on the worker, where lustre
    #    perms and the curator container environment are guaranteed.
    if Shard.has_marker(args.output_path, shard_idx):
        logger.info(f"Shard already complete: {Shard.marker_path(args.output_path, shard_idx)}")
        return

    # 2. Bring up Ray + start the inference server(s).
    ray_client = _ray_client(args)
    inference_server: InferenceServer | None = None
    ts = time.perf_counter(); ray_client.start(); ray_start_s = time.perf_counter() - ts

    try:
        specs = _model_specs(args)
        ts = time.perf_counter()
        cb = dd.DataDesignerConfigBuilder.from_config(args.data_designer_config_file)
        config_load_s = time.perf_counter() - ts
        ts = time.perf_counter()
        inference_server = _start_inference_server(args, specs)
        serve_startup_s = time.perf_counter() - ts

        providers = [
            dd.ModelProvider(
                name=args.provider_name,
                endpoint=inference_server.endpoint,
                provider_type="openai",
                api_key=args.api_key,
            )
        ]

        # 3. Build + run the pipeline. The reader uses the shard-aware partitioner
        #    (already swapped in by enable_slurm_array_partitioning).
        pipeline = SlurmArrayPipeline(
            name="slurm_array_ndd_seeded_sdg",
            description="Seeded DataDesigner generation over one SLURM array shard",
            output_path=args.output_path,
            success_payload={
                "served_models": [
                    {"model_name": s.model_name, "tensor_parallel_size": s.tensor_parallel_size}
                    for s in specs
                ],
                "metrics": {
                    "ray_start_s": round(ray_start_s, 2),
                    "config_load_s": round(config_load_s, 2),
                    "serve_startup_s": round(serve_startup_s, 2),
                },
            },
        )
        pipeline.add_stage(_reader(args, fields))
        pipeline.add_stage(DataDesignerStage(config_builder=cb, model_providers=providers, verbose=args.verbose))
        pipeline.add_stage(_writer(args, writer_path))

        logger.info("\n" + pipeline.describe())
        results = pipeline.run(executor=RayDataExecutor())
        logger.info(f"Shard {shard_idx} completed with {len(results or [])} output task(s)")
    finally:
        if inference_server is not None:
            inference_server.stop()
        ray_client.stop()
        logger.info(f"Shard {shard_idx} elapsed: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
