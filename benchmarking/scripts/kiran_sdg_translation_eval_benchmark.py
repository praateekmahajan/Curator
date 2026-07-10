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

"""Curator-native Kiran SDG translation/evaluation benchmark.

This benchmark keeps Kiran's Data Designer workflow intact, but runs it as a
native Curator pipeline with one multi-model Dynamo InferenceServer:

    JsonlReader -> DataDesignerStage -> JsonlWriter

The default model layout targets a four-GPU allocation:

    2 replicas of gemma-4-12B-it, TP=1
    1 replica  of gemma-4-31B-it, TP=2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import data_designer.config as dd
from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.workflow import WorkflowRunResult
from nemo_curator.stages.synthetic.nemo_data_designer.data_designer import DataDesignerStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.tasks.tasks import Task
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.file_utils import get_all_file_paths_under

if TYPE_CHECKING:
    from nemo_curator.core.serve import DynamoVLLMModelConfig, InferenceServer

TRANSLATION_ALIAS = "gemma12b"
EVALUATION_ALIAS = "gemma31b"
PROVIDER_NAME = "curator-dynamo"
GEMMA4_TRANSFORMERS_RUNTIME_PACKAGE = "transformers>=5.10.1"
PipelineTasks = list[Task] | WorkflowRunResult | Mapping[str, list[Task]] | None


def _jsonl_line_count(paths: list[str]) -> int:
    total = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            total += sum(1 for _ in f)
    return total


def _count_output_rows(output_path: Path) -> int:
    output_files = sorted(str(path) for path in output_path.rglob("*.jsonl"))
    if not output_files:
        return 0
    return _jsonl_line_count(output_files)


def _stage_stat(tasks: PipelineTasks, name: str, default: float = 0.0) -> float:
    try:
        value = TaskPerfUtils.get_aggregated_stage_stat(tasks, "DataDesignerStage", name)
    except Exception as exc:
        logger.warning(f"Could not read DataDesignerStage metric {name}: {exc}")
        return default
    if value is None:
        return default
    return float(value)


def _load_config_builder(
    config_path: Path,
    translation_concurrency: int,
    evaluation_concurrency: int,
) -> dd.DataDesignerConfigBuilder:
    with open(config_path, encoding="utf-8") as f:
        raw_config = json.load(f)

    data_designer_config = json.loads(json.dumps(raw_config["data_designer"]))
    data_designer_config.pop("seed_config", None)
    for column_config in data_designer_config["columns"]:
        column_config.pop("allow_resize", None)
        column_config.pop("skip", None)
        column_config.pop("propagate_skip", None)

    for model_config in data_designer_config["model_configs"]:
        alias = model_config["alias"]
        model_config["provider"] = PROVIDER_NAME
        model_config["skip_health_check"] = True

        if alias == TRANSLATION_ALIAS:
            concurrency = translation_concurrency
        elif alias == EVALUATION_ALIAS:
            concurrency = evaluation_concurrency
        else:
            msg = f"Unexpected model alias in Kiran config: {alias}"
            raise ValueError(msg)

        model_config["inference_parameters"]["max_parallel_requests"] = concurrency

    return dd.DataDesignerConfigBuilder.from_config(data_designer_config)


def _engine_kwargs(  # noqa: PLR0913
    tensor_parallel_size: int,
    max_num_seqs: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_batched_tokens: int | None,
    speculative_model: str | None,
    num_speculative_tokens: int,
    linear_backend: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if linear_backend:
        kwargs["linear_backend"] = linear_backend
    if speculative_model:
        kwargs["speculative_config"] = {
            "model": speculative_model,
            "num_speculative_tokens": num_speculative_tokens,
        }
    return kwargs


def _gemma4_transformers_runtime_env(*, disable_deep_gemm: bool = False) -> dict[str, Any]:
    runtime_env: dict[str, Any] = {"uv": {"packages": [GEMMA4_TRANSFORMERS_RUNTIME_PACKAGE]}}
    if disable_deep_gemm:
        runtime_env["env_vars"] = {"VLLM_USE_DEEP_GEMM": "0"}
    return runtime_env


def _build_model_configs(args: argparse.Namespace) -> list[DynamoVLLMModelConfig]:
    from nemo_curator.core.serve import DynamoVLLMModelConfig

    return [
        DynamoVLLMModelConfig(
            model_identifier=args.translation_model_path or args.translation_model_identifier,
            model_name=args.translation_served_model_name,
            runtime_env=_gemma4_transformers_runtime_env(disable_deep_gemm=args.translation_disable_deep_gemm),
            engine_kwargs=_engine_kwargs(
                tensor_parallel_size=args.translation_tensor_parallel_size,
                max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
                speculative_model=None if args.disable_speculative else args.translation_speculative_model,
                num_speculative_tokens=args.num_speculative_tokens,
                linear_backend=args.translation_linear_backend,
            ),
            num_replicas=args.translation_replicas,
        ),
        DynamoVLLMModelConfig(
            model_identifier=args.evaluation_model_path or args.evaluation_model_identifier,
            model_name=args.evaluation_served_model_name,
            runtime_env=_gemma4_transformers_runtime_env(disable_deep_gemm=args.evaluation_disable_deep_gemm),
            engine_kwargs=_engine_kwargs(
                tensor_parallel_size=args.evaluation_tensor_parallel_size,
                max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
                speculative_model=None if args.disable_speculative else args.evaluation_speculative_model,
                num_speculative_tokens=args.num_speculative_tokens,
                linear_backend=args.evaluation_linear_backend,
            ),
            num_replicas=args.evaluation_replicas,
        ),
    ]


def _start_inference_server(args: argparse.Namespace) -> InferenceServer:
    from nemo_curator.core.serve import DynamoServerConfig, InferenceServer

    server = InferenceServer(
        models=_build_model_configs(args),
        backend=DynamoServerConfig(),
        health_check_timeout_s=args.health_check_timeout_s,
        verbose=args.verbose,
    )
    server.start()
    return server


def run_kiran_sdg_benchmark(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915
    if args.hf_home:
        os.environ.setdefault("HF_HOME", str(args.hf_home))

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    ndd_artifact_path = (
        Path(args.ndd_artifact_path)
        if args.ndd_artifact_path
        else (
            checkpoint_path / "ndd-artifacts" if checkpoint_path is not None else output_path.parent / "ndd-artifacts"
        )
    )

    input_files = get_all_file_paths_under(str(input_path), keep_extensions="jsonl")
    input_files = sorted(input_files)
    if args.num_files is not None:
        input_files = input_files[: args.num_files]
    if not input_files:
        msg = f"No JSONL files found under {input_path}"
        raise ValueError(msg)

    input_row_count = _jsonl_line_count(input_files)
    if args.max_records is not None and args.max_records < input_row_count:
        msg = (
            "--max-records is not implemented for file-level Curator partitioning. "
            f"Use --num-files instead, or process the full {input_row_count} rows."
        )
        raise ValueError(msg)

    required_gpus = (
        args.translation_replicas * args.translation_tensor_parallel_size
        + args.evaluation_replicas * args.evaluation_tensor_parallel_size
    )
    aggregate_translation_fanout = args.client_workers * args.translation_concurrency
    aggregate_evaluation_fanout = args.client_workers * args.evaluation_concurrency

    logger.info(f"Input path: {input_path}")
    logger.info(f"Input files: {len(input_files)}")
    logger.info(f"Input rows: {input_row_count}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"NDD run mode: {args.ndd_run_mode}")
    logger.info(f"NDD artifact path: {ndd_artifact_path}")
    logger.info(f"Required GPUs: {required_gpus}")
    logger.info(f"Translation fanout ~= {aggregate_translation_fanout}")
    logger.info(f"Evaluation fanout ~= {aggregate_evaluation_fanout}")

    config_builder = _load_config_builder(
        config_path=Path(args.config_path),
        translation_concurrency=args.translation_concurrency,
        evaluation_concurrency=args.evaluation_concurrency,
    )

    inference_server = None
    serve_startup_s = 0.0
    run_time_s = 0.0
    output_tasks = []

    run_start_wall = time.time()
    try:
        serve_start = time.perf_counter()
        inference_server = _start_inference_server(args)
        serve_startup_s = time.perf_counter() - serve_start

        model_providers = [
            dd.ModelProvider(
                name=PROVIDER_NAME,
                endpoint=inference_server.endpoint,
                api_key="unused",  # pragma: allowlist secret
            )
        ]

        data_designer_stage = DataDesignerStage(
            config_builder=config_builder,
            model_providers=model_providers,
            run_mode=args.ndd_run_mode,
            artifact_path=ndd_artifact_path if args.ndd_run_mode == "create" else None,
            resume_mode=args.ndd_resume_mode,
            run_config={
                "buffer_size": args.ndd_buffer_size,
                "max_in_flight_tasks": args.ndd_max_in_flight_tasks,
            },
            verbose=args.verbose,
        ).with_(num_workers=args.client_workers)

        pipeline = Pipeline(
            name="kiran_sdg_translation_eval",
            stages=[
                JsonlReader(file_paths=input_files, files_per_partition=args.files_per_partition),
                data_designer_stage,
                JsonlWriter(path=str(output_path), mode="overwrite"),
            ],
        )

        executor = setup_executor(args.executor)
        pipeline_start = time.perf_counter()
        output_tasks = pipeline.run(
            executor,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        )
        run_time_s = time.perf_counter() - pipeline_start
    finally:
        if inference_server is not None:
            inference_server.stop()

    output_row_count = int(_stage_stat(output_tasks, "custom.num_output_records"))
    written_output_row_count = _count_output_rows(output_path)
    if output_row_count == 0:
        output_row_count = written_output_row_count

    total_time_s = time.time() - run_start_wall
    throughput_rows_per_sec = output_row_count / run_time_s if run_time_s > 0 else 0.0

    return {
        "params": {
            **vars(args),
            "input_files": input_files,
            "num_input_files": len(input_files),
            "input_row_count": input_row_count,
            "required_gpus": required_gpus,
            "aggregate_translation_fanout": aggregate_translation_fanout,
            "aggregate_evaluation_fanout": aggregate_evaluation_fanout,
            "ndd_run_mode": args.ndd_run_mode,
            "ndd_artifact_path": str(ndd_artifact_path),
            "ndd_resume_mode": args.ndd_resume_mode,
            "ndd_buffer_size": args.ndd_buffer_size,
            "ndd_max_in_flight_tasks": args.ndd_max_in_flight_tasks,
            "model_layout": {
                "translation": {
                    "model_identifier": args.translation_model_path or args.translation_model_identifier,
                    "source_model_identifier": args.translation_model_identifier,
                    "served_model_name": args.translation_served_model_name,
                    "tensor_parallel_size": args.translation_tensor_parallel_size,
                    "num_replicas": args.translation_replicas,
                },
                "evaluation": {
                    "model_identifier": args.evaluation_model_path or args.evaluation_model_identifier,
                    "source_model_identifier": args.evaluation_model_identifier,
                    "served_model_name": args.evaluation_served_model_name,
                    "tensor_parallel_size": args.evaluation_tensor_parallel_size,
                    "num_replicas": args.evaluation_replicas,
                },
            },
        },
        "metrics": {
            "is_success": True,
            "input_file_count": len(input_files),
            "input_row_count": input_row_count,
            "output_row_count": output_row_count,
            "written_output_row_count": written_output_row_count,
            "serve_startup_s": serve_startup_s,
            "pipeline_time_s": run_time_s,
            "total_time_s": total_time_s,
            "throughput_rows_per_sec": throughput_rows_per_sec,
            "client_workers": args.client_workers,
            "translation_concurrency": args.translation_concurrency,
            "evaluation_concurrency": args.evaluation_concurrency,
            "aggregate_translation_fanout": aggregate_translation_fanout,
            "aggregate_evaluation_fanout": aggregate_evaluation_fanout,
            "ndd_run_mode": args.ndd_run_mode,
            "ndd_buffer_size": args.ndd_buffer_size,
            "ndd_max_in_flight_tasks": args.ndd_max_in_flight_tasks,
            "translation_replicas": args.translation_replicas,
            "evaluation_replicas": args.evaluation_replicas,
            "translation_tensor_parallel_size": args.translation_tensor_parallel_size,
            "evaluation_tensor_parallel_size": args.evaluation_tensor_parallel_size,
            "required_gpus": required_gpus,
            "input_tokens_median_per_record": _stage_stat(
                output_tasks,
                "custom.input_tokens_median_per_record",
            ),
            "output_tokens_median_per_record": _stage_stat(
                output_tasks,
                "custom.output_tokens_median_per_record",
            ),
        },
        "tasks": output_tasks,
    }


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="Kiran SDG translation/evaluation benchmark")
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument(
        "--config-path",
        default="/datasets/praateekm/bigiron_datasets/kiran/en_hi_native_nostruct_g412b.bigiron.json",
    )
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--executor", default="ray_data", choices=["ray_data", "xenna", "ray_actors"])
    parser.add_argument("--num-files", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--client-workers", type=int, default=2)
    parser.add_argument("--translation-concurrency", type=int, default=64)
    parser.add_argument("--evaluation-concurrency", type=int, default=48)
    parser.add_argument("--ndd-run-mode", default="create", choices=["preview", "create"])
    parser.add_argument("--ndd-artifact-path", default=None)
    parser.add_argument("--ndd-resume-mode", default="if_possible", choices=["never", "always", "if_possible"])
    parser.add_argument("--ndd-buffer-size", type=int, default=64)
    parser.add_argument("--ndd-max-in-flight-tasks", type=int, default=192)
    parser.add_argument("--translation-replicas", type=int, default=2)
    parser.add_argument("--evaluation-replicas", type=int, default=1)
    parser.add_argument("--translation-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--evaluation-tensor-parallel-size", type=int, default=2)
    parser.add_argument(
        "--translation-model-identifier",
        default="RedHatAI/gemma-4-12B-it-FP8-Dynamic",
    )
    parser.add_argument(
        "--translation-model-path",
        default=None,
        help=(
            "Optional absolute local model snapshot path. When set, Dynamo/vLLM loads "
            "weights from this path while --translation-served-model-name remains the served name."
        ),
    )
    parser.add_argument(
        "--translation-served-model-name",
        default="google/gemma-4-12B-it",
    )
    parser.add_argument(
        "--evaluation-model-identifier",
        default="RedHatAI/gemma-4-31B-it-FP8-block",
    )
    parser.add_argument(
        "--evaluation-model-path",
        default=None,
        help=(
            "Optional absolute local model snapshot path. When set, Dynamo/vLLM loads "
            "weights from this path while --evaluation-served-model-name remains the served name."
        ),
    )
    parser.add_argument(
        "--evaluation-served-model-name",
        default="google/gemma-4-31B-it",
    )
    parser.add_argument(
        "--translation-speculative-model",
        default="google/gemma-4-12B-it-assistant",
    )
    parser.add_argument(
        "--evaluation-speculative-model",
        default="google/gemma-4-31B-it-assistant",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=4)
    parser.add_argument("--disable-speculative", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--translation-linear-backend", default=None)
    parser.add_argument("--evaluation-linear-backend", default=None)
    parser.add_argument("--translation-disable-deep-gemm", action="store_true")
    parser.add_argument("--evaluation-disable-deep-gemm", action="store_true")
    parser.add_argument("--health-check-timeout-s", type=int, default=1800)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    logger.info("=== Kiran SDG translation/evaluation benchmark starting ===")
    logger.info(f"Arguments: {vars(args)}")

    result_dict: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    success_code = 1
    try:
        result_dict.update(run_kiran_sdg_benchmark(args))
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)

    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
