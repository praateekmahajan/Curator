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

"""Embedding generation benchmark that preserves all non-stale input columns."""

import argparse
import time
from enum import Enum
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from embedding_generation_benchmark import _resolve_max_seq_length
from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.utils.file_utils import get_all_file_paths_and_size_under


class EmbeddingModelVariation(Enum):
    VLLM_TEXT = "vllm_text"
    VLLM_TEXT_PRETOKENIZED = "vllm_text_pretokenized"


def _load_parquet_files(
    input_path: Path,
    dataset_size_gb: float | None,
    dataset_ratio: float | None,
) -> list[str]:
    if dataset_size_gb is not None or dataset_ratio is not None:
        return load_dataset_files(
            input_path,
            dataset_size_gb=dataset_size_gb,
            dataset_ratio=dataset_ratio,
            keep_extensions="parquet",
        )

    return [
        path
        for path, _ in get_all_file_paths_and_size_under(
            str(input_path),
            recurse_subdirectories=True,
            keep_extensions="parquet",
            sort_by_size=False,
        )
    ]


def _get_input_fields(
    first_file: str,
    text_field: str,
    old_embedding_field: str,
    embedding_field: str,
) -> tuple[list[str], list[str]]:
    schema_names = pq.ParquetFile(first_file).schema_arrow.names
    if text_field not in schema_names:
        msg = f"Text field {text_field!r} not found in input schema: {schema_names}"
        raise ValueError(msg)

    read_fields = [name for name in schema_names if name not in {old_embedding_field, embedding_field}]
    output_fields = [*read_fields, embedding_field]
    return read_fields, output_fields


def _collect_parquet_file_metrics(files: list[str]) -> dict[str, Any]:
    total_rows = 0
    total_bytes = 0
    for file_path in files:
        path = Path(file_path)
        total_bytes += path.stat().st_size
        total_rows += pq.ParquetFile(path).metadata.num_rows
    return {
        "num_files": len(files),
        "total_bytes": total_bytes,
        "total_mb": total_bytes / 1e6,
        "num_rows": total_rows,
    }


def _collect_output_metrics(output_path: Path) -> dict[str, Any]:
    output_files = [
        path
        for path, _ in get_all_file_paths_and_size_under(
            str(output_path),
            recurse_subdirectories=True,
            keep_extensions="parquet",
            sort_by_size=False,
        )
    ]
    if not output_files:
        return {
            "output_num_files": 0,
            "output_total_bytes": 0,
            "output_total_mb": 0,
            "output_num_rows": 0,
        }
    metrics = _collect_parquet_file_metrics(output_files)
    return {f"output_{key}": value for key, value in metrics.items()}


def _create_vllm_embedding_stage(  # noqa: PLR0913
    model_identifier: str,
    model_variation: EmbeddingModelVariation,
    text_field: str,
    embedding_field: str,
    max_seq_length: int,
    cache_dir: str | None,
    embedding_gpus_per_worker: float,
    embedding_cpus_per_worker: float,
    embedding_num_workers: int | None,
    vllm_gpu_memory_utilization: float | None,
    vllm_max_port_retries: int,
) -> VLLMEmbeddingModelStage:
    vllm_init_kwargs: dict[str, Any] = {"max_model_len": max_seq_length}
    if vllm_gpu_memory_utilization is not None:
        vllm_init_kwargs["gpu_memory_utilization"] = vllm_gpu_memory_utilization
    vllm_init_kwargs["max_port_retries"] = vllm_max_port_retries

    stage = VLLMEmbeddingModelStage(
        model_identifier=model_identifier,
        text_field=text_field,
        embedding_field=embedding_field,
        pretokenize=model_variation == EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED,
        vllm_init_kwargs=vllm_init_kwargs,
        cache_dir=cache_dir,
    )
    return stage.with_(
        resources=Resources(cpus=embedding_cpus_per_worker, gpus=embedding_gpus_per_worker),
        num_workers=embedding_num_workers,
    )


def run_embedding_generation_full_columns_benchmark(  # noqa: PLR0913
    benchmark_results_path: str,
    input_path: str,
    output_path: str,
    executor: str,
    model_identifier: str,
    model_variation: str,
    cache_dir: str | None,
    dataset_size_gb: float | None,
    dataset_ratio: float | None,
    text_field: str,
    old_embedding_field: str,
    embedding_field: str,
    max_seq_length: int | None,
    files_per_partition: int,
    embedding_gpus_per_worker: float,
    embedding_cpus_per_worker: float,
    embedding_num_workers: int | None,
    vllm_gpu_memory_utilization: float | None,
    vllm_max_port_retries: int,
    writer_mode: str,
    **kwargs: Any,  # noqa: ARG001
) -> dict[str, Any]:
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path).absolute()
    variation = EmbeddingModelVariation(model_variation)

    input_files = _load_parquet_files(input_path_obj, dataset_size_gb, dataset_ratio)
    if not input_files:
        msg = f"No parquet files found under {input_path}"
        raise FileNotFoundError(msg)

    read_fields, output_fields = _get_input_fields(
        input_files[0],
        text_field=text_field,
        old_embedding_field=old_embedding_field,
        embedding_field=embedding_field,
    )
    resolved_max_seq_length = max_seq_length or _resolve_max_seq_length(model_identifier, cache_dir=cache_dir)

    logger.info("Starting full-column embedding generation benchmark")
    logger.info(f"Input path: {input_path_obj}")
    logger.info(f"Output path: {output_path_obj}")
    logger.info(f"Input files: {len(input_files)}")
    logger.info(f"Read fields: {read_fields}")
    logger.info(f"Output fields: {output_fields}")
    logger.info(f"Model: {model_identifier}")
    logger.info(f"Model variation: {variation.value}")
    logger.info(f"Max seq length: {resolved_max_seq_length}")
    logger.info(f"Embedding workers: {embedding_num_workers}")
    logger.info(f"Embedding resources: {embedding_cpus_per_worker} CPUs, {embedding_gpus_per_worker} GPUs")
    logger.info(f"vLLM GPU memory utilization: {vllm_gpu_memory_utilization}")
    logger.info(f"vLLM max port retries: {vllm_max_port_retries}")

    input_metrics = {f"input_{key}": value for key, value in _collect_parquet_file_metrics(input_files).items()}

    reader = ParquetReader(
        file_paths=input_files,
        files_per_partition=files_per_partition,
        fields=read_fields,
        _generate_ids=False,
    )
    embedding_stage = _create_vllm_embedding_stage(
        model_identifier=model_identifier,
        model_variation=variation,
        text_field=text_field,
        embedding_field=embedding_field,
        max_seq_length=resolved_max_seq_length,
        cache_dir=cache_dir,
        embedding_gpus_per_worker=embedding_gpus_per_worker,
        embedding_cpus_per_worker=embedding_cpus_per_worker,
        embedding_num_workers=embedding_num_workers,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_port_retries=vllm_max_port_retries,
    )
    writer = ParquetWriter(path=str(output_path_obj), fields=output_fields, mode=writer_mode)

    run_start_time = time.perf_counter()
    output_tasks = Pipeline(
        name="embedding_generation_full_columns_pipeline",
        stages=[reader, embedding_stage, writer],
    ).run(setup_executor(executor))
    run_time_taken = time.perf_counter() - run_start_time

    num_documents_processed = sum(task._stage_perf[-1].num_items_processed for task in output_tasks)
    throughput_docs_per_sec = num_documents_processed / run_time_taken if run_time_taken > 0 else 0

    logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
    logger.success(f"Processed {num_documents_processed} documents")

    return {
        "params": {
            "max_seq_length": resolved_max_seq_length,
            "input_columns": pq.ParquetFile(input_files[0]).schema_arrow.names,
            "read_columns": read_fields,
            "output_columns": output_fields,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "num_documents_processed": num_documents_processed,
            "throughput_docs_per_sec": throughput_docs_per_sec,
            **input_metrics,
            **_collect_output_metrics(output_path_obj),
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-column embedding generation benchmark")
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--executor", default="ray_data", choices=["ray_data"])
    parser.add_argument("--model-identifier", default="google/embeddinggemma-300m")
    parser.add_argument(
        "--model-variation",
        default="vllm_text",
        choices=[variation.value for variation in EmbeddingModelVariation],
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset-size-gb", type=float, default=None)
    parser.add_argument("--dataset-ratio", type=float, default=None)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--old-embedding-field", default="embedding")
    parser.add_argument("--embedding-field", default="embedding")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--embedding-gpus-per-worker", type=float, default=0.249)
    parser.add_argument("--embedding-cpus-per-worker", type=float, default=1.0)
    parser.add_argument("--embedding-num-workers", type=int, default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--vllm-max-port-retries", type=int, default=8)
    parser.add_argument("--writer-mode", default="error", choices=["ignore", "overwrite", "append", "error"])

    args = parser.parse_args()

    result_dict: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    success_code = 1
    try:
        run_result = run_embedding_generation_full_columns_benchmark(**vars(args))
        result_dict["params"].update(run_result.pop("params", {}))
        result_dict.update(run_result)
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
