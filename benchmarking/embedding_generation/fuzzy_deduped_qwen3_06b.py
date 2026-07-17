# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

_CURATOR_REPO_DIR = Path(__file__).resolve().parents[2]
_BENCHMARKING_SCRIPTS_DIR = _CURATOR_REPO_DIR / "benchmarking" / "scripts"
sys.path.insert(0, str(_CURATOR_REPO_DIR))
sys.path.insert(0, str(_BENCHMARKING_SCRIPTS_DIR))

from benchmarking.embedding_generation.manifest import ManifestFilePartitioningStage
from benchmarking.embedding_generation.writer import MirroredParquetWriter
from benchmarking.scripts.embedding_generation_benchmark import (
    EmbeddingModelVariation,
    _create_embedding_stages,
    _resolve_max_seq_length,
)
from benchmarking.scripts.utils import setup_executor, write_benchmark_results
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
    create_id_generator_actor,
    kill_id_generator_actor,
)
from nemo_curator.stages.text.io.reader import JsonlReaderStage
from nemo_curator.stages.text.modules import MetadataExtractor
from nemo_curator.tasks.utils import TaskPerfUtils


def load_id_path_mapping(path: str | Path) -> dict[str, str]:
    """Load dedup-runtime to ID-registry path-prefix mappings."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        msg = f"Path mapping must contain a JSON list: {path}"
        raise TypeError(msg)

    result: dict[str, str] = {}
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            msg = f"Path mapping record {index} must be a JSON object"
            raise TypeError(msg)
        runtime_prefix = record.get("dedup_path")
        registry_prefix = record.get("container_mounted_dedup_source_path")
        if not isinstance(runtime_prefix, str) or not isinstance(registry_prefix, str):
            msg = (
                f"Path mapping record {index} must contain string dedup_path and "
                "container_mounted_dedup_source_path fields"
            )
            raise TypeError(msg)
        if runtime_prefix in result and result[runtime_prefix] != registry_prefix:
            msg = f"Conflicting mappings for dedup path {runtime_prefix}"
            raise ValueError(msg)
        result[runtime_prefix.rstrip("/")] = registry_prefix.rstrip("/")

    if not result:
        msg = f"Path mapping contains no records: {path}"
        raise ValueError(msg)
    return result


def load_metadata_extractor(path: str | Path) -> MetadataExtractor:
    """Load file-level integer metadata without embedding policy data in code."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        msg = f"Metadata configuration must contain a JSON object: {path}"
        raise TypeError(msg)

    metadata_mapping = payload.get("metadata_mapping")
    output_dtypes = payload.get("output_dtypes")
    task_metadata_field = payload.get("task_metadata_field", "mapping_names")
    text_extraction = payload.get("text_extraction", {})
    if not isinstance(metadata_mapping, dict) or not isinstance(output_dtypes, dict):
        msg = f"Metadata configuration must define metadata_mapping and output_dtypes objects: {path}"
        raise TypeError(msg)
    if not isinstance(task_metadata_field, str):
        msg = f"Metadata configuration task_metadata_field must be a string: {path}"
        raise TypeError(msg)
    if not isinstance(text_extraction, dict):
        msg = f"Metadata configuration text_extraction must be an object: {path}"
        raise TypeError(msg)

    return MetadataExtractor(
        metadata_mapping=metadata_mapping,
        output_dtypes=output_dtypes,
        task_metadata_field=task_metadata_field,
        **text_extraction,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output_path).absolute()
    checkpoint_dir = Path(args.checkpoint_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    id_path_mapping = load_id_path_mapping(args.path_mapping_json)
    metadata_extractor = load_metadata_extractor(args.metadata_mapping_json)
    variation = EmbeddingModelVariation(args.model_variation)
    max_seq_length = _resolve_max_seq_length(args.model_identifier, cache_dir=args.cache_dir)

    reader = JsonlReaderStage(fields=None, _assign_ids=True)
    if args.reader_max_workers is not None:
        reader = reader.with_(ray_stage_spec={RayStageSpecKeys.MAX_WORKERS: args.reader_max_workers})

    embedding_stages = _create_embedding_stages(
        model_identifier=args.model_identifier,
        model_variation=variation,
        model_inference_batch_size=args.model_inference_batch_size,
        model_num_workers=args.model_num_workers,
        model_worker_gpus=args.model_worker_gpus,
        model_gpu_memory_utilization=args.model_gpu_memory_utilization,
        model_kv_cache_memory_bytes=args.model_kv_cache_memory_bytes,
        max_seq_length=max_seq_length,
        embedding_pooling=args.embedding_pooling,
        cache_dir=args.cache_dir,
        # None preserves every input column. The writer removes only text.
        metadata_fields=None,
    )

    writer = MirroredParquetWriter(
        path=str(output_path),
        source_root=args.source_root,
        fields=None,
        write_kwargs={
            "compression": "zstd",
            "compression_level": 3,
            "use_byte_stream_split": ["embeddings.list.element"],
            "use_dictionary": False,
        },
    )
    pipeline = Pipeline(
        name="fuzzy_deduped_embedding_generation",
        stages=[
            ManifestFilePartitioningStage(
                manifest_path=args.manifest_path,
                path_mapping=id_path_mapping,
                required_minimum_files_per_shard=args.require_min_files_per_shard,
                manifest_max_rows=args.manifest_max_rows,
            ),
            reader,
            metadata_extractor,
            *embedding_stages,
            writer,
        ],
    )

    logger.info("Manifest: {}", args.manifest_path)
    logger.info("Output root: {}", output_path)
    logger.info("Checkpoint root: {}", checkpoint_dir)
    logger.info("ID path mappings: {}", len(id_path_mapping))

    started = time.perf_counter()
    create_id_generator_actor(args.id_generator_path, path_mapping=id_path_mapping)
    try:
        output_tasks = (
            pipeline.run(
                setup_executor(args.executor),
                checkpoint_path=checkpoint_dir,
            )
            or []
        )
    finally:
        kill_id_generator_actor()
    elapsed = time.perf_counter() - started

    num_documents = sum(task._stage_perf[-1].num_items_processed for task in output_tasks if task._stage_perf)
    stage_metrics = TaskPerfUtils.collect_stage_metrics(output_tasks)
    return {
        "params": {
            **vars(args),
            "max_seq_length": max_seq_length,
            "id_path_mapping_count": len(id_path_mapping),
            "output_schema": "all input columns except text, plus _curator_dedup_id and embeddings",
            "id_column": CURATOR_DEDUP_ID_STR,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": elapsed,
            "num_documents_processed": num_documents,
            "throughput_docs_per_sec": num_documents / elapsed if elapsed else 0.0,
            "num_output_files": len(output_tasks),
            "stage_names": sorted(stage_metrics),
        },
        "tasks": output_tasks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manifest-sharded fuzzy-deduped embedding generation")
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--path-mapping-json", required=True)
    parser.add_argument("--metadata-mapping-json", required=True)
    parser.add_argument("--id-generator-path", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--require-min-files-per-shard",
        type=int,
        default=1,
        help="Fail if the fixed TOTAL_SHARDS partition gives any shard fewer files",
    )
    parser.add_argument("--manifest-max-rows", type=int, default=None)
    parser.add_argument("--executor", default="ray_data", choices=["ray_data"])
    parser.add_argument("--reader-max-workers", type=int, default=4)
    parser.add_argument("--model-identifier", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--model-variation",
        default=EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED.value,
        choices=[variation.value for variation in EmbeddingModelVariation],
    )
    parser.add_argument("--model-inference-batch-size", type=int, default=10_000)
    parser.add_argument("--model-num-workers", type=int, default=4)
    parser.add_argument("--model-worker-gpus", type=float, default=1.0)
    parser.add_argument("--model-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--model-kv-cache-memory-bytes", type=int, default=34_359_738_368)
    parser.add_argument("--embedding-pooling", default="mean_pooling")
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    exit_code = 1
    try:
        results = run(args)
        exit_code = 0
    finally:
        write_benchmark_results(results, args.benchmark_results_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
