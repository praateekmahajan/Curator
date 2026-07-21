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

"""Benchmark Ray Data KMeans prediction and partitioned output writing."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarking.scripts.utils import write_benchmark_results
from benchmarking.semdedup.kmeans import (
    _input_files,
    _metadata_fields,
    _num_rows,
    _verify_output_metadata,
    _write_batch_size,
)
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.utils import get_available_cpu_gpu_resources
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansPredictWriteStage
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.utils import TaskPerfUtils
from nemo_curator.utils.file_utils import infer_dataset_name_from_path


def _file_group_tasks(input_files: list[str], files_per_task: int) -> list[FileGroupTask]:
    groups = [input_files[start : start + files_per_task] for start in range(0, len(input_files), files_per_task)]
    dataset_name = infer_dataset_name_from_path(input_files[0])
    return [
        FileGroupTask(
            dataset_name=dataset_name,
            data=group,
            _metadata={
                "partition_index": index,
                "total_partitions": len(groups),
                "source_files": group,
            },
            reader_config={},
        )
        for index, group in enumerate(groups)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_files = _input_files([Path(path) for path in args.input_path], args.input_file_limit)
    metadata_fields = _metadata_fields(input_files, args.id_field, args.embedding_field)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=False)

    available_cpus, available_gpus = get_available_cpu_gpu_resources(init_and_shutdown=True)
    cluster_resources = {"CPU": available_cpus, "GPU": available_gpus}
    if available_gpus < 1:
        msg = f"KMeans prediction requires Ray GPU resources; found {cluster_resources}"
        raise RuntimeError(msg)
    worker_count = max(1, int(available_gpus / args.gpu_fraction))
    input_tasks = _file_group_tasks(input_files, args.task_batch_size)

    write_kwargs: dict[str, Any] = {}
    if args.compression != "default":
        write_kwargs["compression"] = None if args.compression == "none" else args.compression

    pipeline = Pipeline(
        name="semdedup_kmeans_predict_benchmark",
        stages=[
            KMeansPredictWriteStage(
                id_field=args.id_field,
                embedding_field=args.embedding_field,
                metadata_fields=metadata_fields,
                output_path=str(output_path),
                centroids_path=args.centroids_path,
                embedding_dim=args.embedding_dim,
                max_samples_per_batch=args.max_samples_per_batch,
                output_embedding_dtype=args.output_embedding_dtype,
                write_batch_size=args.write_batch_size,
                max_output_file_size=(
                    args.max_output_file_size_mb * 1_000_000 if args.max_output_file_size_mb is not None else None
                ),
                prefetch_next_group=args.prefetch_next_group,
                task_batch_size=1,
                gpu_fraction=args.gpu_fraction,
                worker_count=worker_count,
                write_kwargs=write_kwargs,
            ),
        ],
    )

    started = time.perf_counter()
    tasks = pipeline.run(RayDataExecutor(), initial_tasks=input_tasks)
    duration_s = time.perf_counter() - started
    _verify_output_metadata(output_path, metadata_fields)
    task_metrics = TaskPerfUtils.aggregate_task_metrics(tasks)
    rows_processed = _num_rows(task_metrics)
    output_files = list(output_path.rglob("*.parquet"))
    output_bytes = sum(path.stat().st_size for path in output_files)
    return {
        "params": {
            **vars(args),
            "metadata_fields": metadata_fields,
            "input_file_count": len(input_files),
            "input_bytes": sum(Path(path).stat().st_size for path in input_files),
            "ray_cluster_resources": cluster_resources,
            "predict_worker_count": worker_count,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": duration_s,
            "num_documents_processed": rows_processed,
            "throughput_docs_per_sec": rows_processed / duration_s if duration_s else None,
            "output_bytes": output_bytes,
            "output_file_count": len(output_files),
            **task_metrics,
        },
        "tasks": tasks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--centroids-path", required=True)
    parser.add_argument("--input-file-limit", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--embedding-field", default="embeddings")
    parser.add_argument("--max-samples-per-batch", type=int, default=32768)
    parser.add_argument("--output-embedding-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--write-batch-size", type=_write_batch_size, default="auto")
    parser.add_argument("--compression", choices=["default", "zstd", "snappy", "none"], default="default")
    parser.add_argument("--max-output-file-size-mb", type=int)
    parser.add_argument("--prefetch-next-group", action="store_true")
    parser.add_argument("--task-batch-size", type=int, default=48)
    parser.add_argument("--gpu-fraction", type=float, default=1.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    results: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        results = run(args)
        return 0
    finally:
        write_benchmark_results(results, args.benchmark_results_path)


if __name__ == "__main__":
    raise SystemExit(main())
