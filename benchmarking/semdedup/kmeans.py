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

"""Benchmark the semantic-deduplication KMeans stage on existing embeddings."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarking.scripts.utils import write_benchmark_results
from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.utils import get_available_cpu_gpu_resources
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage
from nemo_curator.tasks.utils import TaskPerfUtils


def _write_batch_size(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        batch_size = int(value)
    except ValueError as error:
        msg = "write batch size must be a positive integer or 'auto'"
        raise argparse.ArgumentTypeError(msg) from error
    if batch_size <= 0:
        msg = "write batch size must be positive"
        raise argparse.ArgumentTypeError(msg)
    return batch_size


def _input_files(input_paths: list[Path], input_file_limit: int | None) -> list[str]:
    files = sorted({str(path) for input_path in input_paths for path in input_path.rglob("*.parquet")})
    if not files:
        msg = f"No Parquet inputs found under {input_paths}"
        raise FileNotFoundError(msg)
    if input_file_limit is not None:
        if input_file_limit <= 0:
            msg = "input_file_limit must be positive"
            raise ValueError(msg)
        if len(files) < input_file_limit:
            msg = f"Requested {input_file_limit} input files, but only found {len(files)} under {input_paths}"
            raise ValueError(msg)
        files = files[:input_file_limit]
    return files


def _num_rows(task_metrics: dict[str, Any]) -> int:
    return int(sum(value for key, value in task_metrics.items() if key.endswith("custom.num_rows_sum")))


def _metadata_fields(input_files: list[str], id_field: str, embedding_field: str) -> list[str]:
    input_columns = pq.read_schema(input_files[0]).names
    required = {id_field, embedding_field}
    missing = required.difference(input_columns)
    if missing:
        msg = f"Input is missing required columns {sorted(missing)}: {input_files[0]}"
        raise ValueError(msg)
    return [column for column in input_columns if column not in required]


def _verify_output_metadata(output_path: Path, metadata_fields: list[str]) -> None:
    output_file = next(output_path.rglob("*.parquet"), None)
    if output_file is None:
        msg = f"KMeans did not write any Parquet files under {output_path}"
        raise RuntimeError(msg)
    output_columns = set(pq.read_schema(output_file).names)
    missing = set(metadata_fields).difference(output_columns)
    if missing:
        msg = f"KMeans output is missing metadata columns {sorted(missing)}: {output_file}"
        raise RuntimeError(msg)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_files = _input_files([Path(path) for path in args.input_path], args.input_file_limit)
    metadata_fields = _metadata_fields(input_files, args.id_field, args.embedding_field)
    output_path = Path(args.output_path)
    centroids_path = Path(args.centroids_path)
    output_path.mkdir(parents=True, exist_ok=False)
    centroids_path.mkdir(parents=True, exist_ok=False)

    available_cpus, available_gpus = get_available_cpu_gpu_resources(init_and_shutdown=True)
    cluster_resources = {"CPU": available_cpus, "GPU": available_gpus}
    num_actors = min(len(input_files), int(available_gpus))
    if num_actors < 1:
        msg = f"KMeans requires at least one Ray GPU resource; found {cluster_resources}"
        raise RuntimeError(msg)

    if args.fit_data_fraction is not None and not 0 < args.fit_data_fraction <= 1:
        msg = "fit_data_fraction must be in (0, 1]"
        raise ValueError(msg)
    effective_fit_data_fraction = None if args.fit_data_fraction == 1 else args.fit_data_fraction
    write_kwargs: dict[str, Any] = {}
    if args.compression != "default":
        write_kwargs["compression"] = None if args.compression == "none" else args.compression

    pipeline = Pipeline(
        name="semdedup_kmeans_benchmark",
        stages=[
            KMeansStage(
                n_clusters=args.n_clusters,
                id_field=args.id_field,
                embedding_field=args.embedding_field,
                metadata_fields=metadata_fields,
                input_path=input_files,
                output_path=str(output_path),
                embedding_dim=args.embedding_dim,
                input_filetype="parquet",
                max_iter=args.max_iter,
                tol=args.tol,
                random_state=args.random_state,
                init=args.init,
                oversampling_factor=args.oversampling_factor,
                max_samples_per_batch=args.max_samples_per_batch,
                fit_data_fraction=effective_fit_data_fraction,
                output_embedding_dtype=args.output_embedding_dtype,
                write_batch_size=args.write_batch_size,
                max_output_file_size=(
                    args.max_output_file_size_mb * 1_000_000 if args.max_output_file_size_mb is not None else None
                ),
                prefetch_next_group=args.prefetch_next_group,
                write_kwargs=write_kwargs,
                cache_path=str(centroids_path),
            )
        ],
    )

    started = time.perf_counter()
    tasks = pipeline.run(RayActorPoolExecutor())
    duration_s = time.perf_counter() - started
    _verify_output_metadata(output_path, metadata_fields)
    task_metrics = TaskPerfUtils.aggregate_task_metrics(tasks)
    rows_processed = _num_rows(task_metrics)
    output_files = list(output_path.rglob("*.parquet"))
    output_bytes = sum(path.stat().st_size for path in output_files)
    return {
        "params": {
            **vars(args),
            "effective_fit_data_fraction": effective_fit_data_fraction,
            "metadata_fields": metadata_fields,
            "input_file_count": len(input_files),
            "input_bytes": sum(Path(path).stat().st_size for path in input_files),
            "ray_cluster_resources": cluster_resources,
            "kmeans_actor_count": num_actors,
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
    parser.add_argument("--n-clusters", type=int, default=128)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--embedding-field", default="embeddings")
    parser.add_argument("--fit-data-fraction", type=float)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--init", choices=["k-means||", "random"], default="k-means||")
    parser.add_argument("--oversampling-factor", type=float, default=2.0)
    parser.add_argument("--max-samples-per-batch", type=int, default=32768)
    parser.add_argument("--output-embedding-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--write-batch-size", type=_write_batch_size, default="auto")
    parser.add_argument("--compression", choices=["default", "zstd", "snappy", "none"], default="default")
    parser.add_argument("--max-output-file-size-mb", type=int)
    parser.add_argument("--prefetch-next-group", action="store_true")
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
