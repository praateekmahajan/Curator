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
import math
import time
from pathlib import Path
from typing import Any

import ray
from loguru import logger

from benchmarking.scripts.utils import write_benchmark_results
from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage
from nemo_curator.tasks.utils import TaskPerfUtils


def estimate_kmeans_memory(  # noqa: PLR0913
    *,
    num_rows: int,
    embedding_dim: int,
    n_clusters: int,
    num_actors: int,
    fit_data_fraction: float | None,
    oversampling_factor: float,
    max_samples_per_batch: int,
    safety_factor: float,
) -> dict[str, int | float]:
    """Return a transparent lower-bound estimate for RAFT KMeans GPU memory.

    The sampled embeddings are converted to FP32 and briefly exist both as
    per-group arrays and as one concatenated array. RAFT's distance batch,
    centroid candidates, and final centroids are included separately. The
    safety factor is intentionally explicit because allocator and RAFT
    workspaces are implementation-dependent.
    """
    if num_rows <= 0 or embedding_dim <= 0 or n_clusters <= 0 or num_actors <= 0:
        msg = "num_rows, embedding_dim, n_clusters, and num_actors must be positive"
        raise ValueError(msg)
    if fit_data_fraction is not None and not 0 < fit_data_fraction < 1:
        msg = "fit_data_fraction must be in (0, 1), or None"
        raise ValueError(msg)
    if safety_factor < 1:
        msg = "safety_factor must be at least 1"
        raise ValueError(msg)

    fit_rows = num_rows if fit_data_fraction is None else max(1, math.ceil(num_rows * fit_data_fraction))
    fp32_bytes = 4
    fit_arrays_cluster_bytes = fit_rows * embedding_dim * fp32_bytes * 2
    fit_arrays_per_actor_bytes = math.ceil(fit_arrays_cluster_bytes / num_actors)
    distance_batch_per_actor_bytes = max_samples_per_batch * n_clusters * fp32_bytes
    candidate_centroids_per_actor_bytes = math.ceil(oversampling_factor * n_clusters * 8 * embedding_dim * fp32_bytes)
    centroids_per_actor_bytes = n_clusters * embedding_dim * fp32_bytes
    lower_bound_per_actor_bytes = (
        fit_arrays_per_actor_bytes
        + distance_batch_per_actor_bytes
        + candidate_centroids_per_actor_bytes
        + centroids_per_actor_bytes
    )

    return {
        "input_embeddings_fp16_bytes": num_rows * embedding_dim * 2,
        "fit_rows": fit_rows,
        "fit_arrays_cluster_bytes": fit_arrays_cluster_bytes,
        "fit_arrays_per_actor_bytes": fit_arrays_per_actor_bytes,
        "distance_batch_per_actor_bytes": distance_batch_per_actor_bytes,
        "candidate_centroids_per_actor_bytes": candidate_centroids_per_actor_bytes,
        "centroids_per_actor_bytes": centroids_per_actor_bytes,
        "lower_bound_per_actor_bytes": lower_bound_per_actor_bytes,
        "projected_per_actor_bytes": math.ceil(lower_bound_per_actor_bytes * safety_factor),
        "projected_cluster_bytes": math.ceil(lower_bound_per_actor_bytes * safety_factor) * num_actors,
        "safety_factor": safety_factor,
    }


def _input_files(input_path: Path, input_file_limit: int | None) -> list[str]:
    files = sorted(str(path) for path in input_path.rglob("*.parquet"))
    if not files:
        msg = f"No Parquet inputs found under {input_path}"
        raise FileNotFoundError(msg)
    if input_file_limit is not None:
        if input_file_limit <= 0:
            msg = "input_file_limit must be positive"
            raise ValueError(msg)
        files = files[:input_file_limit]
    return files


def _num_rows(task_metrics: dict[str, Any]) -> int:
    return int(sum(value for key, value in task_metrics.items() if key.endswith("custom.num_rows_sum")))


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_files = _input_files(Path(args.input_path), args.input_file_limit)
    output_path = Path(args.output_path)
    centroids_path = Path(args.centroids_path)
    output_path.mkdir(parents=True, exist_ok=False)
    centroids_path.mkdir(parents=True, exist_ok=False)

    cluster_resources = ray.cluster_resources()
    num_actors = min(len(input_files), int(cluster_resources.get("GPU", 0)))
    if num_actors < 1:
        msg = f"KMeans requires at least one Ray GPU resource; found {cluster_resources}"
        raise RuntimeError(msg)

    memory_estimate = None
    if args.expected_rows is not None:
        memory_estimate = estimate_kmeans_memory(
            num_rows=args.expected_rows,
            embedding_dim=args.embedding_dim,
            n_clusters=args.n_clusters,
            num_actors=num_actors,
            fit_data_fraction=args.fit_data_fraction,
            oversampling_factor=args.oversampling_factor,
            max_samples_per_batch=args.max_samples_per_batch,
            safety_factor=args.memory_safety_factor,
        )
        logger.info(f"Pre-run KMeans memory estimate: {memory_estimate}")

    pipeline = Pipeline(
        name="semdedup_kmeans_benchmark",
        stages=[
            KMeansStage(
                n_clusters=args.n_clusters,
                id_field=args.id_field,
                embedding_field=args.embedding_field,
                input_path=input_files,
                output_path=str(output_path),
                embedding_dim=args.embedding_dim,
                input_filetype="parquet",
                max_iter=args.max_iter,
                tol=args.tol,
                random_state=args.random_state,
                oversampling_factor=args.oversampling_factor,
                max_samples_per_batch=args.max_samples_per_batch,
                fit_data_fraction=args.fit_data_fraction,
                output_embedding_dtype=args.output_embedding_dtype,
                cache_path=str(centroids_path),
            )
        ],
    )

    started = time.perf_counter()
    tasks = pipeline.run(RayActorPoolExecutor())
    duration_s = time.perf_counter() - started
    task_metrics = TaskPerfUtils.aggregate_task_metrics(tasks)
    rows_processed = _num_rows(task_metrics)
    return {
        "params": {
            **vars(args),
            "input_file_count": len(input_files),
            "input_bytes": sum(Path(path).stat().st_size for path in input_files),
            "ray_cluster_resources": cluster_resources,
            "kmeans_actor_count": num_actors,
            "memory_estimate": memory_estimate,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": duration_s,
            "num_documents_processed": rows_processed,
            "throughput_docs_per_sec": rows_processed / duration_s if duration_s else None,
            **task_metrics,
        },
        "tasks": tasks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--centroids-path", required=True)
    parser.add_argument("--input-file-limit", type=int)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--embedding-field", default="embeddings")
    parser.add_argument("--fit-data-fraction", type=float)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--oversampling-factor", type=float, default=2.0)
    parser.add_argument("--max-samples-per-batch", type=int, default=32768)
    parser.add_argument("--output-embedding-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--memory-safety-factor", type=float, default=1.5)
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
