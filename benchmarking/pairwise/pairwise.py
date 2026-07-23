"""Benchmark Curator Pairwise on one existing KMeans centroid partition."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarking.scripts.utils import write_benchmark_results
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.pairwise import PairwiseCosineSimilarityStage
from nemo_curator.stages.deduplication.semantic.ranking import RankingStrategy
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.utils import TaskPerfUtils

RANKING_COLUMNS = ["source_family_id", "quality_rank", "recency_rank"]
RANKING_ASCENDING = [False, False, False, True]


def _pairwise_batch_size(value: str) -> int | str:
    if value == "auto":
        return value
    batch_size = int(value)
    if batch_size <= 0:
        msg = "pairwise batch size must be positive"
        raise argparse.ArgumentTypeError(msg)
    return batch_size


def _input_files(input_path: Path, centroid_id: int) -> list[str]:
    partition = input_path / f"centroid={centroid_id}"
    files = sorted(str(path) for path in partition.rglob("*.parquet"))
    if not files:
        msg = f"No Parquet files found under {partition}"
        raise FileNotFoundError(msg)
    return files


def _validate_schema(input_file: str, id_field: str, embedding_field: str) -> None:
    columns = set(pq.read_schema(input_file).names)
    required = {id_field, embedding_field, *RANKING_COLUMNS}
    missing = required.difference(columns)
    if missing:
        msg = f"Input is missing required columns {sorted(missing)}: {input_file}"
        raise ValueError(msg)


def _phase_metrics(task_metrics: dict[str, Any]) -> dict[str, Any]:
    marker = "custom.pairwise_"
    return {
        key.split(marker, 1)[1].removesuffix("_mean"): value
        for key, value in task_metrics.items()
        if marker in key and key.endswith("_mean")
    }


def _compare_reference(output_file: Path, reference_file: Path) -> dict[str, Any]:
    columns = ["id", "max_id", "cosine_sim_score"]
    output = pq.read_table(output_file, columns=columns)
    reference = pq.read_table(reference_file, columns=columns)
    if output.num_rows != reference.num_rows:
        msg = f"Reference rows {reference.num_rows} != output rows {output.num_rows}"
        raise RuntimeError(msg)

    id_matches = output["id"].equals(reference["id"])
    max_id_matches = output["max_id"].equals(reference["max_id"])
    output_scores = output["cosine_sim_score"].to_numpy(zero_copy_only=False)
    reference_scores = reference["cosine_sim_score"].to_numpy(zero_copy_only=False)
    score_max_abs_diff = float(np.max(np.abs(output_scores - reference_scores), initial=0.0))
    scores_match = np.allclose(output_scores, reference_scores, rtol=1e-6, atol=1e-6)
    if not id_matches or not max_id_matches or not scores_match:
        msg = (
            "Pairwise output differs from reference: "
            f"{id_matches=} {max_id_matches=} {scores_match=} {score_max_abs_diff=}"
        )
        raise RuntimeError(msg)
    return {
        "reference_id_matches": id_matches,
        "reference_max_id_matches": max_id_matches,
        "reference_score_max_abs_diff": score_max_abs_diff,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_path)
    input_files = _input_files(input_path, args.centroid_id)
    _validate_schema(input_files[0], args.id_field, args.embedding_field)
    expected_rows = sum(pq.read_metadata(path).num_rows for path in input_files)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    ranking = RankingStrategy.metadata_based(
        metadata_cols=["source_family_id", "quality_rank", "recency_rank", args.id_field],
        ascending=RANKING_ASCENDING,
    )
    stage = PairwiseCosineSimilarityStage(
        id_field=args.id_field,
        embedding_field=args.embedding_field,
        output_path=str(output_path),
        ranking_strategy=ranking,
        pairwise_batch_size=args.pairwise_batch_size,
        embedding_dim=args.embedding_dim,
        input_embedding_dtype=args.input_embedding_dtype,
        num_additional_neighbors=args.num_additional_neighbors,
        profile=True,
        write_kwargs={"compression": None},
    )
    task = FileGroupTask(
        dataset_name=f"centroid-{args.centroid_id}",
        data=input_files,
        _metadata={"centroid_id": args.centroid_id, "filetype": "parquet"},
    )
    pipeline = Pipeline(name="semdedup_pairwise_largest_cluster", stages=[stage])

    started = time.perf_counter()
    tasks = pipeline.run(RayDataExecutor(), initial_tasks=[task])
    duration_s = time.perf_counter() - started

    output_file = output_path / f"cluster_{args.centroid_id}.parquet"
    if not output_file.exists():
        msg = f"Pairwise output was not written: {output_file}"
        raise RuntimeError(msg)
    output_rows = pq.read_metadata(output_file).num_rows
    if output_rows != expected_rows:
        msg = f"Pairwise output rows {output_rows} != input rows {expected_rows}"
        raise RuntimeError(msg)
    reference_metrics = (
        _compare_reference(output_file, Path(args.reference_output_path)) if args.reference_output_path else {}
    )
    neighbor_metrics = {}
    if args.num_additional_neighbors:
        neighbor_file = output_path / f"cluster_{args.centroid_id}_neighbors.parquet"
        if not neighbor_file.exists():
            msg = f"Additional-neighbor output was not written: {neighbor_file}"
            raise RuntimeError(msg)
        neighbor_schema = set(pq.read_schema(neighbor_file).names)
        required_neighbor_columns = {
            "id",
            "other_id",
            "other_cosine_sim_score",
            "other_neighbor_rank",
            *(f"other_{column}" for column in RANKING_COLUMNS),
        }
        if missing := required_neighbor_columns.difference(neighbor_schema):
            msg = f"Additional-neighbor output is missing columns: {sorted(missing)}"
            raise RuntimeError(msg)
        neighbor_metrics = {
            "additional_neighbor_rows": pq.read_metadata(neighbor_file).num_rows,
            "additional_neighbor_output_bytes": neighbor_file.stat().st_size,
        }

    task_metrics = TaskPerfUtils.aggregate_task_metrics(tasks)
    return {
        "params": {
            **vars(args),
            "ranking_columns": ranking.metadata_cols,
            "ranking_ascending": ranking.ascending,
            "input_file_count": len(input_files),
            "input_bytes": sum(Path(path).stat().st_size for path in input_files),
            "expected_rows": expected_rows,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": duration_s,
            "num_documents_processed": expected_rows,
            "throughput_docs_per_sec": expected_rows / duration_s,
            "output_bytes": output_file.stat().st_size,
            **reference_metrics,
            **neighbor_metrics,
            **_phase_metrics(task_metrics),
        },
        "tasks": tasks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--centroid-id", type=int, required=True)
    parser.add_argument("--id-field", default="_curator_dedup_id")
    parser.add_argument("--embedding-field", default="embeddings")
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--input-embedding-dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--pairwise-batch-size", type=_pairwise_batch_size, default=1024)
    parser.add_argument("--num-additional-neighbors", type=int, choices=range(6), default=0)
    parser.add_argument("--reference-output-path")
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
