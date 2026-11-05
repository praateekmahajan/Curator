# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-embedding scale check for reclustering, diversity and spillable sampling.

Start a four-GPU Ray head and set RAY_ADDRESS=auto. FineWeb score is a synthetic
stand-in for all three math scores: this benchmark makes no quality claim.
One sampling group deliberately stresses skew and external ranking.
"""

import argparse
import json
import resource
import threading
import time
from contextlib import suppress
from pathlib import Path

import duckdb
import pynvml

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.diversity import ClusterDiversityStage
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage
from nemo_curator.stages.deduplication.semantic.pairwise_io import ClusterWiseFilePartitioningStage
from nemo_curator.stages.deduplication.semantic.sampling_disk import sample_fusion_disk


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--sampling-memory", default="512MB")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--fit-fraction", type=float)
    parser.add_argument("--sampling-only", action="store_true")
    parser.add_argument("--compare-gpu", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--input-is-sampling-table",
        action="store_true",
        help="Reuse a prepared single-group FineWeb sampling table; skip clustering and diversity.",
    )
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=args.sampling_only)
    metrics = {"input": args.input, "clusters": args.clusters, "sampling_memory": args.sampling_memory}
    if args.sampling_only:
        metrics.update(json.loads((root / "metrics.json").read_text()))
        metrics["sampling_memory"] = args.sampling_memory
    input_path = args.input
    if args.max_files is not None:
        input_path = [str(p) for p in sorted(Path(args.input).rglob("*.parquet"))[: args.max_files]]
        (root / "input_files.json").write_text(json.dumps(input_path))
    metrics["fit_fraction"] = args.fit_fraction
    metrics["write_embeddings"] = not args.metadata_only
    pynvml.nvmlInit()
    devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    peaks = metrics.get("peak_gpu_bytes", [0] * len(devices))
    stopped = threading.Event()

    def monitor() -> None:
        while not stopped.wait(1):
            for i, device in enumerate(devices):
                peaks[i] = max(peaks[i], pynvml.nvmlDeviceGetMemoryInfo(device).used)
            spill = 0
            for path in (root / "scratch").glob("**/spill/*"):
                with suppress(FileNotFoundError):
                    spill += path.stat().st_size
            metrics["peak_spill_bytes"] = max(metrics.get("peak_spill_bytes", 0), spill)

    threading.Thread(target=monitor, daemon=True).start()

    def record(stage: str, start: float) -> None:
        metrics[stage + "_seconds"] = time.perf_counter() - start
        metrics["peak_gpu_bytes"] = peaks.copy()
        metrics["driver_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)

    if not args.sampling_only and not args.input_is_sampling_table:
        executor = RayActorPoolExecutor()
        start = time.perf_counter()
        pipeline = Pipeline(name="scale_fine_kmeans")
        pipeline.add_stage(
            KMeansStage(
                input_path=input_path,
                output_path=str(root / "fine"),
                id_field="id",
                embedding_field="embedding",
                metadata_fields=["score"],
                n_clusters=args.clusters,
                max_iter=30,
                fit_data_fraction=args.fit_fraction,
                cache_path=str(root / "centroids"),
                write_embeddings=not args.metadata_only,
            )
        )
        pipeline.run(executor)
        metrics["kmeans_output_bytes"] = sum(p.stat().st_size for p in (root / "fine").rglob("*.parquet"))
        record("kmeans", start)
        start = time.perf_counter()
        pipeline = Pipeline(name="scale_diversity")
        pipeline.add_stage(ClusterWiseFilePartitioningStage(str(root / "fine")))
        pipeline.add_stage(
            ClusterDiversityStage(
                centroids_path=str(root / "centroids/kmeans_centroids.npy"),
                output_path=str(root / "diversity"),
                metadata_fields=["score"],
                read_memory_bytes=32 * 1024**2,
            )
        )
        pipeline.run(executor)
        record("diversity", start)
    start = time.perf_counter()
    prepared = Path(args.input) if args.input_is_sampling_table else root / "sampling_input.parquet"
    with duckdb.connect() as db:
        db.execute("SET threads=8")
        if not args.input_is_sampling_table:
            db.execute(
                """COPY (SELECT id, 'fineweb' AS math_topic, 0 AS sampling_shard,
                   least(5,greatest(0,score)) AS ms, least(5,greatest(0,score)) AS rd,
                   least(5,greatest(0,score)) AS cr, diversity_score
                   FROM read_parquet($source)) TO $destination (FORMAT PARQUET, ROW_GROUP_SIZE 65536)""",
                {"source": str(root / "diversity/*.parquet"), "destination": str(prepared)},
            )
        count = db.execute("SELECT count(*) FROM read_parquet(?)", [str(prepared)]).fetchone()[0]
    metrics["rows"] = count
    record("sampling_input", start)
    start = time.perf_counter()
    files = sample_fusion_disk(
        [str(prepared)],
        str(root / "counts"),
        {("fineweb", 0): round(count * 2.5)},
        scratch_path=str(root / "scratch"),
        memory_limit=args.sampling_memory,
        threads=4,
    )
    record("sampling", start)
    with duckdb.connect() as db:
        copies, maximum, unique_rows = db.execute(
            "SELECT sum(total_copies), max(total_copies), count(*) FROM read_parquet(?)", [files]
        ).fetchone()
    if copies != round(count * 2.5) or maximum > 8:  # noqa: PLR2004
        msg = "Sampling violated the exact copy budget or repeat cap"
        raise AssertionError(msg)
    metrics.update(copies=copies, max_repeat=maximum, selected_unique=unique_rows)
    if args.compare_gpu:
        import cudf

        from nemo_curator.stages.deduplication.semantic.sampling import _quality_score, fusion_copies

        start = time.perf_counter()
        frame = cudf.read_parquet(str(prepared)).sort_values("id").reset_index(drop=True)
        frame["total_copies"] = fusion_copies(frame, round(count * 2.5))
        frame["q"] = _quality_score(frame[["ms", "rd", "cr"]].to_cupy(dtype="float64"), (0.45, 0.30, 0.25))
        frame = frame[frame.total_copies > 0]
        frame.to_parquet(str(root / "gpu_counts.parquet"), index=False)
        record("gpu_sampling", start)
        actual = cudf.read_parquet(files).sort_values("id").reset_index(drop=True)
        if not frame[["id", "total_copies"]].reset_index(drop=True).equals(actual[["id", "total_copies"]]):
            msg = "GPU and spill sampling counts differ"
            raise AssertionError(msg)
        metrics["gpu_disk_counts_equal"] = True
        if not frame["q"].reset_index(drop=True).equals(actual["q"]):
            msg = "GPU and spill quality scores differ"
            raise AssertionError(msg)
        metrics["gpu_disk_quality_equal"] = True
    stopped.set()
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
