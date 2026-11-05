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

"""Run each generic GPU curation phase after benchmark_semantic_large.py.

Run --phase prune, filter, fine, diversity, sample, repeat in that order. Each phase
has an independent completion manifest; existing output is never overwritten.
The real corpus's scalar score is used directly, without domain-specific labels.
"""

import argparse
import gc
import json
import random
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import pynvml
from benchmark_semantic_large import record_gpu_memory

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.diversity import ClusterDiversityStage
from nemo_curator.stages.deduplication.semantic.filter import SemanticDuplicateFilterStage
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage
from nemo_curator.stages.deduplication.semantic.pairwise import PairwiseStage
from nemo_curator.stages.deduplication.semantic.pairwise_io import ClusterWiseFilePartitioningStage
from nemo_curator.stages.deduplication.semantic.sampling import (
    QualityDiversitySamplingStage,
    RepeatParquetStage,
    plan_sampling_budgets,
)
from nemo_curator.stages.deduplication.semantic.utils import read_parquet_chunks
from nemo_curator.stages.deduplication.semantic.weighted import WeightedPruningConfig
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.tasks import FileGroupTask


class CapacityCheckedRepeatStage(RepeatParquetStage):
    """Keep headroom on the benchmark's shared output filesystem."""

    def process(self, task: FileGroupTask) -> FileGroupTask:
        if shutil.disk_usage(Path(self.output_path).parent).free < 32 * 1024**3:
            msg = "Repeat output stopped with less than 32 GiB free; partial outputs are retained"
            raise OSError(msg)
        return super().process(task)


def probe_repeat_storage(root: Path, files: list[str]) -> dict:
    """Estimate expansion from actual counts/text, before launching all workers.

    This is a capacity estimate, not a replacement for the full scale run.
    Completed probes can be reused after freeing disk space without rewriting them.
    """
    report = root / "repeat_storage_probe.json"
    if report.exists():
        result = json.loads(report.read_text())
    else:
        output = root / "repeat_storage_probe"
        if output.exists():
            raise FileExistsError(output)

        def text_bytes(path: str) -> int:
            metadata = pq.read_metadata(path)
            columns = [i for i in range(metadata.num_columns) if metadata.schema.column(i).path == "text"]
            return sum(
                metadata.row_group(row).column(column).total_compressed_size
                for row in range(metadata.num_row_groups)
                for column in columns
            )

        with ThreadPoolExecutor(max_workers=8) as readers:
            sizes = dict(zip(files, readers.map(text_bytes, files), strict=True))
        selected = random.Random(42).sample(files, min(16, len(files)))  # noqa: S311
        stage = CapacityCheckedRepeatStage(
            str(output), counts_path=str(root / "sample_counts"), columns=["id", "text"], input_batch_rows=8192
        )
        stage.setup()
        tasks = stage.process(FileGroupTask(data=selected, dataset_name="repeat_storage_probe"))
        written = sum(Path(path).stat().st_size for path in tasks.data)
        selected_bytes = sum(sizes[path] for path in selected)
        result = {
            "source_text_compressed_bytes": sum(sizes.values()),
            "probe_files": selected,
            "probe_input_text_compressed_bytes": selected_bytes,
            "probe_output_bytes": written,
            "projected_output_bytes": round(written / selected_bytes * sum(sizes.values())),
        }
        report.write_text(json.dumps(result, indent=2))
        del stage
        gc.collect()
    free = shutil.disk_usage(root).free
    if result["projected_output_bytes"] * 1.5 + 32 * 1024**3 > free:
        msg = (
            "Repeat capacity probe needs 1.5x estimated output plus 32 GiB headroom; "
            f"free={free}, projected_output_bytes={result['projected_output_bytes']}; see {report}"
        )
        raise OSError(msg)
    return result


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--phase", choices=["prune", "filter", "fine", "diversity", "sample", "repeat"], required=True)
    parser.add_argument("--fine-clusters", type=int, default=3000)
    parser.add_argument("--shards", type=int, default=128)
    parser.add_argument("--target-epochs", type=float, default=2.5)
    parser.add_argument("--survivors-path", help="Optional local/shared output path for the large survivor vectors")
    args = parser.parse_args()
    root = Path(args.run_root)
    directories = {
        "prune": "resident_pairwise",
        "filter": "survivors",
        "fine": "fine",
        "diversity": "diversity",
        "sample": "sample_counts",
        "repeat": "sampled",
    }
    output = root / directories[args.phase]
    if args.phase == "filter" and args.survivors_path:
        output = Path(args.survivors_path)
    if output.exists():
        raise FileExistsError(output)
    if args.phase == "prune":
        metrics = json.loads((root / "metrics.json").read_text())
        if metrics["status"] != "coarse_complete_downstream_pending":
            msg = "Baseline clustering/pruning must complete before the resident comparison"
            raise RuntimeError(msg)
    else:
        previous = list(directories)[list(directories).index(args.phase) - 1]
        if json.loads((root / f"{previous}.json").read_text())["status"] != "complete":
            msg = f"Previous phase {previous} did not complete"
            raise RuntimeError(msg)
    pipeline = Pipeline(name=f"generic_{args.phase}")
    if args.phase == "prune":
        pipeline.add_stage(
            PairwiseStage(
                id_field="id",
                embedding_field="embedding",
                input_path=str(root / "coarse/kmeans_results"),
                output_path=str(output),
                weighted_pruning=WeightedPruningConfig(),
                compute_dtype="float32",
            )
        )
    elif args.phase == "filter":
        pipeline.add_stage(ClusterWiseFilePartitioningStage(str(root / "coarse/kmeans_results")))
        pipeline.add_stage(
            SemanticDuplicateFilterStage(str(root / "resident_pairwise"), str(output), ["id", "embedding", "score"])
        )
    elif args.phase == "fine":
        survivor_path = json.loads((root / "filter.json").read_text())["output_path"]
        pipeline.add_stage(
            KMeansStage(
                n_clusters=args.fine_clusters,
                input_path=survivor_path,
                output_path=str(output),
                id_field="id",
                embedding_field="embedding",
                metadata_fields=["score"],
                cache_path=str(root / "centroids"),
                write_embeddings=False,
                spherical=True,
                init="random",
                fit_data_fraction=1.0,
            )
        )
    elif args.phase == "diversity":
        pipeline.add_stage(ClusterWiseFilePartitioningStage(str(root / "fine")))
        pipeline.add_stage(
            ClusterDiversityStage(
                str(root / "centroids/kmeans_centroids.npy"),
                str(output),
                ["score"],
                group_fields=(),
                sampling_shards=args.shards,
            )
        )
    elif args.phase == "sample":
        groups = json.loads((root / "diversity.json").read_text())["group_counts"]
        counts = {("all", int(shard)): count for shard, count in groups}
        allocated = plan_sampling_budgets(counts, {"all": sum(counts.values())}, args.target_epochs)
        budgets = {(shard,): target for (_, shard), target in allocated.items()}
        pipeline.add_stage(
            FilePartitioningStage(
                file_paths=str(root / "diversity"), files_per_partition=1, file_extensions=[".parquet"]
            )
        )
        pipeline.add_stage(
            QualityDiversitySamplingStage(
                budgets,
                str(output),
                group_fields=("sampling_shard",),
                quality_field="score",
                total_nparts=args.shards,
                rmm_pool_size=None,
            )
        )
    else:
        files = json.loads((root / "input_files.json").read_text())
        pipeline.add_stage(
            FilePartitioningStage(file_paths=files, files_per_partition=1, file_extensions=[".parquet"])
        )
        pipeline.add_stage(
            CapacityCheckedRepeatStage(
                str(output),
                counts_path=str(root / "sample_counts"),
                columns=["id", "text"],
                input_batch_rows=8192,
            )
        )
    client = RayClient(num_gpus=4, num_cpus=24, object_store_memory=8 * 1024**3, include_dashboard=False)
    result = {"phase": args.phase, "status": "running", "arguments": vars(args), "output_path": str(output)}
    report = root / f"{args.phase}.json"
    report.write_text(json.dumps(result, indent=2))
    pynvml.nvmlInit()
    peaks = [0] * pynvml.nvmlDeviceGetCount()
    stopped = threading.Event()
    monitor = threading.Thread(
        target=record_gpu_memory,
        args=(root / f"{args.phase}_gpu_memory.jsonl", stopped, peaks),
        daemon=True,
    )
    monitor.start()
    start = time.perf_counter()
    try:
        if args.phase == "diversity":
            import cudf

            cluster = next(path for path in (root / "fine").iterdir() if path.is_dir())
            probe_files = sorted(str(path) for path in cluster.glob("*.parquet"))[:64]
            columns = ["id", "score", "cosine_dist_to_cent"]
            expected = cudf.concat(list(read_parquet_chunks(probe_files, columns, 256 * 1024**2)), ignore_index=True)
            chunks = list(read_parquet_chunks(probe_files, columns, 256 * 1024**2, files_per_read=64))
            actual = cudf.concat(chunks, ignore_index=True)
            if not actual.equals(expected):
                msg = "Batched reader changed real-cluster data or row order"
                raise AssertionError(msg)  # noqa: TRY301
            result["reader_parity"] = {"files": len(probe_files), "rows": len(actual), "batched_chunks": len(chunks)}
            del expected, actual, chunks
        if args.phase == "repeat":
            result["storage_probe"] = probe_repeat_storage(root, files)
        client.start()
        pipeline_start = time.perf_counter()
        tasks = pipeline.run(XennaExecutor() if args.phase == "prune" else RayActorPoolExecutor())
        result["pipeline_seconds"] = time.perf_counter() - pipeline_start
        (root / f"{args.phase}_task_performance.json").write_text(
            json.dumps(
                [
                    {"task_id": task.task_id, "stages": [stage.to_dict() for stage in task._stage_perf]}
                    for task in tasks
                ],
                indent=2,
            )
        )
        if args.phase == "diversity":
            counts = Counter()
            for task in tasks:
                counts.update(task._metadata["sampling_group_counts"])
            result["group_counts"] = sorted((key[0], count) for key, count in counts.items())
        with ThreadPoolExecutor(max_workers=16) as readers:
            result["output_rows"] = sum(
                readers.map(lambda path: pq.read_metadata(path).num_rows, output.rglob("*.parquet"))
            )
        if args.phase in {"filter", "fine", "diversity"}:
            if args.phase == "filter":
                expected = sum(task._metadata["kept_rows"] for task in tasks)
            else:
                expected = json.loads((root / f"{previous}.json").read_text())["output_rows"]
            if result["output_rows"] != expected:
                msg = f"{args.phase} wrote {result['output_rows']} rows; expected {expected}"
                raise AssertionError(msg)  # noqa: TRY301
        if args.phase == "prune":
            import cudf

            baseline = sorted((root / "coarse/pairwise_results").glob("cluster_*.parquet"))
            actual = {path.name for path in output.glob("cluster_*.parquet")}
            if actual != {path.name for path in baseline}:
                msg = "Resident pruning did not produce exactly the baseline cluster masks"
                raise AssertionError(msg)  # noqa: TRY301
            for path in baseline:
                expected_mask = cudf.read_parquet(path).sort_values("id").reset_index(drop=True)
                actual_mask = cudf.read_parquet(output / path.name).sort_values("id").reset_index(drop=True)
                if not actual_mask.equals(expected_mask):
                    msg = f"Resident pruning differs from baseline for {path.name}"
                    raise AssertionError(msg)  # noqa: TRY301
            result["exact_mask_parity_clusters"] = len(baseline)
        if args.phase == "sample":
            histogram = Counter()
            for frame in read_parquet_chunks(
                sorted(str(path) for path in output.rglob("*.parquet")), ["total_copies"], 256 * 1024**2
            ):
                copies = frame["total_copies"]
                if copies.isnull().any() or copies.dtype.kind not in "iu" or not copies.between(1, 8).all():
                    msg = "Written sample counts must be non-null integers in [1, 8]"
                    raise AssertionError(msg)  # noqa: TRY301
                histogram.update(copies.value_counts().to_pandas().to_dict())
            result["copy_histogram"] = dict(sorted(histogram.items()))
            result["total_copies"] = sum(copies * count for copies, count in histogram.items())
            expected = sum(budgets.values())
            if result["total_copies"] != expected:
                msg = f"Written counts allocate {result['total_copies']} copies; expected {expected}"
                raise AssertionError(msg)  # noqa: TRY301
        if args.phase == "repeat":
            groups = json.loads((root / "diversity.json").read_text())["group_counts"]
            sample_args = json.loads((root / "sample.json").read_text())["arguments"]
            expected = round(sum(count for _, count in groups) * sample_args["target_epochs"])
            if result["output_rows"] != expected:
                msg = f"Repeated output {result['output_rows']} does not match exact corpus budget {expected}"
                raise AssertionError(msg)  # noqa: TRY301
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stopped.set()
        monitor.join(timeout=5)
        result["peak_gpu_bytes"] = peaks
        result["seconds"] = time.perf_counter() - start
        report.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        client.stop()


if __name__ == "__main__":
    main()
