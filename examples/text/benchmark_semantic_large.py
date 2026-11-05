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

"""Large real-vector benchmark; keep original text in source files until late join.

This entry runs coarse spherical fitting, weighted pruning and duplicate IDs.
Its persisted manifest and outputs feed downstream scale-validation phases.
No text or math-quality fields are synthesized here. All selected rows are fit.
"""

import argparse
import json
import random
import resource
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pynvml

from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.semantic.weighted import WeightedPruningConfig
from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow


def record_gpu_memory(path: Path, stopped: threading.Event, peaks: list[int]) -> None:
    """Record visible-device memory once per second until the caller stops it."""
    devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(len(peaks))]
    with path.open("w") as stream:
        while not stopped.wait(1):
            used = [pynvml.nvmlDeviceGetMemoryInfo(device).used for device in devices]
            for i, value in enumerate(used):
                peaks[i] = max(peaks[i], value)
            stream.write(json.dumps({"unix_time": time.time(), "used_bytes": used}) + "\n")
            stream.flush()


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--target-rows", type=int, default=220_000_000)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=False)
    files = sorted(Path(args.input).rglob("*.parquet"))
    random.Random(42).shuffle(files)  # noqa: S311 -- reproducible file selection, not security
    with ThreadPoolExecutor(16) as pool:
        sizes = list(pool.map(lambda p: pq.read_metadata(p).num_rows, files))
    selected, rows = [], 0
    for file, size in zip(files, sizes, strict=True):
        selected.append(str(file))
        rows += size
        if rows >= args.target_rows:
            break
    if rows < args.target_rows:
        msg = f"Requested {args.target_rows} rows, found only {rows}"
        raise ValueError(msg)
    selected.sort()
    (root / "original_input_files.json").write_text(json.dumps(selected, indent=2))
    raw_bytes = rows * 384 * 4
    # Retain both coarse and survivor vectors plus metadata/output headroom.
    if shutil.disk_usage(root).free < raw_bytes * 3:
        msg = "Insufficient result-filesystem space for coarse/survivor vectors and output headroom"
        raise OSError(msg)
    metrics = {
        "input_rows": rows,
        "files": len(selected),
        "raw_embedding_bytes": raw_bytes,
        "embedding_dimensions": 384,
        "fit_data_fraction": 1.0,
        "coarse_clusters": 500,
        "spherical": True,
        "status": "checking_ids",
        "text_policy": "original source text retained for late join; no synthetic replacement",
    }
    pynvml.nvmlInit()
    peaks = [0] * pynvml.nvmlDeviceGetCount()
    stopped = threading.Event()

    def record() -> None:
        metrics["peak_gpu_bytes"] = peaks.copy()
        metrics["driver_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)

    thread = threading.Thread(target=record_gpu_memory, args=(root / "gpu_memory.jsonl", stopped, peaks), daemon=True)
    thread.start()
    record()
    client = RayClient(
        num_gpus=4,
        num_cpus=24,
        object_store_memory=8 * 1024**3,
        ray_temp_dir=str(scratch / "ray"),
        include_dashboard=False,
    )
    try:
        start = time.perf_counter()
        with duckdb.connect() as db:
            db.execute("SET memory_limit='8GB'")
            db.execute("SET threads=8")
            db.execute("SET enable_progress_bar=false")
            db.execute("SET temp_directory=?", [str(scratch / "id_check")])
            db.from_parquet(selected, filename=True).create_view("source")
            invalid = db.execute(
                "SELECT id,count(*),min(filename),max(filename) FROM source GROUP BY id HAVING id IS NULL OR count(*)<>1"
            ).fetchall()
            repairs = []
            if invalid:
                # Preserve the source schema: cuDF batches require matching Parquet schemas.
                # Only copy affected files; never edit original payloads.
                affected = sorted({path for _, _, first, last in invalid for path in (first, last)})
                bad_ids = {row[0] for row in invalid}
                repaired_root = root / "identity_repair"
                repaired_root.mkdir()
                for file_index, path in enumerate(affected):
                    table = pq.read_table(path)
                    ids = table["id"].to_pylist()
                    for row_index, original_id in enumerate(ids):
                        if original_id in bad_ids:
                            ids[row_index] = f"curator-scale:{file_index}:{row_index}"
                            repairs.append(
                                {
                                    "source_file": path,
                                    "source_row": row_index,
                                    "original_id": original_id,
                                    "id": ids[row_index],
                                }
                            )
                    table = table.set_column(table.schema.get_field_index("id"), "id", pa.array(ids))
                    destination = str(repaired_root / f"part-{file_index}.parquet")
                    pq.write_table(table, destination, compression="zstd")
                    selected[selected.index(path)] = destination
                if len(repairs) != sum(row[1] for row in invalid):
                    msg = "Duplicate IDs span additional files; identity repair must be extended before fitting"
                    raise ValueError(msg)  # noqa: TRY301
                db.from_parquet(selected).create_view("repaired")
                remaining = db.execute(
                    "SELECT count(*) FROM (SELECT id FROM repaired GROUP BY id HAVING id IS NULL OR count(*)<>1)"
                ).fetchone()[0]
                if remaining:
                    msg = "Identity repair did not establish globally unique non-null IDs"
                    raise ValueError(msg)  # noqa: TRY301
            (root / "identity_repairs.json").write_text(json.dumps(repairs, indent=2))
            (root / "input_files.json").write_text(json.dumps(selected, indent=2))
            metrics["identity_repairs"] = len(repairs)
        metrics["id_check_seconds"] = time.perf_counter() - start
        metrics["status"] = "coarse_and_weighted_running"
        record()
        client.start()
        start = time.perf_counter()
        SemanticDeduplicationWorkflow(
            input_path=selected,
            output_path=str(root / "coarse"),
            n_clusters=500,
            id_field="id",
            embedding_field="embedding",
            metadata_fields=["score"],
            fit_data_fraction=1.0,
            spherical=True,
            init="random",
            max_iter=300,
            weighted_pruning=WeightedPruningConfig(spill_embeddings=True, scratch_path=str(scratch)),
            pairwise_compute_dtype="float32",
            kmeans_embedding_output_dtype="float32",
        ).run()
        metrics["coarse_and_weighted_seconds"] = time.perf_counter() - start
        metrics["status"] = "coarse_complete_downstream_pending"
        record()
    except Exception as error:
        metrics.update(status="failed", error=f"{type(error).__name__}: {error}")
        record()
        raise
    finally:
        stopped.set()
        thread.join(timeout=5)
        client.stop()


if __name__ == "__main__":
    main()
