# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SLURM-array support for Curator pipelines.

Three pieces, independent of any particular pipeline:

* :class:`Shard` — namespace of static helpers: env detection, paths,
  per-shard ``_SUCCESS/shard_*.json`` marker read/write, and ``completed`` /
  ``missing`` for status/retry tools. Driver-side code should use
  :meth:`Shard.has_marker` to short-circuit before bringing up Ray.
* :class:`SlurmArrayFilePartitioningStage` — ``FilePartitioningStage``
  subclass that keeps only this shard's file groups and skips groups whose
  deterministic writer output already exists. Runs as a normal pipeline
  stage — all filesystem work happens on the worker, not the driver.
* :class:`SlurmArrayPipeline` — ``Pipeline`` subclass that short-circuits
  on an existing marker and writes the marker on clean completion (even
  when the shard produced zero tasks).

Env contract: ``CURATOR_SHARD_INDEX`` and ``CURATOR_NUM_SHARDS`` are
preferred over ``SLURM_ARRAY_*`` so sparse retry submissions
(``--array=3,5,9``) keep the original shard count.
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.io.writer.utils import get_deterministic_hash
from nemo_curator.tasks import FileGroupTask, Task, _EmptyTask
from nemo_curator.utils.file_utils import get_fs

# The text reader's BaseReader appends this to task_id when emitting DocumentBatches;
# the writer then hashes (source_files, task_id_with_suffix) to derive its filename.
# We replicate the same suffix on the partitioner side so the resume check matches.
_WRITER_TASK_ID_SUFFIX = "_processed"


# ---------------------------------------------------------------------------
# Shard — env detection, paths, success markers
# ---------------------------------------------------------------------------


class Shard:
    """Static helpers for the current shard. No state, no instance."""

    @staticmethod
    def env() -> tuple[int, int]:
        """Return ``(shard_index, num_shards)`` from env vars."""
        idx = int(os.environ.get("CURATOR_SHARD_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID") or 0)
        n = 1
        for var in ("CURATOR_NUM_SHARDS", "CURATOR_ORIGINAL_ARRAY_SIZE", "SLURM_ARRAY_TASK_COUNT"):
            if os.environ.get(var):
                n = int(os.environ[var])
                break
        if n < 1 or not (0 <= idx < n):
            msg = f"shard_index={idx} out of range for num_shards={n}"
            raise ValueError(msg)
        return idx, n

    @staticmethod
    def data_path(output_root: str, layout: str = "flat", shard_index: int | None = None) -> str:
        """Writer directory. ``flat`` shares ``<root>/data/``; ``by_shard`` uses ``<root>/data/shard_<i>/``."""
        root = output_root.rstrip("/")
        if layout == "flat":
            return f"{root}/data"
        if layout == "by_shard":
            idx = shard_index if shard_index is not None else Shard.env()[0]
            return f"{root}/data/shard_{idx:05d}"
        msg = f"Unsupported layout: {layout!r}"
        raise ValueError(msg)

    @staticmethod
    def marker_path(output_root: str, shard_index: int) -> str:
        return f"{output_root.rstrip('/')}/_SUCCESS/shard_{shard_index:05d}.json"

    @staticmethod
    def has_marker(output_root: str, shard_index: int | None = None) -> bool:
        idx = shard_index if shard_index is not None else Shard.env()[0]
        path = Shard.marker_path(output_root, idx)
        return get_fs(path).exists(path)

    @staticmethod
    def completed(output_root: str) -> set[int]:
        success_dir = f"{output_root.rstrip('/')}/_SUCCESS"
        fs = get_fs(success_dir)
        if not fs.exists(success_dir):
            return set()
        return {
            int(m.group(1))
            for p in fs.glob(f"{success_dir}/shard_*.json")
            if (m := re.search(r"shard_(\d+)\.json$", str(p)))
        }

    @staticmethod
    def missing(output_root: str, num_shards: int) -> list[int]:
        done = Shard.completed(output_root)
        return [i for i in range(num_shards) if i not in done]

    @staticmethod
    def write_marker(
        output_root: str,
        shard_index: int,
        num_shards: int,
        payload: dict[str, Any] | None = None,
    ) -> str:
        path = Shard.marker_path(output_root, shard_index)
        fs = get_fs(path)
        fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        body = {
            "status": "success",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            **(payload or {}),
        }
        with fs.open(path, "w") as f:
            f.write(json.dumps(body, sort_keys=True, indent=2) + "\n")
        return path


# ---------------------------------------------------------------------------
# Shard- and resume-aware partitioner
# ---------------------------------------------------------------------------


@dataclass
class SlurmArrayFilePartitioningStage(FilePartitioningStage):
    """File partitioner that filters parent-class output through two checks:

    1. ``hash(file_group) % num_shards == shard_index`` — this shard's slice
    2. deterministic output for the group is not already on disk (resume)

    Whole-shard short-circuit on an existing ``_SUCCESS`` marker lives on the
    *driver* (see :meth:`Shard.has_marker`) so we don't need lustre-aware
    permissions or filesystem walks on whatever host launches the pipeline.
    """

    output_path: str = ""
    output_file_extension: str = "jsonl"
    output_layout: str = "flat"  # "flat" or "by_shard"
    shard_index: int | None = None
    num_shards: int | None = None
    skip_existing_outputs: bool = True
    name: str = "slurm_array_file_partitioning"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.output_path:
            msg = "output_path is required"
            raise ValueError(msg)
        self.output_file_extension = self.output_file_extension.lstrip(".")
        if self.output_layout not in ("flat", "by_shard"):
            msg = f"Unsupported output_layout: {self.output_layout!r}"
            raise ValueError(msg)
        env_idx, env_n = Shard.env()
        if self.shard_index is None:
            self.shard_index = env_idx
        if self.num_shards is None:
            self.num_shards = env_n

    def process(self, task: _EmptyTask) -> list[FileGroupTask]:
        all_tasks = super().process(task)
        out_dir = Shard.data_path(self.output_path, self.output_layout, self.shard_index).rstrip("/")
        ext = self.output_file_extension
        fs = get_fs(out_dir)  # one fs object reused across this-shard existence checks

        pending: list[FileGroupTask] = []
        completed = 0
        other_shards = 0
        for ft in all_tasks:
            source_files = list(ft._metadata.get("source_files") or ft.data)
            digest = hashlib.sha256("|".join(sorted(source_files)).encode("utf-8")).hexdigest()
            assigned = int(digest[:16], 16) % self.num_shards
            ft._metadata["slurm_array_assigned_shard"] = assigned
            if assigned != self.shard_index:
                other_shards += 1
                continue
            writer_task_id = f"{ft.task_id}{_WRITER_TASK_ID_SUFFIX}"
            expected = f"{out_dir}/{get_deterministic_hash(source_files, writer_task_id)}.{ext}"
            if self.skip_existing_outputs and fs.exists(expected):
                completed += 1
                continue
            pending.append(ft)

        logger.info(
            f"Shard {self.shard_index}/{self.num_shards}: {len(pending)} pending, "
            f"{completed} already complete, {other_shards} for other shards"
        )
        return pending


# ---------------------------------------------------------------------------
# Reader monkey-patch
# ---------------------------------------------------------------------------


def enable_slurm_array_partitioning(
    reader: CompositeStage,
    *,
    output_path: str,
    output_file_extension: str,
    output_layout: str = "flat",
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> CompositeStage:
    """Replace ``reader.decompose()``'s first FilePartitioningStage with the shard-aware variant.

    Works on any reader composite that decomposes to ``[FilePartitioningStage, …]``
    (JsonlReader, ParquetReader, …). Mutates the reader in-place; returns it
    for fluent use.
    """
    original = reader.decompose

    def decompose() -> list[Any]:
        stages = original()
        for i, stage in enumerate(stages):
            if isinstance(stage, FilePartitioningStage):
                stages[i] = SlurmArrayFilePartitioningStage(
                    file_paths=stage.file_paths,
                    files_per_partition=stage.files_per_partition,
                    blocksize=stage.blocksize,
                    file_extensions=stage.file_extensions,
                    storage_options=stage.storage_options,
                    limit=stage.limit,
                    output_path=output_path,
                    output_file_extension=output_file_extension,
                    output_layout=output_layout,
                    shard_index=shard_index,
                    num_shards=num_shards,
                )
                return stages
        msg = f"{type(reader).__name__} did not decompose to a FilePartitioningStage"
        raise RuntimeError(msg)

    reader.decompose = decompose
    return reader


# ---------------------------------------------------------------------------
# Pipeline subclass — short-circuit + minimal success marker
# ---------------------------------------------------------------------------


class SlurmArrayPipeline(Pipeline):
    """``Pipeline`` that writes one success marker per logical SLURM shard.

    The marker is written even when zero tasks were produced (empty shard)
    so ``status`` / ``retry-missing`` work correctly. Set
    ``success_on_empty=False`` to opt out.

    Pass ``success_payload`` to embed caller-specific metadata (timings,
    served models, etc.) into the marker — the pipeline itself only adds
    the basics (pipeline name, output file list, pipeline run time).
    """

    def __init__(
        self,
        name: str,
        output_path: str,
        description: str | None = None,
        stages: list[Any] | None = None,
        config: dict[str, Any] | None = None,
        success_on_empty: bool = True,
        success_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, stages=stages, config=config)
        self.output_path = output_path
        self.success_on_empty = success_on_empty
        self.success_payload = success_payload or {}

    def update_success_payload(self, payload: dict[str, Any]) -> None:
        self.success_payload.update(payload)

    def run(
        self,
        executor: BaseExecutor | None = None,
        initial_tasks: list[Task] | None = None,
    ) -> list[Task] | None:
        idx, n = Shard.env()
        if Shard.has_marker(self.output_path, idx):
            logger.info(f"Shard {idx}/{n} already has {Shard.marker_path(self.output_path, idx)}; skipping")
            return []

        t0 = time.perf_counter()
        results = super().run(executor=executor, initial_tasks=initial_tasks)
        if not results and not self.success_on_empty:
            return results

        output_files = [p for t in results or [] for p in getattr(t, "data", [])]
        marker = Shard.write_marker(
            self.output_path, idx, n,
            payload={
                "pipeline_name": self.name,
                "pipeline_run_s": round(time.perf_counter() - t0, 2),
                "num_output_tasks": len(results or []),
                "output_files": output_files,
                **self.success_payload,
            },
        )
        logger.info(f"Wrote success marker {marker}")
        return results


__all__ = [
    "Shard",
    "SlurmArrayFilePartitioningStage",
    "SlurmArrayPipeline",
    "enable_slurm_array_partitioning",
]
