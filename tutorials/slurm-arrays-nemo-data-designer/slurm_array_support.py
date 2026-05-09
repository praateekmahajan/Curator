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

"""Tutorial-local SLURM array helpers for Curator seeded-generation runs.

These classes intentionally live in the tutorial instead of Curator core. They
show the small amount of glue needed to run the same seed-data pipeline as a
SLURM array where every array element owns a deterministic subset of input file
groups and can be retried independently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MethodType
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from fsspec.core import url_to_fs
from loguru import logger

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import EmptyTask
from nemo_curator.tasks.utils import TaskPerfUtils

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

    from nemo_curator.backends.base import BaseExecutor
    from nemo_curator.tasks import DocumentBatch, FileGroupTask, Task, _EmptyTask

OutputLayout = Literal["flat", "by_shard"]
ReaderT = TypeVar("ReaderT", bound=CompositeStage)


@dataclass(frozen=True)
class SlurmArrayContext:
    """The logical shard handled by this process.

    ``CURATOR_SHARD_INDEX`` and ``CURATOR_NUM_SHARDS`` are preferred because
    they let retry jobs preserve the original shard count even when only a
    subset of SLURM array indices is resubmitted.
    """

    shard_index: int = 0
    num_shards: int = 1
    slurm_job_id: str | None = None
    slurm_array_job_id: str | None = None
    slurm_array_task_id: str | None = None

    @classmethod
    def from_env(
        cls,
        shard_index: int | None = None,
        num_shards: int | None = None,
    ) -> SlurmArrayContext:
        if shard_index is None:
            if os.environ.get("CURATOR_SHARD_INDEX") is not None:
                shard_index = int(os.environ["CURATOR_SHARD_INDEX"])
            else:
                raw_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
                shard_offset = int(os.environ.get("CURATOR_SHARD_OFFSET", "0"))
                shard_index = raw_task_id - shard_offset

        if num_shards is None:
            num_shards = _first_int_env(
                "CURATOR_NUM_SHARDS",
                "CURATOR_ORIGINAL_ARRAY_SIZE",
                "SLURM_ARRAY_TASK_COUNT",
            )
            if num_shards is None:
                num_shards = _count_from_slurm_min_max()
            if num_shards is None:
                num_shards = 1

        if num_shards < 1:
            msg = f"num_shards must be >= 1, got {num_shards}"
            raise ValueError(msg)
        if shard_index < 0 or shard_index >= num_shards:
            msg = (
                f"shard_index must be in [0, {num_shards}), got {shard_index}. "
                "Use a 0-based SLURM array or set CURATOR_SHARD_INDEX explicitly."
            )
            raise ValueError(msg)

        return cls(
            shard_index=shard_index,
            num_shards=num_shards,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
            slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        )


@dataclass
class SlurmArrayPartitionPlan:
    """Driver-side view of the file groups a shard still needs to process."""

    shard_index: int
    num_shards: int
    already_successful: bool = False
    total_file_groups: int = 0
    pending_tasks: list[FileGroupTask] = field(default_factory=list)
    completed_outputs: int = 0
    other_shard_file_groups: int = 0

    @property
    def pending_file_groups(self) -> int:
        return len(self.pending_tasks)

    def marker_payload(self) -> dict[str, Any]:
        return {
            "already_successful": self.already_successful,
            "total_file_groups": self.total_file_groups,
            "pending_file_groups": self.pending_file_groups,
            "completed_outputs": self.completed_outputs,
            "other_shard_file_groups": self.other_shard_file_groups,
        }


def _first_int_env(*names: str) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return None


def _count_from_slurm_min_max() -> int | None:
    task_min = os.environ.get("SLURM_ARRAY_TASK_MIN")
    task_max = os.environ.get("SLURM_ARRAY_TASK_MAX")
    if task_min is None or task_max is None:
        return None
    return int(task_max) - int(task_min) + 1


def _strip_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value


def data_output_path(output_root: str, layout: OutputLayout = "flat", shard_index: int | None = None) -> str:
    """Return the writer directory for an output root.

    ``flat`` is the default because Curator writer filenames are deterministic
    from source files and task IDs, so concurrent shards do not collide and
    downstream readers can consume one directory.
    """

    root = output_root.rstrip("/")
    if layout == "flat":
        return f"{root}/data"
    if layout == "by_shard":
        if shard_index is None:
            shard_index = SlurmArrayContext.from_env().shard_index
        return f"{root}/data/shard_{shard_index:05d}"
    msg = f"Unsupported output layout: {layout!r}"
    raise ValueError(msg)


def success_dir(output_root: str) -> str:
    return f"{output_root.rstrip('/')}/_SUCCESS"


def success_marker_path(output_root: str, shard_index: int) -> str:
    return f"{success_dir(output_root)}/shard_{shard_index:05d}.json"


def shard_for_source_files(source_files: list[str], num_shards: int) -> int:
    """Map a file group to a stable shard index."""

    combined = "|".join(sorted(source_files))
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def predicted_writer_output_path(  # noqa: PLR0913
    output_root: str,
    source_files: list[str],
    writer_task_id: str,
    output_file_extension: str,
    layout: OutputLayout = "flat",
    shard_index: int | None = None,
) -> str:
    extension = output_file_extension.lstrip(".")
    filename = writer_utils.get_deterministic_hash(source_files, writer_task_id)
    return f"{data_output_path(output_root, layout, shard_index).rstrip('/')}/{filename}.{extension}"


def done_marker_path(output_file_path: str) -> str:
    """Return the writer completion sidecar path for an output file."""

    return f"{output_file_path}.done"


def completed_shards(
    output_root: str,
    storage_options: dict[str, Any] | None = None,
) -> set[int]:
    """Return shard indices that already have success markers."""

    fs, fs_path = url_to_fs(success_dir(output_root), **(storage_options or {}))
    pattern = _fs_join(fs, fs_path, "shard_*.json")
    completed: set[int] = set()
    for marker in fs.glob(pattern):
        match = re.search(r"shard_(\d+)\.json$", str(marker))
        if match:
            completed.add(int(match.group(1)))
    return completed


def missing_shards(
    output_root: str,
    num_shards: int,
    storage_options: dict[str, Any] | None = None,
) -> list[int]:
    done = completed_shards(output_root, storage_options)
    return [idx for idx in range(num_shards) if idx not in done]


def has_success_marker(
    output_root: str,
    context: SlurmArrayContext | None = None,
    storage_options: dict[str, Any] | None = None,
) -> bool:
    context = context or SlurmArrayContext.from_env()
    return _exists(success_marker_path(output_root, context.shard_index), storage_options)


def write_success_marker(
    output_root: str,
    context: SlurmArrayContext | None = None,
    payload: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> str:
    """Write a per-shard success marker and return its path."""

    context = context or SlurmArrayContext.from_env()
    marker_path = success_marker_path(output_root, context.shard_index)
    marker_payload = {
        "status": "success",
        "finished_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - keep Python 3.9 compatible.
        "shard_index": context.shard_index,
        "num_shards": context.num_shards,
        "slurm_job_id": context.slurm_job_id,
        "slurm_array_job_id": context.slurm_array_job_id,
        "slurm_array_task_id": context.slurm_array_task_id,
    }
    if payload:
        marker_payload.update(payload)
    _write_json(marker_path, marker_payload, storage_options)
    return marker_path


def write_done_marker(
    output_file_path: str,
    *,
    payload: dict[str, Any] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> str:
    """Atomically write a completion sidecar for one output file."""

    marker_path = done_marker_path(output_file_path)
    marker_payload = {
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 - keep Python 3.9 compatible.
        "output_file": output_file_path,
    }
    if payload:
        marker_payload.update(payload)
    _write_json(marker_path, marker_payload, storage_options)
    return marker_path


def _exists(path: str, storage_options: dict[str, Any] | None = None) -> bool:
    fs, fs_path = url_to_fs(path, **(storage_options or {}))
    return fs.exists(fs_path)


def _size_bytes(path: str, storage_options: dict[str, Any] | None = None) -> int:
    fs, fs_path = url_to_fs(path, **(storage_options or {}))
    return int(fs.size(fs_path))


def _read_json(
    path: str,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fs, fs_path = url_to_fs(path, **(storage_options or {}))
    with fs.open(fs_path) as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return loaded


def _has_complete_output_marker(
    output_file_path: str,
    min_bytes: int,
    storage_options: dict[str, Any] | None = None,
) -> bool:
    marker_path = done_marker_path(output_file_path)
    if not _exists(marker_path, storage_options):
        return False
    try:
        marker_payload = _read_json(marker_path, storage_options)
        output_size = _size_bytes(output_file_path, storage_options)
    except (OSError, NotImplementedError, json.JSONDecodeError, TypeError) as exc:
        logger.warning(f"Could not validate completion marker {marker_path}: {exc}; treating output as incomplete")
        return False

    marker_size = int(marker_payload.get("size_bytes") or 0)
    required_size = max(min_bytes, marker_size)
    if output_size < required_size:
        logger.warning(
            f"Completion marker {marker_path} says {output_file_path} should have at least "
            f"{required_size} byte(s), but it has {output_size}; treating output as incomplete"
        )
        return False
    return True


def _write_json(
    path: str,
    payload: dict[str, Any],
    storage_options: dict[str, Any] | None = None,
) -> None:
    fs, fs_path = url_to_fs(path, **(storage_options or {}))
    parent = fs.sep.join(fs_path.split(fs.sep)[:-1])
    if parent:
        fs.makedirs(parent, exist_ok=True)

    tmp_path = f"{fs_path}.tmp.{uuid.uuid4().hex}"
    with fs.open(tmp_path, "w") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
        f.write("\n")
    if fs.exists(fs_path):
        fs.rm(fs_path)
    fs.rename(tmp_path, fs_path)


def _fs_join(fs: AbstractFileSystem, root: str, *children: str) -> str:
    path = root.rstrip(fs.sep)
    if not path:
        path = fs.sep
    for child in children:
        clean_child = str(child).strip(fs.sep)
        path = f"{path}{clean_child}" if path.endswith(fs.sep) else f"{path}{fs.sep}{clean_child}"
    return path


@dataclass
class SlurmArrayFilePartitioningStage(FilePartitioningStage):
    """File partitioning with SLURM array sharding and resumability.

    This inherits Curator's normal ``FilePartitioningStage`` so input file
    discovery, partition sizing, and source-file metadata stay identical. The
    tutorial-specific behavior happens after those file groups are created:

    1. assign each file group to one logical shard using a stable hash;
    2. keep only the groups owned by this array element;
    3. skip groups whose expected writer output file already exists;
    4. skip the whole shard when its success marker already exists.
    """

    output_path: str = ""
    output_file_extension: str = "jsonl"
    output_layout: OutputLayout = "flat"
    output_storage_options: dict[str, Any] | None = None
    shard_index: int | None = None
    num_shards: int | None = None
    skip_successful_shard: bool = True
    skip_existing_outputs: bool = True
    min_existing_output_bytes: int = 1
    writer_task_id_suffix: str = "_processed"
    name: str = "slurm_array_file_partitioning"
    context: SlurmArrayContext = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.output_path:
            msg = "output_path is required for SlurmArrayFilePartitioningStage"
            raise ValueError(msg)
        if self.output_layout not in ("flat", "by_shard"):
            msg = f"Unsupported output_layout: {self.output_layout!r}"
            raise ValueError(msg)
        self.output_file_extension = self.output_file_extension.lstrip(".")
        self.output_storage_options = self.output_storage_options or {}
        self.context = SlurmArrayContext.from_env(self.shard_index, self.num_shards)

    def process(self, task: _EmptyTask) -> list[FileGroupTask]:
        return self.plan(task).pending_tasks

    def plan(self, task: _EmptyTask | None = None) -> SlurmArrayPartitionPlan:
        if self.skip_successful_shard and has_success_marker(
            self.output_path,
            self.context,
            self.output_storage_options,
        ):
            logger.info(
                f"Shard {self.context.shard_index}/{self.context.num_shards} already has a success marker; "
                "skipping file partitioning"
            )
            return SlurmArrayPartitionPlan(
                shard_index=self.context.shard_index,
                num_shards=self.context.num_shards,
                already_successful=True,
            )

        all_tasks = super().process(task or EmptyTask)
        return self._build_plan(all_tasks)

    def _build_plan(self, all_tasks: list[FileGroupTask]) -> SlurmArrayPartitionPlan:
        owned_tasks: list[FileGroupTask] = []
        completed_outputs = 0
        skipped_other_shards = 0

        for file_task in all_tasks:
            source_files = list(file_task._metadata.get("source_files") or file_task.data)
            assigned_shard = shard_for_source_files(source_files, self.context.num_shards)
            file_task._metadata.update(
                {
                    "slurm_array_assigned_shard": assigned_shard,
                    "slurm_array_shard_index": self.context.shard_index,
                    "slurm_array_num_shards": self.context.num_shards,
                    "slurm_array_output_root": self.output_path,
                    "slurm_array_output_layout": self.output_layout,
                }
            )

            if assigned_shard != self.context.shard_index:
                skipped_other_shards += 1
                continue

            expected_output = self.expected_output_path(file_task)
            file_task._metadata["slurm_array_expected_output_file"] = expected_output
            file_task._metadata["slurm_array_expected_output_done_marker"] = done_marker_path(expected_output)
            if self.skip_existing_outputs and _has_complete_output_marker(
                expected_output,
                self.min_existing_output_bytes,
                self.output_storage_options,
            ):
                completed_outputs += 1
                continue

            owned_tasks.append(file_task)

        logger.info(
            f"Shard {self.context.shard_index}/{self.context.num_shards}: "
            f"{len(owned_tasks)} pending file group(s), "
            f"{completed_outputs} complete output(s), "
            f"{skipped_other_shards} assigned to other shard(s)"
        )
        return SlurmArrayPartitionPlan(
            shard_index=self.context.shard_index,
            num_shards=self.context.num_shards,
            total_file_groups=len(all_tasks),
            pending_tasks=owned_tasks,
            completed_outputs=completed_outputs,
            other_shard_file_groups=skipped_other_shards,
        )

    def expected_output_path(self, task: FileGroupTask) -> str:
        source_files = list(task._metadata.get("source_files") or task.data)
        writer_task_id = f"{task.task_id}{self.writer_task_id_suffix}"
        return predicted_writer_output_path(
            self.output_path,
            source_files,
            writer_task_id,
            self.output_file_extension,
            self.output_layout,
            self.context.shard_index,
        )


def monkey_patch_reader_partitioning(  # noqa: PLR0913
    reader: ReaderT,
    *,
    output_path: str,
    output_file_extension: str,
    output_layout: OutputLayout = "flat",
    output_storage_options: dict[str, Any] | None = None,
    shard_index: int | None = None,
    num_shards: int | None = None,
    skip_successful_shard: bool = True,
    skip_existing_outputs: bool = True,
    min_existing_output_bytes: int = 1,
    writer_task_id_suffix: str = "_processed",
) -> ReaderT:
    """Replace the first FilePartitioningStage in a reader instance.

    This is the tutorial monkey patch: ``JsonlReader`` and ``ParquetReader`` are
    already the right composite stages, so the example only swaps their first
    decomposed stage for the array-aware version.
    """

    original_decompose = reader.decompose

    def _decompose_with_slurm_partitioning(self: ReaderT) -> list[Any]:
        stages = original_decompose()
        for idx, stage in enumerate(stages):
            if isinstance(stage, FilePartitioningStage):
                stages[idx] = SlurmArrayFilePartitioningStage(
                    file_paths=stage.file_paths,
                    files_per_partition=stage.files_per_partition,
                    blocksize=stage.blocksize,
                    file_extensions=stage.file_extensions,
                    storage_options=stage.storage_options,
                    limit=stage.limit,
                    output_path=output_path,
                    output_file_extension=output_file_extension,
                    output_layout=output_layout,
                    output_storage_options=output_storage_options,
                    shard_index=shard_index,
                    num_shards=num_shards,
                    skip_successful_shard=skip_successful_shard,
                    skip_existing_outputs=skip_existing_outputs,
                    min_existing_output_bytes=min_existing_output_bytes,
                    writer_task_id_suffix=writer_task_id_suffix,
                )
                return stages
        msg = f"{self.__class__.__name__} did not decompose to a FilePartitioningStage"
        raise RuntimeError(msg)

    reader.decompose = MethodType(_decompose_with_slurm_partitioning, reader)
    return reader


def make_slurm_array_jsonl_reader(  # noqa: PLR0913
    *,
    file_paths: str | list[str],
    output_path: str,
    output_file_extension: str = "jsonl",
    output_layout: OutputLayout = "flat",
    files_per_partition: int | None = None,
    blocksize: int | str | None = None,
    fields: list[str] | None = None,
    read_kwargs: dict[str, Any] | None = None,
    **partition_kwargs: Any,  # noqa: ANN401
) -> JsonlReader:
    reader = JsonlReader(
        file_paths=file_paths,
        files_per_partition=files_per_partition,
        blocksize=blocksize,
        fields=fields,
        read_kwargs=read_kwargs,
    )
    return monkey_patch_reader_partitioning(
        reader,
        output_path=output_path,
        output_file_extension=output_file_extension,
        output_layout=output_layout,
        **partition_kwargs,
    )


def make_slurm_array_parquet_reader(  # noqa: PLR0913
    *,
    file_paths: str | list[str],
    output_path: str,
    output_file_extension: str = "parquet",
    output_layout: OutputLayout = "flat",
    files_per_partition: int | None = None,
    blocksize: int | str | None = None,
    fields: list[str] | None = None,
    read_kwargs: dict[str, Any] | None = None,
    **partition_kwargs: Any,  # noqa: ANN401
) -> ParquetReader:
    reader = ParquetReader(
        file_paths=file_paths,
        files_per_partition=files_per_partition,
        blocksize=blocksize,
        fields=fields,
        read_kwargs=read_kwargs,
    )
    return monkey_patch_reader_partitioning(
        reader,
        output_path=output_path,
        output_file_extension=output_file_extension,
        output_layout=output_layout,
        **partition_kwargs,
    )


class _CompletionMarkerWriterMixin:
    """Mixin for tutorial writers that create a done sidecar after a successful write."""

    def process(self, task: DocumentBatch) -> FileGroupTask:
        result = super().process(task)  # type: ignore[misc]
        output_file = result.data[0]
        size_bytes = _size_bytes(output_file, self.storage_options)
        marker_path = write_done_marker(
            output_file,
            payload={
                "size_bytes": size_bytes,
                "num_records": task.num_items,
                "task_id": task.task_id,
                "source_files": list(task._metadata.get("source_files") or []),
            },
            storage_options=self.storage_options,
        )
        result._metadata["completion_marker"] = marker_path
        result._metadata["output_size_bytes"] = size_bytes
        return result


@dataclass
class SlurmArrayJsonlWriter(_CompletionMarkerWriterMixin, JsonlWriter):
    """JSONL writer that writes ``<output>.done`` sidecars for resumability."""

    name: str = "slurm_array_jsonl_writer"


@dataclass
class SlurmArrayParquetWriter(_CompletionMarkerWriterMixin, ParquetWriter):
    """Parquet writer that writes ``<output>.done`` sidecars for resumability."""

    name: str = "slurm_array_parquet_writer"


class SlurmArrayPipeline(Pipeline):
    """Pipeline that writes one success marker per logical SLURM array shard."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        output_path: str,
        description: str | None = None,
        stages: list[Any] | None = None,
        config: dict[str, Any] | None = None,
        success_storage_options: dict[str, Any] | None = None,
        success_on_empty: bool = True,
        success_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name=name, description=description, stages=stages, config=config)
        self.output_path = output_path
        self.success_storage_options = success_storage_options or {}
        self.success_on_empty = success_on_empty
        self.success_payload = success_payload or {}

    def update_success_payload(self, payload: dict[str, Any]) -> None:
        self.success_payload.update(payload)

    def run(self, executor: BaseExecutor | None = None, initial_tasks: list[Task] | None = None) -> list[Task] | None:
        context = SlurmArrayContext.from_env()
        if has_success_marker(self.output_path, context, self.success_storage_options):
            logger.info(
                f"Shard {context.shard_index}/{context.num_shards} already has "
                f"{success_marker_path(self.output_path, context.shard_index)}; skipping pipeline"
            )
            return []

        pipeline_started_at = time.perf_counter()
        results = super().run(executor=executor, initial_tasks=initial_tasks)
        pipeline_run_s = time.perf_counter() - pipeline_started_at
        num_results = len(results or [])
        if num_results > 0 or self.success_on_empty:
            success_payload = dict(self.success_payload)
            upstream_metrics = dict(success_payload.pop("metrics", {}))
            output_files = [path for task in results or [] for path in getattr(task, "data", [])]
            output_bytes = 0
            for output_file in output_files:
                try:
                    output_bytes += _size_bytes(output_file, self.success_storage_options)
                except (OSError, NotImplementedError):
                    logger.warning(f"Could not get size for output file {output_file}")

            metrics = {
                **upstream_metrics,
                "pipeline_run_s": pipeline_run_s,
                "num_output_tasks": num_results,
                "num_output_files": len(output_files),
                "output_bytes": output_bytes,
                **TaskPerfUtils.aggregate_task_metrics(results),
            }
            marker_path = write_success_marker(
                self.output_path,
                context,
                payload={
                    "pipeline_name": self.name,
                    "num_output_tasks": num_results,
                    "output_files": output_files,
                    **success_payload,
                    "metrics": metrics,
                },
                storage_options=self.success_storage_options,
            )
            logger.info(f"Wrote success marker {marker_path}")
        return results


__all__ = [
    "SlurmArrayContext",
    "SlurmArrayFilePartitioningStage",
    "SlurmArrayJsonlWriter",
    "SlurmArrayParquetWriter",
    "SlurmArrayPartitionPlan",
    "SlurmArrayPipeline",
    "completed_shards",
    "data_output_path",
    "done_marker_path",
    "has_success_marker",
    "make_slurm_array_jsonl_reader",
    "make_slurm_array_parquet_reader",
    "missing_shards",
    "monkey_patch_reader_partitioning",
    "predicted_writer_output_path",
    "shard_for_source_files",
    "success_dir",
    "success_marker_path",
    "write_done_marker",
    "write_success_marker",
]
