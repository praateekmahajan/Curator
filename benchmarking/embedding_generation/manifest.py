# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from loguru import logger

from nemo_curator.backends.slurm_array import SlurmArrayConfig, resolve_slurm_array_config
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, FileGroupTask
from nemo_curator.utils.hash_utils import get_deterministic_hash


@dataclass
class ManifestFileGroupTask(FileGroupTask):
    """One physical JSONL file with a stable logical source identity."""

    logical_path: str = ""

    def get_deterministic_id(self) -> str:
        return get_deterministic_hash([self.logical_path])


@dataclass(frozen=True)
class ManifestShardPlan:
    total_files: int
    total_rows: int
    total_shards: int


def _load_manifest_record(raw_line: str, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as error:
        msg = f"Invalid JSON on manifest line {line_number}: {error}"
        raise ValueError(msg) from error

    if not isinstance(record, dict):
        msg = f"Manifest line {line_number} must contain a JSON object"
        raise TypeError(msg)
    return record


def _iter_manifest_records(manifest_path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read either the inventory JSON object or the earlier JSONL representation."""
    path = Path(manifest_path)
    try:
        with path.open(encoding="utf-8") as manifest:
            payload = json.load(manifest)
    except json.JSONDecodeError as error:
        if error.msg != "Extra data":
            msg = f"Invalid JSON manifest {path}: {error}"
            raise ValueError(msg) from error
        with path.open(encoding="utf-8") as manifest:
            for line_number, raw_line in enumerate(manifest, start=1):
                if raw_line.strip():
                    yield line_number, _load_manifest_record(raw_line, line_number)
        return

    if isinstance(payload, dict):
        records = payload.get("files")
    elif isinstance(payload, list):
        records = payload
    else:
        records = None
    if not isinstance(records, list):
        msg = f"JSON manifest {path} must contain a files list"
        raise TypeError(msg)
    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            msg = f"Manifest record {record_number} must contain a JSON object"
            raise TypeError(msg)
        yield record_number, record


def _validate_manifest_record(record: dict[str, Any], line_number: int, expected_index: int) -> None:
    required = {"index", "path", "logical_path", "num_rows"}
    missing = sorted(required - record.keys())
    if missing:
        msg = f"Manifest line {line_number} is missing required fields: {missing}"
        raise ValueError(msg)
    if record["index"] != expected_index:
        msg = f"Manifest line {line_number} has index={record['index']!r}; expected contiguous index {expected_index}"
        raise ValueError(msg)
    if not isinstance(record["path"], str) or not Path(record["path"]).is_absolute():
        msg = f"Manifest line {line_number} path must be an absolute string"
        raise ValueError(msg)
    if not isinstance(record["logical_path"], str) or not record["logical_path"].startswith("/"):
        msg = f"Manifest line {line_number} logical_path must be an absolute string"
        raise ValueError(msg)
    if isinstance(record["num_rows"], bool) or not isinstance(record["num_rows"], int):
        msg = f"Manifest line {line_number} num_rows must be an integer"
        raise TypeError(msg)
    if record["num_rows"] <= 0:
        msg = f"Manifest line {line_number} num_rows must be positive"
        raise ValueError(msg)
    if record.get("error") is not None:
        msg = f"Manifest line {line_number} contains an inventory error: {record['error']}"
        raise ValueError(msg)


def plan_manifest_shards(manifest_path: str | Path, target_rows_per_shard: int) -> ManifestShardPlan:
    """Validate a manifest and calculate the logical shard count."""
    if target_rows_per_shard <= 0:
        msg = "target_rows_per_shard must be positive"
        raise ValueError(msg)

    total_files = 0
    total_rows = 0

    for line_number, record in _iter_manifest_records(manifest_path):
        _validate_manifest_record(record, line_number, line_number - 1)
        if Path(record["path"]).suffix != ".jsonl":
            continue
        num_rows = record["num_rows"]
        if num_rows > target_rows_per_shard:
            msg = (
                f"Manifest index {record['index']} contains {num_rows:,} rows, exceeding "
                f"the {target_rows_per_shard:,}-row whole-file shard limit"
            )
            raise ValueError(msg)
        total_rows += num_rows
        total_files += 1

    if total_files == 0:
        msg = f"Manifest contains no records: {manifest_path}"
        raise ValueError(msg)
    return ManifestShardPlan(
        total_files=total_files,
        total_rows=total_rows,
        total_shards=math.ceil(total_rows / target_rows_per_shard),
    )


def _validate_enriched_fields(record: dict[str, Any], line_number: int) -> None:
    required = {"dedup_path", "id_start", "id_end"}
    missing = sorted(required - record.keys())
    if missing:
        msg = f"Enriched manifest line {line_number} is missing required fields: {missing}"
        raise ValueError(msg)
    if not isinstance(record["dedup_path"], str) or not Path(record["dedup_path"]).is_absolute():
        msg = f"Enriched manifest line {line_number} dedup_path must be an absolute string"
        raise ValueError(msg)
    for field in ("id_start", "id_end"):
        if isinstance(record[field], bool) or not isinstance(record[field], int):
            msg = f"Enriched manifest line {line_number} {field} must be an integer"
            raise TypeError(msg)
    expected_end = record["id_start"] + record["num_rows"] - 1
    if record["id_end"] != expected_end:
        msg = (
            f"Enriched manifest line {line_number} has id range "
            f"[{record['id_start']}, {record['id_end']}], expected end {expected_end}"
        )
        raise ValueError(msg)


def _dedup_path_for_logical_path(logical_path: str, path_mapping: dict[str, str]) -> str | None:
    mappings = sorted(path_mapping.items(), key=lambda item: len(item[1]), reverse=True)
    for dedup_prefix, registry_prefix in mappings:
        normalized_prefix = registry_prefix.rstrip("/")
        if logical_path == normalized_prefix or logical_path.startswith(f"{normalized_prefix}/"):
            return f"{dedup_prefix.rstrip('/')}{logical_path[len(normalized_prefix) :]}"
    return None


def _uses_explicit_shards(records: list[dict[str, Any]]) -> bool:
    flags = ["shard_index" in record for record in records]
    if any(flags) and not all(flags):
        msg = "Either every manifest record must define shard_index or none may define it"
        raise ValueError(msg)
    return all(flags)


def _resolve_record_shard(
    record: dict[str, Any],
    prefix_rows: int,
    total_rows: int,
    slurm_array: SlurmArrayConfig,
    use_explicit_shards: bool,
) -> tuple[int, int]:
    if not use_explicit_shards:
        shard_offset = min(
            prefix_rows * slurm_array.total_shards // total_rows,
            slurm_array.total_shards - 1,
        )
        return slurm_array.minimum_shard_index + shard_offset, shard_offset

    shard_index = record["shard_index"]
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        msg = f"Manifest index {record['index']} shard_index must be an integer"
        raise TypeError(msg)
    shard_offset = shard_index - slurm_array.minimum_shard_index
    if not 0 <= shard_offset < slurm_array.total_shards:
        msg = (
            f"Manifest index {record['index']} has shard_index={shard_index}; expected "
            f"[{slurm_array.minimum_shard_index}, "
            f"{slurm_array.minimum_shard_index + slurm_array.total_shards - 1}]"
        )
        raise ValueError(msg)
    return shard_index, shard_offset


@dataclass
class ManifestFilePartitioningStage(ProcessingStage[EmptyTask, ManifestFileGroupTask]):
    """Emit FPP=1 tasks for the active row-balanced Slurm array shard."""

    manifest_path: str
    path_mapping: dict[str, str]
    required_minimum_files_per_shard: int = 1
    manifest_max_rows: int | None = None
    name: str = "manifest_file_partitioning"
    is_source_stage: bool = True
    is_slurm_array_prepartitioned: bool = True

    def __post_init__(self) -> None:
        if self.required_minimum_files_per_shard <= 0:
            msg = "required_minimum_files_per_shard must be positive"
            raise ValueError(msg)
        if self.manifest_max_rows is not None and self.manifest_max_rows <= 0:
            msg = "manifest_max_rows must be positive when set"
            raise ValueError(msg)
        self.resources = Resources(cpus=0.5)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def num_workers(self) -> int:
        return 1

    def process(self, _: EmptyTask) -> list[ManifestFileGroupTask]:
        slurm_array = resolve_slurm_array_config(is_source_stage=True)
        if slurm_array is None:
            msg = "ManifestFilePartitioningStage requires a Slurm array configuration"
            raise RuntimeError(msg)

        selected_records: list[dict[str, Any]] = []
        total_rows = 0

        for line_number, record in _iter_manifest_records(self.manifest_path):
            _validate_manifest_record(record, line_number, line_number - 1)
            if Path(record["path"]).suffix != ".jsonl":
                continue
            logical_path = record["logical_path"]
            dedup_path = record.get("dedup_path") or _dedup_path_for_logical_path(logical_path, self.path_mapping)
            if not isinstance(dedup_path, str):
                msg = f"Manifest index {record['index']} is not covered by the ID path mapping"
                raise TypeError(msg)
            selected_records.append({**record, "dedup_path": dedup_path})
            total_rows += record["num_rows"]
            if self.manifest_max_rows is not None and total_rows >= self.manifest_max_rows:
                break

        if not selected_records:
            msg = f"Manifest contains no JSONL records: {self.manifest_path}"
            raise ValueError(msg)

        tasks: list[ManifestFileGroupTask] = []
        shard_file_counts = [0] * slurm_array.total_shards
        shard_row_counts = [0] * slurm_array.total_shards
        use_explicit_shards = _uses_explicit_shards(selected_records)
        prefix_rows = 0
        for record in selected_records:
            shard_index, shard_offset = _resolve_record_shard(
                record,
                prefix_rows,
                total_rows,
                slurm_array,
                use_explicit_shards,
            )
            shard_file_counts[shard_offset] += 1
            shard_row_counts[shard_offset] += record["num_rows"]

            if shard_index == slurm_array.shard_index:
                logical_path = record["logical_path"]
                dedup_path = record["dedup_path"]
                tasks.append(
                    ManifestFileGroupTask(
                        dataset_name="fuzzy_deduped_data",
                        data=[dedup_path],
                        logical_path=logical_path,
                        reader_config={},
                        _metadata={
                            "manifest_index": record["index"],
                            "manifest_num_rows": record["num_rows"],
                            "logical_path": logical_path,
                            "source_files": [dedup_path],
                            "inventory_source_file": record["path"],
                            "source_root": record.get("source_root"),
                            "mapping_names": record.get("mapping_names", []),
                        },
                    )
                )
            prefix_rows += record["num_rows"]

        undersized_shards = [
            slurm_array.minimum_shard_index + offset
            for offset, file_count in enumerate(shard_file_counts)
            if file_count < self.required_minimum_files_per_shard
        ]
        if undersized_shards:
            msg = (
                f"Logical shards {undersized_shards[:10]} have fewer than "
                f"{self.required_minimum_files_per_shard} JSONL files; "
                "choose fewer total shards"
            )
            raise ValueError(msg)

        logger.info(
            "Manifest shard {} selected {} files and {:,} rows from {} files and {:,} total rows",
            slurm_array.shard_index,
            len(tasks),
            shard_row_counts[slurm_array.shard_index - slurm_array.minimum_shard_index],
            len(selected_records),
            total_rows,
        )
        return tasks
