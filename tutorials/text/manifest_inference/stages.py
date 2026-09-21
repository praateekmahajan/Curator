# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest source and bounded JSONL reader; no database or full-file scan."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, EmptyTask, FileGroupTask

from .manifest import read_manifest, record_id


@dataclass
class ManifestTask(FileGroupTask):
    def get_deterministic_id(self) -> str:
        return record_id(self._metadata["manifest"])


@dataclass
class ManifestFilePartitioningStage(ProcessingStage[EmptyTask, ManifestTask]):
    manifest: str
    input_dir: str
    name: str = "manifest_partitioning"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, _task: EmptyTask) -> list[ManifestTask]:
        root = Path(self.input_dir).resolve()
        tasks = []
        for record in read_manifest(Path(self.manifest)):
            path = (root / record["input_file"]).resolve()
            if not path.is_relative_to(root):
                msg = "Input escapes input root"
                raise ValueError(msg)
            tasks.append(
                ManifestTask(
                    dataset_name="manifest_inference",
                    data=[str(path)],
                    _metadata={"manifest": record, "source_files": [str(path)]},
                )
            )
        if not tasks:
            msg = "Empty manifest"
            raise ValueError(msg)
        return tasks


@dataclass
class SpecificJsonlReader(ProcessingStage[ManifestTask, DocumentBatch]):
    name: str = "specific_jsonl_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: ManifestTask) -> DocumentBatch:
        record = task._metadata["manifest"]
        path = Path(task.data[0])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size_bytes"], record["mtime_ns"]):
            msg = f"Input changed since manifest generation: {path}"
            raise ValueError(msg)
        rows = []
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            stream.seek(record["start_byte"])
            for _ in range(record["num_rows"]):
                remaining = record["end_byte"] - stream.tell()
                if remaining <= 0:
                    msg = "Manifest range ended before expected row count"
                    raise ValueError(msg)
                raw = stream.readline(remaining)
                digest.update(raw)
                row = json.loads(raw)
                if not isinstance(row, dict):
                    msg = "Each JSONL line must be an object"
                    raise TypeError(msg)
                rows.append(row)
            if stream.tell() != record["end_byte"] or digest.hexdigest() != record["sha256"]:
                msg = "Input range does not match manifest checksum"
                raise ValueError(msg)
        columns = dict.fromkeys(key for row in rows for key in row)
        table = pa.Table.from_pydict({key: [row.get(key) for row in rows] for key in columns})
        return DocumentBatch(
            dataset_name=task.dataset_name, data=table, _metadata=task._metadata, _stage_perf=task._stage_perf
        )
