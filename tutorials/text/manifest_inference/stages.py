# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest source and bounded JSONL reader; no database or full-file scan."""

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.json as paj

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.io.reader.base import ReaderOutput
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.tasks import EmptyTask, FileGroupTask

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
class SpecificJsonlReader(JsonlReaderStage):
    """Read a manifest byte range using the standard reader's Arrow JSON parser."""

    name: str = "specific_jsonl_reader"

    def read_task(self, task: ManifestTask, _read_kwargs: dict | None, fields: list[str] | None) -> ReaderOutput:
        record = task._metadata["manifest"]
        size = record["end_byte"] - record["start_byte"]
        with Path(task.data[0]).open("rb") as stream:
            stream.seek(record["start_byte"])
            payload = stream.read(size)
        if len(payload) != size:
            msg = "Input range ended before the manifest byte boundary"
            raise ValueError(msg)
        table = paj.read_json(
            pa.BufferReader(payload),
            read_options=paj.ReadOptions(block_size=max(size, 1), use_threads=False),
        )
        if table.num_rows != record["num_rows"]:
            msg = "Input range does not match the manifest row count"
            raise ValueError(msg)
        if fields is not None:
            table = table.select(fields)
        return ReaderOutput(table)
