# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from loguru import logger

from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.client_utils import is_remote_url


@dataclass
class MirroredParquetWriter(ParquetWriter):
    """Write one Parquet file at the source-relative JSONL path."""

    source_root: str = ""
    drop_fields: list[str] = field(default_factory=lambda: ["text"])
    name: str = "mirrored_parquet_writer"

    def __post_init__(self) -> None:
        if not self.source_root:
            msg = "source_root is required"
            raise ValueError(msg)
        super().__post_init__()

    def process(self, task: DocumentBatch) -> FileGroupTask:
        source_files = task._metadata.get("source_files")
        if not isinstance(source_files, list) or len(source_files) != 1:
            msg = "MirroredParquetWriter requires exactly one source file per task"
            raise ValueError(msg)

        source_path = Path(source_files[0])
        source_root = Path(self.source_root)
        try:
            relative_path = source_path.relative_to(source_root)
        except ValueError as error:
            msg = f"Source file {source_path} is outside source root {source_root}"
            raise ValueError(msg) from error

        relative_output = PurePosixPath(relative_path.as_posix()).with_suffix(".parquet")
        file_path = self.fs.sep.join([self._fs_path, *relative_output.parts])
        parent_path = posixpath.dirname(file_path)
        self.fs.makedirs(parent_path, exist_ok=True)
        file_path_with_protocol = self.fs.unstrip_protocol(file_path) if is_remote_url(self.path) else file_path

        table = task.to_pyarrow()
        fields_to_drop = [field for field in self.drop_fields if field in table.column_names]
        if fields_to_drop:
            table = table.drop(fields_to_drop)
        output_task = DocumentBatch(
            dataset_name=task.dataset_name,
            data=table,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )

        if self.fs.exists(file_path):
            logger.debug(f"File {file_path_with_protocol} already exists, overwriting it")
        self.write_data(output_task, file_path_with_protocol)
        logger.debug(f"Written {task.num_items} records to {file_path_with_protocol}")

        return FileGroupTask(
            dataset_name=task.dataset_name,
            data=[file_path_with_protocol],
            _metadata={**task._metadata, "format": self.get_file_extension()},
            _stage_perf=task._stage_perf,
        )
