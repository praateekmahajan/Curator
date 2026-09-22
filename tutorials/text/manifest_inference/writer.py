# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish deterministic local/shared-filesystem outputs atomically."""

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.atomic_io import fsync_directory


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]], compression: str | None = None) -> None:
    if compression not in (None, "zstd"):
        msg = f"Unsupported JSONL compression: {compression}"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            encoded = ((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode() for row in rows)
            if compression == "zstd":
                # A manifest task bounds the serialization buffer to at most 512 rows.
                stream.write(pa.Codec("zstd", compression_level=3).compress(b"".join(encoded)))
            else:
                for line in encoded:
                    stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class NamedJsonlWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    output_dir: str
    name: str = "named_jsonl_writer"
    compression: str | None = None
    generation_lineage: dict[str, Any] | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> FileGroupTask:
        root = Path(self.output_dir).resolve()
        path = (root / task._metadata["manifest"]["output_file"]).resolve()
        if self.compression == "zstd":
            path = path.with_name(path.name + ".zst")
        if not path.is_relative_to(root):
            msg = "Output escapes output directory"
            raise ValueError(msg)
        table = task.to_pyarrow()
        if table.num_rows != task._metadata["manifest"]["num_rows"]:
            msg = "Refusing to publish an incomplete task"
            raise ValueError(msg)
        if self.generation_lineage is not None and "generation_lineage" in table.column_names:
            msg = "Refusing to overwrite source column generation_lineage"
            raise ValueError(msg)
        rows = table.to_pylist()
        if self.generation_lineage is not None:
            for row in rows:
                row["generation_lineage"] = self.generation_lineage
        atomic_jsonl(path, rows, compression=self.compression)
        return FileGroupTask(
            dataset_name=task.dataset_name, data=[str(path)], _metadata=task._metadata, _stage_perf=task._stage_perf
        )
