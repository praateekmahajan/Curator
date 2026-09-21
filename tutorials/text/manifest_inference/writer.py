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

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.atomic_io import fsync_directory


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
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

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: DocumentBatch) -> FileGroupTask:
        root = Path(self.output_dir).resolve()
        path = (root / task._metadata["manifest"]["output_file"]).resolve()
        if not path.is_relative_to(root):
            msg = "Output escapes output directory"
            raise ValueError(msg)
        table = task.to_pyarrow()
        if table.num_rows != task._metadata["manifest"]["num_rows"]:
            msg = "Refusing to publish an incomplete task"
            raise ValueError(msg)
        atomic_jsonl(path, table.to_pylist())
        return FileGroupTask(
            dataset_name=task.dataset_name, data=[str(path)], _metadata=task._metadata, _stage_perf=task._stage_perf
        )
