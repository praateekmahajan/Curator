# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use Curator's JSONL serialization with manifest output filenames."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.tasks import DocumentBatch, FileGroupTask


@dataclass
class NamedJsonlWriter(JsonlWriter):
    name: str = "named_jsonl_writer"
    compression: str | None = None
    generation_lineage: dict[str, Any] | None = None

    def process(self, task: DocumentBatch) -> FileGroupTask:
        root = Path(self.path).resolve()
        path = (root / task._metadata["manifest"]["output_file"]).resolve()
        if self.compression == "zstd":
            path = path.with_name(path.name + ".zst")
        if not path.is_relative_to(root):
            msg = "Output escapes output directory"
            raise ValueError(msg)
        if task.num_items != task._metadata["manifest"]["num_rows"]:
            msg = "Refusing to publish an incomplete task"
            raise ValueError(msg)
        frame = task.to_pandas().copy()
        if self.generation_lineage is not None:
            if "generation_lineage" in frame.columns:
                msg = "Refusing to overwrite source column generation_lineage"
                raise ValueError(msg)
            frame["generation_lineage"] = [self.generation_lineage] * len(frame)
        output = DocumentBatch(data=frame, dataset_name=task.dataset_name, _metadata=task._metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.write_kwargs = {**self.write_kwargs, "compression": self.compression}
        self.write_data(output, str(path))
        return FileGroupTask(
            dataset_name=task.dataset_name, data=[str(path)], _metadata=task._metadata, _stage_perf=task._stage_perf
        )
