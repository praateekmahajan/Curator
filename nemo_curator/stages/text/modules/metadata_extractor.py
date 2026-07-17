# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


@dataclass
class MetadataExtractor(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Broadcast configured integer metadata onto every row in a document batch.

    The lookup key comes from task metadata, so file-level provenance can become
    ordinary columns without inspecting or parsing every document.
    """

    metadata_mapping: dict[str, dict[str, int]]
    output_dtypes: dict[str, str]
    task_metadata_field: str = "mapping_names"
    name: str = "metadata_extractor"

    def __post_init__(self) -> None:
        if not self.metadata_mapping:
            msg = "metadata_mapping must not be empty"
            raise ValueError(msg)
        if not self.output_dtypes:
            msg = "output_dtypes must not be empty"
            raise ValueError(msg)

        expected_fields = set(self.output_dtypes)
        for lookup_key, values in self.metadata_mapping.items():
            if set(values) != expected_fields:
                msg = (
                    f"Metadata mapping {lookup_key!r} has fields {sorted(values)}; expected {sorted(expected_fields)}"
                )
                raise ValueError(msg)
            invalid_values = {
                field: value
                for field, value in values.items()
                if isinstance(value, bool) or not isinstance(value, int)
            }
            if invalid_values:
                msg = f"Metadata mapping {lookup_key!r} contains non-integer values: {invalid_values}"
                raise TypeError(msg)

        self._arrow_types: dict[str, pa.DataType] = {}
        for field, dtype in self.output_dtypes.items():
            try:
                arrow_type = pa.type_for_alias(dtype)
            except ValueError as error:
                msg = f"Unsupported dtype {dtype!r} for metadata field {field!r}"
                raise ValueError(msg) from error
            if not pa.types.is_integer(arrow_type):
                msg = f"Metadata field {field!r} must use an integer dtype, got {dtype!r}"
                raise TypeError(msg)
            self._arrow_types[field] = arrow_type

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], list(self.output_dtypes)

    def _resolve_values(self, task: DocumentBatch) -> dict[str, int]:
        raw_keys: Any = task._metadata.get(self.task_metadata_field)
        if isinstance(raw_keys, str):
            lookup_keys = [raw_keys]
        elif isinstance(raw_keys, list) and all(isinstance(key, str) for key in raw_keys):
            lookup_keys = raw_keys
        else:
            msg = f"Task metadata field {self.task_metadata_field!r} must be a string or list of strings"
            raise TypeError(msg)

        matches = [key for key in lookup_keys if key in self.metadata_mapping]
        if len(matches) != 1:
            msg = (
                f"Expected exactly one configured key in task metadata field {self.task_metadata_field!r}; "
                f"found {matches} from {lookup_keys}"
            )
            raise ValueError(msg)
        return self.metadata_mapping[matches[0]]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        values = self._resolve_values(task)
        table = task.to_pyarrow()
        collisions = sorted(set(table.column_names) & set(values))
        if collisions:
            msg = f"MetadataExtractor will not overwrite existing columns: {collisions}"
            raise ValueError(msg)

        for field, value in values.items():
            dtype = self._arrow_types[field]
            column = pa.array(np.full(table.num_rows, value), type=dtype)
            table = table.append_column(field, column)

        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=table,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
