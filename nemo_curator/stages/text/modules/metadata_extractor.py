# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

DEFAULT_SEPARATOR = "\n\n"
MERGE_STRATEGIES = ("separator", "smart")


def _merge_separator(text_blocks: list[str], separator: str) -> str:
    """Join text blocks verbatim with a fixed separator."""
    return separator.join(text_blocks)


def _merge_smart(text_blocks: list[str], _separator: str) -> str:
    """Merge blocks with whitespace-aware boundaries."""
    merged_text: list[str] = []
    if len(text_blocks) > 1:
        for index, block in enumerate(text_blocks):
            if index == 0:
                stripped_block = block.rstrip()
            else:
                stripped_block = block.lstrip()
                if index < len(text_blocks) - 1:
                    stripped_block = stripped_block.rstrip()

                previous_block = text_blocks[index - 1]
                if previous_block.rstrip(" \t").endswith("\n") or block.lstrip(" \t").startswith("\n"):
                    merged_text.append("\n\n")
                else:
                    merged_text.append(" ")

            merged_text.append(stripped_block)
    else:
        merged_text = text_blocks
    return "".join(merged_text)


_MERGERS = {"separator": _merge_separator, "smart": _merge_smart}


@dataclass
class MetadataExtractor(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Broadcast configured integer metadata onto every row in a document batch.

    The lookup key comes from task metadata, so file-level provenance can become
    ordinary columns without inspecting or parsing every document.
    """

    metadata_mapping: dict[str, dict[str, int]]
    output_dtypes: dict[str, str]
    task_metadata_field: str = "mapping_names"
    content_field: str | None = None
    text_field: str = "text"
    separator: str = DEFAULT_SEPARATOR
    merge_strategy: str = "separator"
    retained_input_fields: list[str] | None = None
    name: str = "metadata_extractor"

    def __post_init__(self) -> None:
        if not self.metadata_mapping:
            msg = "metadata_mapping must not be empty"
            raise ValueError(msg)
        if not self.output_dtypes:
            msg = "output_dtypes must not be empty"
            raise ValueError(msg)
        if self.merge_strategy not in _MERGERS:
            msg = f"merge_strategy must be one of {MERGE_STRATEGIES}, got {self.merge_strategy!r}"
            raise ValueError(msg)
        self._merge = _MERGERS[self.merge_strategy]
        self._validate_retained_input_fields()

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

    def _validate_retained_input_fields(self) -> None:
        if self.retained_input_fields is not None:
            if len(self.retained_input_fields) != len(set(self.retained_input_fields)):
                msg = "retained_input_fields must not contain duplicates"
                raise ValueError(msg)
            if self.content_field is not None and self.text_field not in self.retained_input_fields:
                msg = f"retained_input_fields must include configured text field {self.text_field!r}"
                raise ValueError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = list(self.output_dtypes)
        if self.content_field is not None:
            output_fields.append(self.text_field)
        return ["data"], output_fields

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

    def _extract_text(self, document: object) -> str:
        if isinstance(document, dict):
            content = document.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_blocks = [item for item in content if isinstance(item, str)]
                return self._merge(text_blocks, self.separator)
        return ""

    def _add_text_if_missing(self, table: pa.Table) -> pa.Table:
        if self.text_field in table.column_names or self.content_field is None:
            return table
        if self.content_field not in table.column_names:
            msg = (
                f"Input has neither text field {self.text_field!r} nor configured content field {self.content_field!r}"
            )
            raise ValueError(msg)

        texts = [self._extract_text(document) for document in table[self.content_field].to_pylist()]
        return table.append_column(self.text_field, pa.array(texts, type=pa.string()))

    def _prepare_table(self, task: DocumentBatch) -> pa.Table:
        """Extract text before Arrow sees heterogeneous nested payloads."""
        if isinstance(task.data, pd.DataFrame):
            frame = task.data
            if self.text_field not in frame.columns and self.content_field is not None:
                if self.content_field not in frame.columns:
                    msg = (
                        f"Input has neither text field {self.text_field!r} nor configured "
                        f"content field {self.content_field!r}"
                    )
                    raise ValueError(msg)
                frame = frame.copy()
                texts = [self._extract_text(document) for document in frame[self.content_field].tolist()]
                frame[self.text_field] = pd.array(texts, dtype="string[pyarrow]")

            if self.retained_input_fields is not None:
                missing_fields = [field for field in self.retained_input_fields if field not in frame.columns]
                if missing_fields:
                    msg = f"Input is missing retained fields: {missing_fields}"
                    raise ValueError(msg)
                frame = frame[self.retained_input_fields]
            return pa.Table.from_pandas(frame, preserve_index=False)

        table = self._add_text_if_missing(task.to_pyarrow())
        if self.retained_input_fields is not None:
            missing_fields = [field for field in self.retained_input_fields if field not in table.column_names]
            if missing_fields:
                msg = f"Input is missing retained fields: {missing_fields}"
                raise ValueError(msg)
            table = table.select(self.retained_input_fields)
        return table

    def process(self, task: DocumentBatch) -> DocumentBatch:
        values = self._resolve_values(task)
        table = self._prepare_table(task)
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
