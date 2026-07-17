# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.tasks import DocumentBatch

from .base import BaseWriter


@dataclass
class ParquetWriter(BaseWriter):
    """Writer that writes a DocumentBatch to a Parquet file."""

    # Additional kwargs for pandas.DataFrame.to_parquet
    write_kwargs: dict[str, Any] = field(default_factory=dict)
    optimize_embedding_storage: bool = True
    embedding_fields: list[str] = field(default_factory=lambda: ["embeddings"])
    file_extension: str = "parquet"
    name: str = "parquet_writer"

    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        """Write Arrow batches directly and retain pandas compatibility."""
        if isinstance(task.data, pa.Table) and self._write_arrow(task.data, file_path):
            return

        df = task.to_pandas()  # Convert to pandas DataFrame if needed
        if self.fields is not None:
            df = df[self.fields]
        # Build kwargs for to_parquet with explicit options
        write_kwargs = {
            "index": None,
        }

        # Add any additional kwargs, allowing them to override defaults
        write_kwargs.update(self.write_kwargs)
        df.to_parquet(file_path, **write_kwargs)

    def _write_arrow(self, table: pa.Table, file_path: str) -> bool:
        """Write an Arrow table without materializing Python or pandas objects.

        Returns ``False`` when pandas-only options require the compatibility
        path. Embedding defaults are applied only when callers did not provide
        their own Parquet encoding or compression policy.
        """
        if self.fields is not None:
            table = table.select(self.fields)

        embedding_fields = [
            field_name
            for field_name in self.embedding_fields
            if field_name in table.column_names
            and (
                pa.types.is_list(table.schema.field(field_name).type)
                or pa.types.is_large_list(table.schema.field(field_name).type)
            )
            and pa.types.is_floating(table.schema.field(field_name).type.value_type)
        ]
        if not embedding_fields:
            return False

        write_kwargs = self.write_kwargs.copy()
        engine = write_kwargs.pop("engine", None)
        index = write_kwargs.pop("index", None)
        if engine not in {None, "pyarrow"} or index is True:
            return False

        arrow_write_options = {
            "allow_truncated_timestamps",
            "coerce_timestamps",
            "column_encoding",
            "compression",
            "compression_level",
            "data_page_size",
            "data_page_version",
            "dictionary_pagesize_limit",
            "encryption_properties",
            "filesystem",
            "flavor",
            "row_group_size",
            "sorting_columns",
            "store_decimal_as_integer",
            "store_schema",
            "use_byte_stream_split",
            "use_compliant_nested_type",
            "use_deprecated_int96_timestamps",
            "use_dictionary",
            "version",
            "write_batch_size",
            "write_page_checksum",
            "write_page_index",
            "write_statistics",
        }
        if not set(write_kwargs).issubset(arrow_write_options):
            return False

        storage_options = {"compression", "compression_level", "use_byte_stream_split", "use_dictionary"}
        if self.optimize_embedding_storage and embedding_fields and not storage_options.intersection(write_kwargs):
            write_kwargs.update(
                {
                    "compression": "zstd",
                    "compression_level": 3,
                    "use_byte_stream_split": [f"{field_name}.list.element" for field_name in embedding_fields],
                    "use_dictionary": [name for name in table.column_names if name not in embedding_fields],
                }
            )

        pq.write_table(table, file_path, **write_kwargs)
        return True
