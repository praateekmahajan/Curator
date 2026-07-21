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

from typing import TYPE_CHECKING, Literal

import cupy as cp

if TYPE_CHECKING:
    import cudf

from typing import Any

import pyarrow.parquet as pq
from fsspec.parquet import open_parquet_file
from loguru import logger


def get_array_from_df(df: "cudf.DataFrame", embedding_col: str) -> "cp.ndarray":
    """
    Convert a column of lists to a 2D array.
    """
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)


def get_parquet_num_rows_by_file(files: list[str], storage_options: dict[str, Any] | None = None) -> dict[str, int]:
    """Return exact row counts keyed by Parquet file from footer metadata."""
    rows_by_file = {}
    for file in files:
        with open_parquet_file(file, storage_options=storage_options) as f:
            rows_by_file[file] = pq.read_metadata(f).num_rows
    return rows_by_file


def get_parquet_num_rows(files: list[str], storage_options: dict[str, Any] | None = None) -> int:
    """Return the total row count from Parquet file metadata."""
    return sum(get_parquet_num_rows_by_file(files, storage_options).values())


EmbeddingStorageDtype = Literal["auto", "float16", "float32"]


def decode_embedding_array(
    df: "cudf.DataFrame", embedding_col: str, storage_dtype: EmbeddingStorageDtype = "auto"
) -> "cp.ndarray":
    """Return embeddings as FP32, decoding FP16 values stored as uint16 bits."""
    embeddings = get_array_from_df(df, embedding_col)
    if storage_dtype == "auto":
        storage_dtype = "float16" if embeddings.dtype == cp.uint16 else "float32"
    if storage_dtype == "float16":
        if embeddings.dtype != cp.uint16:
            msg = f"Expected uint16 bit storage for float16 embeddings, got {embeddings.dtype}"
            raise TypeError(msg)
        return embeddings.view(cp.float16).astype(cp.float32)
    if storage_dtype == "float32":
        return embeddings.astype(cp.float32, copy=False)
    msg = f"Unsupported embedding storage dtype: {storage_dtype}"
    raise ValueError(msg)


def break_parquet_partition_into_groups(
    files: list[str],
    embedding_dim: int | None = None,
    storage_options: dict[str, Any] | None = None,
    rows_by_file: dict[str, int] | None = None,
) -> list[list[str]]:
    """Break parquet files into groups to avoid cudf 2bn row limit."""
    # TODO: Allow callers to read selected row groups when one file can exceed
    # cuDF's list-child element limit.
    if embedding_dim is None:
        # Default aggressive assumption of 1024 dimensional embedding
        embedding_dim = 1024

    cudf_max_num_rows = 2_000_000_000  # cudf only allows 2bn rows
    cudf_max_num_elements = cudf_max_num_rows / embedding_dim  # cudf considers each element in an array to be a row

    if rows_by_file is not None:
        subgroups = []
        subgroup = []
        subgroup_rows = 0
        for file in files:
            file_rows = rows_by_file[file]
            if subgroup and subgroup_rows + file_rows > cudf_max_num_elements:
                subgroups.append(subgroup)
                subgroup = []
                subgroup_rows = 0
            subgroup.append(file)
            subgroup_rows += file_rows
        if subgroup:
            subgroups.append(subgroup)
        if len(subgroups) > 1:
            logger.debug(f"Broke {len(files)} files into {len(subgroups)} exact-row-count subgroups")
        return subgroups

    # Load the first file and get the number of rows to estimate
    with open_parquet_file(files[0], storage_options=storage_options) as f:
        # Multiply by 1.5 to adjust for skew
        avg_num_rows = pq.read_metadata(f).num_rows * 1.5

    max_files_per_subgroup = int(cudf_max_num_elements / avg_num_rows)
    max_files_per_subgroup = max(1, max_files_per_subgroup)  # Ensure at least 1 file per subgroup

    # Break files into subgroups
    subgroups = [files[i : i + max_files_per_subgroup] for i in range(0, len(files), max_files_per_subgroup)]
    if len(subgroups) > 1:
        logger.debug(
            f"Broke {len(files)} files into {len(subgroups)} subgroups with max {max_files_per_subgroup} files per subgroup"
        )
    return subgroups
