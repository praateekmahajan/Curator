# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cluster-level diversity scores on fine KMeans output, without rereading embeddings."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from zlib import crc32

import cudf
import cupy as cp
import numpy as np
import pylibcudf as plc
from fsspec.utils import get_protocol

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.deduplication.semantic.utils import read_parquet_chunks
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import get_all_file_paths_under, get_fs

_CRC32 = cp.RawKernel(
    r"""
    extern "C" __global__ void crc32_strings(
        const unsigned char* chars, const long long* offsets,
        const unsigned int* table, unsigned int* result, long long n) {
        long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        unsigned int crc = 0xffffffffu;
        for (long long j = offsets[i]; j < offsets[i+1]; ++j)
            crc = (crc >> 8) ^ table[(crc ^ chars[j]) & 255];
        result[i] = crc ^ 0xffffffffu;
    }
    """,
    "crc32_strings",
)


def spark_crc32(values: cudf.Series) -> cudf.Series:
    """Spark-compatible CRC32 of UTF-8 strings, without moving document IDs to CPU.

    Numeric surrogate IDs are encoded as decimal strings. For Forge shard parity,
    supply the original string hash column as the stage's id_field.
    """
    if values.isnull().any():
        msg = "CRC32 sampling keys must not be null"
        raise ValueError(msg)
    strings = values.astype("str")
    column = strings._column.to_pylibcudf()
    result = cp.empty(len(values), dtype=cp.uint32)
    if len(values):
        offset_column = column.child(0)
        dtype = cp.int32 if offset_column.type().id() == plc.TypeId.INT32 else cp.int64
        offsets = cp.asarray(offset_column.data()).view(dtype)
        offsets = offsets[column.offset() : column.offset() + len(values) + 1].astype(cp.int64)
        table = cp.asarray([crc32(bytes([i]), 0xFFFFFFFF) ^ 0xFFFFFFFF for i in range(256)], dtype=cp.uint32)
        _CRC32(
            ((len(values) + 255) // 256,),
            (256,),
            (cp.asarray(column.data()), offsets, table, result, np.int64(len(values))),
        )
    return cudf.Series(result, index=values.index)


def centroid_separation(centroids: np.ndarray) -> np.ndarray:
    """Mean cosine distance to all other centroids in O(KD), not O(K²D)."""
    if centroids.ndim != 2 or not np.isfinite(centroids).all():  # noqa: PLR2004
        msg = "centroids must be a finite two-dimensional array"
        raise ValueError(msg)
    if len(centroids) < 2:  # noqa: PLR2004
        msg = "diversity requires at least two centroids"
        raise ValueError(msg)
    centers = centroids.astype(np.float64)
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    if (norms == 0).any():
        msg = "diversity is undefined for zero-norm centroids"
        raise ValueError(msg)
    centers /= norms
    # sum_j dot(c_i, c_j) = dot(c_i, sum_j c_j); exclude self.
    return 1 - (centers @ centers.sum(axis=0) - np.sum(centers * centers, axis=1)) / (len(centers) - 1)


@dataclass
class ClusterDiversityStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    """One complete fine-cluster task in; projected metadata + diversity files out.

    Use ClusterWiseFilePartitioningStage after KMeansStage (with cache_path set
    to persist centroids). KMeans' cosine distances supply member compactness.
    metadata_fields must carry the ID, quality fields and topic through KMeans;
    embeddings are deliberately not copied into sampling output.
    Optional topic sharding also emits small group counts for global budgets.
    """

    centroids_path: str
    output_path: str
    metadata_fields: list[str]
    id_field: str = "id"
    topic_field: str | None = None
    sampling_shards: int = 128
    read_memory_bytes: int = 256 * 1024**2
    storage_options: dict[str, Any] | None = None
    sampling_hash: str = "crc32"
    group_fields: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        super().__init__(id_generator=None)
        if self.sampling_shards <= 0:
            msg = "sampling_shards must be positive"
            raise ValueError(msg)
        if self.read_memory_bytes <= 0:
            msg = "read_memory_bytes must be positive"
            raise ValueError(msg)
        if self.sampling_hash not in {"crc32", "xxhash32"}:
            msg = "sampling_hash must be crc32 or xxhash32"
            raise ValueError(msg)
        self.name = "ClusterDiversityStage"
        if self.group_fields is not None and self.topic_field is not None:
            msg = "use group_fields or the legacy topic_field, not both"
            raise ValueError(msg)
        self._groups = (
            list(self.group_fields)
            if self.group_fields is not None
            else ([self.topic_field] if self.topic_field else [])
        )
        self.resources = Resources(cpus=1.0, gpus=1.0)

    def setup(self, _: WorkerMetadata | None = None) -> None:
        fs = get_fs(self.centroids_path, self.storage_options)
        with fs.open(self.centroids_path, "rb") as stream:
            self.separation = centroid_separation(np.load(stream, allow_pickle=False))

    def process(self, task: FileGroupTask) -> FileGroupTask:
        cluster_id = int(task._metadata["centroid_id"])
        if not 0 <= cluster_id < len(self.separation):
            msg = "cluster ID is outside the centroid array"
            raise ValueError(msg)
        files = sorted(
            file
            for path in task.data
            for file in get_all_file_paths_under(
                path, recurse_subdirectories=True, keep_extensions=[".parquet"], storage_options=self.storage_options
            )
        )
        count, distance_sum = 0, 0.0
        for distances in self._read_chunks(files, ["cosine_dist_to_cent"]):
            values = distances["cosine_dist_to_cent"]
            if values.isnull().any() or not cp.isfinite(values.values).all().item():
                msg = "centroid distances must be finite and non-null"
                raise ValueError(msg)
            count += len(values)
            distance_sum += float(values.astype("float64").sum())
        score = distance_sum / count * self.separation[cluster_id] if count else 0.0
        fields = list(dict.fromkeys([self.id_field, *self.metadata_fields, *self._groups]))
        outputs, group_counts = [], Counter()
        for i, df in enumerate(self._read_chunks(files, fields)):
            df["centroid"] = cluster_id
            df["diversity_score"] = score
            if self.topic_field or self.group_fields is not None:
                if df[[self.id_field, *self._groups]].isnull().any().any():
                    msg = "sampling topics and IDs must not be null"
                    raise ValueError(msg)
                hashes = (
                    spark_crc32(df[self.id_field])
                    if self.sampling_hash == "crc32"
                    else df[self.id_field].hash_values(method="xxhash32")
                )
                df["sampling_shard"] = hashes % self.sampling_shards
                counts = df.groupby([*self._groups, "sampling_shard"]).size().to_pandas().to_dict()
                group_counts.update({key if isinstance(key, tuple) else (key,): n for key, n in counts.items()})
            output = f"{self.output_path}/cluster_{cluster_id}_{i}.parquet"
            self.write_parquet(df, output, index=False, storage_options=self.storage_options)
            outputs.append(output)
        return FileGroupTask(
            data=outputs,
            dataset_name=task.dataset_name,
            _stage_perf=task._stage_perf,
            _metadata={**task._metadata, "sampling_group_counts": dict(group_counts), "diversity_score": score},
        )

    def _read_chunks(self, files: list[str], columns: list[str]) -> Iterator[cudf.DataFrame]:
        # KMeans outputs have uniform schemas. Remote adapters may prefetch entire files to host.
        files_per_read = 64 if all(get_protocol(path) == "file" for path in files) else 1
        yield from read_parquet_chunks(
            files, columns, self.read_memory_bytes, self.storage_options, files_per_read=files_per_read
        )
