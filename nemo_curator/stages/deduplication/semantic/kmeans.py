# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import cupy as cp
import numpy as np
from cudf.io.parquet import ParquetDatasetWriter

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.deduplication.gpu_utils import get_device_memory_info
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.embedders.utils import create_list_series_from_1d_or_2d_ar
from nemo_curator.tasks import EmptyTask, FileGroupTask
from nemo_curator.utils.file_utils import check_disallowed_kwargs, get_default_file_extensions

from .utils import break_parquet_partition_into_groups, get_array_from_df, get_parquet_num_rows_by_file

if TYPE_CHECKING:
    import cudf

import gc
import os
import random
import time

from loguru import logger

# Column names
L2_DIST_TO_CENT_COL = "l2_dist_to_cent"
COSINE_DIST_TO_CENT_COL = "cosine_dist_to_cent"

# cuDF columns use a 32-bit size type. Keep list child columns below this
# conservative element count when reading or writing embeddings.
CUDF_LIST_COLUMN_MAX_ELEMENTS = 2_000_000_000
AUTO_WRITE_TARGET_MEMORY_FRACTION = 0.80
AUTO_WRITE_BYTES_PER_EMBEDDING_ELEMENT = 24
AUTO_WRITE_FIXED_BYTES_PER_ROW = 128
CENTROID_ARRAY_NDIM = 2


@dataclass
class _LoadedGroup:
    frame: Any
    start: int
    end: int


class _RollingParquetDatasetWriter:
    """Roll partitioned files using conservative per-partition row accounting."""

    def __init__(
        self,
        create_writer: Callable[[int], ParquetDatasetWriter],
        n_partitions: int,
        max_file_size: int | None,
    ) -> None:
        self._create_writer = create_writer
        self._n_partitions = n_partitions
        self._max_file_size = max_file_size
        self._generation = 0
        self._writer = create_writer(self._generation)
        self._rows_by_partition = np.zeros(n_partitions, dtype=np.int64)
        self._target_rows: int | None = None

    def __enter__(self) -> "_RollingParquetDatasetWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _partition_counts(self, frame: "cudf.DataFrame") -> np.ndarray:
        return cp.asnumpy(cp.bincount(frame["centroid"].values, minlength=self._n_partitions))

    def _ensure_target_rows(self, frame: "cudf.DataFrame") -> None:
        if self._max_file_size is None or self._target_rows is not None:
            return
        bytes_per_row = max(1, math.ceil(int(frame.memory_usage().sum()) / len(frame)))
        self._target_rows = max(1, self._max_file_size // bytes_per_row)
        logger.info(
            f"Rolling KMeans output after approximately {self._target_rows} rows per centroid "
            f"(max_file_size={self._max_file_size}, estimated_uncompressed_bytes_per_row={bytes_per_row})"
        )

    def _roll(self) -> None:
        self._writer.close()
        self._generation += 1
        self._writer = self._create_writer(self._generation)
        self._rows_by_partition.fill(0)

    def write_table(self, frame: "cudf.DataFrame") -> None:
        if len(frame) == 0:
            return
        self._ensure_target_rows(frame)
        if self._target_rows is None:
            self._writer.write_table(frame)
            return

        counts = self._partition_counts(frame)
        if counts.max() > self._target_rows:
            num_slices = math.ceil(int(counts.max()) / self._target_rows)
            slice_rows = math.ceil(len(frame) / num_slices)
            for start in range(0, len(frame), slice_rows):
                self.write_table(frame.iloc[start : start + slice_rows])
            return

        if self._rows_by_partition.any() and np.any(self._rows_by_partition + counts > self._target_rows):
            self._roll()
        self._writer.write_table(frame)
        self._rows_by_partition += counts

    def close(self) -> None:
        self._writer.close()


class KMeansReadFitWriteStage(ProcessingStage[FileGroupTask, EmptyTask], DeduplicationIO):
    """KMeans clustering stage that requires RAFT for distributed processing."""

    def __init__(  # noqa: PLR0913
        self,
        id_field: str,
        embedding_field: str,
        output_path: str,
        filetype: Literal["parquet", "jsonl"],
        # KMeans args
        n_clusters: int,
        metadata_fields: list[str] | None = None,
        embedding_dim: int | None = None,
        verbose: bool = False,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
        init: Literal["k-means||", "random"] | np.ndarray = "k-means||",
        n_init: int | Literal["auto"] = 1,
        oversampling_factor: float = 2.0,
        max_samples_per_batch: int = 1 << 15,
        fit_data_fraction: float | None = None,
        output_embedding_dtype: Literal["float16", "float32"] = "float16",
        write_batch_size: int | Literal["auto"] = "auto",
        max_output_file_size: int | None = None,
        prefetch_next_group: bool = False,
        # I/O args
        cache_path: str | None = None,
        read_kwargs: dict[dict] | None = None,
        write_kwargs: dict[dict] | None = None,
    ):
        """KMeans clustering stage that requires RAFT for distributed processing.

        Args:
            id_field (str): The column name of the id column.
            embedding_field (str): The column name of the embedding column.
            output_path (str): The path to the output directory.
            n_clusters (int): The number of clusters to create.
            metadata_fields (list[str] | None): The columns to keep in the output. These columns can be used later to prioritize deduplication.
            embedding_dim (int | None): The dimension of the embedding. This helps us read data into smaller chunks.
            verbose (bool): Whether to print verbose output.
            max_iter (int): The maximum number of iterations to run.
            tol (float): Tolerance for stopping criteria of the kmeans algorithm.
            random_state (int): Seed for the random number generator. Unseeded by default. Does not currently fully guarantee the exact same results.
            init (Literal["k-means||", "random"] | np.ndarray): 'scalable-k-means++' or 'k-means||': Uses fast and stable scalable kmeans++ initialization. 'random': Choose 'n_cluster' observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
            n_init (int | Literal["auto"]): Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
            oversampling_factor (float): The amount of points to sample in scalable k-means++ initialization for potential centroids. Increasing this value can lead to better initial centroids at the cost of memory. The total number of centroids sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.
            max_samples_per_batch (int): The number of data samples to use for batches of the pairwise distance computation. This computation is done throughout both fit predict. The default should suit most cases. The total number of elements in the batched pairwise distance computation is max_samples_per_batch * n_clusters. It might become necessary to lower this number when n_clusters becomes prohibitively large.
            fit_data_fraction (float | None): Fraction of the dataset (in (0, 1)) used to fit the KMeans model. Pass None to fit on the full dataset (single-pass mode). For Parquet, each actor uses footer row counts to sample whole files until it reaches the requested row fraction. Other formats sample by file count. When set, uses a two-pass approach: Pass 1 reads only the embedding column from the sampled files; Pass 2 loads each full original group one at a time to predict labels and write results. If None, all rows are loaded simultaneously.
            output_embedding_dtype: Storage dtype for normalized embeddings written after KMeans. FP16 is encoded as uint16 bit patterns for cuDF compatibility.
            write_batch_size: Maximum rows to materialize and send to the partitioned Parquet writer at once. ``"auto"`` sizes each group from the actor GPU's live memory and embedding width.
            max_output_file_size: Approximate maximum uncompressed bytes per centroid Parquet file.
            prefetch_next_group: Read and normalize one group concurrently with prediction and writing.
            cache_path (str | None): The path to save the centroids to. If None, the centroids will not be saved.
            read_kwargs (dict[dict]): Keyword arguments for the read stage.
            write_kwargs (dict[dict]): Keyword arguments for the write stage.
        """
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.output_path = output_path
        self.filetype = filetype
        self.n_clusters = n_clusters
        self.metadata_fields = metadata_fields if metadata_fields is not None else []
        self.embedding_dim = embedding_dim
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch
        if fit_data_fraction is not None and not 0.0 < fit_data_fraction < 1.0:
            msg = f"fit_data_fraction must be in (0, 1), got {fit_data_fraction}; pass None to fit on the full dataset"
            raise ValueError(msg)
        self.fit_data_fraction = fit_data_fraction
        if output_embedding_dtype not in {"float16", "float32"}:
            msg = f"Unsupported output_embedding_dtype: {output_embedding_dtype}"
            raise ValueError(msg)
        self.output_embedding_dtype = output_embedding_dtype
        if write_batch_size != "auto" and (not isinstance(write_batch_size, int) or write_batch_size <= 0):
            msg = f"write_batch_size must be positive, got {write_batch_size}"
            raise ValueError(msg)
        self.write_batch_size = write_batch_size
        if max_output_file_size is not None and max_output_file_size <= 0:
            msg = f"max_output_file_size must be positive, got {max_output_file_size}"
            raise ValueError(msg)
        self.max_output_file_size = max_output_file_size
        self.prefetch_next_group = prefetch_next_group

        self.cache_path = cache_path
        self.read_kwargs = read_kwargs.copy() if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs is not None else {}
        if self.output_embedding_dtype == "float16":
            self.write_kwargs.setdefault("compression", "zstd")

        check_disallowed_kwargs(self.read_kwargs, ["columns", "assign_id"])
        check_disallowed_kwargs(self.write_kwargs, ["partition_file_name", "partition_cols", "index"])

        self.input_storage_options = self.read_kwargs.pop("storage_options", None)
        self.output_storage_options = self.write_kwargs.pop("storage_options", None)

        self.name = "KMeansStage"
        self.resources = Resources(cpus=1.0, gpus=1.0)

    def process(self, task: FileGroupTask) -> EmptyTask:
        msg = "KMeansReadFitWriteStage does not support single-task processing"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[FileGroupTask]) -> list[EmptyTask]:
        """Process a batch of FileGroupTasks using distributed RAFT KMeans.

        In RAFT mode, each actor processes its assigned tasks, but the KMeans model
        is trained cooperatively across all actors using RAFT communication.

        When fit_data_fraction is set, uses a memory-efficient two-pass approach:
          Pass 1: samples files at the actor level, reads only the embedding column from those files
          Pass 2: loads each (full) original group one at a time for prediction and writing
        Otherwise, loads all groups simultaneously (original behavior).
        """

        if not tasks:
            return []

        groups = self._group_input_files(tasks)

        if self.fit_data_fraction is not None:
            return self._process_batch_two_pass(tasks, groups)
        return self._process_batch_single_pass(tasks, groups)

    def _group_input_files(self, tasks: list[FileGroupTask]) -> list[list[str]]:
        """Group task files into reads that stay below cuDF column limits."""
        all_files = [file for task in tasks for file in task.data]
        if self.filetype == "parquet":
            footer_scan_start = time.perf_counter()
            self._parquet_rows_by_file = get_parquet_num_rows_by_file(
                all_files,
                storage_options=self.input_storage_options,
            )
            footer_scan_time = time.perf_counter() - footer_scan_start
            self._log_metric("kmeans_footer_scan_time", footer_scan_time)
            logger.info(f"Parquet footer scan time: {footer_scan_time:.2f} seconds for {len(all_files)} files")
            groups = break_parquet_partition_into_groups(
                all_files,
                embedding_dim=self.embedding_dim,
                storage_options=self.input_storage_options,
                rows_by_file=self._parquet_rows_by_file,
            )
        elif self.filetype == "jsonl":
            # For JSONL files, just group all files together since we can't easily estimate size
            groups = [all_files]
        else:
            msg = f"Unsupported filetype: {self.filetype}. Only jsonl and parquet are supported."
            raise ValueError(msg)
        return groups

    def _read_group(self, group: list[str], columns: list[str]) -> "cudf.DataFrame":
        """Read a group of files into a cudf DataFrame."""
        if self.filetype == "parquet":
            return self.read_parquet(
                group,
                columns=columns,
                storage_options=self.input_storage_options,
                assign_id=False,
                **self.read_kwargs,
            )
        if self.filetype == "jsonl":
            return self.read_jsonl(
                group,
                columns=columns,
                storage_options=self.input_storage_options,
                assign_id=False,
                **self.read_kwargs,
            )
        msg = f"Unsupported data type: {self.filetype}"
        raise ValueError(msg)

    def _load_contiguous_embeddings(
        self,
        groups: list[list[str]],
        columns: list[str],
        *,
        retain_frames: bool,
    ) -> tuple["cp.ndarray", list[_LoadedGroup]]:
        """Load normalized embeddings, inferring their width from the first group.

        Multiple Parquet groups are copied directly into one preallocated array.
        The allocation uses row counts from the remaining file footers, which are
        an upper bound when read filters are present. A single group needs no copy.
        """
        first_frame = self._read_group(groups[0], columns)
        first_frame = self.normalize_embeddings_col_in_df(first_frame, self.embedding_field)
        first_embeddings = get_array_from_df(first_frame, self.embedding_field)

        if len(groups) == 1:
            return first_embeddings, [_LoadedGroup(first_frame, 0, len(first_frame))]

        if self.filetype != "parquet":
            return self._concatenate_groups(
                groups,
                columns,
                first_frame,
                first_embeddings,
                retain_frames=retain_frames,
            )

        return self._preallocate_parquet_groups(
            groups,
            columns,
            first_frame,
            first_embeddings,
            retain_frames=retain_frames,
        )

    def _concatenate_groups(
        self,
        groups: list[list[str]],
        columns: list[str],
        first_frame: "cudf.DataFrame",
        first_embeddings: "cp.ndarray",
        *,
        retain_frames: bool,
    ) -> tuple["cp.ndarray", list[_LoadedGroup]]:
        """Concatenate formats without cheap row-count metadata."""
        frames = [first_frame]
        embedding_arrays = [first_embeddings]
        for group in groups[1:]:
            frame = self._read_group(group, columns)
            frame = self.normalize_embeddings_col_in_df(frame, self.embedding_field)
            frames.append(frame)
            embedding_arrays.append(get_array_from_df(frame, self.embedding_field))

        embeddings = cp.concatenate(embedding_arrays, axis=0)
        loaded_groups = []
        offset = 0
        for frame in frames:
            end = offset + len(frame)
            if retain_frames:
                loaded_groups.append(_LoadedGroup(frame, offset, end))
            offset = end
        return embeddings, loaded_groups

    def _preallocate_parquet_groups(
        self,
        groups: list[list[str]],
        columns: list[str],
        first_frame: "cudf.DataFrame",
        first_embeddings: "cp.ndarray",
        *,
        retain_frames: bool,
    ) -> tuple["cp.ndarray", list[_LoadedGroup]]:
        """Load multiple Parquet groups into one runtime-sized array."""

        remaining_files = [file for group in groups[1:] for file in group]
        row_capacity = len(first_frame) + sum(self._parquet_rows_by_file[file] for file in remaining_files)
        embedding_width = first_embeddings.shape[1]
        embeddings = cp.empty((row_capacity, embedding_width), dtype=cp.float32)
        loaded_groups: list[_LoadedGroup] = []
        offset = 0

        def append_group(frame: "cudf.DataFrame", group_embeddings: "cp.ndarray") -> None:
            nonlocal offset
            if group_embeddings.shape[1] != embedding_width:
                msg = (
                    f"Embedding width changed from {embedding_width} to {group_embeddings.shape[1]} "
                    "within one KMeans actor"
                )
                raise ValueError(msg)
            end = offset + len(frame)
            if end > row_capacity:
                msg = f"Read {end} rows, exceeding the Parquet metadata row capacity of {row_capacity}"
                raise ValueError(msg)
            embeddings[offset:end] = group_embeddings
            if retain_frames:
                metadata_frame = frame.drop(columns=[self.embedding_field])
                loaded_groups.append(_LoadedGroup(metadata_frame, offset, end))
            offset = end

        append_group(first_frame, first_embeddings)
        del first_frame, first_embeddings

        for group in groups[1:]:
            frame = self._read_group(group, columns)
            frame = self.normalize_embeddings_col_in_df(frame, self.embedding_field)
            group_embeddings = get_array_from_df(frame, self.embedding_field)
            append_group(frame, group_embeddings)
            del frame, group_embeddings

        return embeddings[:offset], loaded_groups

    def _process_batch_single_pass(self, tasks: list[FileGroupTask], groups: list[list[str]]) -> list["EmptyTask"]:
        """Fit all rows from one contiguous array and write retained groups."""
        t0 = time.perf_counter()
        embeddings, loaded_groups = self._load_contiguous_embeddings(
            groups,
            [self.id_field, self.embedding_field, *self.metadata_fields],
            retain_frames=True,
        )

        t1 = time.perf_counter()
        self._log_metrics({"kmeans_read_time": t1 - t0, "num_rows": len(embeddings)})
        logger.debug(f"Read time: {(t1 - t0):.2f} seconds")

        # Fit the model cooperatively across actors, then predict on local data
        self.kmeans._fit(embeddings, sample_weight=None, convert_dtype=False, multigpu=True)

        if self.cache_path is not None and getattr(self, "_actor_index", 0) == 0:
            os.makedirs(self.cache_path, exist_ok=True)
            cp.save(f"{self.cache_path}/kmeans_centroids.npy", self.kmeans.cluster_centers_)
            logger.info(f"Saved {self.n_clusters} KMeans centroids to {self.cache_path}/kmeans_centroids.npy")

        labels = cp.asarray(self.kmeans.labels_).astype(cp.int32, copy=False)

        t2 = time.perf_counter()
        self._log_metric("kmeans_fit_predict_time", t2 - t1)
        logger.info(f"KMeans fit+predict time: {(t2 - t1):.2f} seconds")

        results = []
        with self._create_rolling_dataset_writer(tasks) as writer:
            for i, loaded_group in enumerate(loaded_groups):
                self._write_partitioned_batches(
                    writer,
                    loaded_group.frame,
                    embeddings[loaded_group.start : loaded_group.end],
                    labels[loaded_group.start : loaded_group.end],
                )
                results.append(
                    EmptyTask(
                        dataset_name=f"kmeans_group_{i}",
                        _metadata=None,
                        _stage_perf=[],
                        data=None,
                    )
                )

        t3 = time.perf_counter()
        self._log_metric("kmeans_write_time", t3 - t2)
        logger.info(f"Write time: {(t3 - t2):.2f} seconds")

        return results

    def _process_batch_two_pass(self, tasks: list[FileGroupTask], groups: list[list[str]]) -> list["EmptyTask"]:
        """Memory-efficient two-pass approach for large datasets.

        Pass 1 (_fit_pass): samples fit_data_fraction of the actor's files (across
                all groups), re-chunks them into memory-bounded fit_groups, reads
                only the embedding column, fits the KMeans model, and saves
                centroids if cache_path is set. IO and GPU memory in Pass 1 scale
                with fit_data_fraction.
        Pass 2 (_predict_write_pass): loads each (full) original group one at a
                time, predicts labels, writes, then frees GPU memory before
                loading the next group.

        Peak GPU memory ≈ max(fit_data_fraction x actor_rows, one_group_size) x embedding_dim x 4 bytes,
        instead of total_data x embedding_dim x 4 bytes.
        """
        pass1_read_time = self._fit_pass(groups)
        results, pass2_read_time, total_rows = self._predict_write_pass(tasks, groups)
        self._log_metrics(
            {
                "kmeans_read_time": pass1_read_time + pass2_read_time,
                "kmeans_fit_read_time": pass1_read_time,
                "kmeans_predict_read_time": pass2_read_time,
                "num_rows": total_rows,
            }
        )
        return results

    def _fit_pass(self, groups: list[list[str]]) -> float:
        """Pass 1: sample files at the actor level, read embeddings, fit KMeans,
        and (on actor 0) save centroids.

        Returns:
            Wall-clock seconds spent reading sampled files (for the combined
            kmeans_read_time metric reported by the orchestrator).
        """
        fraction = self.fit_data_fraction

        # Sample files at the actor level (across all groups), then re-chunk into
        # memory-bounded groups. Works for both filetypes: parquet uses the
        # size-aware grouper; jsonl uses a single group (matching process_batch's
        # jsonl path).
        all_files = [f for g in groups for f in g]
        rng = random.Random(self.random_state)  # noqa: S311
        if self.filetype == "parquet":
            shuffled_files = all_files.copy()
            rng.shuffle(shuffled_files)
            target_rows = max(1, round(sum(self._parquet_rows_by_file.values()) * fraction))
            sampled_rows = 0
            fit_files = []
            for file in shuffled_files:
                fit_files.append(file)
                sampled_rows += self._parquet_rows_by_file[file]
                if sampled_rows >= target_rows:
                    break
        else:
            target_n_files = round(len(all_files) * fraction)
            n_files = max(1, target_n_files)
            if target_n_files < 1:
                # RAFT's cooperative _fit needs every actor to contribute at least one row.
                logger.warning(
                    f"fit_data_fraction={fraction} on {len(all_files)} files would sample "
                    "0 files for this actor; bumping to 1 to keep it in the cooperative fit."
                )
            fit_files = rng.sample(all_files, n_files)

        if self.filetype == "parquet":
            fit_groups = break_parquet_partition_into_groups(
                fit_files,
                embedding_dim=self.embedding_dim,
                storage_options=self.input_storage_options,
                rows_by_file=self._parquet_rows_by_file,
            )
        else:  # jsonl
            fit_groups = [fit_files]

        t0 = time.perf_counter()
        sampled_embeddings, fit_group_owners = self._load_contiguous_embeddings(
            fit_groups,
            [self.embedding_field],
            retain_frames=False,
        )
        sampled_rows = len(sampled_embeddings)

        t1 = time.perf_counter()
        pass1_read_time = t1 - t0
        logger.debug(
            f"Pass 1 (sampling) time: {pass1_read_time:.2f}s, "
            f"read {len(fit_files)}/{len(all_files)} files = {sampled_rows} rows"
        )

        logger.info(
            f"Fitting KMeans on {len(sampled_embeddings)} sampled rows "
            f"(fit_data_fraction={fraction:.4f}, {len(fit_files)}/{len(all_files)} files)"
        )

        self.kmeans._fit(sampled_embeddings, sample_weight=None, convert_dtype=False, multigpu=True)
        del sampled_embeddings, fit_group_owners
        gc.collect()
        # Stop the fit-time clock before centroid I/O so the metric isn't skewed
        # by disk write latency on actor 0.
        t_fit_done = time.perf_counter()
        self._log_metric("kmeans_fit_time", t_fit_done - t1)
        logger.info(f"KMeans fit time: {(t_fit_done - t1):.2f} seconds")

        if self.cache_path is not None and getattr(self, "_actor_index", 0) == 0:
            os.makedirs(self.cache_path, exist_ok=True)
            cp.save(f"{self.cache_path}/kmeans_centroids.npy", self.kmeans.cluster_centers_)
            logger.info(f"Saved {self.n_clusters} KMeans centroids to {self.cache_path}/kmeans_centroids.npy")

        return pass1_read_time

    def _predict_write_pass(
        self, tasks: list[FileGroupTask], groups: list[list[str]]
    ) -> tuple[list["EmptyTask"], float, int]:
        """Pass 2: load each full group, predict labels, write results.

        Returns:
            (results, pass2_read_time, total_rows). The orchestrator combines
            pass2_read_time with pass1_read_time into kmeans_read_time, and
            reports total_rows as num_rows.
        """
        t_start = time.perf_counter()
        results: list[EmptyTask] = []
        pass2_read_time = 0.0
        predict_time = 0.0
        write_time = 0.0
        cleanup_time = 0.0
        writer_close_time = 0.0
        total_rows = 0

        writer = self._create_rolling_dataset_writer(tasks)
        prefetch_pool = ThreadPoolExecutor(max_workers=1) if self.prefetch_next_group and groups else None
        prefetched_group = prefetch_pool.submit(self._read_prediction_group, groups[0]) if prefetch_pool else None
        try:
            for i, group in enumerate(groups):
                if prefetched_group is None:
                    df, read_time = self._read_prediction_group(group)
                    pass2_read_time += read_time
                else:
                    read_wait_start = time.perf_counter()
                    df, _ = prefetched_group.result()
                    pass2_read_time += time.perf_counter() - read_wait_start
                    prefetched_group = (
                        prefetch_pool.submit(self._read_prediction_group, groups[i + 1])
                        if prefetch_pool is not None and i + 1 < len(groups)
                        else None
                    )
                embeddings_array = get_array_from_df(df, self.embedding_field)
                total_rows += len(df)

                predict_start = time.perf_counter()
                labels = self._predict_labels(embeddings_array)
                predict_time += time.perf_counter() - predict_start

                write_start = time.perf_counter()
                self._write_partitioned_batches(writer, df, embeddings_array, labels)
                write_time += time.perf_counter() - write_start
                results.append(
                    EmptyTask(
                        dataset_name=f"kmeans_group_{i}",
                        _metadata=None,
                        _stage_perf=[],
                        data=None,
                    )
                )

                cleanup_start = time.perf_counter()
                del df, embeddings_array, labels
                gc.collect()
                cleanup_time += time.perf_counter() - cleanup_start
        finally:
            if prefetch_pool is not None:
                prefetch_pool.shutdown()
            close_start = time.perf_counter()
            writer.close()
            writer_close_time = time.perf_counter() - close_start

        t_end = time.perf_counter()
        predict_write_time = (t_end - t_start) - pass2_read_time
        self._log_metrics(
            {
                "kmeans_predict_write_time": predict_write_time,
                "kmeans_predict_time": predict_time,
                "kmeans_write_time": write_time,
                "kmeans_writer_close_time": writer_close_time,
                "kmeans_cleanup_time": cleanup_time,
            }
        )
        logger.info(
            f"Pass 2 total time: {(t_end - t_start):.2f} seconds "
            f"(read: {pass2_read_time:.2f}s, predict: {predict_time:.2f}s, "
            f"write: {write_time:.2f}s, close: {writer_close_time:.2f}s, cleanup: {cleanup_time:.2f}s)"
        )

        return results, pass2_read_time, total_rows

    def _predict_labels(self, embeddings: "cp.ndarray") -> "cp.ndarray":
        from cuml import using_output_type

        with using_output_type("cupy"):
            return self.kmeans.predict(embeddings, convert_dtype=False).astype(cp.int32)

    def _read_prediction_group(self, group: list[str]) -> tuple["cudf.DataFrame", float]:
        read_start = time.perf_counter()
        frame = self._read_group(group, [self.id_field, self.embedding_field, *self.metadata_fields])
        frame = self.normalize_embeddings_col_in_df(frame, self.embedding_field)
        return frame, time.perf_counter() - read_start

    def setup(self, _: WorkerMetadata | None = None) -> None:
        from cuml.cluster.kmeans import KMeans as cumlKMeans

        if not hasattr(self, "_raft_handle"):
            msg = "RAFT handle not found. Make sure the stage is initialized with RAFT"
            raise ValueError(msg)

        self.kmeans = cumlKMeans(
            handle=self._raft_handle,
            output_type="cupy",
            init=self.init,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            n_init=self.n_init,
            oversampling_factor=self.oversampling_factor,
            max_samples_per_batch=self.max_samples_per_batch,
        )

    @staticmethod
    def normalize_embeddings_col_in_df(df: "cudf.DataFrame", embedding_col: str) -> "cudf.DataFrame":
        embeddings = get_array_from_df(df, embedding_col)
        # RAFT KMeans expects float32, while cached embeddings may be stored as float16.
        needs_float32_conversion = embeddings.dtype != cp.float32
        if needs_float32_conversion:
            embeddings = embeddings.astype(cp.float32)
        embeddings /= cp.linalg.norm(embeddings, axis=1, keepdims=True)
        if needs_float32_conversion:
            df[embedding_col] = create_list_series_from_1d_or_2d_ar(embeddings, index=df.index)
        return df

    def _encode_embeddings_for_write(self, df: "cudf.DataFrame", embedding_col: str) -> "cudf.DataFrame":
        if self.output_embedding_dtype == "float32":
            return df
        embeddings = get_array_from_df(df, embedding_col)
        fp16_bits = embeddings.astype(cp.float16).view(cp.uint16)
        df[embedding_col] = create_list_series_from_1d_or_2d_ar(fp16_bits, index=df.index)
        return df

    def _create_dataset_writer(self, tasks: list[FileGroupTask], generation: int) -> ParquetDatasetWriter:
        """Create one incremental partitioned writer per KMeans actor."""
        supported_kwargs = {"compression", "statistics"}
        unsupported_kwargs = set(self.write_kwargs).difference(supported_kwargs)
        if unsupported_kwargs:
            msg = f"Chunked KMeans output does not support write kwargs {sorted(unsupported_kwargs)}"
            raise ValueError(msg)
        generation_suffix = f"_{generation}" if self.max_output_file_size is not None else ""
        return ParquetDatasetWriter(
            self.output_path,
            partition_cols=["centroid"],
            index=False,
            file_name_prefix=f"{tasks[0].task_id}{generation_suffix}.parquet",
            storage_options=self.output_storage_options,
            **self.write_kwargs,
        )

    def _create_rolling_dataset_writer(self, tasks: list[FileGroupTask]) -> _RollingParquetDatasetWriter:
        return _RollingParquetDatasetWriter(
            create_writer=lambda generation: self._create_dataset_writer(tasks, generation),
            n_partitions=self.n_clusters,
            max_file_size=self.max_output_file_size,
        )

    def _write_partitioned_batches(
        self,
        writer: _RollingParquetDatasetWriter,
        source_frame: "cudf.DataFrame",
        embeddings: "cp.ndarray",
        labels: "cp.ndarray",
    ) -> None:
        """Materialize distances and encoded embeddings in bounded row batches."""
        write_batch_size = self._resolve_write_batch_size(embeddings.shape[1])
        for start in range(0, len(source_frame), write_batch_size):
            end = min(start + write_batch_size, len(source_frame))
            frame = source_frame.iloc[start:end].copy(deep=False)
            frame[self.embedding_field] = create_list_series_from_1d_or_2d_ar(
                embeddings[start:end],
                index=frame.index,
            )
            frame["centroid"] = labels[start:end]
            frame = self._assign_distances(frame, self.embedding_field, self._cluster_centers())
            frame = self._encode_embeddings_for_write(frame, self.embedding_field)
            writer.write_table(frame)
            del frame

    def _resolve_write_batch_size(self, embedding_width: int) -> int:
        """Bound a write batch by cuDF's size type and the actor's live memory."""
        list_column_limit = max(1, CUDF_LIST_COLUMN_MAX_ELEMENTS // embedding_width)
        if self.write_batch_size != "auto":
            return min(self.write_batch_size, list_column_limit)

        memory_info = get_device_memory_info()
        if memory_info is None:
            fallback = min(100_000, list_column_limit)
            logger.warning(f"Could not query GPU memory; using write_batch_size={fallback}")
            return fallback

        free_memory, total_memory = memory_info
        used_memory = total_memory - free_memory
        target_headroom = max(0, int(total_memory * AUTO_WRITE_TARGET_MEMORY_FRACTION) - used_memory)
        estimated_bytes_per_row = (
            AUTO_WRITE_BYTES_PER_EMBEDDING_ELEMENT * embedding_width + AUTO_WRITE_FIXED_BYTES_PER_ROW
        )
        memory_limit = max(1, target_headroom // estimated_bytes_per_row)
        batch_size = min(list_column_limit, memory_limit)
        logger.info(
            f"Auto write batch size: {batch_size} rows "
            f"(embedding_width={embedding_width}, free={free_memory}, total={total_memory}, "
            f"list_column_limit={list_column_limit})"
        )
        return batch_size

    def _cluster_centers(self) -> "cp.ndarray":
        return self.kmeans.cluster_centers_

    @staticmethod
    def _assign_distances(df: "cudf.DataFrame", embedding_col: str, centroids: "cp.ndarray") -> "cudf.DataFrame":
        """
        Computes the L2 distance to nearest centroid to each embedding in the DataFrame.
        Embeddings are normalized. For cosine we'll need to normalize the centroids as well.
        """
        normalized_embeddings = get_array_from_df(df, embedding_col)
        # We normalize the centroids as well for cosine distance
        normalized_centroids = centroids / cp.linalg.norm(centroids, axis=1, keepdims=True)

        df[L2_DIST_TO_CENT_COL] = cp.sqrt(
            cp.sum((normalized_embeddings - centroids[df["centroid"].values]) ** 2, axis=1)
        )
        df[COSINE_DIST_TO_CENT_COL] = 1 - (
            cp.sum(
                normalized_embeddings * normalized_centroids[df["centroid"].values],
                axis=1,
            )
        )
        return df

    def ray_stage_spec(self) -> dict[str, Any]:
        return {
            "is_raft_actor": True,
        }


class KMeansPredictWriteStage(KMeansReadFitWriteStage):
    """Predict centroids and write file-task batches without a RAFT collective.

    Each Ray Data worker loads the saved centroids once during setup, then streams
    its assigned file batches through the existing bounded reader and rolling
    Parquet writer. Input embeddings remain local to the worker GPU.
    """

    def __init__(  # noqa: PLR0913
        self,
        id_field: str,
        embedding_field: str,
        output_path: str,
        centroids_path: str,
        filetype: Literal["parquet", "jsonl"] = "parquet",
        metadata_fields: list[str] | None = None,
        embedding_dim: int | None = None,
        max_samples_per_batch: int = 1 << 15,
        output_embedding_dtype: Literal["float16", "float32"] = "float16",
        write_batch_size: int | Literal["auto"] = "auto",
        max_output_file_size: int | None = None,
        prefetch_next_group: bool = False,
        task_batch_size: int = 1,
        gpu_fraction: float = 1.0,
        worker_count: int | None = None,
        read_kwargs: dict[dict] | None = None,
        write_kwargs: dict[dict] | None = None,
    ) -> None:
        if task_batch_size <= 0:
            msg = f"task_batch_size must be positive, got {task_batch_size}"
            raise ValueError(msg)
        if not 0.0 < gpu_fraction <= 1.0:
            msg = f"gpu_fraction must be in (0, 1], got {gpu_fraction}"
            raise ValueError(msg)
        if worker_count is not None and worker_count <= 0:
            msg = f"worker_count must be positive, got {worker_count}"
            raise ValueError(msg)

        self.centroids_path = centroids_path
        super().__init__(
            id_field=id_field,
            embedding_field=embedding_field,
            output_path=output_path,
            filetype=filetype,
            n_clusters=1,  # Replaced from the centroid array during setup.
            metadata_fields=metadata_fields,
            embedding_dim=embedding_dim,
            max_samples_per_batch=max_samples_per_batch,
            fit_data_fraction=None,
            output_embedding_dtype=output_embedding_dtype,
            write_batch_size=write_batch_size,
            max_output_file_size=max_output_file_size,
            prefetch_next_group=prefetch_next_group,
            read_kwargs=read_kwargs,
            write_kwargs=write_kwargs,
        )
        self.name = "KMeansPredictWriteStage"
        self.resources = Resources(cpus=1.0, gpus=gpu_fraction)
        self.batch_size = task_batch_size
        self.worker_count = worker_count

    def setup(self, _: WorkerMetadata | None = None) -> None:
        centroids = cp.load(self.centroids_path)
        if centroids.ndim != CENTROID_ARRAY_NDIM or centroids.shape[0] == 0 or centroids.shape[1] == 0:
            msg = f"Expected a non-empty 2D centroid array, got shape={centroids.shape}"
            raise ValueError(msg)
        if centroids.dtype != cp.float32:
            centroids = centroids.astype(cp.float32)

        self.n_clusters = int(centroids.shape[0])
        self._centroid_embedding_width = int(centroids.shape[1])
        self._centroids = cp.ascontiguousarray(centroids)
        logger.info(
            f"Loaded {self.n_clusters} centroids with embedding width {centroids.shape[1]} from {self.centroids_path}"
        )

    def _predict_labels(self, embeddings: "cp.ndarray") -> "cp.ndarray":
        from cuml.metrics import pairwise_distances

        if embeddings.shape[1] != self._centroid_embedding_width:
            msg = (
                f"Input embedding width {embeddings.shape[1]} does not match centroid width "
                f"{self._centroid_embedding_width}"
            )
            raise ValueError(msg)
        labels = cp.empty(len(embeddings), dtype=cp.int32)
        for start in range(0, len(embeddings), self.max_samples_per_batch):
            end = min(start + self.max_samples_per_batch, len(embeddings))
            distances = pairwise_distances(
                embeddings[start:end],
                self._centroids,
                metric="sqeuclidean",
                convert_dtype=False,
            )
            labels[start:end] = cp.argmin(distances, axis=1).astype(cp.int32)
            del distances
        return labels

    def _cluster_centers(self) -> "cp.ndarray":
        return self._centroids

    def process_batch(self, tasks: list[FileGroupTask]) -> list[EmptyTask]:
        if len(tasks) == 0:
            return []

        groups = self._group_input_files(tasks)
        _, read_time, total_rows = self._predict_write_pass(tasks, groups)
        self._log_metrics(
            {
                "kmeans_read_time": read_time,
                "kmeans_predict_read_time": read_time,
                "num_rows": total_rows,
            }
        )
        # The stage is a sink: output Parquet files are the materialized result.
        # Emit one completion task per input batch so batch metrics are not
        # duplicated when a batch is split into multiple cuDF-safe groups.
        return [EmptyTask(dataset_name=tasks[0].dataset_name)]

    def ray_stage_spec(self) -> dict[str, Any]:
        """This prediction-only stage does not participate in a RAFT collective."""
        # Input tasks contain only file paths, so spreading actors across nodes
        # improves I/O concurrency without moving materialized data.
        return {
            RayStageSpecKeys.MAX_TASKS_IN_FLIGHT_PER_ACTOR: 1,
            RayStageSpecKeys.RAY_REMOTE_ARGS: {"scheduling_strategy": "SPREAD"},
        }

    def num_workers(self) -> int | None:
        return self.worker_count


@dataclass
class KMeansStage(CompositeStage[EmptyTask, EmptyTask]):
    """KMeans clustering stage that requires RAFT for distributed processing."""

    n_clusters: int
    id_field: str
    embedding_field: str
    input_path: str | list[str]
    output_path: str
    metadata_fields: list[str] | None = None
    verbose: bool = False
    embedding_dim: int | None = None
    # I/O args
    input_filetype: Literal["jsonl", "parquet"] = "parquet"
    input_file_extensions: list[str] | None = None
    read_kwargs: dict[dict] | None = None
    write_kwargs: dict[dict] | None = None
    # KMeans args
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42
    init: Literal["k-means||", "random"] | np.ndarray = "k-means||"
    n_init: int | Literal["auto"] = 1
    oversampling_factor: float = 2.0
    max_samples_per_batch: int = 1 << 15
    fit_data_fraction: float | None = None
    output_embedding_dtype: Literal["float16", "float32"] = "float16"
    write_batch_size: int | Literal["auto"] = "auto"
    max_output_file_size: int | None = None
    prefetch_next_group: bool = False
    cache_path: str | None = None
    """KMeans clustering stage that requires RAFT for distributed processing.

    Args:
        n_clusters (int): The number of clusters to create.
        id_field (str): The column name of the id column.
        embedding_field (str): The column name of the embedding column.
        input_path (str | list[str]): The path to the input directory.
        output_path (str): The path to the output directory.
        metadata_fields (list[str] | None): The columns to keep in the output. These columns can be used later to prioritize deduplication.
        verbose (bool): Whether to print verbose output.
        embedding_dim (int | None): The dimension of the embedding. This helps us read data into smaller chunks.
        input_filetype (Literal["jsonl", "parquet"]): The type of the input file
        read_kwargs (dict[dict]): Keyword arguments for the read stage.
        write_kwargs (dict[dict]): Keyword arguments for the write stage.
        max_iter (int): The maximum number of iterations to run.
        tol (float): Tolerance for stopping criteria of the kmeans algorithm.
        random_state (int): Seed for the random number generator. Unseeded by default. Does not currently fully guarantee the exact same results.
        init (Literal["k-means||", "random"] | np.ndarray): 'scalable-k-means++' or 'k-means||': Uses fast and stable scalable kmeans++ initialization. 'random': Choose 'n_cluster' observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        n_init (int | Literal["auto"]): Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
        oversampling_factor (float): The amount of points to sample in scalable k-means++ initialization for potential centroids. Increasing this value can lead to better initial centroids at the cost of memory. The total number of centroids sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.
        max_samples_per_batch (int): The number of data samples to use for batches of the pairwise distance computation. This computation is done throughout both fit predict. The default should suit most cases. The total number of elements in the batched pairwise distance computation is max_samples_per_batch * n_clusters. It might become necessary to lower this number when n_clusters becomes prohibitively large.
        fit_data_fraction (float | None): Fraction of the dataset (in (0, 1)) used to fit the KMeans model. Pass None to fit on the full dataset (single-pass mode). For Parquet, each actor uses footer row counts to sample whole files until it reaches the requested row fraction. Other formats sample by file count. When set, uses a two-pass approach: Pass 1 reads only the embedding column from the sampled files; Pass 2 loads each full original group one at a time to predict labels and write results. If None, all rows are loaded simultaneously.
        output_embedding_dtype: Storage dtype for normalized embeddings written after KMeans. Defaults to FP16 encoded as uint16 bits.
        write_batch_size: Maximum rows to materialize and write at once. ``"auto"`` sizes each group from the actor GPU's live memory and embedding width.
        max_output_file_size: Approximate maximum uncompressed bytes per centroid Parquet file.
        prefetch_next_group: Read and normalize one group concurrently with prediction and writing.
        cache_path (str | None): The path to save the centroids to. If None, the centroids will not be saved.
    """

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        # Validate eagerly so bad values surface at construction, not later in
        # decompose() / on a worker.
        if self.fit_data_fraction is not None and not 0.0 < self.fit_data_fraction < 1.0:
            msg = f"fit_data_fraction must be in (0, 1), got {self.fit_data_fraction}; pass None to fit on the full dataset"
            raise ValueError(msg)
        if self.output_embedding_dtype not in {"float16", "float32"}:
            msg = f"Unsupported output_embedding_dtype: {self.output_embedding_dtype}"
            raise ValueError(msg)
        if self.write_batch_size != "auto" and (
            not isinstance(self.write_batch_size, int) or self.write_batch_size <= 0
        ):
            msg = f"write_batch_size must be positive, got {self.write_batch_size}"
            raise ValueError(msg)
        if self.max_output_file_size is not None and self.max_output_file_size <= 0:
            msg = f"max_output_file_size must be positive, got {self.max_output_file_size}"
            raise ValueError(msg)

    def decompose(self) -> list[ProcessingStage]:
        # Set default file extensions based on input_filetype if not provided
        file_extensions = self.input_file_extensions or get_default_file_extensions(self.input_filetype)

        return [
            FilePartitioningStage(
                file_paths=self.input_path,
                file_extensions=file_extensions,
                files_per_partition=1,  # We set this to one, and then the RaftActor will break it up into smaller groups
                storage_options=self.read_kwargs.get("storage_options") if self.read_kwargs is not None else None,
            ),
            KMeansReadFitWriteStage(
                id_field=self.id_field,
                embedding_field=self.embedding_field,
                output_path=self.output_path,
                filetype=self.input_filetype,
                n_clusters=self.n_clusters,
                metadata_fields=self.metadata_fields,
                verbose=self.verbose,
                embedding_dim=self.embedding_dim,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                init=self.init,
                n_init=self.n_init,
                oversampling_factor=self.oversampling_factor,
                max_samples_per_batch=self.max_samples_per_batch,
                fit_data_fraction=self.fit_data_fraction,
                output_embedding_dtype=self.output_embedding_dtype,
                write_batch_size=self.write_batch_size,
                max_output_file_size=self.max_output_file_size,
                prefetch_next_group=self.prefetch_next_group,
                read_kwargs=self.read_kwargs,
                write_kwargs=self.write_kwargs,
                cache_path=self.cache_path,
            ),
        ]
