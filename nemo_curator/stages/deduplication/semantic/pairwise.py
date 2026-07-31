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

import gc
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import cudf
import cupy as cp
import numpy as np
import torch
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask, FileGroupTask
from nemo_curator.utils.file_utils import check_disallowed_kwargs

from .pairwise_io import ClusterWiseFilePartitioningStage
from .ranking import RankingStrategy
from .utils import EmbeddingStorageDtype, break_parquet_partition_into_groups, decode_embedding_array

PairwiseBatchSize = int | Literal["auto"]
_AUTO_BATCH_MEMORY_FRACTION = 0.8
_AUTO_BATCH_OVERHEAD_FACTOR = 1.25
_AUTO_BATCH_ALIGNMENT = 256
_MAX_ADDITIONAL_NEIGHBORS = 5


def _release_cached_gpu_memory() -> None:
    """Return unused allocator blocks so reused actors do not retain a previous cluster's peak."""
    gc.collect()
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()


def _resolve_pairwise_batch_size(cluster_reps: "torch.Tensor", batch_size: PairwiseBatchSize) -> int:
    if batch_size != "auto":
        if not isinstance(batch_size, int) or batch_size <= 0:
            msg = f"pairwise_batch_size must be a positive integer or 'auto', got {batch_size!r}"
            raise ValueError(msg)
        return min(batch_size, cluster_reps.shape[0])

    free_memory, _ = torch.cuda.mem_get_info(cluster_reps.device)
    bytes_per_column = cluster_reps.shape[0] * cluster_reps.element_size()
    memory_budget = int(free_memory * _AUTO_BATCH_MEMORY_FRACTION)
    max_batch_size = int(memory_budget / (bytes_per_column * _AUTO_BATCH_OVERHEAD_FACTOR))
    if max_batch_size <= 0:
        msg = f"Insufficient free GPU memory for one Pairwise similarity column: {free_memory=} {bytes_per_column=}"
        raise MemoryError(msg)
    if max_batch_size >= _AUTO_BATCH_ALIGNMENT:
        max_batch_size = max_batch_size // _AUTO_BATCH_ALIGNMENT * _AUTO_BATCH_ALIGNMENT
    return min(max_batch_size, cluster_reps.shape[0])


@dataclass
class _PairwiseSimilarityResult:
    max_similarity: "cp.ndarray"
    max_indices: "cp.ndarray"
    top_similarity: "cp.ndarray | None" = None
    top_indices: "cp.ndarray | None" = None


def _pairwise_cosine_similarity_batched(
    cluster_reps: "torch.Tensor",
    batch_size: int = 1024,
    num_additional_neighbors: int = 0,
) -> _PairwiseSimilarityResult:
    """
    Computes pairwise cosine similarity between cluster items,
    considering only earlier rows as candidate neighbors.
    This function is useful for large clusters where the pairwise similarity matrix
    does not fit into memory.
    We use a batched approach to compute the pairwise similarity matrix in batches.
    Memory requirements are O(N*B) where N is the number of items in the cluster and B is the batch size
    instead of O(N^2) for the full matrix.

    TODO: In future we can estimate memory requirement and calculate batch size dynamically.
    """
    device = "cuda"

    cluster_reps = cluster_reps.to(device)
    max_similarity = torch.zeros(cluster_reps.shape[0], dtype=torch.float32, device=device)
    max_indices = torch.zeros(cluster_reps.shape[0], dtype=torch.int64, device=device)
    top_width = num_additional_neighbors + 1 if num_additional_neighbors else 0
    top_similarity = (
        torch.full((cluster_reps.shape[0], top_width), -torch.inf, dtype=torch.float32, device=device)
        if top_width
        else None
    )
    top_indices = (
        torch.zeros((cluster_reps.shape[0], top_width), dtype=torch.int64, device=device) if top_width else None
    )
    workspace_width = min(batch_size, cluster_reps.shape[0])
    pairwise_sim_workspace = torch.empty(
        (cluster_reps.shape[0], workspace_width),
        dtype=cluster_reps.dtype,
        device=device,
    )
    batch_workspace = torch.empty(
        (workspace_width, cluster_reps.shape[1]),
        dtype=cluster_reps.dtype,
        device=device,
    )
    invalid_batch_triangle = torch.ones(
        (workspace_width, workspace_width),
        dtype=torch.bool,
        device=device,
    ).tril_()
    for start_idx in range(0, cluster_reps.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, cluster_reps.shape[0])
        batch = cluster_reps[start_idx:end_idx]
        # Rows at or beyond end_idx cannot precede any query in this batch.
        # Restricting the candidate prefix preserves the exact result while
        # avoiding nearly half of the dense dot products over the full loop.
        candidate_reps = cluster_reps[:end_idx]
        batch_width = end_idx - start_idx
        batch_workspace[:batch_width].copy_(batch)
        if batch_width < workspace_width:
            batch_workspace[batch_width:].zero_()
        torch.mm(candidate_reps, batch_workspace.T, out=pairwise_sim_workspace[:end_idx])
        pairwise_sim_matrix = pairwise_sim_workspace[:end_idx, :batch_width]
        pairwise_sim_matrix[start_idx:end_idx].masked_fill_(
            invalid_batch_triangle[:batch_width, :batch_width],
            -torch.inf,
        )
        max_values_and_indices = torch.max(pairwise_sim_matrix, dim=0)
        if start_idx == 0:
            max_values_and_indices.values[0] = 0.0
            max_values_and_indices.indices[0] = 0
        max_similarity[start_idx:end_idx] = max_values_and_indices[0]
        max_indices[start_idx:end_idx] = max_values_and_indices[1]
        if top_width:
            batch_top_width = min(top_width, end_idx)
            batch_columns = torch.arange(end_idx - start_idx, device=device)
            selected = max_values_and_indices
            for neighbor_idx in range(batch_top_width):
                top_similarity[start_idx:end_idx, neighbor_idx] = selected.values
                top_indices[start_idx:end_idx, neighbor_idx] = selected.indices
                pairwise_sim_matrix[selected.indices, batch_columns] = -torch.inf
                if neighbor_idx + 1 < batch_top_width:
                    selected = torch.max(pairwise_sim_matrix, dim=0)
        del batch, candidate_reps, pairwise_sim_matrix

    return _PairwiseSimilarityResult(
        max_similarity=cp.asarray(max_similarity),
        max_indices=cp.asarray(max_indices),
        top_similarity=cp.asarray(top_similarity) if top_similarity is not None else None,
        top_indices=cp.asarray(top_indices) if top_indices is not None else None,
    )


def pairwise_cosine_similarity_batched(
    cluster_reps: "torch.Tensor",
    batch_size: int = 1024,
) -> tuple["cp.ndarray", "cp.ndarray"] | tuple[np.ndarray, np.ndarray]:
    """Return each row's most similar earlier row."""
    result = _pairwise_cosine_similarity_batched(cluster_reps, batch_size)
    return result.max_similarity, result.max_indices


class _PairwiseProfiler:
    """Synchronize and record opt-in Pairwise phase metrics."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.metrics: dict[str, float] = {}

    def start(self) -> float:
        if self.enabled:
            torch.cuda.synchronize()
        return time.perf_counter()

    def end(self, name: str, started: float) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize()
        self.metrics[f"pairwise_{name}_time"] = time.perf_counter() - started
        free_memory, total_memory = torch.cuda.mem_get_info()
        self.metrics[f"pairwise_gpu_used_after_{name}_bytes"] = total_memory - free_memory


def _build_additional_neighbors(  # noqa: PLR0913
    ranked_metadata_df: "cudf.DataFrame",
    id_field: str,
    ranking_columns: list[str],
    max_indices: "cp.ndarray",
    top_similarity: "cp.ndarray",
    top_indices: "cp.ndarray",
    num_additional_neighbors: int,
) -> "cudf.DataFrame":
    num_rows, top_width = top_similarity.shape
    query_positions = cp.repeat(cp.arange(num_rows, dtype=cp.int64), top_width)
    candidate_positions = top_indices.reshape(-1)
    candidate_scores = top_similarity.reshape(-1)
    max_positions = cp.repeat(max_indices, top_width)
    valid = cp.isfinite(candidate_scores) & (candidate_positions != max_positions)
    valid_2d = valid.reshape(num_rows, top_width)
    neighbor_ranks = cp.cumsum(valid_2d, axis=1)
    valid &= neighbor_ranks.reshape(-1) <= num_additional_neighbors
    query_positions = query_positions[valid]
    candidate_positions = candidate_positions[valid]
    candidate_scores = candidate_scores[valid]
    neighbor_ranks = neighbor_ranks.reshape(-1)[valid] + 1

    ids = ranked_metadata_df[id_field]
    additional_neighbors_df = cudf.DataFrame(
        {
            "id": ids.iloc[query_positions].reset_index(drop=True),
            "other_id": ids.iloc[candidate_positions].reset_index(drop=True),
            "other_cosine_sim_score": candidate_scores,
            "other_neighbor_rank": neighbor_ranks.astype(cp.int8),
        }
    )
    for column in ranking_columns:
        additional_neighbors_df[f"other_{column}"] = (
            ranked_metadata_df[column].iloc[candidate_positions].reset_index(drop=True)
        )
    return additional_neighbors_df


class PairwiseCosineSimilarityStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    """Pairwise cosine similarity stage that computes similarity within clusters."""

    def __init__(  # noqa: PLR0913
        self,
        id_field: str,
        embedding_field: str,
        output_path: str,
        ranking_strategy: RankingStrategy,
        pairwise_batch_size: PairwiseBatchSize = 1024,
        verbose: bool = False,
        embedding_dim: int | None = None,
        input_embedding_dtype: EmbeddingStorageDtype = "auto",
        num_additional_neighbors: int = 0,
        profile: bool = False,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the pairwise cosine similarity stage.

        Args:
            id_field: The column name of the id column.
            embedding_field: The column name of the embedding column.
            output_path: The path to the output directory.
            ranking_strategy: Strategy for ranking/sorting clusters before similarity computation.
            pairwise_batch_size: Batch size for pairwise similarity computation, or ``"auto"`` to use 80% of the
                free GPU memory with an overhead allowance.
            verbose: Whether to print verbose output.
            embedding_dim: Embedding dimension for memory estimation.
            input_embedding_dtype: Storage dtype of the embedding list. ``auto`` detects uint16 FP16 bit storage.
            num_additional_neighbors: Number of earlier-ranked neighbors, excluding the maximum, to write to an
                analytical sidecar. Must be between 0 and 5.
            profile: Synchronize GPU work and report detailed phase timings and memory snapshots.
            read_kwargs: Kwargs for reading parquet files.
            write_kwargs: Kwargs for writing parquet files.
        """
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.output_path = output_path
        self.pairwise_batch_size = pairwise_batch_size
        self.embedding_dim = embedding_dim
        if input_embedding_dtype not in {"auto", "float16", "float32"}:
            msg = f"Unsupported input_embedding_dtype: {input_embedding_dtype}"
            raise ValueError(msg)
        self.input_embedding_dtype = input_embedding_dtype
        if not 0 <= num_additional_neighbors <= _MAX_ADDITIONAL_NEIGHBORS:
            msg = (
                f"num_additional_neighbors must be between 0 and {_MAX_ADDITIONAL_NEIGHBORS}, "
                f"got {num_additional_neighbors}"
            )
            raise ValueError(msg)
        self.num_additional_neighbors = num_additional_neighbors
        if pairwise_batch_size != "auto" and (not isinstance(pairwise_batch_size, int) or pairwise_batch_size <= 0):
            msg = f"pairwise_batch_size must be a positive integer or 'auto', got {pairwise_batch_size!r}"
            raise ValueError(msg)
        self.profile = profile
        self.ranking_strategy = ranking_strategy
        self.verbose = verbose
        self.read_kwargs = read_kwargs.copy() if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs is not None else {}
        check_disallowed_kwargs(self.read_kwargs, ["columns", "assign_id"])
        check_disallowed_kwargs(self.write_kwargs, ["index"])
        self.input_storage_options = self.read_kwargs.pop("storage_options", None) if self.read_kwargs else None
        self.output_storage_options = self.write_kwargs.pop("storage_options", None) if self.write_kwargs else None
        self.name = "PairwiseCosineSimilarityStage"
        self.resources = Resources(cpus=1.0, gpus=1.0)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """Process one cluster and release cached allocations before this actor accepts another."""
        try:
            return self._process(task)
        finally:
            _release_cached_gpu_memory()

    def _process(self, task: FileGroupTask) -> FileGroupTask:  # noqa: C901, PLR0915
        """Process a PairwiseFileGroupTask to compute pairwise similarities."""
        if task._metadata.get("filetype") != "parquet":
            msg = f"PairwiseCosineSimilarityStage only supports parquet files, got {task._metadata.get('filetype')}"
            raise ValueError(msg)

        cluster_id = task._metadata.get("centroid_id")
        output_path = os.path.join(self.output_path, f"cluster_{cluster_id}.parquet")
        if cluster_id is None:
            msg = "centroid_id not found in task metadata"
            raise ValueError(msg)

        t1 = time.perf_counter()
        profiler = _PairwiseProfiler(self.profile)

        # Read all file groups and concatenate
        dfs = []
        num_rows = 0

        # Break input files into groups to avoid the 2bn list-child-element limit.
        # TODO: Split an individually oversized Parquet file at row-group boundaries;
        # file-level grouping cannot make that case safe.
        phase_started = profiler.start()
        file_groups = break_parquet_partition_into_groups(
            task.data, embedding_dim=self.embedding_dim, storage_options=self.input_storage_options
        )
        profiler.end("footer_grouping", phase_started)

        # Determine which columns to read based on ranking strategy
        additional_cols = self.ranking_strategy.metadata_cols if self.ranking_strategy.strategy == "sort" else []

        # We do the list(dict.fromkeys(...)) to remove duplicates from the list of columns to read, in case additional_cols contains self.id_field
        metadata_cols = list(dict.fromkeys([self.id_field, *additional_cols]))
        phase_started = profiler.start()
        for file_group in file_groups:
            # Read required columns including metadata columns for ranking
            df = self.read_parquet(
                file_group,
                columns=[*metadata_cols, self.embedding_field],
                assign_id=False,
                storage_options=self.input_storage_options,
                **self.read_kwargs,
            )
            dfs.append(df)
            num_rows += len(df)
        profiler.end("read", phase_started)

        if not dfs:
            logger.warning(f"No data found for cluster {cluster_id}")
            return FileGroupTask(
                dataset_name=task.dataset_name,
                _metadata=task._metadata,
                _stage_perf=task._stage_perf,
                data=[],
            )

        num_rows = sum(len(df) for df in dfs)

        # Handle single item clusters
        if num_rows == 1:
            result_df = cudf.DataFrame(
                {
                    "id": dfs[0][self.id_field],
                    "max_id": dfs[0][self.id_field],
                    "cosine_sim_score": cudf.Series([0], dtype="float32"),
                }
            )
            self.write_parquet(
                result_df, output_path, storage_options=self.output_storage_options, index=False, **self.write_kwargs
            )
            return FileGroupTask(
                dataset_name=task.dataset_name,
                _metadata={
                    **task._metadata,
                    "centroid_id": cluster_id,
                },
                _stage_perf=task._stage_perf,
                data=[os.path.join(self.output_path, f"cluster_{cluster_id}.parquet")],
            )

        # Cannot concatenate dataframes with embeddings due to cudf 2bn row limit
        # Instead, concatenate metadata columns and handle embeddings separately
        metadata_dfs, embedding_arrays = [], []
        phase_started = profiler.start()
        for df in dfs:
            metadata_dfs.append(df[metadata_cols])
            embedding_arrays.append(decode_embedding_array(df, self.embedding_field, self.input_embedding_dtype))
        profiler.end("decode", phase_started)

        phase_started = profiler.start()
        metadata_cluster_df = cudf.concat(metadata_dfs, ignore_index=True).reset_index(drop=True)
        profiler.end("metadata_concat", phase_started)

        # Add original index to track reordering
        metadata_cluster_df["_original_idx"] = metadata_cluster_df.index

        phase_started = profiler.start()
        ranked_metadata_df = self.ranking_strategy.rank_cluster(metadata_cluster_df)
        profiler.end("rank", phase_started)
        # Get reorder indices from the ranked dataframe (TODO: we get it to CPU, but maybe we can do it on GPU todo)
        phase_started = profiler.start()
        reorder_indices = ranked_metadata_df["_original_idx"].to_arrow().to_pylist()
        profiler.end("reorder_indices_to_host", phase_started)
        # Remove the helper column
        ranked_metadata_df = ranked_metadata_df.drop(columns=["_original_idx"])

        # Convert numpy arrays to torch tensors before concatenating
        phase_started = profiler.start()
        concatenated_embeddings = torch.cat([torch.as_tensor(arr, device="cuda") for arr in embedding_arrays], dim=0)
        profiler.end("embedding_concat", phase_started)
        phase_started = profiler.start()
        cluster_embeddings = concatenated_embeddings[reorder_indices]
        profiler.end("embedding_reorder", phase_started)

        ids = ranked_metadata_df[self.id_field]

        # Compute pairwise similarities
        free_memory_before_similarity, _ = torch.cuda.mem_get_info(cluster_embeddings.device)
        resolved_batch_size = _resolve_pairwise_batch_size(cluster_embeddings, self.pairwise_batch_size)
        if self.pairwise_batch_size == "auto":
            logger.info(
                f"Auto-selected Pairwise batch size {resolved_batch_size} for cluster {cluster_id} "
                f"with {num_rows} rows and {free_memory_before_similarity} free GPU bytes"
            )
        phase_started = profiler.start()
        similarity_result = _pairwise_cosine_similarity_batched(
            cluster_embeddings,
            resolved_batch_size,
            self.num_additional_neighbors,
        )
        profiler.end("similarity", phase_started)
        # The dense O(N*B) Torch workspace is dead after the helper returns, but
        # Torch's caching allocator otherwise keeps it reserved while cuDF/RMM
        # materializes neighbor metadata and encodes Parquet.
        torch.cuda.empty_cache()
        max_similarity = similarity_result.max_similarity
        max_indices = similarity_result.max_indices

        # Convert indices back to IDs
        phase_started = profiler.start()
        max_indices_id = ids.iloc[max_indices].reset_index(drop=True)

        # Create result dataframe
        points_to_remove_df = cudf.DataFrame(
            {
                "id": ids,
                "max_id": max_indices_id,
                "cosine_sim_score": max_similarity,
            }
        )
        profiler.end("result_build", phase_started)

        neighbor_output_path = None
        additional_neighbor_count = 0
        if self.num_additional_neighbors:
            phase_started = profiler.start()
            neighbor_output_path = os.path.join(self.output_path, f"cluster_{cluster_id}_neighbors.parquet")
            if similarity_result.top_similarity is None or similarity_result.top_indices is None:
                msg = "Additional-neighbor tensors were not produced"
                raise RuntimeError(msg)
            ranking_columns = [column for column in self.ranking_strategy.metadata_cols if column != self.id_field]
            additional_neighbors_df = _build_additional_neighbors(
                ranked_metadata_df,
                self.id_field,
                ranking_columns,
                max_indices,
                similarity_result.top_similarity,
                similarity_result.top_indices,
                self.num_additional_neighbors,
            )
            additional_neighbor_count = len(additional_neighbors_df)
            profiler.end("additional_neighbors_build", phase_started)

        # Write results
        phase_started = profiler.start()
        self.write_parquet(
            points_to_remove_df,
            output_path,
            storage_options=self.output_storage_options,
            index=False,
            **self.write_kwargs,
        )
        if neighbor_output_path is not None:
            self.write_parquet(
                additional_neighbors_df,
                neighbor_output_path,
                storage_options=self.output_storage_options,
                index=False,
                **self.write_kwargs,
            )
        profiler.end("write", phase_started)

        t2 = time.perf_counter()
        self._log_metrics(
            {
                "pairwise_total_time": t2 - t1,
                "pairwise_num_rows": num_rows,
                "pairwise_input_file_count": len(task.data),
                "pairwise_file_group_count": len(file_groups),
                "pairwise_batch_size": resolved_batch_size,
                "pairwise_batch_size_auto": float(self.pairwise_batch_size == "auto"),
                "pairwise_gpu_free_before_similarity_bytes": free_memory_before_similarity,
                "pairwise_num_additional_neighbors": self.num_additional_neighbors,
                "pairwise_additional_neighbor_count": additional_neighbor_count,
                **profiler.metrics,
            }
        )
        if self.verbose:
            logger.debug(
                f"Pairwise computation for cluster {cluster_id} with {num_rows} rows done in {(t2 - t1):.2f} seconds"
            )

        return FileGroupTask(
            dataset_name=task.dataset_name,
            _metadata={
                **task._metadata,
                "centroid_id": cluster_id,
                "neighbor_output_path": neighbor_output_path,
            },
            _stage_perf=task._stage_perf,
            data=[output_path],
        )


@dataclass
class PairwiseStage(CompositeStage[EmptyTask, FileGroupTask]):
    """Pairwise similarity stage for semantic deduplication."""

    # Required parameters
    id_field: str
    embedding_field: str
    input_path: str  # Path to kmeans output
    output_path: str
    # Ranking strategy
    ranking_strategy: RankingStrategy | None = None

    # Optional parameters
    embedding_dim: int | None = None
    input_embedding_dtype: EmbeddingStorageDtype = "auto"
    num_additional_neighbors: int = 0
    profile: bool = False
    pairwise_batch_size: PairwiseBatchSize = 1024
    verbose: bool = False
    read_kwargs: dict[str, Any] | None = None
    write_kwargs: dict[str, Any] | None = None
    # Ranking (for backward compatibility)
    which_to_keep: Literal["hard", "easy", "random"] = "hard"
    sim_metric: Literal["cosine", "l2"] = "cosine"
    random_seed: int = 42

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        if self.input_embedding_dtype not in {"auto", "float16", "float32"}:
            msg = f"Unsupported input_embedding_dtype: {self.input_embedding_dtype}"
            raise ValueError(msg)
        if self.ranking_strategy is None:
            if self.which_to_keep == "random":
                self.ranking_strategy = RankingStrategy(
                    metadata_cols=[], strategy="random", random_seed=self.random_seed
                )
            else:
                if self.sim_metric not in {"cosine", "l2"}:
                    msg = f"Invalid similarity metric: {self.sim_metric}. Only 'cosine' and 'l2' are supported."
                    raise ValueError(msg)
                if self.which_to_keep not in {"hard", "easy"}:
                    msg = f"Invalid which_to_keep value: {self.which_to_keep}. Supported: 'hard', 'easy', 'random'"
                    raise ValueError(msg)
                distance_col = "cosine_dist_to_cent" if self.sim_metric == "cosine" else "l2_dist_to_cent"
                # Determine sort order for ranking within cluster:
                # - "hard": Keep outliers farthest from centroid (descending distance, i.e., ascending=False)
                # - "easy": Keep representatives closest to centroid (ascending distance, i.e., ascending=True)
                # - "random": Handled above, not used here
                ascending = False if self.which_to_keep == "hard" else True  # noqa: SIM211

                # For distance-based ranking, explicitly add ID column as tie-breaker to maintain
                # compatibility with original semantic deduplication behavior
                self.ranking_strategy = RankingStrategy(
                    metadata_cols=[distance_col, self.id_field],
                    ascending=[ascending, ascending],  # Same sort order for both distance and ID
                )

    def decompose(self) -> list[ProcessingStage]:
        return [
            ClusterWiseFilePartitioningStage(
                input_path=self.input_path,
                storage_options=self.read_kwargs.get("storage_options") if self.read_kwargs else None,
            ),
            PairwiseCosineSimilarityStage(
                id_field=self.id_field,
                embedding_field=self.embedding_field,
                output_path=self.output_path,
                pairwise_batch_size=self.pairwise_batch_size,
                verbose=self.verbose,
                ranking_strategy=self.ranking_strategy,
                embedding_dim=self.embedding_dim,
                input_embedding_dtype=self.input_embedding_dtype,
                num_additional_neighbors=self.num_additional_neighbors,
                profile=self.profile,
                read_kwargs=self.read_kwargs,
                write_kwargs=self.write_kwargs,
            ),
        ]
