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

"""Quality/diversity fusion sampling, using Curator's existing GPU shuffle."""

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import cudf
import cupy as cp
import numpy as np
import pyarrow as pa

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.semantic.utils import read_parquet_chunks
from nemo_curator.stages.deduplication.shuffle_utils.rapidsmpf_shuffler import pylibcudf_to_cudf_dataframe
from nemo_curator.stages.deduplication.shuffle_utils.stage import ShuffleStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.file_utils import get_all_file_paths_under, get_fs


def plan_sampling_budgets(  # noqa: C901
    group_counts: dict[tuple[str, int], int],
    pre_dedup_topic_counts: dict[str, int],
    target_epochs: float = 2.5,
    topic_multipliers: dict[str, float] | None = None,
    max_repeat: int = 8,
) -> dict[tuple[str, int], int]:
    """Freeze pre-dedup topic proportions, then apportion exact topic/shard budgets.

    Only small group counts reach the driver. Impossible budgets fail explicitly
    instead of silently losing copies or redistributing across topics.
    """
    if not np.isfinite(target_epochs) or target_epochs < 0 or max_repeat <= 0:
        msg = "target_epochs must be finite/non-negative and max_repeat positive"
        raise ValueError(msg)
    if any(n <= 0 for n in group_counts.values()) or any(n < 0 for n in pre_dedup_topic_counts.values()):
        msg = "group counts must be positive and pre-dedup counts non-negative"
        raise ValueError(msg)
    if {topic for topic, _ in group_counts} - pre_dedup_topic_counts.keys():
        msg = "every surviving topic needs a pre-dedup count"
        raise ValueError(msg)
    topics = sorted(pre_dedup_topic_counts)
    weights = np.array([pre_dedup_topic_counts[t] * (topic_multipliers or {}).get(t, 1.0) for t in topics])
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        msg = "topic weights must be finite/non-negative with positive total"
        raise ValueError(msg)

    def apportion(w: np.ndarray, target: int) -> np.ndarray:
        raw = w / w.sum() * target
        counts = np.floor(raw).astype(np.int64)
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[: target - int(counts.sum())]] += 1
        return counts

    total = round(target_epochs * sum(group_counts.values()))
    topic_budgets = apportion(weights, total)
    budgets = {}
    for topic, target in zip(topics, topic_budgets, strict=True):
        groups = sorted(key for key in group_counts if key[0] == topic)
        if not groups:
            if target:
                msg = f"topic {topic!r} has a positive budget but no survivors"
                raise ValueError(msg)
            continue
        sizes = np.array([group_counts[key] for key in groups])
        for key, budget in zip(groups, apportion(sizes, int(target)), strict=True):
            if budget > group_counts[key] * max_repeat:
                msg = f"budget for {key!r} exceeds max_repeat capacity"
                raise ValueError(msg)
            budgets[key] = int(budget)
    return budgets


def _quality_score(values: cp.ndarray, weights: tuple[float, float, float]) -> cp.ndarray:
    # Preserve Forge's operation order: a dot product changes ties for discrete labels.
    return cp.clip(
        weights[0] * values[:, 0] / 5.0 + weights[1] * values[:, 1] / 5.0 + weights[2] * values[:, 2] / 5.0, 0, 1
    )


def fusion_copies(  # noqa: PLR0913
    frame: cudf.DataFrame,
    target: int,
    max_repeat: int = 8,
    temperature: float = 2.0,
    diversity_weight: float = 1.0,
    quality_weights: tuple[float, float, float] = (0.45, 0.30, 0.25),
    gate_threshold: float = 3.0,
) -> cp.ndarray:
    """Forge math-18 fusion formula; frame must already be in deterministic ID order.

    Inputs: ms, rd, cr in [0,5], diversity_score. Population z-scores and average
    percentile ranks are computed within this complete topic/shard group.
    """
    if len(quality_weights) != 3 or not np.isfinite([*quality_weights, gate_threshold]).all():  # noqa: PLR2004
        msg = "quality_weights must contain three finite values and gate_threshold must be finite"
        raise ValueError(msg)
    values = frame[["ms", "rd", "cr", "diversity_score"]].to_cupy(dtype=cp.float64)
    if not cp.isfinite(values).all().item() or ((values[:, :3] < 0) | (values[:, :3] > 5)).any().item():  # noqa: PLR2004
        msg = "quality fields must be finite in [0,5]; diversity must be finite"
        raise ValueError(msg)
    return allocate_copies(
        _quality_score(values, quality_weights),
        values[:, 3],
        target,
        gate=cp.maximum(values[:, 2] - gate_threshold, 0),
        max_repeat=max_repeat,
        temperature=temperature,
        diversity_weight=diversity_weight,
    )


def allocate_copies(  # noqa: C901, PLR0913
    quality: cp.ndarray,
    diversity: cp.ndarray,
    target: int,
    *,
    gate: cp.ndarray | None = None,
    max_repeat: int = 8,
    temperature: float = 2.0,
    diversity_weight: float = 1.0,
) -> cp.ndarray:
    """Allocate an exact repeat budget on GPU from caller-supplied scalar scores.

    Inputs are aligned one-dimensional arrays in deterministic tie-break order.
    Quality may use any finite scale; population z-scoring removes its units.
    Positive standardized diversity boosts quality. An optional nonnegative gate
    controls that boost and is mean-normalized; None enables it for every row.
    No label names, subject domains or document text are required.
    """
    quality = cp.asarray(quality, dtype=cp.float64)
    diversity = cp.asarray(diversity, dtype=cp.float64)
    if quality.ndim != 1 or diversity.shape != quality.shape:
        msg = "quality and diversity must be aligned one-dimensional arrays"
        raise ValueError(msg)
    n = len(quality)
    gate = cp.ones(n, dtype=cp.float64) if gate is None else cp.asarray(gate, dtype=cp.float64)
    if gate.shape != quality.shape:
        msg = "gate must align with quality and diversity"
        raise ValueError(msg)
    if not bool(cp.all(cp.isfinite(quality) & cp.isfinite(diversity) & cp.isfinite(gate) & (gate >= 0))):
        msg = "scores must be finite and gates nonnegative"
        raise ValueError(msg)
    if isinstance(target, bool) or not isinstance(target, int) or not isinstance(max_repeat, int) or max_repeat <= 0:
        msg = "target and max_repeat must be integers, with max_repeat positive"
        raise ValueError(msg)
    if not np.isfinite([temperature, diversity_weight]).all() or temperature < 0 or diversity_weight < 0:
        msg = "temperature and diversity_weight must be finite and non-negative"
        raise ValueError(msg)
    if not 0 <= target <= n * max_repeat:
        msg = "copy budget exceeds group capacity"
        raise ValueError(msg)
    copies = cp.zeros(n, dtype=cp.int64)
    if not n or not target:
        return copies

    def zscore(x: cp.ndarray) -> cp.ndarray:
        std = x.std()
        return (x - x.mean()) / cp.where(std > 0, std, 1.0)

    gate = gate / cp.maximum(gate.mean(), 1e-9)
    composite = zscore(quality) + diversity_weight * gate * cp.maximum(zscore(diversity), 0)
    ranks = cudf.Series(composite).rank(method="average", pct=True).values
    weights = cp.exp((ranks - ranks.max()) * temperature)
    remaining = target
    while remaining:
        active_weights = cp.where(copies < max_repeat, weights, 0)
        weight_sum = active_weights.sum()
        if not (weight_sum > 0).item():
            msg = "sampling weights underflowed before exhausting the budget; reduce temperature"
            raise ValueError(msg)
        raw = active_weights / weight_sum * remaining
        additions = cp.floor(raw).astype(cp.int64)
        remainder = remaining - int(additions.sum().item())
        order = cp.argsort(cp.where(copies < max_repeat, -(raw - additions), cp.inf))
        additions[order[:remainder]] += 1
        copies = cp.minimum(copies + additions, max_repeat)
        remaining = target - int(copies.sum().item())
    return copies


class QualityDiversitySamplingStage(ShuffleStage):
    """Shuffle complete caller-defined groups, then allocate per-document repeat counts.

    Requires unique IDs and prevalidated quality metadata. Input should omit
    embeddings. Unlike Forge's exploded table this writes compact counts first;
    RepeatParquetStage expands them later in bounded GPU batches.
    Caller-defined score columns are preserved; total_copies is replaced.
    Hash shuffle partitions must fit GPU memory on extraction (as in ShuffleStage).
    """

    name = "QualityDiversitySamplingStage"

    def __init__(  # noqa: PLR0913
        self,
        budgets: dict[tuple[Any, ...], int],
        output_path: str,
        id_field: str = "id",
        group_fields: tuple[str, ...] = ("group", "sampling_shard"),
        quality_field: str = "quality_score",
        diversity_field: str = "diversity_score",
        gate_field: str | None = None,
        max_repeat: int = 8,
        output_rows_per_file: int = 65_536,
        temperature: float = 2.0,
        diversity_weight: float = 1.0,
        **shuffle_kwargs,
    ) -> None:
        if isinstance(max_repeat, bool) or not isinstance(max_repeat, int) or max_repeat <= 0:
            msg = "max_repeat must be a positive integer"
            raise ValueError(msg)
        self.budgets = budgets
        self.id_field = id_field
        if not group_fields or len(set(group_fields)) != len(group_fields):
            msg = "group_fields must be nonempty and unique"
            raise ValueError(msg)
        self.group_fields = list(group_fields)
        self.quality_field = quality_field
        self.diversity_field = diversity_field
        self.gate_field = gate_field
        self.max_repeat = max_repeat
        self.fusion_kwargs = {
            "temperature": temperature,
            "diversity_weight": diversity_weight,
        }
        if output_rows_per_file <= 0:
            msg = "output_rows_per_file must be positive"
            raise ValueError(msg)
        self.output_rows_per_file = output_rows_per_file
        super().__init__(shuffle_on=self.group_fields, output_path=output_path, **shuffle_kwargs)

    def sample_partition(self, frame: cudf.DataFrame) -> cudf.DataFrame:
        if frame[self.group_fields].isnull().any().any():
            msg = "sampling topics and shards must not be null"
            raise ValueError(msg)
        results = []
        for raw_key, unsorted_group in frame.groupby(self.group_fields):
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            if key not in self.budgets:
                msg = f"missing sampling budget for {key!r}"
                raise ValueError(msg)
            group = unsorted_group.sort_values(self.id_field).reset_index(drop=True)
            if group[self.id_field].isnull().any() or group[self.id_field].duplicated().any():
                msg = "sampling IDs must be non-null and unique within each group"
                raise ValueError(msg)
            group["total_copies"] = allocate_copies(
                group[self.quality_field].values,
                group[self.diversity_field].values,
                self.budgets[key],
                max_repeat=self.max_repeat,
                gate=group[self.gate_field].values if self.gate_field else None,
                **self.fusion_kwargs,
            )
            results.append(group[group.total_copies > 0])
        if not results:
            frame["total_copies"] = cudf.Series([], dtype="int64")
            return frame
        return cudf.concat(results, ignore_index=True)

    def extract_and_write(self) -> list[FileGroupTask]:
        self._check_actor_obj()
        tasks = []
        write_kwargs = {"index": False, **self.write_kwargs}
        for partition_id, partition in self._actor_obj.extract():
            frame = pylibcudf_to_cudf_dataframe(partition, column_names=self.output_columns)
            frame = self.sample_partition(frame)
            for start in range(0, len(frame), self.output_rows_per_file):
                path = f"{self.output_path}/part.{partition_id}.{start}.parquet"
                frame.iloc[start : start + self.output_rows_per_file].to_parquet(path, **write_kwargs)
                tasks.append(
                    FileGroupTask(data=[path], dataset_name=self.dataset_name, _metadata={"filetype": "parquet"})
                )
        return tasks


class FusionSamplingStage(QualityDiversitySamplingStage):
    """Compatibility adapter for the original math-18 score recipe.

    New pipelines should use QualityDiversitySamplingStage with their own scalar
    quality and optional gate columns. This adapter keeps reference experiments
    separate from the domain-independent GPU allocator.
    """

    name = "FusionSamplingStage"

    def __init__(  # noqa: PLR0913
        self,
        budgets: dict[tuple[str, int], int],
        output_path: str,
        id_field: str = "id",
        topic_field: str = "math_topic",
        quality_weights: tuple[float, float, float] = (0.45, 0.30, 0.25),
        gate_threshold: float = 3.0,
        **kwargs,
    ) -> None:
        if len(quality_weights) != 3 or not np.isfinite([*quality_weights, gate_threshold]).all():  # noqa: PLR2004
            msg = "quality_weights must contain three finite values and gate_threshold must be finite"
            raise ValueError(msg)
        self.quality_weights = quality_weights
        self.gate_threshold = gate_threshold
        super().__init__(
            budgets,
            output_path,
            id_field=id_field,
            group_fields=(topic_field, "sampling_shard"),
            quality_field="q",
            gate_field="_fusion_gate",
            **kwargs,
        )

    def sample_partition(self, frame: cudf.DataFrame) -> cudf.DataFrame:
        frame = frame.copy(deep=False)
        if "_fusion_gate" in frame:
            msg = "_fusion_gate is reserved by the math-18 compatibility adapter"
            raise ValueError(msg)
        values = frame[["ms", "rd", "cr"]].to_cupy(dtype=cp.float64)
        if not bool(cp.all(cp.isfinite(values) & (values >= 0) & (values <= 5))):  # noqa: PLR2004
            msg = "quality fields must be finite in [0,5]"
            raise ValueError(msg)
        frame["q"] = _quality_score(values, self.quality_weights)
        frame["_fusion_gate"] = cp.maximum(frame["cr"].values - self.gate_threshold, 0)
        return super().sample_partition(frame).drop(columns=["_fusion_gate"])


def _repeat_rows(frame: cudf.DataFrame, max_repeat: int) -> cudf.DataFrame:
    """Expand a bounded GPU frame; retain a per-document copy index."""
    if "copy_index" in frame:
        frame = frame.drop(columns=["copy_index"])
    copies = frame["total_copies"]
    if copies.dtype.kind not in "iu" or copies.isnull().any():
        msg = "total_copies must be non-null integers"
        raise ValueError(msg)
    if ((copies < 0) | (copies > max_repeat)).any():
        msg = "total_copies is outside [0, max_repeat]"
        raise ValueError(msg)
    counts = copies.values.astype(cp.int64)
    indices = cp.repeat(cp.arange(len(frame), dtype=cp.int64), counts)
    result = frame.iloc[indices].reset_index(drop=True)
    result["copy_index"] = cp.arange(len(indices), dtype=cp.int64) - cp.repeat(cp.cumsum(counts) - counts, counts)
    return result


@dataclass
class RepeatDocumentsStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Expand counts without Python document loops; preserves (ID, copy_index) identity.

    Configure upstream reader batch size so max_repeat * batch rows fits memory.
    """

    max_repeat: int = 8
    name: str = "RepeatDocumentsStage"

    def __post_init__(self) -> None:
        super().__init__()
        self.resources = Resources(cpus=1.0, gpus=1.0)
        if isinstance(self.max_repeat, bool) or not isinstance(self.max_repeat, int) or self.max_repeat <= 0:
            msg = "max_repeat must be a positive integer"
            raise ValueError(msg)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        # DocumentBatch is CPU-backed; file pipelines avoid this boundary via RepeatParquetStage.
        table = task.to_pyarrow()
        schema = pa.schema([field for field in table.schema if field.name != "copy_index"])
        schema = schema.append(pa.field("copy_index", pa.int64()))
        result = _repeat_rows(cudf.DataFrame.from_arrow(table), self.max_repeat).to_arrow().cast(schema)
        return DocumentBatch(
            data=result, dataset_name=task.dataset_name, _metadata=task._metadata, _stage_perf=task._stage_perf
        )


@dataclass
class RepeatParquetStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Stream counts to repeated Parquet rows without materializing a complete file.

    Uses at most input_batch_rows * max_repeat rows per expansion. This is not
    a byte limit: individual long documents and Parquet pages can exceed it.
    Output files must not exist; failed tasks can leave partial files to inspect.
    Optional counts_path enables a late ID join: the sorted ID/count table must
    fit each GPU, while original text is read in chunks and never shuffled.
    """

    output_path: str
    max_repeat: int = 8
    input_batch_rows: int = 1024
    storage_options: dict[str, Any] | None = None
    read_memory_bytes: int = 256 * 1024**2
    counts_path: str | None = None
    id_field: str = "id"
    columns: list[str] | None = None
    name: str = "RepeatParquetStage"

    def __post_init__(self) -> None:
        super().__init__()
        self._repeat = RepeatDocumentsStage(max_repeat=self.max_repeat)
        self.resources = Resources(cpus=1.0, gpus=1.0)
        if (
            isinstance(self.input_batch_rows, bool)
            or not isinstance(self.input_batch_rows, int)
            or self.input_batch_rows <= 0
        ):
            msg = "input_batch_rows must be a positive integer"
            raise ValueError(msg)
        if self.read_memory_bytes <= 0:
            msg = "read_memory_bytes must be positive"
            raise ValueError(msg)

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self.counts_path is not None:
            files = get_all_file_paths_under(
                self.counts_path,
                recurse_subdirectories=True,
                keep_extensions=[".parquet"],
                storage_options=self.storage_options,
            )
            if not files:
                self._counts = cudf.DataFrame({self.id_field: [], "total_copies": cudf.Series([], dtype="int64")})
                return
            self._counts = (
                cudf.concat(
                    list(
                        read_parquet_chunks(
                            files,
                            [self.id_field, "total_copies"],
                            self.read_memory_bytes,
                            self.storage_options,
                        )
                    ),
                    ignore_index=True,
                )
                .sort_values(self.id_field)
                .reset_index(drop=True)
            )
            if self._counts.isnull().any().any() or self._counts[self.id_field].duplicated().any():
                msg = "late-join counts require unique non-null IDs and non-null counts"
                raise ValueError(msg)
            copies = self._counts["total_copies"]
            if copies.dtype.kind not in "iu" or ((copies < 0) | (copies > self.max_repeat)).any():
                msg = "late-join counts must be integers in [0, max_repeat]"
                raise ValueError(msg)

    def _attach_counts(self, frame: cudf.DataFrame) -> cudf.DataFrame:
        if self.counts_path is None:
            return frame
        if frame[self.id_field].isnull().any():
            msg = "late-join input IDs must be non-null"
            raise ValueError(msg)
        if not len(self._counts):
            frame = frame.iloc[:0].copy()
            frame["total_copies"] = cudf.Series([], dtype="int64")
            return frame
        # Binary lookup reuses the sorted GPU index instead of rebuilding a corpus-wide
        # hash table for every text chunk. Exact string comparison avoids hash collisions.
        positions = self._counts[self.id_field].searchsorted(frame[self.id_field])
        candidates = self._counts.iloc[cp.minimum(positions, len(self._counts) - 1)].reset_index(drop=True)
        frame = frame.reset_index(drop=True)
        matched = candidates[self.id_field] == frame[self.id_field]
        frame["total_copies"] = candidates["total_copies"]
        return frame[matched]

    def process(self, task: FileGroupTask) -> FileGroupTask:
        fs = get_fs(self.output_path, self.storage_options)
        fs.makedirs(self.output_path, exist_ok=True)
        outputs = []
        for index, path in enumerate(task.data):
            output = f"{self.output_path}/{task.task_id}_{index}.parquet"
            if fs.exists(output):
                raise FileExistsError(output)
            with ExitStack() as stack:
                writer = None
                for chunk in read_parquet_chunks([path], self.columns, self.read_memory_bytes, self.storage_options):
                    joined = self._attach_counts(chunk)
                    for start in range(0, len(joined), self.input_batch_rows):
                        expanded = _repeat_rows(joined.iloc[start : start + self.input_batch_rows], self.max_repeat)
                        if not len(expanded):
                            continue
                        if writer is None:
                            sink = stack.enter_context(fs.open(output, "wb"))
                            writer = stack.enter_context(
                                cudf.io.parquet.ParquetWriter(
                                    sink,
                                    compression="zstd",
                                    index=False,
                                    row_group_size_rows=self.input_batch_rows * self.max_repeat,
                                )
                            )
                        writer.write_table(expanded)
                if writer is not None:
                    outputs.append(output)
        return FileGroupTask(
            data=outputs, dataset_name=task.dataset_name, _metadata=task._metadata, _stage_perf=task._stage_perf
        )
