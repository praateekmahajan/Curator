# modality: text
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

import runpy
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

with suppress(ImportError):
    import cudf

    from nemo_curator.stages.deduplication.semantic.sampling import (
        RepeatDocumentsStage,
        RepeatParquetStage,
        fusion_copies,
        plan_sampling_budgets,
    )
    from nemo_curator.tasks import DocumentBatch, FileGroupTask


@pytest.mark.gpu
def test_budget_and_repeat_contract() -> None:
    budgets = plan_sampling_budgets({("a", 0): 3, ("a", 1): 2, ("b", 0): 5}, {"a": 80, "b": 20})
    assert budgets == {("a", 0): 12, ("a", 1): 8, ("b", 0): 5}
    with pytest.raises(ValueError, match="capacity"):
        plan_sampling_budgets({("a", 0): 1, ("b", 0): 99}, {"a": 99, "b": 1})
    with pytest.raises(ValueError, match="no survivors"):
        plan_sampling_budgets({("a", 0): 10}, {"a": 10, "b": 10})
    task = DocumentBatch(
        data=pa.table({"id": ["a", "b", "c"], "total_copies": [2, 0, 3], "copy_index": [-99, -99, -99]}),
        dataset_name="math",
    )
    output = RepeatDocumentsStage().process(task).to_pandas()
    assert output.id.tolist() == ["a", "a", "c", "c", "c"]
    assert output.copy_index.tolist() == [0, 1, 0, 1, 2]


@pytest.mark.gpu
@pytest.mark.parametrize("batch_rows", [1, 3])
@pytest.mark.parametrize("copies", [[2, 0, 3, 8, 1], [0, 0, 0, 0, 0], []])
def test_streamed_repeat_matches_batch(tmp_path: Path, batch_rows: int, copies: list[int]) -> None:
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "id": pa.array(range(len(copies)), type=pa.int64()),
            "text": pa.array([None if i == 1 else f"document {i}" for i in range(len(copies))], type=pa.string()),
            "total_copies": pa.array(copies, type=pa.int64()),
            "copy_index": pa.array([-99] * len(copies), type=pa.int64()),
        }
    )
    source = tmp_path / "counts.parquet"
    pq.write_table(table, source, row_group_size=2)
    task = FileGroupTask(data=[str(source)], dataset_name="math", _metadata={"source": "preserved"})
    stage = RepeatParquetStage(output_path=str(tmp_path / "repeated"), input_batch_rows=batch_rows)
    result = stage.process(task)
    assert result._metadata == task._metadata
    if not sum(copies):
        assert result.data == []
        return
    expected = RepeatDocumentsStage().process(DocumentBatch(data=table, dataset_name="math")).to_pyarrow()
    assert pq.read_table(result.data[0]).equals(expected)
    footer = pq.read_metadata(result.data[0])
    assert all(footer.row_group(i).num_rows <= batch_rows * 8 for i in range(footer.num_row_groups))
    with pytest.raises(FileExistsError):
        stage.process(task)


@pytest.mark.gpu
def test_late_join_real_payload_and_repeat(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    counts = tmp_path / "counts"
    counts.mkdir()
    pq.write_table(pa.table({"id": ["c", "a"], "total_copies": [3, 2]}), counts / "part.parquet")
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"id": ["a", "b", "c", "z"], "text": ["é", "unused", "中文", "unused"]}), source)
    stage = RepeatParquetStage(str(tmp_path / "out"), counts_path=str(counts), columns=["id", "text"])
    stage.setup()
    result = stage.process(FileGroupTask(data=[str(source)], dataset_name="generic"))
    output = pq.read_table(result.data[0]).to_pydict()
    assert output == {
        "id": ["a", "a", "c", "c", "c"],
        "text": ["é", "é", "中文", "中文", "中文"],
        "total_copies": [2, 2, 3, 3, 3],
        "copy_index": [0, 1, 0, 1, 2],
    }


@pytest.mark.gpu
def test_late_join_counts_exceed_open_file_limit(tmp_path: Path) -> None:
    import resource

    import pyarrow.parquet as pq

    counts = tmp_path / "counts"
    counts.mkdir()
    for index in range(1100):
        pq.write_table(pa.table({"id": [index], "total_copies": [2]}), counts / f"{index}.parquet")
    stage = RepeatParquetStage(str(tmp_path / "out"), counts_path=str(counts))
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, limits[1]))
        stage.setup()
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)
    joined = stage._attach_counts(cudf.DataFrame({"id": [1099, 0, 1100]}))
    assert joined.to_pandas().to_dict("list") == {"id": [1099, 0], "total_copies": [2, 2]}


@pytest.mark.gpu
def test_empty_and_invalid_late_join_counts(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    counts = tmp_path / "counts"
    counts.mkdir()
    stage = RepeatParquetStage(str(tmp_path / "out"), counts_path=str(counts))
    stage.setup()
    assert len(stage._attach_counts(cudf.DataFrame({"id": ["a"], "text": ["original"]}))) == 0
    pq.write_table(pa.table({"id": ["a"], "total_copies": [-1]}), counts / "part.parquet")
    with pytest.raises(ValueError, match="integers"):
        stage.setup()


@pytest.mark.gpu
@pytest.mark.parametrize("target", [0, 1, 150, 750, 800])
def test_fusion_matches_numpy(target: int) -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.random((100, 4)), columns=["ms", "rd", "cr", "diversity_score"])
    frame[["ms", "rd", "cr"]] *= 5
    q = frame[["ms", "rd", "cr"]].to_numpy() @ np.array([0.45, 0.30, 0.25]) / 5
    gate = np.maximum(frame.cr.to_numpy() - 3, 0)
    gate /= max(gate.mean(), 1e-9)
    div = frame.diversity_score.to_numpy()
    composite = (q - q.mean()) / q.std() + gate * np.maximum((div - div.mean()) / div.std(), 0)
    ranks = pd.Series(composite).rank(pct=True).to_numpy()
    weights = np.exp((ranks - ranks.max()) * 2)
    expected = np.zeros(100, dtype=np.int64)
    while expected.sum() < target:
        w = np.where(expected < 8, weights, 0)
        raw = w / w.sum() * (target - expected.sum())
        add = np.floor(raw).astype(np.int64)
        remainder = int(target - expected.sum() - add.sum())
        order = np.argsort(np.where(expected < 8, -(raw - add), np.inf), kind="stable")
        add[order[:remainder]] += 1
        expected = np.minimum(expected + add, 8)
    np.testing.assert_array_equal(fusion_copies(cudf.from_pandas(frame), target).get(), expected)


@pytest.mark.gpu
def test_generic_gpu_sampling_custom_schema(tmp_path: Path) -> None:
    from nemo_curator.stages.deduplication.semantic.sampling import QualityDiversitySamplingStage, allocate_copies

    frame = cudf.DataFrame(
        {
            "document_key": np.arange(20),
            "source_bucket": ["a"] * 10 + ["b"] * 10,
            "education": np.linspace(0, 5, 20),
            "novelty": np.linspace(1, 0, 20),
            "text": [f"original {i}" for i in range(20)],
        }
    )
    stage = QualityDiversitySamplingStage(
        budgets={("a",): 15, ("b",): 20},
        output_path=str(tmp_path),
        id_field="document_key",
        group_fields=("source_bucket",),
        quality_field="education",
        diversity_field="novelty",
    )
    result = stage.sample_partition(frame)
    assert result.groupby("source_bucket").total_copies.sum().to_pandas().to_dict() == {"a": 15, "b": 20}
    assert set(result.columns) == {*frame.columns, "total_copies"}
    for key, target in (("a", 15), ("b", 20)):
        group = frame[frame.source_bucket == key]
        expected = allocate_copies(group.education.values, group.novelty.values, target)
        actual = result[result.source_bucket == key].sort_values("document_key")
        np.testing.assert_array_equal(actual.total_copies.to_numpy(), expected[expected > 0].get())
    assert not result.document_key.duplicated().any()


@pytest.mark.gpu
def test_generic_gpu_scores_affine_invariance_and_validation() -> None:
    import cupy as cp

    from nemo_curator.stages.deduplication.semantic.sampling import allocate_copies

    quality = cp.arange(16, dtype=cp.float64)
    diversity = cp.sin(quality)
    expected = allocate_copies(quality, diversity, 35)
    cp.testing.assert_array_equal(expected, allocate_copies(quality * 7 + 123, diversity, 35))
    with pytest.raises(ValueError, match="aligned"):
        allocate_copies(quality, diversity[:2], 10)
    with pytest.raises(ValueError, match="nonnegative"):
        allocate_copies(quality, diversity, 10, gate=-cp.ones(16))


@pytest.mark.gpu
@pytest.mark.parametrize("sampling_memory_limit", [None, "128MB"])
def test_math18_end_to_end(tmp_path: Path, sampling_memory_limit: str | None) -> None:
    from nemo_curator.stages.deduplication.semantic.weighted import WeightedPruningConfig

    if sampling_memory_limit is not None:
        pytest.importorskip("duckdb")
    example = Path(__file__).resolve().parents[4] / "examples/text/semdedup_math18.py"
    run_math18 = runpy.run_path(str(example))["run_math18"]

    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 16)).astype(np.float32)
    x[32:] = x[:32]  # Exact pairs must be pruned, regardless of coarse assignments.
    source = pd.DataFrame(
        {
            "id": np.arange(len(x)),
            "embeddings": list(x),
            "text": [f"document {i}" for i in range(len(x))],
            "source": ["original-source"] * len(x),
            "math_topic": ["contest_problem_solving", "general_mixed"] * 32,
            "ms": rng.uniform(0, 5, len(x)),
            "rd": rng.uniform(0, 5, len(x)),
            "cr": rng.uniform(0, 5, len(x)),
        }
    )
    input_path = tmp_path / "input"
    input_path.mkdir()
    source.to_parquet(input_path / "part.parquet", index=False)
    output = run_math18(
        str(input_path),
        str(tmp_path / "run"),
        {"contest_problem_solving": 32, "general_mixed": 32},
        coarse_clusters=3,
        fine_clusters=4,
        sampling_shards=2,
        sampling_memory_limit=sampling_memory_limit,
        max_repeat=5,
        weighted_pruning=WeightedPruningConfig(spill_embeddings=sampling_memory_limit is not None),
    )
    result = pd.read_parquet(output)
    assert len(result) == 80
    assert result.groupby("math_topic").size().to_dict() == {"contest_problem_solving": 63, "general_mixed": 17}
    assert result.groupby("id").size().max() <= 5
    assert not result.duplicated(["id", "copy_index"]).any()
    assert "embeddings" not in result
    assert "diversity_score" in result
    assert (result.source == "original-source").all()
    np.testing.assert_array_equal(
        result.q, np.clip(0.45 * result.ms / 5 + 0.30 * result.rd / 5 + 0.25 * result.cr / 5, 0, 1)
    )
    assert result.id.min() >= 32  # Weighted removal visits ascending ID, retaining the later pair member.
