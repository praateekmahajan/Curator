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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_wide_text_output_with_small_buffer_budget(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    import pyarrow.parquet as pq

    from nemo_curator.stages.deduplication.semantic.sampling_disk import sample_fusion_disk

    rows = 32768
    frame = pd.DataFrame(
        {
            "id": np.arange(rows),
            "text": [f"{i}:" + "x" * 4096 for i in range(rows)],
            "math_topic": "math",
            "sampling_shard": 0,
            "ms": 3.0,
            "rd": 3.0,
            "cr": 3.0,
            "diversity_score": 0.0,
        }
    )
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False)
    outputs = sample_fusion_disk(
        [str(source)],
        str(tmp_path / "out"),
        {("math", 0): rows},
        scratch_path=str(tmp_path / "scratch"),
        memory_limit="128MB",
        threads=2,
        max_repeat=1,
    )
    actual = pd.concat([pd.read_parquet(path) for path in outputs]).sort_values("id")
    assert actual.text.tolist() == frame.text.tolist()
    assert (actual.total_copies == 1).all()
    # Byte-target groups must split this payload even though its row count is small.
    assert sum(pq.read_metadata(path).num_row_groups for path in outputs) > 2


@pytest.mark.gpu
@pytest.mark.parametrize("epochs", [0, 0.1, 2.5, 7.5, 8])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"quality_weights": (2.0, 1.0, 0.0), "gate_threshold": 4.5, "temperature": 5.0, "diversity_weight": 0.25},
        {"temperature": 0.0, "diversity_weight": 0.0},
    ],
)
def test_disk_matches_gpu(tmp_path: Path, epochs: float, dtype: str, options: dict) -> None:
    pytest.importorskip("duckdb")
    import cudf

    from nemo_curator.stages.deduplication.semantic.sampling import fusion_copies
    from nemo_curator.stages.deduplication.semantic.sampling_disk import sample_fusion_disk

    rng = np.random.default_rng(37)
    frame = pd.DataFrame(rng.uniform(0, 5, (1000, 4)), columns=["ms", "rd", "cr", "diversity_score"]).astype(dtype)
    frame.iloc[:100] = 3  # Equal composite scores exercise deterministic remainder ties.
    frame["id"] = np.arange(len(frame))
    frame["math_topic"] = "math"
    frame["sampling_shard"] = frame.id % 2
    frame["text"] = "retained"
    frame["q"] = -99.0  # A stale input score must be replaced, not trusted or duplicated.
    frame["total_copies"] = -99
    frame["copies"] = "original-metadata"  # Must not shadow the internal allocation column.
    files, budgets, expected = [], {}, []
    for shard, group in frame.groupby("sampling_shard"):
        target = round(len(group) * epochs)
        budgets[("math", shard)] = target
        counts = fusion_copies(cudf.from_pandas(group), target, **options).get()
        expected.extend(zip(group.id, counts, strict=True))
    shuffled = frame.sample(frac=1, random_state=1)
    for i, indices in enumerate(np.array_split(np.arange(len(frame)), 3)):
        part = shuffled.iloc[indices]
        path = tmp_path / f"input{i}.parquet"
        part.to_parquet(path, index=False)
        files.append(str(path))
    outputs = sample_fusion_disk(
        files,
        str(tmp_path / "out"),
        budgets,
        scratch_path=str(tmp_path / "scratch"),
        memory_limit="128MB",
        threads=1,
        **options,
    )
    actual = (
        pd.concat([pd.read_parquet(p) for p in outputs])
        if outputs
        else pd.DataFrame(columns=["id", "total_copies", "text"])
    )
    assert dict(zip(actual.id, actual.total_copies, strict=True)) == {i: c for i, c in expected if c}
    assert (actual.text == "retained").all()
    if actual.empty:
        return
    assert list(actual.columns).count("total_copies") == 1
    assert "total_copies_1" not in actual
    assert (actual.copies == "original-metadata").all()
    values = actual[["ms", "rd", "cr"]].to_numpy(dtype=np.float64)
    weights = options.get("quality_weights", (0.45, 0.30, 0.25))
    expected_q = np.clip(
        weights[0] * values[:, 0] / 5 + weights[1] * values[:, 1] / 5 + weights[2] * values[:, 2] / 5, 0, 1
    )
    np.testing.assert_array_equal(actual.q.to_numpy(), expected_q)


@pytest.mark.gpu
@pytest.mark.parametrize("constant", [True, False])
def test_extreme_temperature(tmp_path: Path, constant: bool) -> None:
    pytest.importorskip("duckdb")
    import cudf

    from nemo_curator.stages.deduplication.semantic.sampling import fusion_copies
    from nemo_curator.stages.deduplication.semantic.sampling_disk import sample_fusion_disk

    frame = pd.DataFrame(
        {
            "id": range(4),
            "ms": 1.0 if constant else np.arange(4, dtype=float),
            "rd": 1.0,
            "cr": 1.0,
            "diversity_score": 0.0,
            "math_topic": "math",
            "sampling_shard": 0,
        }
    )
    path = tmp_path / "input.parquet"
    frame.to_parquet(path, index=False)
    if constant:
        np.testing.assert_array_equal(fusion_copies(cudf.from_pandas(frame), 8, 2, temperature=10000).get(), [2] * 4)
        files = sample_fusion_disk(
            [str(path)],
            str(tmp_path / "out"),
            {("math", 0): 8},
            scratch_path=str(tmp_path / "scratch"),
            temperature=10000,
            max_repeat=2,
        )
        assert pd.concat([pd.read_parquet(f) for f in files]).total_copies.sum() == 8
    else:
        with pytest.raises(ValueError, match="underflowed"):
            fusion_copies(cudf.from_pandas(frame), 8, 2, temperature=10000)
        with pytest.raises(ValueError, match="underflowed"):
            sample_fusion_disk(
                [str(path)],
                str(tmp_path / "out"),
                {("math", 0): 8},
                scratch_path=str(tmp_path / "scratch"),
                temperature=10000,
                max_repeat=2,
            )
