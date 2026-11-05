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

from contextlib import suppress
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
import pytest

with suppress(ImportError):
    from nemo_curator.stages.deduplication.semantic.diversity import (
        ClusterDiversityStage,
        centroid_separation,
        spark_crc32,
    )
    from nemo_curator.tasks import FileGroupTask


@pytest.mark.gpu
def test_spark_crc32_utf8_and_slices() -> None:
    import cudf

    strings = ["prefix", "", "123456789", "é数学🐬", "a\0b", "f" * 64, "suffix"]
    values = cudf.Series(strings, index=np.arange(len(strings)) + 20)
    for selection in (values, values.iloc[1:-1], values.iloc[:0], cudf.Series(["", ""])):
        actual = spark_crc32(selection).to_pandas()
        expected = selection.to_pandas().map(lambda s: crc32(s.encode("utf-8"))).astype("uint32")
        pd.testing.assert_series_equal(actual, expected)
    with pytest.raises(ValueError, match="null"):
        spark_crc32(cudf.Series(["a", None]))


@pytest.mark.gpu
def test_separation_matches_dense() -> None:
    centers = np.random.default_rng(4).normal(size=(57, 13))
    normalized = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    distances = 1 - normalized @ normalized.T
    np.fill_diagonal(distances, 0)
    np.testing.assert_allclose(centroid_separation(centers), distances.sum(axis=1) / 56, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("read_memory_bytes", [1, 256 * 1024**2])
def test_complete_cluster_diversity(tmp_path: Path, read_memory_bytes: int) -> None:
    centers = np.eye(3)
    np.save(tmp_path / "centers.npy", centers)
    files = []
    for i, distances in enumerate(([0.1, 0.3], [0.8])):
        path = tmp_path / f"input{i}.parquet"
        pd.DataFrame(
            {"id": [f"{i}-{j}" for j in range(len(distances))], "math_topic": "math", "cosine_dist_to_cent": distances}
        ).to_parquet(path)
        files.append(str(path))
    stage = ClusterDiversityStage(
        str(tmp_path / "centers.npy"),
        str(tmp_path / "output"),
        [],
        topic_field="math_topic",
        read_memory_bytes=read_memory_bytes,
    )
    stage.setup()
    result = stage.process(FileGroupTask(data=files, dataset_name="math", _metadata={"centroid_id": 0}))
    output = pd.concat([pd.read_parquet(path) for path in result.data])
    np.testing.assert_allclose(output.diversity_score, 0.4)
    np.testing.assert_array_equal(output.sampling_shard, output.id.map(lambda s: crc32(s.encode()) % 128))
    assert sum(result._metadata["sampling_group_counts"].values()) == 3
    assert "embeddings" not in output


@pytest.mark.gpu
@pytest.mark.parametrize("groups", [(), ("source",)])
def test_generic_diversity_groups(tmp_path: Path, groups: tuple[str, ...]) -> None:
    np.save(tmp_path / "centers.npy", np.eye(2))
    path = tmp_path / "input.parquet"
    pd.DataFrame({"id": ["a", "b"], "source": ["web", "code"], "cosine_dist_to_cent": [0.2, 0.4]}).to_parquet(path)
    stage = ClusterDiversityStage(
        str(tmp_path / "centers.npy"),
        str(tmp_path / "out"),
        ["source"],
        group_fields=groups,
    )
    stage.setup()
    result = stage.process(FileGroupTask(data=[str(path)], dataset_name="generic", _metadata={"centroid_id": 0}))
    counts = result._metadata["sampling_group_counts"]
    assert sum(counts.values()) == 2
    assert all(len(key) == len(groups) + 1 for key in counts)
    output = pd.concat([pd.read_parquet(p) for p in result.data])
    assert "math_topic" not in output
    np.testing.assert_allclose(output.diversity_score, 0.3)


@pytest.mark.gpu
def test_large_file_is_chunked(tmp_path: Path) -> None:
    np.save(tmp_path / "centers.npy", np.eye(2))
    source = pd.DataFrame({"id": np.arange(100_000), "cosine_dist_to_cent": np.linspace(0, 1, 100_000)})
    path = tmp_path / "single.parquet"
    source.to_parquet(path, row_group_size=4096, index=False)
    stage = ClusterDiversityStage(str(tmp_path / "centers.npy"), str(tmp_path / "out"), [], read_memory_bytes=4096)
    stage.setup()
    result = stage.process(FileGroupTask(data=[str(path)], dataset_name="skew", _metadata={"centroid_id": 0}))
    assert len(result.data) > 1
    output = pd.concat([pd.read_parquet(p) for p in result.data])
    assert len(output) == len(source)
    np.testing.assert_allclose(output.diversity_score, 0.5)
