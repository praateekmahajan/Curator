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

import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("id_field", ["id", "document_key"])
def test_filter_preserves_vectors_and_validates_mask(tmp_path: Path, id_field: str) -> None:
    import cudf

    from nemo_curator.stages.deduplication.semantic.filter import SemanticDuplicateFilterStage
    from nemo_curator.tasks import FileGroupTask

    frame = cudf.DataFrame(
        {"id": ["a", "b", "c"], "embedding": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "score": [2.0, 3.0, 4.0]}
    )
    frame = frame.rename(columns={"id": id_field})
    source = tmp_path / "input.parquet"
    frame.to_parquet(source, index=False, row_group_size_rows=1)
    masks = tmp_path / "masks"
    masks.mkdir()
    mask = cudf.DataFrame({"id": ["a", "b", "c"], "removed": [False, True, False]})
    mask.to_parquet(masks / "cluster_0.parquet", index=False)
    task = FileGroupTask(data=[str(source)], dataset_name="generic", _metadata={"centroid_id": 0})
    stage = SemanticDuplicateFilterStage(
        str(masks),
        str(tmp_path / "out"),
        list(frame.columns),
        id_field=id_field,
        read_memory_bytes=1,
    )
    result = stage.process(task)
    actual = cudf.read_parquet(result.data)
    assert actual.to_arrow().equals(frame.iloc[[0, 2]].reset_index(drop=True).to_arrow())
    assert result._metadata["input_rows"] == 3
    assert result._metadata["kept_rows"] == 2
    mask["id"] = ["a", "a", "c"]
    mask.to_parquet(masks / "cluster_0.parquet", index=False)
    with pytest.raises(ValueError, match="unique"):
        stage.process(task)
