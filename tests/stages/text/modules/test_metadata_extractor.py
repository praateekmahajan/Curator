# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pyarrow as pa
import pytest

from nemo_curator.stages.text.modules import MetadataExtractor
from nemo_curator.tasks import DocumentBatch


def _extractor() -> MetadataExtractor:
    return MetadataExtractor(
        metadata_mapping={
            "source_a": {
                "source_family_id": 0,
                "source_dataset_id": 10,
                "source_priority": 0,
                "quality_rank": 13,
                "recency_rank": 1,
            },
            "source_b": {
                "source_family_id": 1,
                "source_dataset_id": 20,
                "source_priority": 1,
                "quality_rank": 0,
                "recency_rank": 0,
            },
        },
        output_dtypes={
            "source_family_id": "int8",
            "source_dataset_id": "int16",
            "source_priority": "int8",
            "quality_rank": "int8",
            "recency_rank": "int8",
        },
    )


@pytest.mark.parametrize(
    ("data", "mapping_names"),
    [
        (pd.DataFrame({"text": ["one", "two"]}), ["source_a"]),
        (pa.table({"text": ["one", "two"]}), "source_b"),
    ],
)
def test_broadcasts_integer_metadata(
    data: pd.DataFrame | pa.Table,
    mapping_names: str | list[str],
) -> None:
    batch = DocumentBatch(dataset_name="test", data=data, _metadata={"mapping_names": mapping_names})

    result = _extractor().process(batch).to_pyarrow()

    assert result.num_rows == 2
    assert result.schema.field("source_family_id").type == pa.int8()
    assert result.schema.field("source_dataset_id").type == pa.int16()
    expected_priority = 0 if mapping_names == ["source_a"] else 1
    assert result["source_priority"].to_pylist() == [expected_priority, expected_priority]


def test_rejects_unknown_mapping() -> None:
    batch = DocumentBatch(
        dataset_name="test",
        data=pa.table({"text": ["one"]}),
        _metadata={"mapping_names": ["unknown"]},
    )

    with pytest.raises(ValueError, match="exactly one configured key"):
        _extractor().process(batch)


def test_rejects_existing_output_column() -> None:
    batch = DocumentBatch(
        dataset_name="test",
        data=pa.table({"text": ["one"], "quality_rank": [99]}),
        _metadata={"mapping_names": ["source_a"]},
    )

    with pytest.raises(ValueError, match="will not overwrite"):
        _extractor().process(batch)
