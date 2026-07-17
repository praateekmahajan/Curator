# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.modules import MetadataExtractor
from nemo_curator.tasks import DocumentBatch


def _extractor(
    merge_strategy: str = "separator",
    retained_input_fields: list[str] | None = None,
) -> MetadataExtractor:
    return MetadataExtractor(
        metadata_mapping={
            "source_a": {
                "source_family_id": 0,
                "quality_rank": 130,
                "recency_rank": 1,
            },
            "source_b": {
                "source_family_id": 1,
                "quality_rank": 0,
                "recency_rank": 0,
            },
        },
        output_dtypes={
            "source_family_id": "int8",
            "quality_rank": "int16",
            "recency_rank": "int8",
        },
        content_field="multimodal_document",
        merge_strategy=merge_strategy,
        retained_input_fields=retained_input_fields,
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
    assert result.schema.field("quality_rank").type == pa.int16()
    expected_family = 0 if mapping_names == ["source_a"] else 1
    assert result["source_family_id"].to_pylist() == [expected_family, expected_family]


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


def test_preserves_existing_text_from_reader_shaped_pandas_input() -> None:
    batch = DocumentBatch(
        dataset_name="test",
        data=pd.DataFrame(
            {
                "_curator_dedup_id": pd.array([7], dtype="int64[pyarrow]"),
                "text": ["already present"],
                "other": [99],
            }
        ),
        _metadata={"mapping_names": ["source_a"]},
    )

    output_batch = _extractor(retained_input_fields=["_curator_dedup_id", "text"]).process(batch)
    result = output_batch.to_pyarrow()

    assert result["_curator_dedup_id"].to_pylist() == [7]
    assert result["text"].to_pylist() == ["already present"]
    assert "other" not in result.column_names
    assert VLLMEmbeddingModelStage("unused").validate_input(output_batch)


def test_extracts_text_blocks_and_ignores_non_text_items() -> None:
    extractor = _extractor()

    assert extractor._extract_text({"content": ["first", {"kind": "non_text"}, "second"]}) == "first\n\nsecond"


def test_extracts_text_and_preserves_source_document() -> None:
    batch = DocumentBatch(
        dataset_name="test",
        data=pa.Table.from_pylist(
            [
                {
                    "multimodal_document": {
                        "content": ["first", "second"],
                    },
                    "other": 7,
                }
            ]
        ),
        _metadata={"mapping_names": ["source_a"]},
    )

    output_batch = _extractor().process(batch)
    result = output_batch.to_pyarrow()

    assert result["text"].to_pylist() == ["first\n\nsecond"]
    assert result["other"].to_pylist() == [7]
    assert "multimodal_document" in result.column_names
    assert VLLMEmbeddingModelStage("unused").validate_input(output_batch)


def test_extracts_reader_shaped_heterogeneous_content_before_arrow_conversion(tmp_path: Path) -> None:
    input_path = tmp_path / "structured.jsonl"
    input_path.write_text('{"multimodal_document":{"content":["first",{"kind":"non_text"},"second"]},"other":99}\n')
    frame = JsonlReaderStage().read_data([str(input_path)])
    assert isinstance(frame, pd.DataFrame)
    frame["_curator_dedup_id"] = pd.array([7], dtype="int64[pyarrow]")

    batch = DocumentBatch(
        dataset_name="test",
        data=frame,
        _metadata={"mapping_names": ["source_b"]},
    )

    output_batch = _extractor(retained_input_fields=["_curator_dedup_id", "text"]).process(batch)
    result = output_batch.to_pyarrow()

    assert result["_curator_dedup_id"].to_pylist() == [7]
    assert result["text"].to_pylist() == ["first\n\nsecond"]
    assert "multimodal_document" not in result.column_names
    assert "other" not in result.column_names
    assert VLLMEmbeddingModelStage("unused").validate_input(output_batch)


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        (["first", "second"], "first second"),
        (["first\n", "second"], "first\n\nsecond"),
        (["first", "\nsecond"], "first\n\nsecond"),
        (["  unchanged  "], "  unchanged  "),
        ([], ""),
    ],
)
def test_smart_merge(blocks: list[str], expected: str) -> None:
    extractor = _extractor(merge_strategy="smart")

    assert extractor._extract_text({"content": blocks}) == expected


def test_fails_when_neither_text_nor_content_is_available() -> None:
    batch = DocumentBatch(
        dataset_name="test",
        data=pa.table({"other": [7]}),
        _metadata={"mapping_names": ["source_a"]},
    )

    with pytest.raises(ValueError, match="neither text field"):
        _extractor().process(batch)
