# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import benchmarking.embedding_generation.manifest as manifest_module
from benchmarking.embedding_generation.fuzzy_deduped_qwen3_06b import (
    MetadataExtractingJsonlReaderStage,
    _configure_object_store_memory_limit,
    _embedding_metadata_fields,
)
from benchmarking.embedding_generation.manifest import ManifestFilePartitioningStage
from benchmarking.embedding_generation.prepare_smoke_manifest import prepare_smoke_manifest
from benchmarking.embedding_generation.writer import MirroredParquetWriter
from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.slurm_array import SlurmArrayConfig
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.text.modules import MetadataExtractor
from nemo_curator.tasks import DocumentBatch, EmptyTask, FileGroupTask


def test_metadata_extracting_reader_emits_compact_embedding_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(
            {
                CURATOR_DEDUP_ID_STR: 7,
                "multimodal_document": {"content": ["first", "second"]},
                "unused": "drop me",
            }
        )
        + "\n"
    )
    extractor = MetadataExtractor(
        metadata_mapping={
            "source_a": {
                "source_family_id": 1,
                "quality_rank": 180,
                "recency_rank": 0,
            }
        },
        output_dtypes={
            "source_family_id": "int8",
            "quality_rank": "int16",
            "recency_rank": "int8",
        },
        content_field="multimodal_document",
        retained_input_fields=[CURATOR_DEDUP_ID_STR, "text"],
    )
    reader = MetadataExtractingJsonlReaderStage(metadata_extractor=extractor)
    task = FileGroupTask(
        dataset_name="test",
        data=[str(input_path)],
        _metadata={"mapping_names": ["source_a"]},
    )

    result = reader.process(task).to_pyarrow()

    assert result.column_names == [
        CURATOR_DEDUP_ID_STR,
        "text",
        "source_family_id",
        "quality_rank",
        "recency_rank",
    ]
    assert result["text"].to_pylist() == ["first\n\nsecond"]
    assert result["quality_rank"].to_pylist() == [180]
    assert "unused" not in result.column_names


def test_metadata_extracting_reader_declares_generated_fields() -> None:
    extractor = MetadataExtractor(
        metadata_mapping={"source_a": {"source_family_id": 1}},
        output_dtypes={"source_family_id": "int8"},
        content_field="multimodal_document",
        retained_input_fields=[CURATOR_DEDUP_ID_STR, "text"],
    )
    reader = MetadataExtractingJsonlReaderStage(
        _assign_ids=True,
        metadata_extractor=extractor,
    )

    assert reader.outputs() == (["data"], [CURATOR_DEDUP_ID_STR, "source_family_id", "text"])


@pytest.mark.parametrize(
    ("keep_text", "expected"),
    [
        (False, [CURATOR_DEDUP_ID_STR, "source_family_id"]),
        (True, [CURATOR_DEDUP_ID_STR, "source_family_id", "text"]),
    ],
)
def test_embedding_metadata_fields_drop_text_unless_requested(keep_text: bool, expected: list[str]) -> None:
    extractor = MetadataExtractor(
        metadata_mapping={"source_a": {"source_family_id": 1}},
        output_dtypes={"source_family_id": "int8"},
        content_field="multimodal_document",
    )

    assert _embedding_metadata_fields(extractor, keep_text) == expected


def test_configure_object_store_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    context = type("Context", (), {})()
    monkeypatch.setattr(
        "benchmarking.embedding_generation.fuzzy_deduped_qwen3_06b.DataContext.get_current",
        lambda: context,
    )

    _configure_object_store_memory_limit(0.7)

    assert context.override_object_store_memory_limit_fraction == 0.7


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_configure_object_store_memory_limit_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="must be in"):
        _configure_object_store_memory_limit(fraction)


def test_production_config_bounds_object_store_and_drops_text() -> None:
    config_path = Path(__file__).parents[3] / "benchmarking/embedding_generation/fuzzy-deduped-qwen3-0p6b.yaml"
    config = yaml.safe_load(config_path.read_text())
    args = config["entries"][0]["args"]

    assert config["object_store_size"] == 96 * 1024**3
    assert "--override-object-store-memory-limit-fraction=0.7" in args
    assert "--keep-text" not in args


def test_metadata_extraction_uses_only_reader_task_boundary(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"text":"hello"}\n')
    extractor = MetadataExtractor(
        metadata_mapping={"source_a": {"source_family_id": 1}},
        output_dtypes={"source_family_id": "int8"},
    )
    reader = MetadataExtractingJsonlReaderStage(metadata_extractor=extractor)
    task = FileGroupTask(
        dataset_name="test",
        data=[str(input_path)],
        _metadata={"mapping_names": ["source_a"]},
    )
    task._set_task_id("0", "source")

    [result] = BaseStageAdapter(reader).process_batch([task])

    assert result.task_id == "0_source_0"
    assert len(result._stage_perf) == 1
    assert result.to_pyarrow()["source_family_id"].to_pylist() == [1]


def test_writer_emits_only_generated_fields(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_file = source_root / "nested" / "input.jsonl"
    output_root = tmp_path / "output"
    task = DocumentBatch(
        dataset_name="test",
        data=pa.table(
            {
                CURATOR_DEDUP_ID_STR: [1],
                "embeddings": [[0.1, 0.2]],
                "quality_rank": [130],
                "adlr_id": [99],
                "sample_id": ["legacy"],
                "text": ["temporary"],
            }
        ),
        _metadata={"source_files": [str(source_file)]},
    )
    writer = MirroredParquetWriter(
        path=str(output_root),
        source_root=str(source_root),
        fields=[CURATOR_DEDUP_ID_STR, "embeddings", "quality_rank"],
        drop_fields=[],
    )

    output_task = writer.process(task)

    table = pq.read_table(output_task.data[0])
    assert table.column_names == [CURATOR_DEDUP_ID_STR, "embeddings", "quality_rank"]


def test_prepare_smoke_manifest_and_explicit_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_manifest = tmp_path / "runtime.jsonl"
    metadata_mapping = tmp_path / "metadata.json"
    output_manifest = tmp_path / "smoke.jsonl"
    records = []
    for index in range(6):
        family_id = index % 2
        records.append(
            {
                "index": index,
                "path": f"/inventory/{index}.jsonl",
                "logical_path": f"/logical/{index}.jsonl",
                "num_rows": index + 1,
                "id_start": index * 10,
                "id_end": index * 10 + index,
                "mapping_names": [f"family_{family_id}"],
                "error": None,
            }
        )
    input_manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    id_generator = tmp_path / "id_generator.json"
    id_generator.write_text(
        json.dumps(
            {
                "next_id": 60,
                "batch_registry": {
                    str(uuid.uuid5(uuid.NAMESPACE_URL, record["path"])): [record["id_start"], record["id_end"]]
                    for record in records
                },
            }
        )
    )
    metadata_mapping.write_text(
        json.dumps(
            {
                "metadata_mapping": {
                    "family_0": {"source_family_id": 0},
                    "family_1": {"source_family_id": 1},
                }
            }
        )
    )

    summary = prepare_smoke_manifest(
        input_manifest=input_manifest,
        metadata_mapping=metadata_mapping,
        output_manifest=output_manifest,
        first_family_id=0,
        second_family_id=1,
        files_per_shard=2,
    )

    assert summary["total_files"] == 6
    assert summary["shards"]["0"]["family_ids"] == [0]
    assert summary["shards"]["1"]["family_ids"] == [1]
    assert summary["shards"]["2"]["family_ids"] == [0, 1]

    selected_families = []
    for shard_index in range(3):

        def resolve_config(is_source_stage: bool, index: int = shard_index) -> SlurmArrayConfig:
            assert is_source_stage
            return SlurmArrayConfig(shard_index=index, total_shards=3)

        monkeypatch.setattr(
            manifest_module,
            "resolve_slurm_array_config",
            resolve_config,
        )
        tasks = ManifestFilePartitioningStage(
            manifest_path=str(output_manifest),
            id_generator_path=str(id_generator),
            required_minimum_files_per_shard=2,
        ).process(EmptyTask())
        selected_families.append(
            sorted(int(task._metadata["mapping_names"][0].removeprefix("family_")) for task in tasks)
        )

    assert selected_families == [[0, 0], [1, 1], [0, 1]]


def test_prepare_smoke_manifest_selects_disjoint_row_targets(tmp_path: Path) -> None:
    input_manifest = tmp_path / "runtime.jsonl"
    metadata_mapping = tmp_path / "metadata.json"
    output_manifest = tmp_path / "smoke.jsonl"
    records = [
        {
            "index": index,
            "path": f"/inventory/{index}.jsonl",
            "logical_path": f"/logical/{index}.jsonl",
            "num_rows": 40,
            "mapping_names": [f"family_{index % 2}"],
            "error": None,
        }
        for index in range(16)
    ]
    input_manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    metadata_mapping.write_text(
        json.dumps(
            {
                "metadata_mapping": {
                    "family_0": {"source_family_id": 0},
                    "family_1": {"source_family_id": 1},
                }
            }
        )
    )

    summary = prepare_smoke_manifest(
        input_manifest=input_manifest,
        metadata_mapping=metadata_mapping,
        output_manifest=output_manifest,
        first_family_id=0,
        second_family_id=1,
        files_per_shard=2,
        target_rows_per_shard=100,
    )

    assert summary["shards"]["0"]["num_rows"] >= 100
    assert summary["shards"]["1"]["num_rows"] >= 100
    assert summary["shards"]["2"]["num_rows"] >= 100
    selected = [json.loads(line) for line in output_manifest.read_text().splitlines()]
    assert len({record["inventory_index"] for record in selected}) == len(selected)


def test_smoke_writer_can_retain_text(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_file = source_root / "input.jsonl"
    writer = MirroredParquetWriter(
        path=str(tmp_path / "output"),
        source_root=str(source_root),
        fields=[CURATOR_DEDUP_ID_STR, "text", "embeddings"],
        drop_fields=[],
    )
    task = DocumentBatch(
        dataset_name="test",
        data=pa.table(
            {
                CURATOR_DEDUP_ID_STR: [1],
                "text": ["embedded text"],
                "embeddings": [[0.1, 0.2]],
                "legacy_id": [99],
            }
        ),
        _metadata={"source_files": [str(source_file)]},
    )

    output_task = writer.process(task)

    table = pq.read_table(output_task.data[0])
    assert table.column_names == [CURATOR_DEDUP_ID_STR, "text", "embeddings"]
