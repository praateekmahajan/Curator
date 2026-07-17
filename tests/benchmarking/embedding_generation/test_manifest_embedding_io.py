# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarking.embedding_generation.manifest import ManifestIdAssignmentStage
from benchmarking.embedding_generation.prepare_manifest import prepare_manifest
from benchmarking.embedding_generation.writer import MirroredParquetWriter
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.tasks import DocumentBatch


def _write_input_manifest(path: Path) -> None:
    records = [
        {
            "index": 0,
            "path": "/inventory/a.jsonl",
            "logical_path": "/logical/a.jsonl",
            "num_rows": 2,
            "mapping_names": ["source_a"],
            "error": None,
        },
        {
            "index": 1,
            "path": "/inventory/b.jsonl",
            "logical_path": "/logical/nested/b.jsonl",
            "num_rows": 3,
            "mapping_names": ["source_b"],
            "error": None,
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_prepare_manifest_writes_ranges_and_reverse_lookup(tmp_path: Path) -> None:
    input_manifest = tmp_path / "input.jsonl"
    path_mapping = tmp_path / "mapping.json"
    output_manifest = tmp_path / "runtime.jsonl"
    output_registry = tmp_path / "id_generator.json"
    _write_input_manifest(input_manifest)
    path_mapping.write_text(
        json.dumps(
            [
                {
                    "dedup_path": "/dedup",
                    "container_mounted_dedup_source_path": "/logical",
                }
            ]
        )
    )

    summary = prepare_manifest(
        input_manifest=input_manifest,
        path_mapping=path_mapping,
        output_manifest=output_manifest,
        output_id_registry=output_registry,
        start_id=100,
    )

    records = [json.loads(line) for line in output_manifest.read_text().splitlines()]
    assert [(record["id_start"], record["id_end"]) for record in records] == [(100, 101), (102, 104)]
    assert [record["dedup_path"] for record in records] == ["/dedup/a.jsonl", "/dedup/nested/b.jsonl"]
    assert summary == {"num_files": 2, "num_rows": 5, "start_id": 100, "next_id": 105}

    registry = json.loads(output_registry.read_text())
    first_key = str(uuid.uuid5(uuid.NAMESPACE_URL, "/dedup/a.jsonl"))
    assert registry["next_id"] == 105
    assert registry["batch_registry"][first_key] == [100, 101]
    assert registry["id_lookup"][0] == {
        "key": first_key,
        "manifest_index": 0,
        "path": "/dedup/a.jsonl",
        "id_start": 100,
        "id_end": 101,
        "num_rows": 2,
    }


def test_manifest_id_assignment_uses_exact_row_offset() -> None:
    task = DocumentBatch(
        dataset_name="test",
        data=pa.table({"text": ["a", "b", "c"]}),
        _metadata={"id_start": 20, "id_end": 22, "manifest_num_rows": 3},
    )

    result = ManifestIdAssignmentStage().process(task).to_pyarrow()

    assert result[CURATOR_DEDUP_ID_STR].to_pylist() == [20, 21, 22]


def test_manifest_id_assignment_rejects_row_count_mismatch() -> None:
    task = DocumentBatch(
        dataset_name="test",
        data=pa.table({"text": ["a", "b"]}),
        _metadata={"id_start": 20, "id_end": 22, "manifest_num_rows": 3},
    )

    with pytest.raises(ValueError, match="reader produced 2"):
        ManifestIdAssignmentStage().process(task)


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
                "source_priority": [1],
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
        fields=[CURATOR_DEDUP_ID_STR, "embeddings", "source_priority"],
        drop_fields=[],
    )

    output_task = writer.process(task)

    table = pq.read_table(output_task.data[0])
    assert table.column_names == [CURATOR_DEDUP_ID_STR, "embeddings", "source_priority"]
