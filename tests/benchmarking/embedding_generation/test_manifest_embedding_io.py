# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import benchmarking.embedding_generation.manifest as manifest_module
from benchmarking.embedding_generation.manifest import ManifestFilePartitioningStage
from benchmarking.embedding_generation.prepare_manifest import prepare_manifest
from benchmarking.embedding_generation.prepare_smoke_manifest import prepare_smoke_manifest
from benchmarking.embedding_generation.writer import MirroredParquetWriter
from nemo_curator.backends.slurm_array import SlurmArrayConfig
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.tasks import DocumentBatch, EmptyTask


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
                "dedup_path": f"/dedup/{index}.jsonl",
                "num_rows": index + 1,
                "id_start": index * 10,
                "id_end": index * 10 + index,
                "mapping_names": [f"family_{family_id}"],
                "error": None,
            }
        )
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
            path_mapping={"/dedup": "/logical"},
            required_minimum_files_per_shard=2,
        ).process(EmptyTask())
        selected_families.append(
            sorted(int(task._metadata["mapping_names"][0].removeprefix("family_")) for task in tasks)
        )

    assert selected_families == [[0, 0], [1, 1], [0, 1]]


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
