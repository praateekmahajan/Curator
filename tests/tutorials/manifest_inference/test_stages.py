# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.slurm_array import SlurmArrayConfig, slurm_array_shard_for_task
from nemo_curator.tasks import EmptyTask
from tutorials.text.manifest_inference.manifest import generate_manifest, shard_loads
from tutorials.text.manifest_inference.stages import ManifestFilePartitioningStage, SpecificJsonlReader


def test_reader_ranges_and_planner_match_real_adapter(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "part.jsonl").write_text('{"a":1}\n{"a":2,"later":"é"}\n{"a":3}\n')
    manifest = tmp_path / "manifest.jsonl"
    generate_manifest(source, manifest, 2)
    stage = ManifestFilePartitioningStage(str(manifest), str(source))
    stage.is_source_stage = True
    root = EmptyTask()
    tasks = BaseStageAdapter(stage)._post_process_task_ids([root], stage.process(root))
    loads = [0] * 5
    config = SlurmArrayConfig(shard_index=0, total_shards=5, minimum_shard_index=0)
    for task in tasks:
        loads[slurm_array_shard_for_task(task, config)] += task._metadata["manifest"]["num_rows"]
    assert loads == shard_loads(manifest, 5)
    reader = SpecificJsonlReader()
    assert reader.process(tasks[0]).to_pyarrow().to_pylist() == [{"a": 1, "later": None}, {"a": 2, "later": "é"}]
    assert reader.process(tasks[1]).to_pyarrow().to_pylist() == [{"a": 3}]


def test_bad_row_fails_only_its_range(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "part.jsonl").write_text('{"a":1}\n{"a":broken}\n{"a":3}\n')
    manifest = tmp_path / "manifest.jsonl"
    generate_manifest(source, manifest, 1)
    tasks = ManifestFilePartitioningStage(str(manifest), str(source)).process(EmptyTask())
    reader = SpecificJsonlReader()
    assert reader.process(tasks[0]).to_pyarrow().to_pylist() == [{"a": 1}]
    with pytest.raises(pa.ArrowInvalid):
        reader.process(tasks[1])
    assert reader.process(tasks[2]).to_pyarrow().to_pylist() == [{"a": 3}]
