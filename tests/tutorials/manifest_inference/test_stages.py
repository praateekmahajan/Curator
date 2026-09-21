# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

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


def test_reader_detects_same_size_same_mtime_corruption(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    path = source / "part.jsonl"
    path.write_text('{"a":1}\n')
    manifest = tmp_path / "manifest.jsonl"
    generate_manifest(source, manifest)
    task = ManifestFilePartitioningStage(str(manifest), str(source)).process(EmptyTask())[0]
    stat = path.stat()
    path.write_text('{"a":2}\n')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="checksum"):
        SpecificJsonlReader().process(task)
