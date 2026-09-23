# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

import nemo_curator.backends.base as base_module
from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.slurm_array import SlurmArrayConfig, slurm_array_shard_for_task
from nemo_curator.tasks import EmptyTask
from tutorials.text.manifest_inference.manifest import generate_manifest, read_manifest, record_id
from tutorials.text.manifest_inference.recovery import build_recovery_manifest, parse_shard_indices
from tutorials.text.manifest_inference.stages import ManifestFilePartitioningStage, ManifestTask


def _source_tasks(manifest: Path, input_dir: Path) -> list[ManifestTask]:
    stage = ManifestFilePartitioningStage(str(manifest), str(input_dir))
    stage.is_source_stage = True
    root = EmptyTask()
    return BaseStageAdapter(stage)._post_process_task_ids([root], stage.process(root))


def test_parse_shard_indices():
    assert parse_shard_indices("7,2-4,9") == (2, 3, 4, 7, 9)
    with pytest.raises(ValueError, match="Duplicate"):
        parse_shard_indices("1,1")
    with pytest.raises(ValueError, match="ascending"):
        parse_shard_indices("3-1")


def test_builds_canonical_subset_and_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "part.jsonl").write_text("".join(json.dumps({"text": str(i)}) + "\n" for i in range(24)))
    manifest = tmp_path / "manifest.jsonl"
    generate_manifest(input_dir, manifest, max_num_rows=1)
    original_tasks = _source_tasks(manifest, input_dir)
    config = SlurmArrayConfig(shard_index=0, total_shards=4)
    populated_shards = sorted({slurm_array_shard_for_task(task, config) for task in original_tasks})
    selected_shards = tuple(populated_shards[:2])
    excluded_shards = tuple(populated_shards[2:])
    output = tmp_path / "recovery.jsonl"
    plan_file = tmp_path / "recovery-plan.json"

    plan = build_recovery_manifest(
        manifest=manifest,
        input_dir=input_dir,
        output=output,
        plan_file=plan_file,
        shards=selected_shards,
        total_shards=4,
        exclude_shards=excluded_shards,
    )

    recovery_records = list(read_manifest(output))
    recovery_tasks = _source_tasks(output, input_dir)
    expected = [task for task in original_tasks if slurm_array_shard_for_task(task, config) in selected_shards]
    assert [record_id(record) for record in recovery_records] == [
        record_id(task._metadata["manifest"]) for task in expected
    ]
    assert [task.task_id for task in recovery_tasks] == [task.task_id for task in expected]
    assert plan_file.read_text() == json.dumps(plan, indent=2, sort_keys=True) + "\n"
    assert plan["recovery_tasks"] == len(expected)
    assert plan["recovery_rows"] == len(expected)
    assert plan["selected_shards"] == list(selected_shards)
    assert plan["excluded_shards"] == list(excluded_shards)
    assert plan["requires_original_checkpoint"] is True
    assert plan["requires_slurm_source_filtering_disabled"] is True

    completed_source = expected[0].get_source_id()
    checkpoint_queries = []
    checkpoint_updates = []
    monkeypatch.setattr(
        base_module,
        "completed_resumability_sources",
        lambda source_ids: checkpoint_queries.append(source_ids) or {completed_source},
    )
    monkeypatch.setattr(base_module, "flush_resumability_deltas", checkpoint_updates.extend)
    survivors = BaseStageAdapter(ManifestFilePartitioningStage(str(output), str(input_dir)))._source_counters(
        recovery_tasks
    )
    assert completed_source in checkpoint_queries[0]
    assert all(task.get_source_id() != completed_source for task in survivors)
    assert {source_id for _, source_id, _ in checkpoint_updates} == {
        task.get_source_id() for task in recovery_tasks if task.get_source_id() != completed_source
    }


def test_rejects_overlap_range_and_overwrite(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "part.jsonl").write_text('{"text":"a"}\n')
    manifest = tmp_path / "manifest.jsonl"
    generate_manifest(input_dir, manifest, max_num_rows=1)
    output = tmp_path / "recovery.jsonl"
    plan_file = tmp_path / "plan.json"

    with pytest.raises(ValueError, match="overlap"):
        build_recovery_manifest(manifest, input_dir, output, plan_file, (0,), 1, exclude_shards=(0,))
    with pytest.raises(ValueError, match="outside"):
        build_recovery_manifest(manifest, input_dir, output, plan_file, (1,), 1)

    build_recovery_manifest(manifest, input_dir, output, plan_file, (0,), 1)
    with pytest.raises(FileExistsError):
        build_recovery_manifest(manifest, input_dir, output, tmp_path / "second-plan.json", (0,), 1)
