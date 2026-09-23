# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.slurm_array import SlurmArrayConfig, slurm_array_shard_for_task
from nemo_curator.tasks import EmptyTask
from tutorials.text.manifest_inference.manifest import generate_manifest
from tutorials.text.manifest_inference.stages import ManifestFilePartitioningStage
from tutorials.text.manifest_inference.verification import _validate_row, verify_manifest_outputs


def _row(success: bool) -> dict:
    return {
        "curator_question": "question",
        "updated_test_answer": "answer" if success else None,
        "updated_test_reasoning": "",
        "updated_test_metadata": {
            "finish_reason": "stop" if success else None,
            "prompt_tokens": 2 if success else None,
            "completion_tokens": 1 if success else None,
            "attempt_count": 1,
            "retry_count": 0,
            "is_success": success,
            "failed_reason": None if success else "context_length_exceeded: too long",
        },
        "generation_lineage": {"request_kwargs": {"max_tokens": 8192}},
    }


def _setup_outputs(tmp_path: Path) -> tuple[Path, Path, Path, dict[int, list[dict]]]:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "part.jsonl").write_text("\n".join(json.dumps({"value": i}) for i in range(6)) + "\n")
    manifest = tmp_path / "manifest.jsonl"
    generate_manifest(input_dir, manifest, 2)
    source = ManifestFilePartitioningStage(str(manifest), str(input_dir))
    source.is_source_stage = True
    initial = EmptyTask()
    tasks = BaseStageAdapter(source)._post_process_task_ids([initial], source.process(initial))
    assignment = SlurmArrayConfig(shard_index=0, total_shards=2)
    by_shard: dict[int, list[dict]] = {0: [], 1: []}
    output_dir = tmp_path / "output"
    for task in tasks:
        record = task._metadata["manifest"]
        by_shard[slurm_array_shard_for_task(task, assignment)].append(record)
        path = output_dir / (record["output_file"] + ".zst")
        path.parent.mkdir(parents=True, exist_ok=True)
        with pa.output_stream(str(path), compression="zstd") as stream:
            rows = [_row(True) for _ in range(record["num_rows"])]
            stream.write(("\n".join(json.dumps(row) for row in rows) + "\n").encode())
    return manifest, input_dir, output_dir, by_shard


def test_verifies_success_and_explicit_failure_then_marks_completion(tmp_path: Path) -> None:
    manifest, input_dir, output_dir, by_shard = _setup_outputs(tmp_path)
    shard = next(value for value, records in by_shard.items() if records)
    record = by_shard[shard][0]
    path = output_dir / (record["output_file"] + ".zst")
    with pa.output_stream(str(path), compression="zstd") as stream:
        rows = [_row(False), *[_row(True) for _ in range(record["num_rows"] - 1)]]
        stream.write(("\n".join(json.dumps(row) for row in rows) + "\n").encode())
    checkpoint = tmp_path / "checkpoint"
    report = tmp_path / "report.json"

    result = verify_manifest_outputs(
        original_manifest=manifest,
        input_dir=input_dir,
        output_dir=output_dir,
        model_key="test",
        total_shards=2,
        shards=(shard,),
        checkpoint_path=checkpoint,
        report_path=report,
    )

    assert result["failed_rows"] == 1
    assert result["successful_rows"] == result["rows"] - 1
    assert len(result["completion_manifests"]) == 1
    assert json.loads(report.read_text()) == result


def test_invalid_failed_row_does_not_mark_completion(tmp_path: Path) -> None:
    manifest, input_dir, output_dir, by_shard = _setup_outputs(tmp_path)
    shard = next(value for value, records in by_shard.items() if records)
    record = by_shard[shard][0]
    path = output_dir / (record["output_file"] + ".zst")
    bad_row = _row(False)
    bad_row["updated_test_metadata"]["failed_reason"] = ""
    with pa.output_stream(str(path), compression="zstd") as stream:
        rows = [bad_row, *[_row(True) for _ in range(record["num_rows"] - 1)]]
        stream.write(("\n".join(json.dumps(row) for row in rows) + "\n").encode())
    checkpoint = tmp_path / "checkpoint"

    with pytest.raises(ValueError, match="failed row requires failed_reason"):
        verify_manifest_outputs(
            original_manifest=manifest,
            input_dir=input_dir,
            output_dir=output_dir,
            model_key="test",
            total_shards=2,
            shards=(shard,),
            checkpoint_path=checkpoint,
        )

    assert not (checkpoint / ".nemo_curator_metadata" / ".slurm_array_completion").exists()


def test_invalid_prompt_can_be_recorded_without_an_api_attempt() -> None:
    row = _row(False)
    row["updated_test_metadata"].update(attempt_count=0, failed_reason="invalid_prompt: empty question")
    assert _validate_row(row, "test", 8192, "test row") is False


def test_systemic_failure_cannot_be_verified_as_terminal() -> None:
    row = _row(False)
    row["updated_test_metadata"]["failed_reason"] = "http_503: unavailable"
    with pytest.raises(ValueError, match="non-terminal failure"):
        _validate_row(row, "test", 8192, "test row")
