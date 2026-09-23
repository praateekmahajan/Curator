# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify generated manifest outputs and optionally mark logical shards complete."""

import argparse
import json
from pathlib import Path

import pyarrow as pa

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.slurm_array import (
    SlurmArrayConfig,
    build_slurm_array_completion_manifest,
    slurm_array_shard_for_task,
)
from nemo_curator.tasks import EmptyTask
from nemo_curator.utils.atomic_io import write_json_atomically

from .inference import SUCCESSFUL_FINISH_REASONS, TERMINAL_ROW_FAILURE_PREFIXES
from .recovery import parse_shard_indices
from .stages import ManifestFilePartitioningStage

METADATA_FIELDS = {
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "attempt_count",
    "retry_count",
    "is_success",
    "failed_reason",
}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_row(row: dict, model_key: str, expected_max_tokens: int, location: str) -> bool:
    _require(isinstance(row, dict), f"{location}: row must be a JSON object")
    required_fields = {
        f"updated_{model_key}_answer",
        f"updated_{model_key}_reasoning",
        f"updated_{model_key}_metadata",
        "generation_lineage",
    }
    _require(required_fields <= set(row), f"{location}: missing generated fields")
    answer = row[f"updated_{model_key}_answer"]
    reasoning = row[f"updated_{model_key}_reasoning"]
    metadata = row[f"updated_{model_key}_metadata"]
    _require(answer is None or isinstance(answer, str), f"{location}: answer must be text or null")
    _require(reasoning is None or isinstance(reasoning, str), f"{location}: reasoning must be text or null")
    _require(isinstance(metadata, dict), f"{location}: metadata must be an object")
    _require(set(metadata) == METADATA_FIELDS, f"{location}: metadata fields do not match the schema")
    _require(isinstance(metadata["is_success"], bool), f"{location}: is_success must be boolean")
    attempts = metadata["attempt_count"]
    retries = metadata["retry_count"]
    _require(_is_int(attempts) and attempts >= 0, f"{location}: attempt_count must be non-negative")
    _require(_is_int(retries) and retries >= 0, f"{location}: retry_count must be non-negative")
    _require(retries == max(0, attempts - 1), f"{location}: retry_count does not match attempt_count")
    for field in ("prompt_tokens", "completion_tokens"):
        value = metadata[field]
        _require(
            value is None or (_is_int(value) and value >= 0),
            f"{location}: {field} must be a non-negative integer or null",
        )
    finish_reason = metadata["finish_reason"]
    _require(
        finish_reason is None or isinstance(finish_reason, str),
        f"{location}: finish_reason must be text or null",
    )
    if metadata["is_success"]:
        _require(attempts >= 1, f"{location}: a successful row must have an API attempt")
        _require(isinstance(answer, str) and bool(answer.strip()), f"{location}: successful answer is empty")
        _require(finish_reason in SUCCESSFUL_FINISH_REASONS, f"{location}: invalid successful finish_reason")
        _require(metadata["prompt_tokens"] is not None, f"{location}: successful prompt_tokens is null")
        _require(
            _is_int(metadata["completion_tokens"]) and metadata["completion_tokens"] > 0,
            f"{location}: successful completion_tokens must be positive",
        )
        _require(metadata["failed_reason"] is None, f"{location}: successful failed_reason must be null")
    else:
        failed_reason = metadata["failed_reason"]
        _require(
            isinstance(failed_reason, str) and bool(failed_reason.strip()),
            f"{location}: failed row requires failed_reason",
        )
        prefix = failed_reason.partition(":")[0]
        _require(prefix in TERMINAL_ROW_FAILURE_PREFIXES, f"{location}: non-terminal failure {prefix!r}")
    _require(
        f"updated_{model_key}_request_latency_s" not in row,
        f"{location}: deprecated request latency column is present",
    )
    _require("request_latency_s" not in metadata, f"{location}: deprecated request latency metadata is present")
    lineage = row["generation_lineage"]
    _require(isinstance(lineage, dict), f"{location}: generation_lineage must be an object")
    request_kwargs = lineage.get("request_kwargs")
    _require(isinstance(request_kwargs, dict), f"{location}: lineage request_kwargs must be an object")
    _require(
        request_kwargs.get("max_tokens") == expected_max_tokens,
        f"{location}: lineage max_tokens does not match {expected_max_tokens}",
    )
    return metadata["is_success"]


def _source_tasks(manifest: Path, input_dir: Path) -> list:
    source = ManifestFilePartitioningStage(str(manifest), str(input_dir))
    source.is_source_stage = True
    initial = EmptyTask()
    # Assign IDs directly so an active or explicitly disabled Slurm-array
    # environment cannot filter the original manifest during verification.
    return BaseStageAdapter(source)._post_process_task_ids([initial], source.process(initial))


def verify_manifest_outputs(  # noqa: PLR0913
    *,
    original_manifest: Path,
    input_dir: Path,
    output_dir: Path,
    model_key: str,
    total_shards: int,
    shards: tuple[int, ...],
    expected_max_tokens: int = 8192,
    checkpoint_path: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    """Verify every original output assigned to ``shards``.

    Completion manifests are published only after every selected output passes
    row-count, schema, success/failure, and lineage validation.
    """
    _require(total_shards > 0, "total_shards must be positive")
    _require(expected_max_tokens > 0, "expected_max_tokens must be positive")
    _require(bool(shards) and len(shards) == len(set(shards)), "shards must be nonempty and unique")
    _require(
        all(0 <= shard < total_shards for shard in shards),
        f"shards must be in [0, {total_shards - 1}]",
    )

    selected_shards = set(shards)
    per_shard = {shard: {"rows": 0, "tasks": 0, "successful_rows": 0, "failed_rows": 0} for shard in shards}
    assignment = SlurmArrayConfig(shard_index=0, total_shards=total_shards)
    for task in _source_tasks(original_manifest, input_dir):
        shard = slurm_array_shard_for_task(task, assignment)
        if shard not in selected_shards:
            continue
        record = task._metadata["manifest"]
        path = output_dir / (record["output_file"] + ".zst")
        with pa.input_stream(str(path), compression="zstd") as stream:
            contents = [json.loads(line) for line in stream.read().splitlines()]
        _require(
            len(contents) == record["num_rows"],
            f"{path}: found {len(contents)} rows, expected {record['num_rows']}",
        )
        successful_rows = sum(
            _validate_row(row, model_key, expected_max_tokens, f"{path}:row {index}")
            for index, row in enumerate(contents)
        )
        counts = per_shard[shard]
        counts["rows"] += len(contents)
        counts["tasks"] += 1
        counts["successful_rows"] += successful_rows
        counts["failed_rows"] += len(contents) - successful_rows

    _require(
        all(counts["rows"] > 0 and counts["tasks"] > 0 for counts in per_shard.values()),
        "one or more selected shards have no assigned manifest records",
    )
    combined = {
        field: sum(counts[field] for counts in per_shard.values())
        for field in ("rows", "tasks", "successful_rows", "failed_rows")
    }

    completion_manifests = []
    if checkpoint_path is not None:
        for shard in shards:
            completion = build_slurm_array_completion_manifest(
                checkpoint_path=checkpoint_path,
                shard_index=shard,
                total_shards=total_shards,
                minimum_shard_index=0,
            )
            if completion is None:
                msg = "completion tracking unexpectedly disabled"
                raise RuntimeError(msg)
            completion_file = completion.mark_completed()
            if completion_file is None:
                msg = "completion manifest was not written"
                raise RuntimeError(msg)
            completion_manifests.append(str(completion_file))

    result = {
        "verified": True,
        "model": model_key,
        "shard": shards[0] if len(shards) == 1 else None,
        "shards": list(shards),
        "total_shards": total_shards,
        **combined,
        "per_shard": {str(shard): counts for shard, counts in per_shard.items()},
        "completion_manifests": completion_manifests,
    }
    if report_path is not None:
        write_json_atomically(report_path, result, indent=2)
    return result


def _parse_shards(value: str) -> tuple[int, ...]:
    try:
        return parse_shard_indices(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--original-manifest", type=Path, required=True)
    result.add_argument("--input-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--model-key", required=True)
    result.add_argument("--total-shards", type=int, required=True)
    result.add_argument("--shards", type=_parse_shards, required=True)
    result.add_argument("--checkpoint-path", type=Path)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--expected-max-tokens", type=int, default=8192)
    return result


def main() -> None:
    args = parser().parse_args()
    result = verify_manifest_outputs(
        original_manifest=args.original_manifest,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_key=args.model_key,
        total_shards=args.total_shards,
        shards=args.shards,
        checkpoint_path=args.checkpoint_path,
        report_path=args.report,
        expected_max_tokens=args.expected_max_tokens,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
