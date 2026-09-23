# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a manifest that bundles selected logical shards for checkpointed recovery.

The generated manifest contains every source task canonically assigned to the
selected shards. Run it with the original checkpoint path and with Slurm source
filtering disabled; Curator's resumability checkpoint will skip sources that
already completed.
"""

import argparse
import json
from pathlib import Path

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.backends.slurm_array import SlurmArrayConfig, slurm_array_shard_for_task
from nemo_curator.tasks import EmptyTask

from .manifest import record_id
from .stages import ManifestFilePartitioningStage

RANGE_BOUND_COUNT = 2
DISTINCT_PATH_COUNT = 3


def parse_shard_indices(value: str) -> tuple[int, ...]:
    """Parse comma-separated shard indices and inclusive ranges."""
    indices: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            msg = f"Invalid empty shard component in {value!r}"
            raise ValueError(msg)
        if "-" not in part:
            try:
                indices.append(int(part))
            except ValueError as e:
                msg = f"Invalid shard index {part!r}"
                raise ValueError(msg) from e
            continue
        bounds = part.split("-")
        if len(bounds) != RANGE_BOUND_COUNT or not all(bounds):
            msg = f"Invalid shard range {part!r}"
            raise ValueError(msg)
        try:
            start, end = (int(bound) for bound in bounds)
        except ValueError as e:
            msg = f"Invalid shard range {part!r}"
            raise ValueError(msg) from e
        if start > end:
            msg = f"Shard range must be ascending: {part!r}"
            raise ValueError(msg)
        indices.extend(range(start, end + 1))
    if len(set(indices)) != len(indices):
        msg = f"Duplicate shard index in {value!r}"
        raise ValueError(msg)
    return tuple(sorted(indices))


def _validate_shards(indices: tuple[int, ...], config: SlurmArrayConfig, label: str) -> None:
    minimum = config.minimum_shard_index
    maximum = minimum + config.total_shards - 1
    invalid = [index for index in indices if not minimum <= index <= maximum]
    if invalid:
        msg = f"{label} {invalid} outside original shard range [{minimum}, {maximum}]"
        raise ValueError(msg)


def _validate_request(  # noqa: PLR0913
    manifest: Path,
    input_dir: Path,
    output: Path,
    plan_file: Path,
    shards: tuple[int, ...],
    total_shards: int,
    minimum_shard_index: int,
    exclude_shards: tuple[int, ...],
) -> SlurmArrayConfig:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if len({manifest, output, plan_file}) != DISTINCT_PATH_COUNT:
        msg = "Manifest, recovery output, and plan file must be different paths"
        raise ValueError(msg)
    if output.exists():
        raise FileExistsError(output)
    if plan_file.exists():
        raise FileExistsError(plan_file)
    if total_shards <= 0:
        msg = "total_shards must be positive"
        raise ValueError(msg)
    if minimum_shard_index < 0:
        msg = "minimum_shard_index must be non-negative"
        raise ValueError(msg)
    if not shards:
        msg = "At least one recovery shard is required"
        raise ValueError(msg)
    config = SlurmArrayConfig(minimum_shard_index, total_shards, minimum_shard_index)
    _validate_shards(shards, config, "Recovery shards")
    _validate_shards(exclude_shards, config, "Excluded shards")
    overlap = sorted(set(shards) & set(exclude_shards))
    if overlap:
        msg = f"Recovery shards overlap excluded shards: {overlap}"
        raise ValueError(msg)
    return config


def _select_records(
    manifest: Path,
    input_dir: Path,
    shards: tuple[int, ...],
    config: SlurmArrayConfig,
) -> tuple[list[dict], dict[int, dict[str, int]], int, int]:
    stage = ManifestFilePartitioningStage(str(manifest), str(input_dir))
    stage.is_source_stage = True
    root = EmptyTask()
    tasks = BaseStageAdapter(stage)._post_process_task_ids([root], stage.process(root))
    selected: list[dict] = []
    shard_counts = {index: {"tasks": 0, "rows": 0} for index in shards}
    selected_ids: set[str] = set()
    source_rows = 0
    for task in tasks:
        record = task._metadata["manifest"]
        source_rows += record["num_rows"]
        shard_index = slurm_array_shard_for_task(task, config)
        if shard_index not in shard_counts:
            continue
        identity = record_id(record)
        if identity in selected_ids:
            msg = f"Duplicate selected manifest record identity: {identity}"
            raise ValueError(msg)
        selected_ids.add(identity)
        selected.append(record)
        shard_counts[shard_index]["tasks"] += 1
        shard_counts[shard_index]["rows"] += record["num_rows"]
    if not selected:
        msg = "Selected recovery shards contain no manifest tasks"
        raise ValueError(msg)
    return selected, shard_counts, len(tasks), source_rows


def build_recovery_manifest(  # noqa: PLR0913
    manifest: Path,
    input_dir: Path,
    output: Path,
    plan_file: Path,
    shards: tuple[int, ...],
    total_shards: int,
    minimum_shard_index: int = 0,
    exclude_shards: tuple[int, ...] = (),
) -> dict:
    """Select full original logical shards while preserving source identities."""
    manifest = manifest.resolve()
    input_dir = input_dir.resolve()
    output = output.resolve()
    plan_file = plan_file.resolve()
    config = _validate_request(
        manifest,
        input_dir,
        output,
        plan_file,
        shards,
        total_shards,
        minimum_shard_index,
        exclude_shards,
    )
    selected, shard_counts, source_tasks, source_rows = _select_records(manifest, input_dir, shards, config)

    plan = {
        "source_manifest": str(manifest),
        "input_dir": str(input_dir),
        "recovery_manifest": str(output),
        "minimum_shard_index": minimum_shard_index,
        "total_shards": total_shards,
        "selected_shards": list(shards),
        "excluded_shards": list(exclude_shards),
        "source_tasks": source_tasks,
        "source_rows": source_rows,
        "recovery_tasks": len(selected),
        "recovery_rows": sum(record["num_rows"] for record in selected),
        "selected_shard_counts": {str(index): counts for index, counts in shard_counts.items()},
        "requires_original_checkpoint": True,
        "requires_slurm_source_filtering_disabled": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record) + "\n")
    with plan_file.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return plan


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--input-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--plan-file", type=Path, required=True)
    result.add_argument("--shards", required=True, help="Comma-separated original shard indices or ranges")
    result.add_argument("--exclude-shards", default="", help="Shards that must be disjoint from --shards")
    result.add_argument("--total-shards", type=int, required=True)
    result.add_argument("--minimum-shard-index", type=int, default=0)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        shards = parse_shard_indices(args.shards)
        exclude_shards = parse_shard_indices(args.exclude_shards) if args.exclude_shards else ()
        plan = build_recovery_manifest(
            manifest=args.manifest,
            input_dir=args.input_dir,
            output=args.output,
            plan_file=args.plan_file,
            shards=shards,
            total_shards=args.total_shards,
            minimum_shard_index=args.minimum_shard_index,
            exclude_shards=exclude_shards,
        )
    except (FileNotFoundError, FileExistsError, NotADirectoryError, OSError, ValueError) as e:
        parser().error(str(e))
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
