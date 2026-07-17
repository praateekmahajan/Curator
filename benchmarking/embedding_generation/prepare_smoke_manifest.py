# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarking.embedding_generation.manifest import (
    _iter_manifest_records,
    _validate_manifest_record,
)


def _load_family_mapping(path: str | Path, family_field: str) -> dict[str, int]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata_mapping"), dict):
        msg = f"Metadata configuration must contain a metadata_mapping object: {path}"
        raise TypeError(msg)

    result: dict[str, int] = {}
    for mapping_name, values in payload["metadata_mapping"].items():
        if not isinstance(mapping_name, str) or not isinstance(values, dict):
            msg = "Metadata mapping keys must be strings and values must be objects"
            raise TypeError(msg)
        family_id = values.get(family_field)
        if isinstance(family_id, bool) or not isinstance(family_id, int):
            msg = f"Metadata mapping {mapping_name!r} lacks integer field {family_field!r}"
            raise TypeError(msg)
        result[mapping_name] = family_id
    return result


def _record_family(record: dict[str, Any], family_mapping: dict[str, int]) -> int:
    mapping_names = record.get("mapping_names")
    if not isinstance(mapping_names, list) or not all(isinstance(name, str) for name in mapping_names):
        msg = f"Manifest index {record['index']} mapping_names must be a list of strings"
        raise TypeError(msg)
    matches = [family_mapping[name] for name in mapping_names if name in family_mapping]
    if len(matches) != 1:
        msg = f"Manifest index {record['index']} expected one metadata mapping match, found {matches}"
        raise ValueError(msg)
    return matches[0]


def _take_smallest(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(records) < count:
        msg = f"Requested {count} files, but only {len(records)} matching files are available"
        raise ValueError(msg)
    return sorted(records, key=lambda record: (record["num_rows"], record["index"]))[:count]


def _take_until_target_rows(
    records: list[dict[str, Any]],
    target_rows: int,
    minimum_files: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_rows = 0
    for record in records:
        selected.append(record)
        selected_rows += record["num_rows"]
        if len(selected) >= minimum_files and selected_rows >= target_rows:
            return selected
    msg = (
        f"Could not select at least {minimum_files} files and {target_rows} rows; "
        f"only {len(selected)} files and {selected_rows} rows are available"
    )
    raise ValueError(msg)


def _select_shard_records(
    by_family: dict[int, list[dict[str, Any]]],
    first_family_id: int,
    second_family_id: int,
    files_per_shard: int,
    target_rows_per_shard: int | None,
) -> dict[int, list[dict[str, Any]]]:
    mixed_per_family = files_per_shard // 2
    if target_rows_per_shard is None:
        required_per_family = files_per_shard + mixed_per_family
        selected_by_family = {
            family_id: _take_smallest(records, required_per_family) for family_id, records in by_family.items()
        }
        return {
            0: selected_by_family[first_family_id][:files_per_shard],
            1: selected_by_family[second_family_id][:files_per_shard],
            2: [
                *selected_by_family[first_family_id][files_per_shard:],
                *selected_by_family[second_family_id][files_per_shard:],
            ],
        }

    ordered_by_family = {
        family_id: sorted(records, key=lambda record: record["index"]) for family_id, records in by_family.items()
    }
    pure_first = _take_until_target_rows(ordered_by_family[first_family_id], target_rows_per_shard, files_per_shard)
    pure_second = _take_until_target_rows(ordered_by_family[second_family_id], target_rows_per_shard, files_per_shard)
    mixed_target_per_family = (target_rows_per_shard + 1) // 2
    mixed_first = _take_until_target_rows(
        ordered_by_family[first_family_id][len(pure_first) :], mixed_target_per_family, mixed_per_family
    )
    mixed_second = _take_until_target_rows(
        ordered_by_family[second_family_id][len(pure_second) :], mixed_target_per_family, mixed_per_family
    )
    return {
        0: pure_first,
        1: pure_second,
        2: [*mixed_first, *mixed_second],
    }


def prepare_smoke_manifest(  # noqa: PLR0913
    input_manifest: str | Path,
    metadata_mapping: str | Path,
    output_manifest: str | Path,
    first_family_id: int,
    second_family_id: int,
    files_per_shard: int = 16,
    family_field: str = "source_family_id",
    target_rows_per_shard: int | None = None,
) -> dict[str, Any]:
    """Build three explicit shards: first-family, second-family, and mixed."""
    if files_per_shard <= 0 or files_per_shard % 2:
        msg = "files_per_shard must be a positive even integer"
        raise ValueError(msg)
    if first_family_id == second_family_id:
        msg = "first_family_id and second_family_id must differ"
        raise ValueError(msg)
    if target_rows_per_shard is not None and target_rows_per_shard <= 0:
        msg = "target_rows_per_shard must be positive"
        raise ValueError(msg)

    output_manifest = Path(output_manifest)
    if output_manifest.exists():
        msg = f"Refusing to overwrite existing smoke manifest: {output_manifest}"
        raise FileExistsError(msg)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    family_mapping = _load_family_mapping(metadata_mapping, family_field)
    by_family: dict[int, list[dict[str, Any]]] = {first_family_id: [], second_family_id: []}
    for line_number, record in _iter_manifest_records(input_manifest):
        _validate_manifest_record(record, line_number, line_number - 1)
        family_id = _record_family(record, family_mapping)
        if family_id in by_family:
            by_family[family_id].append(record)

    shard_records = _select_shard_records(
        by_family,
        first_family_id,
        second_family_id,
        files_per_shard,
        target_rows_per_shard,
    )

    summary: dict[str, Any] = {"total_files": 0, "total_rows": 0, "shards": {}}
    output_tmp = output_manifest.with_suffix(f"{output_manifest.suffix}.incomplete")
    try:
        with output_tmp.open("x", encoding="utf-8") as output:
            output_index = 0
            for shard_index, records in shard_records.items():
                shard_rows = 0
                for record in records:
                    smoke_record = {
                        **record,
                        "inventory_index": record["index"],
                        "index": output_index,
                        "shard_index": shard_index,
                    }
                    output.write(json.dumps(smoke_record, separators=(",", ":")) + "\n")
                    output_index += 1
                    shard_rows += record["num_rows"]
                summary["shards"][str(shard_index)] = {
                    "num_files": len(records),
                    "num_rows": shard_rows,
                    "family_ids": sorted({_record_family(record, family_mapping) for record in records}),
                }
                summary["total_files"] += len(records)
                summary["total_rows"] += shard_rows
        output_tmp.replace(output_manifest)
    except BaseException:
        output_tmp.unlink(missing_ok=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an explicit three-shard embedding smoke manifest")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--metadata-mapping", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--first-family-id", type=int, required=True)
    parser.add_argument("--second-family-id", type=int, required=True)
    parser.add_argument("--files-per-shard", type=int, default=16)
    parser.add_argument("--family-field", default="source_family_id")
    parser.add_argument("--target-rows-per-shard", type=int)
    args = parser.parse_args()
    summary = prepare_smoke_manifest(
        input_manifest=args.input_manifest,
        metadata_mapping=args.metadata_mapping,
        output_manifest=args.output_manifest,
        first_family_id=args.first_family_id,
        second_family_id=args.second_family_id,
        files_per_shard=args.files_per_shard,
        family_field=args.family_field,
        target_rows_per_shard=args.target_rows_per_shard,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
