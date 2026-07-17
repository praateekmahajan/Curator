# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from benchmarking.embedding_generation.manifest import _iter_manifest_records, _validate_manifest_record


def _load_path_mappings(path: str | Path) -> list[tuple[str, str]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        msg = f"Path mapping must contain a JSON list: {path}"
        raise TypeError(msg)

    mappings: list[tuple[str, str]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            msg = f"Path mapping record {index} must be a JSON object"
            raise TypeError(msg)
        dedup_prefix = record.get("dedup_path")
        logical_prefix = record.get("container_mounted_dedup_source_path")
        if not isinstance(dedup_prefix, str) or not isinstance(logical_prefix, str):
            msg = (
                f"Path mapping record {index} must contain string dedup_path and "
                "container_mounted_dedup_source_path fields"
            )
            raise TypeError(msg)
        mappings.append((dedup_prefix.rstrip("/"), logical_prefix.rstrip("/")))
    return sorted(mappings, key=lambda item: len(item[1]), reverse=True)


def _resolve_dedup_path(logical_path: str, mappings: list[tuple[str, str]]) -> str:
    for dedup_prefix, logical_prefix in mappings:
        if logical_path == logical_prefix or logical_path.startswith(f"{logical_prefix}/"):
            return f"{dedup_prefix}{logical_path[len(logical_prefix) :]}"
    msg = f"Logical path is not covered by the dataset path mapping: {logical_path}"
    raise ValueError(msg)


def _batch_key(path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


def _validate_new_batch_key(key: str, dedup_path: str, batch_registry: dict[str, tuple[int, int]]) -> None:
    if key in batch_registry:
        msg = f"Duplicate deduplicated path in manifest: {dedup_path}"
        raise ValueError(msg)


def prepare_manifest(
    input_manifest: str | Path,
    path_mapping: str | Path,
    output_manifest: str | Path,
    output_id_registry: str | Path,
    start_id: int = 0,
) -> dict[str, int]:
    """Write an enriched runtime manifest and a compatible, reversible ID registry."""
    if start_id < 0:
        msg = "start_id must be non-negative"
        raise ValueError(msg)

    output_manifest = Path(output_manifest)
    output_id_registry = Path(output_id_registry)
    if output_manifest.exists() or output_id_registry.exists():
        msg = "Refusing to overwrite an existing enriched manifest or ID registry"
        raise FileExistsError(msg)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_id_registry.parent.mkdir(parents=True, exist_ok=True)

    mappings = _load_path_mappings(path_mapping)
    cursor = start_id
    batch_registry: dict[str, tuple[int, int]] = {}
    id_lookup: list[dict[str, Any]] = []
    manifest_tmp = output_manifest.with_suffix(f"{output_manifest.suffix}.incomplete")
    registry_tmp = output_id_registry.with_suffix(f"{output_id_registry.suffix}.incomplete")

    try:
        with manifest_tmp.open("x", encoding="utf-8") as output:
            for line_number, record in _iter_manifest_records(input_manifest):
                _validate_manifest_record(record, line_number, line_number - 1)
                if Path(record["path"]).suffix != ".jsonl":
                    continue

                dedup_path = _resolve_dedup_path(record["logical_path"], mappings)
                id_start = cursor
                id_end = id_start + record["num_rows"] - 1
                key = _batch_key(dedup_path)
                _validate_new_batch_key(key, dedup_path, batch_registry)

                enriched = {
                    **record,
                    "dedup_path": dedup_path,
                    "id_start": id_start,
                    "id_end": id_end,
                }
                output.write(json.dumps(enriched, separators=(",", ":")) + "\n")
                batch_registry[key] = (id_start, id_end)
                id_lookup.append(
                    {
                        "key": key,
                        "manifest_index": record["index"],
                        "path": dedup_path,
                        "id_start": id_start,
                        "id_end": id_end,
                        "num_rows": record["num_rows"],
                    }
                )
                cursor = id_end + 1

        registry_tmp.write_text(
            json.dumps(
                {
                    "next_id": cursor,
                    "batch_registry": batch_registry,
                    "id_lookup": id_lookup,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        manifest_tmp.replace(output_manifest)
        registry_tmp.replace(output_id_registry)
    except BaseException:
        manifest_tmp.unlink(missing_ok=True)
        registry_tmp.unlink(missing_ok=True)
        raise

    return {
        "num_files": len(id_lookup),
        "num_rows": cursor - start_id,
        "start_id": start_id,
        "next_id": cursor,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an embedding manifest and row-ID registry")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--path-mapping", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-id-registry", required=True)
    parser.add_argument("--start-id", type=int, default=0)
    args = parser.parse_args()
    summary = prepare_manifest(
        input_manifest=args.input_manifest,
        path_mapping=args.path_mapping,
        output_manifest=args.output_manifest,
        output_id_registry=args.output_id_registry,
        start_id=args.start_id,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
