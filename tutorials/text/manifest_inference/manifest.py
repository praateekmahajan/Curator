# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream uncompressed JSONL into a portable, byte-indexed task manifest."""

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath


def record_id(record: dict) -> str:
    """Identity includes the range and its content; contains no lineage separators."""
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def read_manifest(path: Path) -> Iterator[dict]:
    """Yield validated records without loading source data."""
    names = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            for field in ("input_file", "output_file"):
                value = PurePosixPath(record[field])
                if value.is_absolute() or ".." in value.parts or not value.parts:
                    msg = f"Unsafe {field}: {value}"
                    raise ValueError(msg)
            if record["output_file"] in names:
                msg = f"Duplicate output: {record['output_file']}"
                raise ValueError(msg)
            names.add(record["output_file"])
            if not (0 <= record["start_byte"] < record["end_byte"] <= record["size_bytes"]):
                msg = "Invalid byte range"
                raise ValueError(msg)
            if record["num_rows"] != record["end_line"] - record["start_line"] or record["num_rows"] <= 0:
                msg = "Invalid row range"
                raise ValueError(msg)
            yield record


def generate_manifest(input_dir: Path, output: Path, max_num_rows: int = 512) -> dict:  # noqa: C901, PLR0915 — one sequential streaming pass
    """One input line in memory; publish only a complete manifest, without overwriting."""
    input_dir, output = input_dir.resolve(), output.resolve()
    if max_num_rows < 1:
        msg = "max_num_rows must be positive"
        raise ValueError(msg)
    if output.is_relative_to(input_dir):
        msg = "Put the manifest outside the input directory"
        raise ValueError(msg)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_rows = total_tasks = total_files = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as dst:
            temporary = Path(dst.name)
            for path in sorted(input_dir.rglob("*.jsonl")):
                if not path.resolve().is_relative_to(input_dir):
                    msg = f"Input symlink escapes input directory: {path}"
                    raise ValueError(msg)
                before = path.stat()
                relative = path.relative_to(input_dir).as_posix()
                start_byte = start_line = line_number = task_index = 0
                digest = hashlib.sha256()
                with path.open("rb") as src:
                    while raw := src.readline():
                        if not raw.strip():
                            msg = f"Blank JSONL line: {relative}:{line_number + 1}"
                            raise ValueError(msg)
                        digest.update(raw)
                        line_number += 1
                        end_byte = src.tell()
                        if line_number - start_line == max_num_rows or end_byte == before.st_size:
                            record = {
                                "input_file": relative,
                                "output_file": f"{relative}/{path.stem}_task_{task_index}.jsonl",
                                "start_line": start_line,
                                "end_line": line_number,
                                "start_byte": start_byte,
                                "end_byte": end_byte,
                                "num_rows": line_number - start_line,
                                "size_bytes": before.st_size,
                                "mtime_ns": before.st_mtime_ns,
                                "sha256": digest.hexdigest(),
                            }
                            dst.write(json.dumps(record) + "\n")
                            total_tasks += 1
                            task_index += 1
                            start_byte, start_line = end_byte, line_number
                            digest = hashlib.sha256()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    msg = f"Input changed during indexing: {path}"
                    raise ValueError(msg)
                total_rows += line_number
                total_files += 1
            if total_tasks == 0:
                msg = "No nonempty JSONL inputs found"
                raise ValueError(msg)
            dst.flush()
            os.fsync(dst.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"files": total_files, "tasks": total_tasks, "rows": total_rows}


def shard_loads(manifest: Path, shards: int) -> list[int]:
    """Match Curator's hash of the source lineage ID (EmptyTask root is '0')."""
    if shards < 1:
        msg = "shards must be positive"
        raise ValueError(msg)
    loads = [0] * shards
    for record in read_manifest(manifest):
        digest = hashlib.sha256(f"0_{record_id(record)}".encode()).hexdigest()
        loads[int(digest[:16], 16) % shards] += record["num_rows"]
    return loads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--input-dir", type=Path, required=True)
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument("--max-num-rows", type=int, default=512)
    plan = commands.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--shards", type=int, required=True)
    plan.add_argument("--rows-per-second", type=float, required=True)
    plan.add_argument("--throughput-fraction", type=float, default=0.8)
    plan.add_argument("--setup-minutes", type=float, default=30)
    plan.add_argument("--target-minutes", type=float, default=180)
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps(generate_manifest(args.input_dir, args.manifest, args.max_num_rows), indent=2))
    else:
        if args.rows_per_second <= 0 or not 0 < args.throughput_fraction <= 1 or args.setup_minutes < 0:
            parser.error("Invalid throughput or setup budget")
        loads = shard_loads(args.manifest, args.shards)
        minutes = args.setup_minutes + max(loads) / (args.rows_per_second * args.throughput_fraction * 60)
        print(
            json.dumps(
                {
                    "shards": args.shards,
                    "rows": sum(loads),
                    "min_rows": min(loads),
                    "max_rows": max(loads),
                    "estimated_max_minutes": minutes,
                    "within_target": minutes <= args.target_minutes,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
