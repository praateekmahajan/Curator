# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from tutorials.text.manifest_inference.manifest import generate_manifest, read_manifest, record_id, shard_loads


def test_streaming_ranges_and_portable_identity(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    payload = b'{"text":"a"}\r\n' + '{"text":"é"}\n{"text":"last"}'.encode()
    (source / "part.jsonl").write_bytes(payload)
    manifest = tmp_path / "manifest.jsonl"
    assert generate_manifest(source, manifest, 2) == {"files": 1, "tasks": 2, "rows": 3}
    records = list(read_manifest(manifest))
    assert [(r["start_line"], r["end_line"]) for r in records] == [(0, 2), (2, 3)]
    assert records[0]["end_byte"] == records[1]["start_byte"]
    assert records[1]["end_byte"] == len(payload)
    assert len({record_id(r) for r in records}) == 2
    assert str(source) not in manifest.read_text()
    assert sum(shard_loads(manifest, 7)) == 3
    with pytest.raises(FileExistsError):
        generate_manifest(source, manifest, 2)


def test_blank_line_does_not_publish_partial_manifest(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "bad.jsonl").write_text("{}\n\n")
    manifest = tmp_path / "manifest.jsonl"
    with pytest.raises(ValueError, match="Blank"):
        generate_manifest(source, manifest, 1)
    assert not manifest.exists()


def test_reject_duplicate_output_and_path_escape(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "part.jsonl").write_text('{"a":1}\n')
    manifest = tmp_path / "manifest.jsonl"
    generate_manifest(source, manifest)
    record = next(read_manifest(manifest))
    manifest.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        list(read_manifest(manifest))
    record["input_file"] = "../escape.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="Unsafe"):
        list(read_manifest(manifest))
