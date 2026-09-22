# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.tasks import DocumentBatch
from tutorials.text.manifest_inference.writer import NamedJsonlWriter, atomic_jsonl


@pytest.mark.parametrize("compression", [None, "zstd"])
def test_atomic_replay_and_failed_serialization_preserves_old_output(tmp_path: Path, compression: str | None):
    output = tmp_path / ("task.jsonl.zst" if compression else "task.jsonl")
    atomic_jsonl(output, [{"text": "é", "a": 1}], compression=compression)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="Out of range float"):
        atomic_jsonl(output, [{"a": 2}, {"a": float("nan")}], compression=compression)
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]
    atomic_jsonl(output, [{"a": 3}], compression=compression)
    with pa.input_stream(str(output), compression=compression) as stream:
        assert json.loads(stream.read()) == {"a": 3}


def test_compressed_writer_preserves_rows_and_adds_nested_lineage(tmp_path: Path):
    lineage = {"model": "test", "request_kwargs": {"max_tokens": 8192, "extra_body": {"top_k": 20}}}
    task = DocumentBatch(
        data=pa.Table.from_pylist([{"question": "é", "answer": "hello"}]),
        _metadata={"manifest": {"output_file": "part.jsonl/task.jsonl", "num_rows": 1}},
    )
    writer = NamedJsonlWriter(str(tmp_path), compression="zstd", generation_lineage=lineage)
    result = writer.process(task)
    assert result.data == [str(tmp_path / "part.jsonl/task.jsonl.zst")]
    with pa.input_stream(result.data[0], compression="zstd") as stream:
        assert json.loads(stream.read()) == {"question": "é", "answer": "hello", "generation_lineage": lineage}
    task.data = pa.Table.from_pylist([{"generation_lineage": "source value"}])
    with pytest.raises(ValueError, match="overwrite source column"):
        writer.process(task)
