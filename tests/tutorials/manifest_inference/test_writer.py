# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pyarrow as pa
import pytest

from nemo_curator.tasks import DocumentBatch
from tutorials.text.manifest_inference.writer import NamedJsonlWriter


def test_compressed_writer_preserves_rows_and_adds_nested_lineage(tmp_path: Path):
    lineage = {"model": "test", "request_kwargs": {"max_tokens": 8192, "extra_body": {"top_k": 20}}}
    task = DocumentBatch(
        dataset_name="test",
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
