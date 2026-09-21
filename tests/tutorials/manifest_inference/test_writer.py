# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from tutorials.text.manifest_inference.writer import atomic_jsonl


def test_atomic_replay_and_failed_serialization_preserves_old_output(tmp_path: Path):
    output = tmp_path / "task.jsonl"
    atomic_jsonl(output, [{"text": "é", "a": 1}])
    original = output.read_bytes()
    with pytest.raises(ValueError, match="Out of range float"):
        atomic_jsonl(output, [{"a": 2}, {"a": float("nan")}])
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]
    atomic_jsonl(output, [{"a": 3}])
    assert json.loads(output.read_text()) == {"a": 3}
