# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pyarrow as pa
import pytest

from nemo_curator.tasks import DocumentBatch, FailedTask
from tutorials.text.manifest_inference import inference


def test_exhausted_row_fails_task_with_durable_row_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class Client:
        def __init__(self, **kwargs):
            self.client = SimpleNamespace(with_options=lambda **_kw: self.client, close=AsyncMock())

        def setup(self) -> None:
            pass

        async def query_model_response(self, **kwargs) -> SimpleNamespace:
            msg = "service unavailable"
            raise RuntimeError(msg)

    monkeypatch.setattr(inference, "AsyncOpenAIClient", Client)
    stage = inference.ResumableInferenceStage(
        model_alias="qwen",
        model={
            "model": "test",
            "endpoint": "http://unused/v1",
            "replicas": 1,
            "gpus_per_replica": 1,
            "chat_template_kwargs": {},
        },
        prompt_field="question",
        generation={"max_tokens": 4, "temperature": 0, "top_p": 1, "seed": 42},
        max_concurrent_requests=1,
        max_retries=1,
        retry_base_delay_s=0,
        failure_dir=str(tmp_path),
    )
    result = stage.process(
        DocumentBatch(
            dataset_name="test", data=pa.table({"question": ["hello"]}), _metadata={"manifest": {"start_line": 512}}
        )
    )
    assert isinstance(result, FailedTask)
    diagnostic = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert diagnostic["failed_rows"][0]["line"] == 512
    assert diagnostic["failed_rows"][0]["attempts"] == 2
    assert "service unavailable" in diagnostic["failed_rows"][0]["reason"]
