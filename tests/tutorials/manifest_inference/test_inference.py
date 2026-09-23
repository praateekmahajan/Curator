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


def test_exhausted_transport_failure_keeps_task_pending_with_durable_row_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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


def test_terminal_row_failures_are_serialized_with_successful_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tutorials.text.manifest_inference.writer import NamedJsonlWriter

    class ContextLengthError(RuntimeError):
        status_code = 400

    class Client:
        def __init__(self, **kwargs):
            self.client = SimpleNamespace(with_options=lambda **_kw: self.client, close=AsyncMock())

        def setup(self) -> None:
            pass

        async def query_model_response(self, *, messages: list[dict[str, str]], **kwargs) -> SimpleNamespace:
            prompt = messages[0]["content"]
            if prompt == "too long":
                msg = "this model's maximum context length is 32768 tokens, but the prompt contains 40000 input tokens"
                raise ContextLengthError(msg)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="" if prompt == "empty" else "answer", reasoning=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
            )

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
        max_concurrent_requests=3,
        max_retries=0,
        failure_dir=str(tmp_path / "failures"),
    )
    manifest = {
        "input_file": "part.jsonl",
        "output_file": "part.jsonl/task.jsonl",
        "start_line": 10,
        "num_rows": 3,
    }
    result = stage.process(
        DocumentBatch(
            dataset_name="test",
            data=pa.table({"question": ["ok", "too long", "empty"]}),
            _metadata={"manifest": manifest},
        )
    )

    assert isinstance(result, DocumentBatch)
    rows = result.to_pyarrow().to_pylist()
    assert rows[0]["updated_qwen_answer"] == "answer"
    assert rows[0]["updated_qwen_metadata"]["is_success"] is True
    assert rows[1]["updated_qwen_answer"] is None
    assert rows[1]["updated_qwen_metadata"]["is_success"] is False
    assert rows[1]["updated_qwen_metadata"]["failed_reason"].startswith("context_length_exceeded:")
    assert rows[2]["updated_qwen_answer"] == ""
    assert rows[2]["updated_qwen_metadata"]["is_success"] is False
    assert rows[2]["updated_qwen_metadata"]["failed_reason"] == "empty_completion: Completion answer is empty"
    assert not (tmp_path / "failures").exists()

    output = NamedJsonlWriter(
        str(tmp_path / "output"), compression="zstd", generation_lineage={"request_kwargs": {"max_tokens": 4}}
    ).process(result)
    with pa.input_stream(output.data[0], compression="zstd") as stream:
        serialized_rows = [json.loads(line) for line in stream.read().decode().splitlines()]
    assert len(serialized_rows) == 3
    assert serialized_rows[1]["updated_qwen_metadata"]["is_success"] is False
    assert serialized_rows[1]["updated_qwen_metadata"]["failed_reason"].startswith("context_length_exceeded:")
    assert serialized_rows[2]["updated_qwen_metadata"]["failed_reason"] == (
        "empty_completion: Completion answer is empty"
    )
    from tutorials.text.manifest_inference.verification import _validate_row

    assert [_validate_row(row, "qwen", 4, "serialized test row") for row in serialized_rows] == [
        True,
        False,
        False,
    ]


def test_systemic_failure_keeps_mixed_batch_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class ServerError(RuntimeError):
        status_code = 503

    class Client:
        def __init__(self, **kwargs):
            self.client = SimpleNamespace(with_options=lambda **_kw: self.client, close=AsyncMock())

        def setup(self) -> None:
            pass

        async def query_model_response(self, *, messages: list[dict[str, str]], **kwargs) -> SimpleNamespace:
            prompt = messages[0]["content"]
            if prompt == "server error":
                msg = "service unavailable"
                raise ServerError(msg)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="" if prompt == "empty" else "answer", reasoning=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
            )

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
        max_concurrent_requests=2,
        max_retries=0,
        failure_dir=str(tmp_path),
    )
    result = stage.process(
        DocumentBatch(
            dataset_name="test",
            data=pa.table({"question": ["empty", "server error"]}),
            _metadata={"manifest": {"start_line": 20}},
        )
    )

    assert isinstance(result, FailedTask)
    diagnostic = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert [row["line"] for row in diagnostic["failed_rows"]] == [20, 21]
    assert diagnostic["failed_rows"][0]["reason"].startswith("empty_completion:")
    assert diagnostic["failed_rows"][1]["reason"].startswith("http_503:")


@pytest.mark.parametrize("alias", ["qwen", "deepseek"])
def test_output_metadata_is_nested_and_answers_stay_aligned(alias: str, monkeypatch: pytest.MonkeyPatch):
    class Client:
        def __init__(self, **kwargs):
            self.client = SimpleNamespace(with_options=lambda **_kw: self.client, close=AsyncMock())

        def setup(self) -> None:
            pass

        async def query_model_response(self, *, messages: list[dict[str, str]], **kwargs) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=messages[0]["content"] + " answer", reasoning=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
            )

    monkeypatch.setattr(inference, "AsyncOpenAIClient", Client)
    stage = inference.NativeVLLMClientStage(
        model_alias=alias,
        model={"model": "test", "endpoint": "http://unused/v1", "chat_template_kwargs": {}},
        prompt_field="question",
        generation={"max_tokens": 8192},
        max_concurrent_requests=2,
    )
    output = stage.process(
        DocumentBatch(dataset_name="test", data=pa.table({"question": ["first", "second"]}))
    ).to_pyarrow()
    assert output.column_names == [
        "question",
        f"updated_{alias}_answer",
        f"updated_{alias}_reasoning",
        f"updated_{alias}_metadata",
    ]
    for row in output.to_pylist():
        assert row[f"updated_{alias}_answer"] == row["question"] + " answer"
        assert row[f"updated_{alias}_metadata"] == {
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "attempt_count": 1,
            "retry_count": 0,
            "is_success": True,
            "failed_reason": None,
        }
