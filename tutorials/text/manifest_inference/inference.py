# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async inference shared with the native vLLM benchmark."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import pyarrow as pa
from loguru import logger

from nemo_curator.models.client import AsyncOpenAIClient
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, FailedTask, Task

from .manifest import record_id

SUCCESSFUL_FINISH_REASONS = {"stop", "length"}
TERMINAL_ROW_FAILURE_PREFIXES = frozenset(
    {
        "context_length_exceeded",
        "empty_completion",
        "invalid_prompt",
        "invalid_response",
        "reasoning_budget_exhausted",
        "unexpected_finish_reason",
        "zero_completion_tokens",
    }
)


class ModelSettings(TypedDict):
    model: str
    endpoint: str
    replicas: int
    gpus_per_replica: int
    chat_template_kwargs: dict[str, Any]
    generation_overrides: NotRequired[dict[str, Any]]
    request_extra_body: NotRequired[dict[str, Any]]
    request_timeout_s: NotRequired[float]


class InputSettings(TypedDict):
    paths: list[str]
    prompt_field: str


class GenerationSettings(TypedDict):
    max_tokens: int
    temperature: float
    top_p: float
    seed: int


class WorkloadSettings(TypedDict):
    input: InputSettings
    generation: GenerationSettings
    models: dict[str, ModelSettings]


class BenchmarkResults(TypedDict):
    params: dict[str, Any]
    metrics: dict[str, Any]
    tasks: list[Task]


class InvalidCompletionError(RuntimeError):
    """A syntactically valid API response that is not a usable completion."""


class ReasoningBudgetExhaustedError(InvalidCompletionError):
    """The output cap was consumed by reasoning before a final answer."""


def _output_columns(alias: str) -> dict[str, str]:
    return {name: f"updated_{alias}_{name}" for name in ("answer", "reasoning", "metadata")}


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ReasoningBudgetExhaustedError):
        return False
    if isinstance(exc, InvalidCompletionError):
        return True
    if isinstance(exc, ValueError):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        return True
    return status in {408, 409, 429} or status >= HTTPStatus.INTERNAL_SERVER_ERROR


def _failed_reason(exc: Exception) -> str:
    """Return a bounded, single-line reason suitable for a dataset column."""
    message = " ".join(str(exc).split()) or "no error message"
    status = getattr(exc, "status_code", None)
    if isinstance(exc, ValueError) and message.startswith("Invalid prompt in field"):
        prefix = "invalid_prompt"
    elif isinstance(exc, ReasoningBudgetExhaustedError):
        prefix = "reasoning_budget_exhausted"
    elif isinstance(exc, InvalidCompletionError) and message == "Completion answer is empty":
        prefix = "empty_completion"
    elif isinstance(exc, InvalidCompletionError) and message == "Completion token usage is zero":
        prefix = "zero_completion_tokens"
    elif isinstance(exc, InvalidCompletionError) and message.startswith("Unexpected finish reason"):
        prefix = "unexpected_finish_reason"
    elif isinstance(exc, InvalidCompletionError):
        prefix = "invalid_response"
    elif status == HTTPStatus.BAD_REQUEST and "maximum context length" in message:
        prefix = "context_length_exceeded"
    elif status is not None:
        prefix = f"http_{status}"
    else:
        prefix = type(exc).__name__
    return f"{prefix}: {message}"[:1000]


def _is_terminal_row_failure(reason: str | None) -> bool:
    """Return whether a request-specific failure is safe to persist in the output."""
    return isinstance(reason, str) and reason.partition(":")[0] in TERMINAL_ROW_FAILURE_PREFIXES


@dataclass
class NativeVLLMClientStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    model_alias: str
    model: ModelSettings
    prompt_field: str
    generation: GenerationSettings
    max_concurrent_requests: int
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    name: str = "native_vllm_client"
    # This stage spends nearly all of its time awaiting the remote vLLM server.
    # A fractional CPU prevents Ray's resource budget from limiting client tasks.
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5, gpus=0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], list(_output_columns(self.model_alias).values())

    async def query(  # noqa: C901, PLR0915 — bounded request/retry lifecycle
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int, int, float, float]:
        client = AsyncOpenAIClient(
            base_url=self.model["endpoint"],
            api_key="unused",  # pragma: allowlist secret
            max_concurrent_requests=self.max_concurrent_requests,
            max_retries=0,
            timeout=self.model.get("request_timeout_s", 1200),
        )
        client.setup()
        client.client = client.client.with_options(max_retries=0)
        pending = iter(enumerate(rows))
        results: dict[int, dict[str, Any]] = {}
        total_retries = 0
        requests_retried = 0
        first_request_started_unix_s = float("inf")
        last_response_finished_unix_s = 0.0

        async def consume() -> None:  # noqa: C901, PLR0912, PLR0915 — per-row response validation
            nonlocal first_request_started_unix_s
            nonlocal last_response_finished_unix_s
            nonlocal requests_retried, total_retries
            for index, row in pending:
                first_request_started_unix_s = min(first_request_started_unix_s, time.time())
                prompt = row.get(self.prompt_field)
                last_exc: Exception | None = None
                attempts_made = 0
                diagnostic_answer: str | None = None
                diagnostic_reasoning: str | None = None
                diagnostic_finish_reason: str | None = None
                diagnostic_prompt_tokens: int | None = None
                diagnostic_completion_tokens: int | None = None
                for attempt in range(self.max_retries + 1):
                    try:
                        if not isinstance(prompt, str) or not prompt.strip():
                            msg = f"Invalid prompt in field {self.prompt_field}"
                            raise ValueError(msg)  # noqa: TRY301
                        attempts_made += 1
                        extra_body = {
                            **self.model.get("request_extra_body", {}),
                            "chat_template_kwargs": self.model["chat_template_kwargs"],
                        }
                        response = await client.query_model_response(
                            model=self.model["model"],
                            messages=[{"role": "user", "content": prompt}],
                            generation_config={
                                **self.generation,
                                **self.model.get("generation_overrides", {}),
                                "extra_kwargs": {"extra_body": extra_body},
                            },
                        )
                        if response.usage is None or len(response.choices) != 1:
                            msg_0 = "Response must contain usage and exactly one choice"
                            raise InvalidCompletionError(msg_0)  # noqa: TRY301
                        choice = response.choices[0]
                        answer = choice.message.content or ""
                        finish_reason = choice.finish_reason or ""
                        reasoning = getattr(choice.message, "reasoning", None) or getattr(
                            choice.message, "reasoning_content", None
                        )
                        diagnostic_answer = answer if isinstance(answer, str) else None
                        diagnostic_reasoning = reasoning if isinstance(reasoning, str) else None
                        diagnostic_finish_reason = finish_reason if isinstance(finish_reason, str) else None
                        diagnostic_prompt_tokens = (
                            response.usage.prompt_tokens if isinstance(response.usage.prompt_tokens, int) else None
                        )
                        diagnostic_completion_tokens = (
                            response.usage.completion_tokens
                            if isinstance(response.usage.completion_tokens, int)
                            else None
                        )
                        if not isinstance(answer, str):
                            msg_0 = "Completion answer must be text"
                            raise InvalidCompletionError(msg_0)  # noqa: TRY301
                        if reasoning is not None and not isinstance(reasoning, str):
                            msg_0 = "Completion reasoning must be text or null"
                            raise InvalidCompletionError(msg_0)  # noqa: TRY301
                        if finish_reason not in SUCCESSFUL_FINISH_REASONS:
                            msg_0 = f"Unexpected finish reason: {finish_reason!r}"
                            raise InvalidCompletionError(msg_0)  # noqa: TRY301
                        if not isinstance(response.usage.prompt_tokens, int) or response.usage.prompt_tokens < 0:
                            msg_0 = "Prompt token usage must be a non-negative integer"
                            raise InvalidCompletionError(msg_0)  # noqa: TRY301
                        if not isinstance(response.usage.completion_tokens, int):
                            msg_0 = "Completion token usage must be an integer"
                            raise InvalidCompletionError(msg_0)  # noqa: TRY301
                        if (
                            finish_reason == "length"
                            and not answer.strip()
                            and isinstance(reasoning, str)
                            and reasoning.strip()
                        ):
                            msg_0 = "Completion exhausted max tokens in reasoning before final answer"
                            raise ReasoningBudgetExhaustedError(msg_0)  # noqa: TRY301
                        if not answer.strip():
                            msg_0 = "Completion answer is empty"
                            raise InvalidCompletionError(msg_0)  # noqa: TRY301
                        if response.usage.completion_tokens <= 0:
                            msg_0 = "Completion token usage is zero"
                            raise InvalidCompletionError(msg_0)  # noqa: TRY301
                    except Exception as exc:  # noqa: BLE001 — classify API errors for bounded retries
                        last_exc = exc
                        if attempt == self.max_retries or not _is_retryable(exc):
                            break
                        total_retries += 1
                        delay = self.retry_base_delay_s * 2**attempt
                        logger.warning(
                            "Retrying {} request {} after attempt {}/{} in {:.1f}s: {}",
                            self.model_alias,
                            index,
                            attempt + 1,
                            self.max_retries + 1,
                            delay,
                            str(exc)[:300],
                        )
                        await asyncio.sleep(delay)
                        continue

                    results[index] = {
                        "answer": answer,
                        "reasoning": reasoning or "",
                        "finish_reason": finish_reason,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "attempt_count": attempts_made,
                        "retry_count": attempt,
                        "is_success": True,
                        "failed_reason": None,
                    }
                    break

                if index not in results:
                    if last_exc is None:
                        last_exc = RuntimeError("Request ended without a response or exception")
                    results[index] = {
                        "answer": diagnostic_answer,
                        "reasoning": diagnostic_reasoning,
                        "finish_reason": diagnostic_finish_reason,
                        "prompt_tokens": diagnostic_prompt_tokens,
                        "completion_tokens": diagnostic_completion_tokens,
                        "attempt_count": attempts_made,
                        "retry_count": max(0, attempts_made - 1),
                        "is_success": False,
                        "failed_reason": _failed_reason(last_exc),
                    }
                    logger.error(
                        "Recording failed {} request {} after {} API attempt(s): {}",
                        self.model_alias,
                        index,
                        attempts_made,
                        results[index]["failed_reason"],
                    )
                if results[index]["retry_count"] > 0:
                    requests_retried += 1
                last_response_finished_unix_s = max(last_response_finished_unix_s, time.time())

        try:
            await asyncio.gather(*(consume() for _ in range(min(self.max_concurrent_requests, len(rows)))))
            if len(results) != len(rows):
                msg = f"Expected {len(rows)} terminal row results, received {len(results)}"
                raise RuntimeError(msg)
            return (
                [results[index] for index in range(len(rows))],
                total_retries,
                requests_retried,
                0.0 if first_request_started_unix_s == float("inf") else first_request_started_unix_s,
                last_response_finished_unix_s,
            )
        finally:
            await client.client.close()

    def process(self, task: DocumentBatch) -> DocumentBatch:
        table = task.to_pyarrow()
        output_columns = _output_columns(self.model_alias)
        collisions = set(output_columns.values()) & set(table.column_names)
        if collisions:
            msg = f"Refusing to overwrite existing columns: {sorted(collisions)}"
            raise ValueError(msg)
        (
            updates,
            total_retries,
            requests_retried,
            first_request_started_unix_s,
            last_response_finished_unix_s,
        ) = asyncio.run(self.query(table.to_pylist()))
        metadata_types = pa.struct(
            [
                ("finish_reason", pa.string()),
                ("prompt_tokens", pa.int64()),
                ("completion_tokens", pa.int64()),
                ("attempt_count", pa.int64()),
                ("retry_count", pa.int64()),
                ("is_success", pa.bool_()),
                ("failed_reason", pa.string()),
            ]
        )
        for name in ("answer", "reasoning"):
            table = table.append_column(
                output_columns[name], pa.array([row[name] for row in updates], type=pa.string())
            )
        table = table.append_column(
            output_columns["metadata"],
            pa.array(
                [{field.name: row[field.name] for field in metadata_types} for row in updates], type=metadata_types
            ),
        )
        successful_updates = [row for row in updates if row["is_success"]]
        failed_updates = [row for row in updates if not row["is_success"]]
        failed_reasons = [row["failed_reason"] for row in failed_updates]
        self._log_metrics(
            {
                "num_requests": len(updates),
                "num_successful_completions": len(successful_updates),
                "num_failed_completions": len(failed_updates),
                "num_api_attempts": sum(row["attempt_count"] for row in updates),
                "num_input_tokens": sum(row["prompt_tokens"] or 0 for row in successful_updates),
                "num_output_tokens": sum(row["completion_tokens"] or 0 for row in successful_updates),
                "num_truncated_responses": sum(row["finish_reason"] == "length" for row in successful_updates),
                "num_empty_responses": sum(reason.startswith("empty_completion:") for reason in failed_reasons),
                "num_zero_token_responses": sum(
                    reason.startswith("zero_completion_tokens:") for reason in failed_reasons
                ),
                "num_context_length_exceeded": sum(
                    reason.startswith("context_length_exceeded:") for reason in failed_reasons
                ),
                "num_invalid_prompts": sum(reason.startswith("invalid_prompt:") for reason in failed_reasons),
                "request_retries": total_retries,
                "requests_retried": requests_retried,
                "first_request_started_unix_s": first_request_started_unix_s,
                "last_response_finished_unix_s": last_response_finished_unix_s,
            }
        )
        return DocumentBatch(
            data=table,
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class ResumableInferenceStage(NativeVLLMClientStage):
    """Persist terminal row failures while retrying tasks with systemic failures."""

    failure_dir: str = ""

    def process(self, task: DocumentBatch) -> DocumentBatch | FailedTask:
        if not self.failure_dir:
            msg = "failure_dir is required"
            raise ValueError(msg)
        result = super().process(task)
        failed = []
        requires_task_retry = False
        record = task._metadata["manifest"]
        for offset, row in enumerate(result.to_pyarrow().to_pylist()):
            metadata = row[f"updated_{self.model_alias}_metadata"]
            if not metadata["is_success"]:
                reason = metadata["failed_reason"]
                failed.append(
                    {
                        "line": record["start_line"] + offset,
                        "reason": reason,
                        "attempts": metadata["attempt_count"],
                    }
                )
                requires_task_retry = requires_task_retry or not _is_terminal_row_failure(reason)
        if failed and requires_task_retry:
            path = Path(self.failure_dir) / f"{record_id(record)}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "manifest": record,
                        "task_id": task.task_id,
                        "model": self.model["model"],
                        "failed_rows": failed,
                    }
                )
                + "\n"
            )
            return FailedTask()
        if failed:
            logger.warning(
                "Persisting {} terminal {} row failure(s) with failure metadata",
                len(failed),
                self.model_alias,
            )
        return result
