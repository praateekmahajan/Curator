# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared native-vLLM benchmark aggregation from task performance records."""

from typing import Any

from nemo_curator.tasks import Task
from nemo_curator.tasks.utils import TaskPerfUtils


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _compute_native_vllm_metrics(  # noqa: PLR0913 — preserve benchmark API with writer/latency options
    tasks: list[Task],
    elapsed: float,
    gpu_count: int,
    expected_requests: int,
    client_parallelism: int,
    *,
    writer_stage_name: str = "parquet_writer",
    include_mean_latency: bool = True,
) -> dict[str, Any]:
    """Compute model-neutral benchmark metrics from additive task statistics."""
    task_metrics = TaskPerfUtils.aggregate_task_metrics(tasks, prefix="task")
    prefix = "task_native_vllm_client_custom"

    def custom_sum(name: str) -> float:
        return float(task_metrics.get(f"{prefix}.{name}_sum", 0.0))

    num_requests = int(custom_sum("num_requests"))
    num_successful_completions = int(custom_sum("num_successful_completions"))
    num_failed_completions = int(custom_sum("num_failed_completions"))
    num_api_attempts = int(custom_sum("num_api_attempts"))
    num_input_tokens = int(custom_sum("num_input_tokens"))
    num_output_tokens = int(custom_sum("num_output_tokens"))
    num_truncated_responses = int(custom_sum("num_truncated_responses"))
    num_empty_responses = int(custom_sum("num_empty_responses"))
    num_zero_token_responses = int(custom_sum("num_zero_token_responses"))
    num_context_length_exceeded = int(custom_sum("num_context_length_exceeded"))
    num_invalid_prompts = int(custom_sum("num_invalid_prompts"))
    num_request_retries = int(custom_sum("request_retries"))
    num_requests_retried = int(custom_sum("requests_retried"))
    request_latency_sum_s = custom_sum("request_latency_sum_s")
    num_rows_written = int(task_metrics.get(f"task_{writer_stage_name}_num_items_processed_sum", 0.0))
    stage_metrics = TaskPerfUtils.collect_stage_metrics(tasks).get("native_vllm_client", {})
    request_starts = stage_metrics.get("custom.first_request_started_unix_s", [])
    response_finishes = stage_metrics.get("custom.last_response_finished_unix_s", [])
    positive_request_starts = [float(value) for value in request_starts if float(value) > 0]
    service_window_started_unix_s = min(positive_request_starts, default=0.0)
    service_window_finished_unix_s = max(map(float, response_finishes), default=0.0)
    service_window_time_s = max(0.0, service_window_finished_unix_s - service_window_started_unix_s)
    service_window_rows_per_sec = _safe_div(num_requests, service_window_time_s)
    service_window_successful_rows_per_sec = _safe_div(num_successful_completions, service_window_time_s)
    service_window_output_tokens_per_sec = _safe_div(num_output_tokens, service_window_time_s)
    is_complete = bool(num_requests == expected_requests and num_rows_written == expected_requests)
    success = bool(
        is_complete
        and num_successful_completions == expected_requests
        and num_failed_completions == 0
        and num_empty_responses == 0
    )
    metrics = {
        "is_success": success,
        "is_complete": is_complete,
        "time_taken_s": elapsed,
        "expected_requests": expected_requests,
        "num_requests": num_requests,
        "num_successful_completions": num_successful_completions,
        "num_failed_completions": num_failed_completions,
        "num_api_attempts": num_api_attempts,
        "num_rows_written": num_rows_written,
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "num_truncated_responses": num_truncated_responses,
        "num_empty_responses": num_empty_responses,
        "num_zero_token_responses": num_zero_token_responses,
        "num_context_length_exceeded": num_context_length_exceeded,
        "num_invalid_prompts": num_invalid_prompts,
        "num_request_retries": num_request_retries,
        "num_requests_retried": num_requests_retried,
        "num_gpus": gpu_count,
        "num_client_workers": client_parallelism,
        "rows_per_sec": _safe_div(num_requests, elapsed),
        "rows_per_sec_per_gpu": _safe_div(num_requests, elapsed * gpu_count),
        "successful_rows_per_sec": _safe_div(num_successful_completions, elapsed),
        "successful_rows_per_sec_per_gpu": _safe_div(num_successful_completions, elapsed * gpu_count),
        "requests_per_sec": _safe_div(num_requests, elapsed),
        "requests_per_sec_per_gpu": _safe_div(num_requests, elapsed * gpu_count),
        "input_tokens_per_sec": _safe_div(num_input_tokens, elapsed),
        "input_tokens_per_sec_per_gpu": _safe_div(num_input_tokens, elapsed * gpu_count),
        "output_tokens_per_sec": _safe_div(num_output_tokens, elapsed),
        "output_tokens_per_sec_per_gpu": _safe_div(num_output_tokens, elapsed * gpu_count),
        "mean_input_tokens": _safe_div(num_input_tokens, num_successful_completions),
        "mean_output_tokens": _safe_div(num_output_tokens, num_successful_completions),
        "mean_request_latency_s": _safe_div(request_latency_sum_s, num_requests),
        "retry_rate": _safe_div(num_requests_retried, num_requests),
        "failure_rate": _safe_div(num_failed_completions, num_requests),
        "gpu_hours_per_million_requests": _safe_div(
            gpu_count * 1_000_000 * elapsed,
            num_requests * 3600,
        ),
        "gpu_hours_per_million_successful_rows": _safe_div(
            gpu_count * 1_000_000 * elapsed,
            num_successful_completions * 3600,
        ),
        "service_window_started_unix_s": service_window_started_unix_s,
        "service_window_finished_unix_s": service_window_finished_unix_s,
        "service_window_time_s": service_window_time_s,
        "service_window_rows_per_sec": service_window_rows_per_sec,
        "service_window_rows_per_sec_per_gpu": _safe_div(service_window_rows_per_sec, gpu_count),
        "service_window_successful_rows_per_sec": service_window_successful_rows_per_sec,
        "service_window_successful_rows_per_sec_per_gpu": _safe_div(service_window_successful_rows_per_sec, gpu_count),
        "service_window_output_tokens_per_sec": service_window_output_tokens_per_sec,
        "service_window_output_tokens_per_sec_per_gpu": _safe_div(service_window_output_tokens_per_sec, gpu_count),
    }
    if not include_mean_latency:
        metrics.pop("mean_request_latency_s")
    return metrics
