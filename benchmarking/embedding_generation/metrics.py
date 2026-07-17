# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def summarize_vllm_stage_metrics(
    stage_metrics: dict[str, dict[str, Any]],
    model_worker_gpus: float,
) -> dict[str, float | int]:
    """Derive comparable per-GPU throughput and sequence metrics for vLLM stages."""
    gpu_stage_metrics = [values for stage_name, values in stage_metrics.items() if stage_name.endswith("_vllm")]
    process_time_sum = sum(
        float(values["process_time"].sum()) for values in gpu_stage_metrics if "process_time" in values
    )
    num_items_processed = sum(
        int(values["num_items_processed"].sum()) for values in gpu_stage_metrics if "num_items_processed" in values
    )
    input_tokens = sum(
        int(values["custom.input_tokens"].sum()) for values in gpu_stage_metrics if "custom.input_tokens" in values
    )
    models_per_gpu = 1.0 / model_worker_gpus if model_worker_gpus else 0.0
    throughput_items = models_per_gpu * num_items_processed / process_time_sum if process_time_sum > 0 else 0.0
    throughput_tokens = models_per_gpu * input_tokens / process_time_sum if process_time_sum > 0 else 0.0
    sequence_length_mean = input_tokens / num_items_processed if num_items_processed > 0 else 0.0
    return {
        "gpu_stage_process_time_sum_s": process_time_sum,
        "gpu_stage_num_items_processed": num_items_processed,
        "gpu_stage_input_tokens": input_tokens,
        "input_sequence_length_mean": sequence_length_mean,
        "models_per_gpu": models_per_gpu,
        "throughput_items_per_gpu_second": throughput_items,
        "throughput_input_tokens_per_gpu_second": throughput_tokens,
    }
