# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from benchmarking.scripts.embedding_generation_benchmark import summarize_vllm_stage_metrics


def test_summarize_vllm_stage_metrics() -> None:
    metrics = summarize_vllm_stage_metrics(
        {
            "reader": {"process_time": np.asarray([1.0])},
            "model_vllm": {
                "process_time": np.asarray([2.0, 3.0]),
                "num_items_processed": np.asarray([100.0, 150.0]),
                "custom.input_tokens": np.asarray([1_000.0, 2_000.0]),
            },
        },
        model_worker_gpus=0.25,
    )

    assert metrics["gpu_stage_process_time_sum_s"] == 5.0
    assert metrics["gpu_stage_num_items_processed"] == 250
    assert metrics["gpu_stage_input_tokens"] == 3_000
    assert metrics["input_sequence_length_mean"] == 12.0
    assert metrics["models_per_gpu"] == 4.0
    assert metrics["throughput_items_per_gpu_second"] == 200.0
    assert metrics["throughput_input_tokens_per_gpu_second"] == 2_400.0
