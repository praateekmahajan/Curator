# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner entrypoint for the manifest inference tutorial."""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from utils import write_benchmark_results

from tutorials.text.manifest_inference.pipeline import parser, run


def add_benchmark_metrics(
    results: dict[str, Any], args: argparse.Namespace, started_unix_s: float, finished_unix_s: float
) -> None:
    """Metrics are best effort and must not change the inference outcome."""
    try:
        from native_vllm_metrics import _compute_native_vllm_metrics

        config = yaml.safe_load(os.path.expandvars(args.workload_config.read_text()))["manifest_inference"]
        model = config["models"][args.model_key]
        basic = results["metrics"]
        metrics = _compute_native_vllm_metrics(
            results["tasks"],
            elapsed=basic["time_taken_s"],
            gpu_count=model["replicas"] * model["gpus_per_replica"],
            expected_requests=basic["rows_written_this_attempt"],
            client_parallelism=args.client_workers,
            writer_stage_name="named_jsonl_writer",
            include_mean_latency=False,
        )
        # Failed and checkpoint-skipped sources are absent from returned tasks.
        # Preserve the pipeline's authoritative status, including failed-task markers.
        metrics.update(
            pipeline_started_unix_s=started_unix_s,
            pipeline_finished_unix_s=finished_unix_s,
            max_individual_pipeline_wall_time_s=basic["time_taken_s"],
        )
        metrics.pop("is_success")
        metrics["is_complete"] = metrics["is_complete"] and basic["is_success"]
        metrics["benchmark_metrics_scope"] = "completed_tasks_this_attempt"
        metrics["benchmark_metrics_aggregation_success"] = True
        results["metrics"].update(metrics)
    except Exception:
        logger.exception("Benchmark metric aggregation failed; retaining basic metrics and task records")
        results["metrics"]["benchmark_metrics_aggregation_success"] = False


def main() -> int:
    arguments = parser()
    arguments.add_argument("--benchmark-results-path", type=Path, required=True)
    args = arguments.parse_args()
    started_unix_s = time.time()
    try:
        results = run(args)
    except Exception:
        logger.exception("Manifest inference failed; incomplete tasks remain pending")
        write_benchmark_results(
            {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}, args.benchmark_results_path
        )
        return 1
    add_benchmark_metrics(results, args, started_unix_s, time.time())
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
