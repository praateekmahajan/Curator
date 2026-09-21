# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner entrypoint for the manifest inference tutorial."""

from pathlib import Path

from loguru import logger
from utils import write_benchmark_results

from tutorials.text.manifest_inference.pipeline import parser, run


def main() -> int:
    arguments = parser()
    arguments.add_argument("--benchmark-results-path", type=Path, required=True)
    args = arguments.parse_args()
    try:
        results = run(args)
    except Exception:
        logger.exception("Manifest inference failed; incomplete tasks remain pending")
        write_benchmark_results(
            {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}, args.benchmark_results_path
        )
        return 1
    write_benchmark_results(results, args.benchmark_results_path)
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
