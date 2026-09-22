# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one model and one logical manifest shard with durable source checkpoints."""

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from nemo_curator.backends.failed_task_markers import (
    FAILED_TASKS_DIR_ENV_VAR,
    configure_failed_task_manifest_dir,
    configure_slurm_array_failed_task_manifest_dir,
    failed_task_manifest_exists,
)
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.backends.slurm_array import SlurmArrayConfig
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline

from .inference import GenerationSettings, ModelSettings, ResumableInferenceStage
from .manifest import read_manifest
from .stages import ManifestFilePartitioningStage, SpecificJsonlReader
from .writer import NamedJsonlWriter


def build_pipeline(
    args: argparse.Namespace,
    model: ModelSettings,
    generation: GenerationSettings,
    failure_dir: Path,
    generation_lineage: dict | None = None,
) -> Pipeline:
    pipeline = Pipeline(name=f"manifest_{args.model_key}")
    pipeline.add_stage(ManifestFilePartitioningStage(str(args.manifest), str(args.input_dir)))
    pipeline.add_stage(SpecificJsonlReader())
    pipeline.add_stage(
        ResumableInferenceStage(
            model_alias=args.model_key,
            model=model,
            prompt_field=args.prompt_field,
            generation=generation,
            max_concurrent_requests=args.max_concurrent_requests,
            max_retries=args.max_retries,
            retry_base_delay_s=args.retry_base_delay_s,
            failure_dir=str(failure_dir),
        ).with_(num_workers=args.client_workers)
    )
    pipeline.add_stage(
        NamedJsonlWriter(
            str(args.output_dir), compression=args.output_compression, generation_lineage=generation_lineage
        )
    )
    # The benchmark runner can start Ray before this driver's attempt path exists.
    # Explicit stage environments propagate shard/failure settings to those workers.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("NEMO_CURATOR_SLURM_ARRAY_") or key == FAILED_TASKS_DIR_ENV_VAR
    }
    for stage in pipeline.stages:
        stage.runtime_env = {"env_vars": environment}
    return pipeline


def run(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(os.path.expandvars(args.workload_config.read_text()))["manifest_inference"]
    model = config["models"][args.model_key]
    generation_lineage = None
    if args.server_cli_file is not None:
        server = json.loads(args.server_cli_file.read_text())
        if server["model"] != model["model"]:
            msg = "Captured server model does not match request model"
            raise ValueError(msg)
        generation_lineage = {
            **server,
            "request_kwargs": {
                **config["generation"],
                **model.get("generation_overrides", {}),
                "extra_body": {
                    **model.get("request_extra_body", {}),
                    "chat_template_kwargs": model["chat_template_kwargs"],
                },
            },
        }
    if "${" in model["endpoint"]:
        msg = "Unresolved model endpoint"
        raise ValueError(msg)
    if (
        min(args.client_workers, args.max_concurrent_requests) < 1
        or args.max_retries < 0
        or args.retry_base_delay_s < 0
    ):
        msg = "Invalid concurrency/retry settings"
        raise ValueError(msg)
    expected_rows = sum(record["num_rows"] for record in read_manifest(args.manifest))
    shard = SlurmArrayConfig.from_env()
    checkpoint = args.checkpoint_path.resolve()
    os.environ.pop(FAILED_TASKS_DIR_ENV_VAR, None)
    attempt = (
        configure_slurm_array_failed_task_manifest_dir(checkpoint, shard.shard_index)
        if shard
        else configure_failed_task_manifest_dir(checkpoint)
    )
    failures = attempt / "tasks"
    pipeline = build_pipeline(args, model, config["generation"], failures, generation_lineage)
    client = RayClient()
    started = time.monotonic()
    try:
        client.start()
        tasks = pipeline.run(RayDataExecutor(), checkpoint_path=checkpoint)
        successful = not failed_task_manifest_exists(attempt)
        return {
            "params": {**vars(args), "generation_lineage": generation_lineage},
            "tasks": tasks or [],
            "metrics": {
                "is_success": successful,
                "time_taken_s": time.monotonic() - started,
                "manifest_rows": expected_rows,
                "tasks_written_this_attempt": len(tasks or []),
                "rows_written_this_attempt": sum(t._metadata["manifest"]["num_rows"] for t in tasks or []),
            },
        }
    finally:
        client.stop()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "input-dir", "output-dir", "checkpoint-path", "workload-config"):
        result.add_argument(f"--{name}", type=Path, required=True)
    result.add_argument("--model-key", required=True)
    result.add_argument("--prompt-field", default="curator_question")
    result.add_argument("--client-workers", type=int, default=64)
    result.add_argument("--max-concurrent-requests", type=int, required=True)
    result.add_argument("--max-retries", type=int, default=3)
    result.add_argument("--retry-base-delay-s", type=float, default=1)
    result.add_argument("--output-compression", choices=["zstd"])
    result.add_argument("--server-cli-file", type=Path)
    return result


def main() -> int:
    results = run(parser().parse_args())
    print(json.dumps(results["metrics"], indent=2))
    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
