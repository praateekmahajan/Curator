# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU Curator pipelines against independently managed OpenAI-compatible servers."""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.models.client import AsyncOpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.tasks import DocumentBatch, Task


class ModelSettings(TypedDict):
    model: str
    endpoint: str
    replicas: int
    gpus_per_replica: int
    chat_template_kwargs: dict[str, bool]


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


class ResponseRecord(TypedDict):
    model_alias: str
    model: str
    source_id: str
    prompt: str
    response: str
    reasoning: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    request_latency_s: float


class BenchmarkResults(TypedDict):
    params: dict[str, Any]
    metrics: dict[str, float]
    tasks: list[Task]


@dataclass
class NativeVLLMClientStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    models: dict[str, ModelSettings]
    prompt_field: str
    generation: GenerationSettings
    concurrency: dict[str, int]
    name: str = "native_vllm_client"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1, gpus=0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.prompt_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    async def query(self, rows: list[dict[str, Any]]) -> list[ResponseRecord]:
        async def query_model(alias: str, config: ModelSettings) -> list[ResponseRecord]:
            limit = self.concurrency[alias]
            client = AsyncOpenAIClient(
                base_url=config["endpoint"],
                api_key="unused",  # pragma: allowlist secret
                max_concurrent_requests=limit,
                max_retries=0,
                timeout=600,
            )
            client.setup()
            # Disable the SDK's additional retry layer so failed runs cannot look successful.
            client.client = client.client.with_options(max_retries=0)
            pending = iter(enumerate(rows))
            results: dict[int, ResponseRecord] = {}

            async def consume() -> None:
                for index, row in pending:
                    prompt = row[self.prompt_field]
                    if not isinstance(prompt, str) or not prompt.strip():
                        msg = f"Invalid prompt in field {self.prompt_field}"
                        raise ValueError(msg)
                    started = time.perf_counter()
                    response = await client.query_model_response(
                        model=config["model"],
                        messages=[{"role": "user", "content": prompt}],
                        generation_config={
                            **self.generation,
                            "extra_kwargs": {"extra_body": {"chat_template_kwargs": config["chat_template_kwargs"]}},
                        },
                    )
                    if response.usage is None or len(response.choices) != 1:
                        msg = "Server must return token usage and exactly one choice"
                        raise ValueError(msg)
                    choice = response.choices[0]
                    results[index] = {
                        "model_alias": alias,
                        "model": config["model"],
                        "source_id": str(row.get("curator_int_id", index)),
                        "prompt": prompt,
                        "response": choice.message.content or "",
                        "reasoning": getattr(choice.message, "reasoning", None)
                        or getattr(choice.message, "reasoning_content", None)
                        or "",
                        "finish_reason": choice.finish_reason,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "request_latency_s": time.perf_counter() - started,
                    }

            try:
                await asyncio.gather(*(consume() for _ in range(min(limit, len(rows)))))
                return [results[index] for index in range(len(rows))]
            finally:
                await client.client.close()

        batches = await asyncio.gather(*(query_model(alias, config) for alias, config in self.models.items()))
        return [row for batch in batches for row in batch]

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return DocumentBatch(
            data=pa.Table.from_pylist(asyncio.run(self.query(task.to_pyarrow().to_pylist()))),
            dataset_name=task.dataset_name,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


def _run_pipeline(  # noqa: PLR0913
    config: WorkloadSettings,
    aliases: list[str],
    output: Path,
    batch_size: int,
    replicas: int,
    shared: bool,
) -> tuple[list[Task], float]:
    models = {alias: config["models"][alias] for alias in aliases}
    workers = replicas if shared else replicas * models[aliases[0]]["replicas"]
    concurrency = {alias: batch_size * (model["replicas"] if shared else 1) for alias, model in models.items()}
    pipeline = Pipeline(name="native_vllm_" + "_".join(aliases))
    pipeline.add_stage(
        JsonlReader(
            file_paths=config["input"]["paths"],
            # Two 64-row files let each shared worker keep 8 x 16 Qwen requests active.
            files_per_partition=2 if shared else 1,
            fields=["curator_int_id", config["input"]["prompt_field"]],
        )
    )
    pipeline.add_stage(
        NativeVLLMClientStage(
            models=models,
            prompt_field=config["input"]["prompt_field"],
            generation=config["generation"],
            concurrency=concurrency,
        ).with_(num_workers=workers)
    )
    pipeline.add_stage(ParquetWriter(path=str(output), mode="error"))
    started = time.perf_counter()
    tasks = pipeline.run(setup_executor("ray_data"))
    if tasks is None:
        msg = "Pipeline returned no output tasks"
        raise RuntimeError(msg)
    return tasks, time.perf_counter() - started


def _compute_native_vllm_metrics(paths: list[Path], elapsed: float) -> dict[str, float]:
    metrics: dict[str, dict[str, float]] = {}
    for path in paths:
        for file in Path(path).glob("*.parquet"):
            table = pq.read_table(
                file,
                columns=[
                    "model_alias",
                    "prompt_tokens",
                    "completion_tokens",
                    "request_latency_s",
                    "finish_reason",
                    "response",
                ],
            )
            for row in table.to_pylist():
                total = metrics.setdefault(
                    row["model_alias"],
                    {
                        "requests": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_sum_s": 0.0,
                        "length_truncated": 0,
                        "empty_responses": 0,
                    },
                )
                total["requests"] += 1
                total["prompt_tokens"] += row["prompt_tokens"]
                total["completion_tokens"] += row["completion_tokens"]
                total["latency_sum_s"] += row["request_latency_s"]
                total["length_truncated"] += row["finish_reason"] == "length"
                total["empty_responses"] += not row["response"].strip()
    flat = {"pipeline_wall_time_s": elapsed}
    for alias, total in metrics.items():
        count = total["requests"]
        total.update(
            requests_per_s=count / elapsed,
            input_tokens_per_s=total["prompt_tokens"] / elapsed,
            output_tokens_per_s=total["completion_tokens"] / elapsed,
            mean_input_tokens=total["prompt_tokens"] / count,
            mean_output_tokens=total["completion_tokens"] / count,
            mean_request_latency_s=total["latency_sum_s"] / count,
            gpu_hours_per_million_requests=8 * 1_000_000 * elapsed / count / 3600,
        )
        flat.update({f"{alias}_{key}": value for key, value in total.items()})
    if not metrics:
        msg = "No output requests were written"
        raise ValueError(msg)
    return flat


def run_native_vllm_benchmark(args: argparse.Namespace) -> BenchmarkResults:
    """Run the configured CPU pipeline and return standard benchmark artifacts."""
    replicas = args.num_client_replicas_per_model_replica
    config = cast(
        "WorkloadSettings", yaml.safe_load(os.path.expandvars(args.workload_config.read_text()))["native_vllm"]
    )
    if args.input_path is not None:
        config["input"]["paths"] = [str(args.input_path)]
    for model in config["models"].values():
        if "${" in model["endpoint"] or model["replicas"] < 1:
            msg = "Model endpoints must be resolved and replica counts positive"
            raise ValueError(msg)
    root = args.benchmark_results_path
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    started_unix = time.time()
    started = time.perf_counter()
    aliases = list(config["models"]) if args.layout == "shared" else [args.layout]
    output = root / "parquet" / args.layout
    paths.append(output)
    runs = [_run_pipeline(config, aliases, output, args.batch_size, replicas, args.layout == "shared")]
    elapsed = time.perf_counter() - started
    metrics = _compute_native_vllm_metrics(paths, elapsed)
    metrics["pipeline_started_unix_s"] = started_unix
    metrics["pipeline_finished_unix_s"] = started_unix + elapsed
    metrics["max_individual_pipeline_wall_time_s"] = max(duration for _, duration in runs)
    metrics["is_success"] = 1.0
    logger.success("Native vLLM benchmark completed in {:.2f}s", elapsed)
    return {
        "params": {**vars(args), "executor": "ray_data", "workload": config},
        "metrics": metrics,
        "tasks": [task for tasks, _ in runs for task in tasks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-config", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, help="Override with a fixed benchmark sample directory")
    parser.add_argument("--layout", choices=["qwen", "deepseek", "shared"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-client-replicas-per-model-replica", type=int, required=True)
    parser.add_argument("--benchmark-results-path", type=Path, required=True)
    args = parser.parse_args()
    replicas = args.num_client_replicas_per_model_replica
    if min(args.batch_size, replicas) < 1:
        parser.error("Batch size and client replicas must be positive")
    try:
        results = run_native_vllm_benchmark(args)
    except Exception:
        logger.exception("Native vLLM benchmark failed")
        write_benchmark_results({"params": vars(args), "metrics": {"is_success": 0}}, args.benchmark_results_path)
        return 1
    write_benchmark_results(results, args.benchmark_results_path)
    print(json.dumps(results["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
