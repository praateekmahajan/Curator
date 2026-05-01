# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: PLR0913, ANN401

"""Nemotron-CC SDG benchmark using NeMo Data Designer.

Translates step_4-sdg.py (Nemotron repo): AsyncOpenAIClient + GenerationConfig
become NDD's ModelProvider + ModelConfig + ChatCompletionInferenceParams, and
the per-task SDG stages come from
``nemo_curator.stages.synthetic.nemotron_cc.nemo_data_designer.nemotron_cc``.
Preprocessing/postprocessing helpers are shared with the existing tutorial.

Backends (matches ndd_benchmark.py):
  --inference-server-type  ray-serve | dynamo | nvidia-nim
  --engine-kwargs          JSON vLLM kwargs, e.g. '{"tensor_parallel_size": 4}'
  --autoscaling-config     JSON Ray Serve autoscaling, e.g. '{"min_replicas": 1, "max_replicas": 4}'
                           For ``dynamo``, autoscaling is unsupported: ``min_replicas`` must
                           equal ``max_replicas`` and is used as a static ``num_replicas``.
  --model-path             Optional absolute path to a local model snapshot dir. When set
                           (ray-serve/dynamo only), used as ``model_identifier`` so vLLM
                           loads weights from disk; ``--model-id`` is still used as the
                           served model name in /v1/models. Ignored for ``nvidia-nim``.

Usage:
    python benchmarking/run.py --config benchmarking/nemotron_cc_sdg_ndd.yaml
    python benchmarking/run.py --config benchmarking/nemotron_cc_sdg_ndd.yaml \\
        --entries "diverse_qa and ray_serve"
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import data_designer.config as dd
from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.synthetic.nemotron_cc.nemo_data_designer.nemotron_cc import (
    DistillStage,
    DiverseQAStage,
    ExtractKnowledgeStage,
    KnowledgeListStage,
)
from nemo_curator.stages.synthetic.nemotron_cc.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
)
from nemo_curator.stages.text.filters import Filter
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.core.serve import InferenceServer

# Reuse the shared preprocessing/postprocessing helpers from the SDG tutorial.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tutorials" / "synthetic" / "nemotron_cc"))
from nemotron_cc_pipelines import (  # noqa: E402
    add_distill_postprocessing_pipeline,
    add_diverse_qa_postprocessing_pipeline,
    add_extract_knowledge_postprocessing_pipeline,
    add_knowledge_list_postprocessing_pipeline,
    add_preprocessing_pipeline,
)

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "diverse_qa": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": DIVERSE_QA_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1000,
        "max_output_tokens": 600,
        "stage_cls": DiverseQAStage,
        "output_field": "diverse_qa",
        "postprocessing_fn": add_diverse_qa_postprocessing_pipeline,
    },
    "distill": {
        "system_prompt": NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
        "prompt_template": DISTILL_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 10,
        "max_input_tokens": 2000,
        "max_output_tokens": 1600,
        "stage_cls": DistillStage,
        "output_field": "distill",
        "postprocessing_fn": add_distill_postprocessing_pipeline,
    },
    "extract_knowledge": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1400,
        "max_output_tokens": 1400,
        "stage_cls": ExtractKnowledgeStage,
        "output_field": "extract_knowledge",
        "postprocessing_fn": add_extract_knowledge_postprocessing_pipeline,
    },
    "knowledge_list": {
        "system_prompt": NEMOTRON_CC_SYSTEM_PROMPT,
        "prompt_template": KNOWLEDGE_LIST_PROMPT_TEMPLATE,
        "min_document_tokens": 30,
        "min_segment_tokens": 30,
        "max_input_tokens": 1000,
        "max_output_tokens": 600,
        "stage_cls": KnowledgeListStage,
        "output_field": "knowledge_list",
        "postprocessing_fn": add_knowledge_list_postprocessing_pipeline,
    },
}

DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 0.9


# DocumentJoiner downstream uses document_id_field='id' to regroup segments back
# into documents. Real Nemotron-CC parquet has no `id` column; mint one if absent.
@processing_stage(name="add-document-id")
def add_document_id(batch: DocumentBatch) -> DocumentBatch:
    df = batch.to_pandas()
    if "id" not in df.columns:
        df["id"] = range(len(df))
        batch.data = df
    return batch


# ---------------------------------------------------------------------------
# InferenceServer helpers (mirrors ndd_benchmark.py)
# ---------------------------------------------------------------------------


def _start_ray_serve_inference_server(
    model_id: str,
    engine_kwargs: dict[str, Any] | None = None,
    autoscaling_config: dict[str, Any] | None = None,
    model_path: str | None = None,
    port: int | None = None,
) -> "InferenceServer":
    """Start a local Ray Serve-backed InferenceServer and return it.

    If ``model_path`` is set, vLLM loads weights from that local path while
    ``model_id`` is used as the served name in ``/v1/models``.
    """
    from nemo_curator.core.serve import InferenceServer, RayServeModelConfig

    engine_kwargs = engine_kwargs or {}
    autoscaling_config = autoscaling_config or {"min_replicas": 1, "max_replicas": 1}

    server_config = RayServeModelConfig(
        model_identifier=model_path or model_id,
        model_name=model_id if model_path else None,
        deployment_config={"autoscaling_config": autoscaling_config},
        engine_kwargs=engine_kwargs,
    )

    server_kwargs: dict[str, Any] = {"models": [server_config]}
    if port is not None:
        server_kwargs["port"] = port
    server = InferenceServer(**server_kwargs)
    server.start()
    return server


def _start_dynamo_inference_server(
    model_id: str,
    engine_kwargs: dict[str, Any] | None = None,
    autoscaling_config: dict[str, Any] | None = None,
    model_path: str | None = None,
    port: int | None = None,
) -> "InferenceServer":
    """Start a local Dynamo-backed InferenceServer and return it.

    Dynamo has no autoscaling — ``min_replicas`` and ``max_replicas`` (when
    supplied) must match and are used as a static ``num_replicas``.
    If ``model_path`` is set, vLLM loads weights from that local path while
    ``model_id`` is used as the served name in ``/v1/models``.
    """
    from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer

    engine_kwargs = engine_kwargs or {}
    num_replicas = 1
    if autoscaling_config:
        min_r = autoscaling_config.get("min_replicas", 1)
        max_r = autoscaling_config.get("max_replicas", min_r)
        if min_r != max_r:
            msg = (
                f"Dynamo backend does not support autoscaling; min_replicas ({min_r}) "
                f"must equal max_replicas ({max_r})."
            )
            raise ValueError(msg)
        num_replicas = min_r

    model_config = DynamoVLLMModelConfig(
        model_identifier=model_path or model_id,
        model_name=model_id if model_path else None,
        engine_kwargs=engine_kwargs,
        num_replicas=num_replicas,
    )
    server_kwargs: dict[str, Any] = {"models": [model_config], "backend": DynamoServerConfig()}
    if port is not None:
        server_kwargs["port"] = port
    server = InferenceServer(**server_kwargs)
    server.start()
    return server


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(  # noqa: C901, PLR0912, PLR0915
    task: str,
    inference_server_type: str,
    model_id: str,
    model_path: str | None,
    base_url: str,
    api_key: str,
    tokenizer: str,
    input_path: str,
    output_path: str,
    output_format: str,
    executor: str,
    engine_kwargs: dict[str, Any] | None,
    autoscaling_config: dict[str, Any] | None,
    max_parallel_requests: int,
    ndd_timeout: int,
    server_port: int | None,
    bucket_field: str,
    min_bucket_threshold: int | None,
    num_files: int | None,
    **kwargs: Any,  # noqa: ARG001
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    output_path_obj = Path(output_path).absolute()
    output_path_obj.mkdir(parents=True, exist_ok=True)

    input_files = load_dataset_files(Path(input_path), dataset_ratio=1.0, keep_extensions="parquet")
    if num_files is not None and num_files > 0:
        logger.info(f"Using {num_files} of {len(input_files)} input files")
        input_files = input_files[:num_files]
    logger.info(f"Input files: {len(input_files)}")
    if not input_files:
        msg = f"No parquet files found under {input_path}"
        raise FileNotFoundError(msg)

    inference_server = None
    serve_startup_s = 0.0
    model_providers = None

    if inference_server_type in ("ray-serve", "dynamo"):
        os.environ.setdefault("NVIDIA_API_KEY", "none")
        logger.info(f"Starting local {inference_server_type} InferenceServer with engine_kwargs={engine_kwargs}")
        serve_start = time.perf_counter()
        starter = (
            _start_ray_serve_inference_server
            if inference_server_type == "ray-serve"
            else _start_dynamo_inference_server
        )
        inference_server = starter(
            model_id,
            engine_kwargs,
            autoscaling_config,
            model_path=model_path,
            port=server_port,
        )
        serve_startup_s = time.perf_counter() - serve_start
        logger.info(f"InferenceServer ready at {inference_server.endpoint} (startup: {serve_startup_s:.1f}s)")

        provider_name = "local"
        model_providers = [
            dd.ModelProvider(
                name=provider_name,
                endpoint=inference_server.endpoint,
                provider_type="openai",
                api_key="unused",  # pragma: allowlist secret
            )
        ]
        provider_skip_health_check = True
    elif inference_server_type == "nvidia-nim":
        if not api_key:
            api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            msg = "NVIDIA_API_KEY (or --api-key) is required for --inference-server-type=nvidia-nim."
            raise OSError(msg)
        provider_name = "nvidia"
        model_providers = [
            dd.ModelProvider(
                name=provider_name,
                endpoint=base_url,
                provider_type="openai",
                api_key=api_key,
            )
        ]
        provider_skip_health_check = False
    else:
        msg = f"Unknown inference_server_type: {inference_server_type}"
        raise ValueError(msg)

    try:
        model_alias = model_id
        task_config = TASK_CONFIGS[task]
        inference_params = dd.ChatCompletionInferenceParams(
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            max_tokens=task_config["max_output_tokens"],
            max_parallel_requests=max_parallel_requests,
            timeout=ndd_timeout,
        )
        model_config = dd.ModelConfig(
            alias=model_alias,
            model=model_id,
            provider=provider_name,
            inference_parameters=inference_params,
            skip_health_check=provider_skip_health_check,
        )

        # Helpers in nemotron_cc_pipelines.py read tokenizer + hf_token off args.
        helper_args = argparse.Namespace(
            tokenizer=AutoTokenizer.from_pretrained(tokenizer),
            hf_token=os.environ.get("HF_TOKEN", ""),
        )

        pipeline = Pipeline(
            name=f"nemotron_cc_sdg_ndd_{task}",
            description=f"Nemotron-CC SDG (NDD): {task}",
        )
        pipeline.add_stage(
            ParquetReader(
                file_paths=input_files,
                read_kwargs={"engine": "pyarrow", "dtype_backend": "pyarrow"},
            )
        )
        pipeline.add_stage(add_document_id)

        if min_bucket_threshold is not None and bucket_field:
            threshold = int(min_bucket_threshold)
            pipeline.add_stage(
                Filter(
                    filter_fn=lambda x, _t=threshold: (
                        x is not None and not (isinstance(x, float) and math.isnan(x)) and int(x) >= _t
                    ),
                    filter_field=bucket_field,
                ),
            )

        pipeline = add_preprocessing_pipeline(
            pipeline=pipeline,
            text_field="text",
            system_prompt=task_config["system_prompt"],
            user_prompt_template=task_config["prompt_template"],
            min_document_tokens=task_config["min_document_tokens"],
            min_segment_tokens=task_config["min_segment_tokens"],
            max_input_tokens=task_config["max_input_tokens"],
            args=helper_args,
        )

        pipeline.add_stage(
            task_config["stage_cls"](
                input_field="text",
                output_field=task_config["output_field"],
                model_alias=model_alias,
                model_configs=[model_config],
                model_providers=model_providers,
                system_prompt=task_config["system_prompt"],
            )
        )
        pipeline = task_config["postprocessing_fn"](
            pipeline=pipeline,
            llm_response_field=task_config["output_field"],
            args=helper_args,
        )

        if output_format == "jsonl":
            pipeline.add_stage(JsonlWriter(path=str(output_path_obj)))
        else:
            pipeline.add_stage(ParquetWriter(path=str(output_path_obj)))

        executor_obj = setup_executor(executor)

        logger.info(f"Starting Nemotron-CC SDG NDD pipeline (task={task})...")
        run_start = time.perf_counter()
        output_tasks = pipeline.run(executor_obj)
        run_time = time.perf_counter() - run_start

    finally:
        if inference_server is not None:
            try:
                inference_server.stop()
            except Exception as e:
                logger.warning(f"InferenceServer.stop() raised: {e}")

    logger.success(f"Completed task={task} in {run_time:.2f}s")

    return {
        "metrics": {
            "is_success": True,
            "task": task,
            "inference_server_type": inference_server_type,
            "model_id": model_id,
            "time_taken_s": run_time,
            "serve_startup_s": serve_startup_s,
            "num_input_files": len(input_files),
            "num_files": num_files or "all",
            "max_parallel_requests": max_parallel_requests,
            "ndd_timeout": ndd_timeout,
        },
        "tasks": output_tasks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Nemotron-CC SDG benchmark (NDD)")
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-path", required=True, help="Directory containing parquet files with a 'text' column.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()))
    parser.add_argument(
        "--inference-server-type",
        required=True,
        choices=["ray-serve", "dynamo", "nvidia-nim"],
        help="Model serving backend.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Optional absolute path to a local model snapshot dir (ray-serve/dynamo only). "
            "When set, vLLM loads weights from this path; --model-id remains the served name."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="https://integrate.api.nvidia.com/v1",
        help="Endpoint URL (only used with --inference-server-type=nvidia-nim).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NVIDIA_API_KEY", ""),
        help="API key (only used with --inference-server-type=nvidia-nim).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace tokenizer name/path. Defaults to --model-id.",
    )
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data", "ray_actors"])
    parser.add_argument("--output-format", default="parquet", choices=["jsonl", "parquet"])

    parser.add_argument(
        "--engine-kwargs",
        type=str,
        default=None,
        help="JSON string of vLLM engine kwargs (e.g. '{\"tensor_parallel_size\": 4}').",
    )
    parser.add_argument(
        "--autoscaling-config",
        type=str,
        default=None,
        help=(
            'JSON string of Ray Serve autoscaling config (e.g. \'{"min_replicas": 1, "max_replicas": 4}\'). '
            "For dynamo, min_replicas must equal max_replicas (used as static num_replicas)."
        ),
    )

    parser.add_argument(
        "--max-parallel-requests",
        type=int,
        default=4,
        help=(
            "NDD per-process concurrency cap (semaphore in DataDesigner). Each Curator stage actor "
            "runs its own DataDesigner, so total in-flight = num_actors * this value."
        ),
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help=(
            "Port to bind the InferenceServer's OpenAI HTTP endpoint (Ray Serve / Dynamo frontend). "
            "Default lets InferenceServer pick (8000 → next free). Set explicitly when running "
            "ray-serve and dynamo concurrently on the same host network to avoid port collisions: "
            "e.g. ray-serve=8000 (HAProxy ingress) + dynamo=8001."
        ),
    )
    parser.add_argument(
        "--ndd-timeout",
        type=int,
        default=600,
        help=(
            "httpx read/connect/write timeout (seconds) inside NDD's HttpModelClient. "
            "Default in NDD is 60s, which is too tight when vLLM queues are deep."
        ),
    )

    parser.add_argument(
        "--bucket-field",
        default="bucketed_results",
        help="Column name for the Nemotron-CC quality bucket.",
    )
    parser.add_argument(
        "--min-bucket-threshold",
        type=int,
        default=None,
        help="If set, keep only rows where <bucket-field> >= threshold (skip filter otherwise).",
    )

    parser.add_argument("--num-files", type=int, default=None, help="Limit number of input files (default: all).")

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_id

    logger.info(f"=== Nemotron-CC SDG NDD Benchmark Starting ===\n{vars(args)}")

    engine_kwargs = json.loads(args.engine_kwargs) if args.engine_kwargs else None
    autoscaling_config = json.loads(args.autoscaling_config) if args.autoscaling_config else None

    result_dict: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    success = 1
    try:
        result_dict.update(
            run_benchmark(
                task=args.task,
                inference_server_type=args.inference_server_type,
                model_id=args.model_id,
                model_path=args.model_path,
                base_url=args.base_url,
                api_key=args.api_key,
                tokenizer=args.tokenizer,
                input_path=args.input_path,
                output_path=args.output_path,
                output_format=args.output_format,
                executor=args.executor,
                engine_kwargs=engine_kwargs,
                autoscaling_config=autoscaling_config,
                max_parallel_requests=args.max_parallel_requests,
                ndd_timeout=args.ndd_timeout,
                server_port=args.server_port,
                bucket_field=args.bucket_field,
                min_bucket_threshold=args.min_bucket_threshold,
                num_files=args.num_files,
            )
        )
        success = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success


if __name__ == "__main__":
    raise SystemExit(main())
