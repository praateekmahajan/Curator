# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# ruff: noqa: PLR0913

"""Nemotron-Parse PDF benchmark with explicit stages and server inference modes.

Inference modes
---------------
in_process_vllm
    Runs vLLM inside each Curator GPU actor — the same path as the existing
    ``nemotron_parse_pdf_benchmark.py``.  Suitable for single-node benchmarks.

inference_server_base64
    Encodes each page image as a base64 data-URL and calls a running Dynamo
    InferenceServer via the OpenAI chat-completions API.  No shared filesystem
    required.

inference_server_file_url
    Writes PNG page images to a shared directory, then passes an HTTP file-URL
    to the server.  Requires ``--inference-server-file-url-prefix`` so Dynamo
    workers can fetch the images over the network (Dynamo vLLM rejects
    ``file://`` URLs).

Serving Nemotron-Parse with InferenceServer (Dynamo backend)
------------------------------------------------------------
Nemotron-Parse requires two non-default Dynamo frontend flags:

* ``dyn_chat_processor=vllm`` — Dynamo's native-Rust processor serializes
  OpenAI multimodal ``content`` arrays rather than flattening them, corrupting
  the model's pass-through chat template.  The vLLM processor renders
  multimodal content correctly.

* ``chat_template_content_format=string`` — Forces vLLM's processor to emit
  a plain string instead of a structured content object, matching what the
  model was trained on.

Set these explicitly via ``--dynamo-frontend-kwargs`` or let this script apply
its defaults.  Example::

    python nemotron_parse_pdf_benchmark_new.py \\
        --manifest manifest.jsonl \\
        --pdf-dir /path/to/pdfs \\
        --output-dir ./output \\
        --benchmark-results-path results.json \\
        --inference-mode inference_server_base64 \\
        --model-path /path/to/NVIDIA-Nemotron-Parse-v1.2 \\
        --engine-kwargs '{"trust_remote_code": true, "dtype": "bfloat16", "limit_mm_per_prompt": {"image": 1}, "enable_prefix_caching": false}' \\
        --dynamo-frontend-kwargs '{"dyn_chat_processor": "vllm", "trust_remote_code": true, "chat_template_content_format": "string"}'

InferenceServer setup (Python API)::

    from nemo_curator.core.serve import (
        DynamoRouterConfig,
        DynamoServerConfig,
        DynamoVLLMModelConfig,
        InferenceServer,
    )

    server = InferenceServer(
        models=[
            DynamoVLLMModelConfig(
                model_identifier="/path/to/NVIDIA-Nemotron-Parse-v1.2",
                engine_kwargs={
                    "trust_remote_code": True,
                    "dtype": "bfloat16",
                    "limit_mm_per_prompt": {"image": 1},
                    # Nemotron-Parse (encoder-decoder) crashes with prefix caching on vLLM 0.19.
                    "enable_prefix_caching": False,
                },
                dynamo_kwargs={"enable_multimodal": True},
            )
        ],
        backend=DynamoServerConfig(
            request_plane="tcp",
            router=DynamoRouterConfig(
                router_kwargs={
                    # Required: native-Rust processor corrupts multimodal content arrays.
                    "dyn_chat_processor": "vllm",
                    "trust_remote_code": True,
                    "chat_template_content_format": "string",
                }
            ),
            subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": "180"},
        ),
    )
    server.start()
    # server.endpoint -> "http://<host>:<port>/v1"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import pyarrow as pa
from loguru import logger
from openai import AsyncOpenAI
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
    DEFAULT_MODEL_PATH,
    NemotronParseInferenceStage,
    build_task_prompt,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import PDFPartitioningStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.utils import TaskPerfUtils

if TYPE_CHECKING:
    import pandas as pd

    from nemo_curator.core.serve import InferenceServer

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tutorials" / "interleaved" / "nemotron_parse_pdf"))

from main import create_nemotron_parse_pdf_argparser  # noqa: E402

InferenceMode = Literal["in_process_vllm", "inference_server_file_url", "inference_server_base64"]
ServerRequestMode = Literal["file_url", "base64"]
PROC_SIZE_DIMS = 2


@dataclass
class _ServerPageResult:
    text: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None
    retries: int = 0
    error: str | None = None


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _safe_path_component(value: object, *, max_len: int = 96) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return (cleaned or "item")[:max_len]


def _parse_json_arg(value: str | None, *, arg_name: str) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        msg = f"{arg_name} must be valid JSON: {e}"
        raise ValueError(msg) from e
    if not isinstance(parsed, dict):
        msg = f"{arg_name} must decode to a JSON object"
        raise TypeError(msg)
    return parsed


def _parse_proc_size(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != PROC_SIZE_DIMS:
        msg = "Processor size must be formatted as HEIGHT,WIDTH, for example 2048,1664"
        raise argparse.ArgumentTypeError(msg)
    return parts[0], parts[1]


def _dynamo_num_replicas(autoscaling_config: dict[str, Any] | None) -> int:
    if not autoscaling_config:
        return 1
    min_replicas = int(autoscaling_config.get("min_replicas", 1))
    max_replicas = int(autoscaling_config.get("max_replicas", min_replicas))
    if min_replicas != max_replicas:
        msg = (
            "Dynamo backend does not support autoscaling for this benchmark; "
            f"min_replicas ({min_replicas}) must equal max_replicas ({max_replicas})."
        )
        raise ValueError(msg)
    return min_replicas


def _dynamo_gpu_count(engine_kwargs: dict[str, Any] | None, autoscaling_config: dict[str, Any] | None) -> int:
    engine_kwargs = engine_kwargs or {}
    tensor_parallel_size = int(engine_kwargs.get("tensor_parallel_size", 1))
    pipeline_parallel_size = int(engine_kwargs.get("pipeline_parallel_size", 1))
    return _dynamo_num_replicas(autoscaling_config) * tensor_parallel_size * pipeline_parallel_size


def _start_dynamo_inference_server(
    *,
    model_id: str,
    engine_kwargs: dict[str, Any] | None = None,
    frontend_kwargs: dict[str, Any] | None = None,
    autoscaling_config: dict[str, Any] | None = None,
    model_path: str | None = None,
    request_timeout_s: int = 180,
) -> InferenceServer:
    """Start a local Dynamo-backed InferenceServer and return it.

    ``frontend_kwargs`` must include ``dyn_chat_processor="vllm"`` for
    Nemotron-Parse — see module docstring for details.
    """
    from nemo_curator.core.serve import DynamoRouterConfig, DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer

    model_config = DynamoVLLMModelConfig(
        model_identifier=model_path or model_id,
        model_name=model_id if model_path else None,
        engine_kwargs=engine_kwargs or {},
        dynamo_kwargs={"enable_multimodal": True},
        num_replicas=_dynamo_num_replicas(autoscaling_config),
    )
    server = InferenceServer(
        models=[model_config],
        backend=DynamoServerConfig(
            request_plane="tcp",
            router=DynamoRouterConfig(router_kwargs=frontend_kwargs or {}),
            subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": str(request_timeout_s)},
        ),
    )
    server.start()
    return server


@dataclass
class NemotronParseInferenceServerStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Call a Dynamo/OpenAI-compatible Nemotron-Parse server asynchronously.

    The stage intentionally uses the same stage name and additive custom metric
    names as NemotronParseInferenceStage, so benchmark aggregation can compare
    in-process vLLM and server-backed inference with the same keys.
    """

    endpoint: str
    model_name: str
    request_mode: ServerRequestMode
    image_dir: str
    file_url_prefix: str | None = None
    text_in_pic: bool = False
    task_prompt: str | None = None
    proc_size: tuple[int, int] = (2048, 1664)
    concurrency: int = 2
    request_timeout_s: float = 180.0
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    max_tokens: int = 9000
    stream_response: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    accounting_num_gpus: int = 1
    stage_workers: int | None = None
    name: str = "nemotron_parse_inference"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if self.task_prompt is None:
            self.task_prompt = build_task_prompt(text_in_pic=self.text_in_pic)
        self.accounting_num_gpus = max(1, int(self.accounting_num_gpus))
        self.concurrency = max(1, int(self.concurrency))
        if self.stage_workers is not None:
            self.stage_workers = max(1, int(self.stage_workers))
        self._image_dir = Path(self.image_dir).absolute()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def num_workers(self) -> int | None:
        return self.stage_workers

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": False}

    def xenna_stage_spec(self) -> dict[str, Any]:
        if self.stage_workers is None:
            return {}
        return {"num_workers_per_node": self.stage_workers}

    def _image_ref_for_row(
        self, *, task_id: str, row_idx: int, row: pd.Series, image_bytes: bytes
    ) -> tuple[str, Path | None]:
        if self.request_mode == "base64":
            content_type = str(row.get("content_type") or "image/png")
            encoded = base64.b64encode(image_bytes).decode("ascii")
            return f"data:{content_type};base64,{encoded}", None

        sample_id = _safe_path_component(row.get("sample_id", "sample"))
        position = _safe_path_component(row.get("position", row_idx), max_len=24)
        digest = hashlib.blake2b(image_bytes, digest_size=8).hexdigest()
        filename = f"{_safe_path_component(task_id)}_{row_idx:06d}_{sample_id}_p{position}_{digest}.png"
        image_path = self._image_dir / filename
        self._image_dir.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)

        if self.file_url_prefix:
            return f"{self.file_url_prefix.rstrip('/')}/{quote(filename)}", image_path
        return image_path.resolve().as_uri(), image_path

    @staticmethod
    def _cleanup_image_paths(image_paths: list[Path]) -> None:
        for image_path in image_paths:
            with contextlib.suppress(OSError):
                image_path.unlink(missing_ok=True)

    @staticmethod
    def _response_text(choice: object) -> str:
        content = getattr(getattr(choice, "message", None), "content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    async def _query_one_streaming(self, client: AsyncOpenAI, messages: list[dict[str, Any]]) -> _ServerPageResult:
        stream = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            top_p=1.0,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body or None,
            stream=True,
            stream_options={"include_usage": True},
        )

        text_parts: list[str] = []
        prompt_tokens = 0
        output_tokens = 0
        finish_reason: str | None = None
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or prompt_tokens)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or output_tokens)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)

        return _ServerPageResult(
            text="".join(text_parts),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    async def _query_one_nonstreaming(self, client: AsyncOpenAI, messages: list[dict[str, Any]]) -> _ServerPageResult:
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            top_p=1.0,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body or None,
        )
        choice = response.choices[0] if response.choices else None
        usage = response.usage
        return _ServerPageResult(
            text=self._response_text(choice) if choice is not None else "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=getattr(choice, "finish_reason", None) if choice is not None else None,
        )

    async def _query_one(self, client: AsyncOpenAI, image_ref: str) -> _ServerPageResult:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.task_prompt},
                    {"type": "image_url", "image_url": {"url": image_ref}},
                ],
            }
        ]

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.stream_response:
                    result = await self._query_one_streaming(client, messages)
                else:
                    result = await self._query_one_nonstreaming(client, messages)
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries:
                    logger.warning(f"Server inference failed after {attempt + 1} attempt(s): {e}")
                    break
                await asyncio.sleep(self.retry_base_delay_s * (2**attempt))
            else:
                result.retries = attempt
                return result

        return _ServerPageResult(retries=self.max_retries, error=str(last_error) if last_error else "unknown")

    async def _infer_async(self, image_refs: list[str]) -> list[_ServerPageResult]:
        client = AsyncOpenAI(
            api_key="unused",  # pragma: allowlist secret
            base_url=self.endpoint.rstrip("/"),
            timeout=self.request_timeout_s,
        )
        semaphore = asyncio.Semaphore(self.concurrency)

        async def _bounded_query(idx: int, image_ref: str) -> tuple[int, _ServerPageResult]:
            async with semaphore:
                return idx, await self._query_one(client, image_ref)

        try:
            pairs = await asyncio.gather(*[_bounded_query(idx, ref) for idx, ref in enumerate(image_refs)])
        finally:
            await client.close()

        return [result for _, result in sorted(pairs, key=lambda pair: pair[0])]

    def _infer(self, image_refs: list[str]) -> list[_ServerPageResult]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._infer_async(image_refs))

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(self._infer_async(image_refs))).result()

    def _build_metrics(
        self,
        results: list[_ServerPageResult],
        *,
        request_time_s: float,
        num_input_pages: int,
        num_valid_pages: int,
    ) -> dict[str, float]:
        output_texts = [result.text for result in results]
        total_output_tokens = float(sum(result.output_tokens for result in results))
        total_output_chars = float(sum(len(text) for text in output_texts))
        return {
            "vllm_inference_time": request_time_s * self.accounting_num_gpus,
            "inference_server_request_time": request_time_s,
            "num_input_pages": float(num_input_pages),
            "num_valid_pages": float(num_valid_pages),
            "num_skipped_pages": float(num_input_pages - num_valid_pages),
            "total_prompt_tokens": float(sum(result.prompt_tokens for result in results)),
            "total_output_tokens": total_output_tokens,
            "avg_output_tokens_per_page": _safe_div(total_output_tokens, num_valid_pages),
            "total_output_chars": total_output_chars,
            "avg_output_chars_per_page": _safe_div(total_output_chars, num_valid_pages),
            "num_output_length_truncated": float(sum(result.finish_reason == "length" for result in results)),
            "num_empty_outputs": float(sum(not text.strip() for text in output_texts)),
            "num_request_errors": float(sum(result.error is not None for result in results)),
            "vllm_retries": float(sum(result.retries for result in results)),
        }

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        task_df = task.to_pandas()
        image_refs: list[str] = []
        image_paths_to_cleanup: list[Path] = []
        valid_mask: list[bool] = []

        for row_idx, row in task_df.iterrows():
            raw_bytes = row.get("binary_content")
            if raw_bytes is None:
                valid_mask.append(False)
                continue
            try:
                image_bytes = bytes(raw_bytes)
            except TypeError:
                valid_mask.append(False)
                continue
            if not image_bytes:
                valid_mask.append(False)
                continue

            try:
                image_ref, image_path = self._image_ref_for_row(
                    task_id=task.task_id, row_idx=int(row_idx), row=row, image_bytes=image_bytes
                )
                image_refs.append(image_ref)
                if image_path is not None:
                    image_paths_to_cleanup.append(image_path)
                valid_mask.append(True)
            except OSError as e:
                logger.warning(f"Could not prepare image reference for page {row_idx} in {task.task_id}: {e}")
                valid_mask.append(False)

        if not image_refs:
            return None

        request_t0 = time.perf_counter()
        try:
            results = self._infer(image_refs)
            request_time_s = time.perf_counter() - request_t0
        finally:
            self._cleanup_image_paths(image_paths_to_cleanup)
        self._log_metrics(
            self._build_metrics(
                results,
                request_time_s=request_time_s,
                num_input_pages=len(valid_mask),
                num_valid_pages=len(image_refs),
            )
        )

        valid_text_iter = iter(result.text for result in results)
        task_df["text_content"] = [next(valid_text_iter) if is_valid else "" for is_valid in valid_mask]

        metadata = dict(task._metadata)
        metadata["proc_size"] = list(self.proc_size)
        metadata["model_path"] = self.model_name
        metadata["inference_mode"] = f"inference_server_{self.request_mode}"
        metadata["inference_server_endpoint"] = self.endpoint

        return InterleavedBatch(
            task_id=f"{task.task_id}_inferred",
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(task_df, preserve_index=False),
            _metadata=metadata,
            _stage_perf=task._stage_perf,
        )


def create_explicit_nemotron_parse_pdf_pipeline(
    args: argparse.Namespace,
    *,
    inference_stage: ProcessingStage[InterleavedBatch, InterleavedBatch],
) -> Pipeline:
    """Build PDF processing pipeline with individually visible execution stages."""
    pipeline = Pipeline(
        name="nemotron_parse_pdf_explicit",
        description="PDF -> render pages -> infer -> postprocess -> interleaved parquet",
    )
    pipeline.add_stage(
        PDFPartitioningStage(
            manifest_path=args.manifest,
            pdfs_per_task=args.pdfs_per_task,
            max_pdfs=args.max_pdfs,
            dataset_name=args.dataset_name,
            file_name_field=args.file_name_field,
            file_names_field=args.file_names_field,
            url_field=args.url_field,
        )
    )
    pipeline.add_stage(
        PDFPreprocessStage(
            zip_base_dir=args.zip_base_dir,
            pdf_dir=args.pdf_dir,
            jsonl_base_dir=args.jsonl_base_dir,
            dpi=args.dpi,
            max_pages=args.max_pages,
        ).with_(resources=Resources(cpus=1.0))
    )
    pipeline.add_stage(inference_stage)
    pipeline.add_stage(
        NemotronParsePostprocessStage(min_crop_px=args.min_crop_size).with_(resources=Resources(cpus=1.0))
    )
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_dir,
            materialize_on_write=False,
        ).with_(resources=Resources(cpus=1.0))
    )
    return pipeline


def _compute_pdf_parse_metrics(
    output_tasks: list,
    run_time_taken: float,
    *,
    accounting_num_gpus: int,
) -> dict[str, float]:
    """Compute benchmark-level throughput metrics from additive task stats."""
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    metric_prefix = "task_nemotron_parse_inference_custom"
    accounting_num_gpus = max(1, int(accounting_num_gpus))

    total_num_pages = task_metrics.get(f"{metric_prefix}.num_valid_pages_sum", 0.0)
    total_input_pages = task_metrics.get(f"{metric_prefix}.num_input_pages_sum", 0.0)
    total_skipped_pages = task_metrics.get(f"{metric_prefix}.num_skipped_pages_sum", 0.0)
    total_output_tokens = task_metrics.get(f"{metric_prefix}.total_output_tokens_sum", 0.0)
    total_output_chars = task_metrics.get(f"{metric_prefix}.total_output_chars_sum", 0.0)
    vllm_inference_time_sum = task_metrics.get(f"{metric_prefix}.vllm_inference_time_sum", 0.0)
    server_request_time_sum = task_metrics.get(f"{metric_prefix}.inference_server_request_time_sum", 0.0)
    inference_wall_time_s = _safe_div(vllm_inference_time_sum, accounting_num_gpus)
    inference_pages_sec = _safe_div(total_num_pages, inference_wall_time_s)
    pipeline_pages_sec = _safe_div(total_num_pages, run_time_taken)

    return {
        "total_num_pages": total_num_pages,
        "total_input_pages": total_input_pages,
        "total_skipped_pages": total_skipped_pages,
        "total_output_tokens": total_output_tokens,
        "total_output_chars": total_output_chars,
        "vllm_inference_time_sum": vllm_inference_time_sum,
        "inference_server_request_time_sum": server_request_time_sum,
        "accounting_num_gpus": float(accounting_num_gpus),
        "inference_wall_time_s": inference_wall_time_s,
        "inference_pages_sec": inference_pages_sec,
        "pipeline_pages_sec": pipeline_pages_sec,
        "inference_pages_per_sec": inference_pages_sec,
        "inference_pages_per_sec_per_gpu": _safe_div(total_num_pages, vllm_inference_time_sum),
        "inference_output_tokens_per_sec": _safe_div(total_output_tokens, inference_wall_time_s),
        "inference_output_tokens_per_sec_per_gpu": _safe_div(total_output_tokens, vllm_inference_time_sum),
        "pipeline_pages_per_sec": pipeline_pages_sec,
        "pipeline_output_tokens_per_sec": _safe_div(total_output_tokens, run_time_taken),
    }


def _count_processed_pdfs(output_tasks: list[FileGroupTask]) -> int:
    source_entries: set[str] = set()
    for task in output_tasks:
        source_entries.update(task._metadata.get("source_files", []))
    return len(source_entries)


def _server_request_mode(inference_mode: InferenceMode) -> ServerRequestMode:
    if inference_mode == "inference_server_file_url":
        return "file_url"
    if inference_mode == "inference_server_base64":
        return "base64"
    msg = f"{inference_mode} is not an inference-server mode"
    raise ValueError(msg)


def _create_inference_stage(
    args: argparse.Namespace,
    *,
    endpoint: str | None,
    served_model_name: str,
    output_dir: Path,
    server_gpu_count: int,
    extra_body: dict[str, Any],
    engine_kwargs: dict[str, Any] | None = None,
) -> ProcessingStage[InterleavedBatch, InterleavedBatch]:
    if args.inference_mode == "in_process_vllm":
        if args.backend != "vllm":
            msg = "--inference-mode=in_process_vllm requires --backend=vllm"
            raise ValueError(msg)
        return NemotronParseInferenceStage(
            model_path=args.model_path,
            text_in_pic=args.text_in_pic,
            backend="vllm",
            inference_batch_size=args.inference_batch_size,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            engine_kwargs=engine_kwargs,
        ).with_(resources=Resources(cpus=1.0, gpus=1.0))

    if endpoint is None:
        msg = "endpoint is required for inference-server modes"
        raise ValueError(msg)

    image_dir = args.inference_server_image_dir or str(output_dir / "_inference_server_page_images")
    return NemotronParseInferenceServerStage(
        endpoint=endpoint,
        model_name=served_model_name,
        request_mode=_server_request_mode(args.inference_mode),
        image_dir=image_dir,
        file_url_prefix=args.inference_server_file_url_prefix,
        text_in_pic=args.text_in_pic,
        proc_size=args.inference_server_proc_size,
        concurrency=args.inference_server_concurrency,
        request_timeout_s=args.inference_server_request_timeout_s,
        max_retries=args.inference_server_max_retries,
        max_tokens=args.inference_server_max_tokens,
        stream_response=args.inference_server_stream,
        extra_body=extra_body,
        accounting_num_gpus=server_gpu_count,
        stage_workers=args.inference_server_stage_workers,
    )


def run_nemotron_parse_pdf_benchmark(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915
    """Run the Nemotron-Parse PDF benchmark and collect metrics."""
    executor = setup_executor(args.executor)

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.inference_mode == "inference_server_file_url":
        if args.inference_server_file_url_prefix is None:
            msg = (
                "Dynamo vLLM rejects local file:// image URLs. "
                "Use --inference-mode=inference_server_base64, or provide an HTTP(S) "
                "--inference-server-file-url-prefix that Dynamo workers can fetch."
            )
            raise ValueError(msg)
        image_dir = Path(args.inference_server_image_dir or output_dir / "_inference_server_page_images").absolute()
        image_dir.mkdir(parents=True, exist_ok=True)
        args.inference_server_image_dir = str(image_dir)

    engine_kwargs = _parse_json_arg(args.engine_kwargs, arg_name="--engine-kwargs")
    frontend_kwargs = _parse_json_arg(args.dynamo_frontend_kwargs, arg_name="--dynamo-frontend-kwargs")
    autoscaling_config = _parse_json_arg(args.autoscaling_config, arg_name="--autoscaling-config")
    extra_body = _parse_json_arg(args.inference_server_extra_body, arg_name="--inference-server-extra-body") or {}
    if args.inference_mode != "in_process_vllm":
        engine_kwargs = dict(engine_kwargs or {})
        engine_kwargs.setdefault("trust_remote_code", True)
        engine_kwargs.setdefault("dtype", "bfloat16")
        engine_kwargs.setdefault("limit_mm_per_prompt", {"image": 1})
        # Nemotron-Parse is encoder-decoder multimodal; vLLM 0.19 crashes in
        # CrossAttentionManager when prefix caching is enabled.
        engine_kwargs.setdefault("enable_prefix_caching", False)
        frontend_kwargs = dict(frontend_kwargs or {})
        # Dynamo's native-Rust processor serializes multimodal content arrays
        # instead of flattening them, corrupting Nemotron-Parse's pass-through
        # chat template. Set these explicitly — no auto-injection in the backend.
        frontend_kwargs.setdefault("dyn_chat_processor", "vllm")
        frontend_kwargs.setdefault("trust_remote_code", True)
        frontend_kwargs.setdefault("chat_template_content_format", "string")

    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"PDF source: zip_base_dir={args.zip_base_dir}, pdf_dir={args.pdf_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Inference mode: {args.inference_mode}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"PDFs per task: {args.pdfs_per_task}, max PDFs: {args.max_pdfs}")

    inference_server: InferenceServer | None = None
    serve_startup_s = 0.0
    server_gpu_count = 1
    accounting_num_gpus = max(1, int(args.accounting_num_gpus or 1))
    served_model_name = args.model_id or args.model_path or DEFAULT_MODEL_PATH

    try:
        if args.inference_mode != "in_process_vllm":
            server_gpu_count = _dynamo_gpu_count(engine_kwargs, autoscaling_config)
            accounting_num_gpus = max(1, int(args.accounting_num_gpus or server_gpu_count))
            logger.info(
                "Starting Dynamo InferenceServer "
                f"served_model_name={served_model_name}, engine_kwargs={engine_kwargs}, "
                f"frontend_kwargs={frontend_kwargs}, "
                f"autoscaling_config={autoscaling_config}, server_gpus={server_gpu_count}, "
                f"accounting_gpus={accounting_num_gpus}"
            )
            serve_t0 = time.perf_counter()
            inference_server = _start_dynamo_inference_server(
                model_id=served_model_name,
                model_path=args.model_path,
                engine_kwargs=engine_kwargs,
                frontend_kwargs=frontend_kwargs,
                autoscaling_config=autoscaling_config,
                request_timeout_s=max(1, int(args.inference_server_request_timeout_s)),
            )
            serve_startup_s = time.perf_counter() - serve_t0
            logger.info(f"InferenceServer ready at {inference_server.endpoint} (startup: {serve_startup_s:.1f}s)")

        inference_stage = _create_inference_stage(
            args,
            endpoint=inference_server.endpoint if inference_server else None,
            served_model_name=served_model_name,
            output_dir=output_dir,
            server_gpu_count=accounting_num_gpus,
            extra_body=extra_body,
            engine_kwargs=engine_kwargs,
        )
        pipeline = create_explicit_nemotron_parse_pdf_pipeline(args, inference_stage=inference_stage)

        run_start_time = time.perf_counter()
        success = False
        output_tasks: list[FileGroupTask] = []

        try:
            logger.info("Running Nemotron-Parse PDF pipeline...")
            logger.info(f"Pipeline description:\n{pipeline.describe()}")

            output_tasks = pipeline.run(executor) or []
            run_time_taken = time.perf_counter() - run_start_time

            num_pdfs_processed = _count_processed_pdfs(output_tasks)
            pdf_parse_metrics = _compute_pdf_parse_metrics(
                output_tasks,
                run_time_taken,
                accounting_num_gpus=accounting_num_gpus,
            )

            logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
            logger.success(f"Processed {num_pdfs_processed} PDFs")
            logger.success(f"Inference throughput: {pdf_parse_metrics['inference_pages_sec']:.2f} pages/s")
            logger.success(f"Pipeline throughput: {pdf_parse_metrics['pipeline_pages_sec']:.2f} pages/s")
            success = True

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Benchmark failed: {e}")
            logger.debug(f"Full traceback:\n{error_traceback}")
            run_time_taken = time.perf_counter() - run_start_time
            num_pdfs_processed = 0
            pdf_parse_metrics = {}

    finally:
        if inference_server is not None:
            with contextlib.suppress(Exception):
                inference_server.stop()

    return {
        "params": {
            "executor": args.executor,
            "manifest": args.manifest,
            "pdf_dir": args.pdf_dir,
            "zip_base_dir": args.zip_base_dir,
            "jsonl_base_dir": args.jsonl_base_dir,
            "output_dir": str(output_dir),
            "benchmark_results_path": str(args.benchmark_results_path),
            "inference_mode": args.inference_mode,
            "model_path": args.model_path,
            "model_id": args.model_id,
            "served_model_name": served_model_name,
            "backend": args.backend,
            "pdfs_per_task": args.pdfs_per_task,
            "max_pdfs": args.max_pdfs,
            "dpi": args.dpi,
            "max_pages": args.max_pages,
            "inference_batch_size": args.inference_batch_size,
            "max_num_seqs": args.max_num_seqs,
            "inference_server_concurrency": args.inference_server_concurrency,
            "inference_server_stage_workers": args.inference_server_stage_workers,
            "inference_server_request_timeout_s": args.inference_server_request_timeout_s,
            "inference_server_max_retries": args.inference_server_max_retries,
            "inference_server_max_tokens": args.inference_server_max_tokens,
            "inference_server_stream": args.inference_server_stream,
            "inference_server_proc_size": args.inference_server_proc_size,
            "inference_server_file_url_prefix": args.inference_server_file_url_prefix,
            "engine_kwargs": engine_kwargs,
            "dynamo_frontend_kwargs": frontend_kwargs,
            "autoscaling_config": autoscaling_config,
            "server_accounting_gpus": server_gpu_count,
            "accounting_num_gpus": accounting_num_gpus,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_pdfs_processed": num_pdfs_processed,
            "num_output_tasks": len(output_tasks),
            "throughput_pdfs_per_sec": _safe_div(num_pdfs_processed, run_time_taken),
            "serve_startup_s": serve_startup_s,
            "server_gpu_count": server_gpu_count,
            **pdf_parse_metrics,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = create_nemotron_parse_pdf_argparser()

    parser.add_argument(
        "--benchmark-results-path",
        type=Path,
        required=True,
        help="Path to write benchmark results",
    )
    parser.add_argument(
        "--executor",
        default="xenna",
        choices=["xenna", "ray_data"],
        help="Executor to use for pipeline execution",
    )
    parser.add_argument(
        "--inference-mode",
        default="in_process_vllm",
        choices=["in_process_vllm", "inference_server_file_url", "inference_server_base64"],
        help="Inference path to benchmark",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Served model name for Dynamo modes. Defaults to --model-path.",
    )
    parser.add_argument(
        "--engine-kwargs",
        type=str,
        default=None,
        help="JSON string of vLLM/Dynamo engine kwargs, for example '{\"tensor_parallel_size\": 1}'",
    )
    parser.add_argument(
        "--dynamo-frontend-kwargs",
        type=str,
        default=None,
        help=(
            "JSON string of Dynamo frontend kwargs. Server modes default to "
            '\'{"dyn_chat_processor": "vllm", "trust_remote_code": true, '
            '"chat_template_content_format": "string"}\' to match vLLM serve preprocessing.'
        ),
    )
    parser.add_argument(
        "--autoscaling-config",
        type=str,
        default=None,
        help='JSON string with static Dynamo replicas, for example \'{"min_replicas": 4, "max_replicas": 4}\'',
    )
    parser.add_argument(
        "--inference-server-concurrency",
        type=int,
        default=2,
        help="Async page requests per server-inference stage worker",
    )
    parser.add_argument(
        "--inference-server-stage-workers",
        type=int,
        default=None,
        help="Optional fixed client-side inference stage worker count for server modes",
    )
    parser.add_argument(
        "--inference-server-request-timeout-s",
        type=float,
        default=180.0,
        help="Per-request timeout for server inference",
    )
    parser.add_argument(
        "--inference-server-max-retries",
        type=int,
        default=3,
        help="Retries per page request for server inference",
    )
    parser.add_argument(
        "--inference-server-max-tokens",
        type=int,
        default=9000,
        help="Max generated tokens per page for server inference",
    )
    parser.add_argument(
        "--inference-server-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use streaming chat completions and aggregate chunks client-side. "
            "This avoids Dynamo's non-streaming chat aggregator for vLLM-processed multimodal requests."
        ),
    )
    parser.add_argument(
        "--inference-server-extra-body",
        type=str,
        default='{"top_k": 1, "repetition_penalty": 1.1, "skip_special_tokens": false}',
        help="JSON object passed as OpenAI extra_body for server inference",
    )
    parser.add_argument(
        "--inference-server-image-dir",
        default=None,
        help="Directory used to write PNG page images for inference_server_file_url mode",
    )
    parser.add_argument(
        "--inference-server-file-url-prefix",
        default=None,
        help="Optional URL prefix for file_url mode. Defaults to file:// URIs for the image directory.",
    )
    parser.add_argument(
        "--inference-server-proc-size",
        type=_parse_proc_size,
        default=(2048, 1664),
        help="Processor size HEIGHT,WIDTH used by postprocess in server modes",
    )
    parser.add_argument(
        "--accounting-num-gpus",
        type=int,
        default=None,
        help=(
            "GPU count used to normalize inference_pages_sec. Defaults to Dynamo replica GPU count "
            "for server modes and 1 for in-process mode."
        ),
    )

    args = parser.parse_args()

    logger.info("=== Nemotron-Parse PDF Explicit Pipeline Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    results: dict[str, Any] = {
        "params": vars(args),
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        results = run_nemotron_parse_pdf_benchmark(args)
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
