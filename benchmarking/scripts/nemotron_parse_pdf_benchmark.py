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

# ruff: noqa: ANN401, PLR0913

"""Nemotron-Parse PDF benchmark with explicit in-process and server modes."""

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
from server_utils import (
    ServerBackend,
    build_multimodal_chat_messages,
    default_dynamo_frontend_kwargs,
    default_nemotron_parse_engine_kwargs,
    parse_json_arg,
    server_gpu_count,
    start_inference_server,
    static_autoscaling_config,
)
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

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "tutorials" / "interleaved" / "nemotron_parse_pdf"))

from main import create_nemotron_parse_pdf_argparser  # noqa: E402

InferenceMode = Literal["in_process_vllm", "dynamo_http", "ray_serve_http", "ray_serve_grpc"]
ServerImageRequestMode = Literal["base64", "file_url"]
ServerTransport = Literal["http", "ray_serve_handle"]
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


def _parse_proc_size(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != PROC_SIZE_DIMS:
        msg = "Processor size must be formatted as HEIGHT,WIDTH, for example 2048,1664"
        raise argparse.ArgumentTypeError(msg)
    return parts[0], parts[1]


def _server_backend_for_mode(inference_mode: str) -> ServerBackend | None:
    if inference_mode in {"in_process_vllm", "external_http"}:
        return None
    if inference_mode == "dynamo_http":
        return "dynamo"
    if inference_mode in {"ray_serve_http", "ray_serve_grpc"}:
        return "ray_serve"
    msg = f"Unknown inference mode: {inference_mode}"
    raise ValueError(msg)


def _resolve_inference_endpoint(
    inference_mode: str,
    *,
    configured_endpoint: str | None,
    managed_endpoint: str | None,
) -> str | None:
    if inference_mode == "external_http":
        if not configured_endpoint:
            msg = "--inference-server-endpoint is required with --inference-mode=external_http"
            raise ValueError(msg)
        return configured_endpoint
    return managed_endpoint


def _server_transport_for_mode(inference_mode: str) -> ServerTransport:
    if inference_mode == "ray_serve_grpc":
        return "ray_serve_handle"
    return "http"


def _extract_usage(payload: dict[str, Any]) -> tuple[int, int]:
    usage = payload.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


@dataclass
class NemotronParseInferenceServerStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """CPU client stage that calls Dynamo or Ray Serve for page-image inference."""

    endpoint: str
    model_name: str
    server_transport: ServerTransport
    image_request_mode: ServerImageRequestMode
    image_dir: str
    file_url_prefix: str | None = None
    ray_serve_app_name: str = "default"
    text_in_pic: bool = False
    task_prompt: str | None = None
    proc_size: tuple[int, int] = (2048, 1664)
    request_batch_size: int = 4
    request_timeout_s: float = 180.0
    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    max_tokens: int = 9000
    stream_response: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    accounting_num_gpus: int = 1
    client_num_workers: int | None = 16
    name: str = "nemotron_parse_inference"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        if self.task_prompt is None:
            self.task_prompt = build_task_prompt(text_in_pic=self.text_in_pic)
        self.accounting_num_gpus = max(1, int(self.accounting_num_gpus))
        self.request_batch_size = int(self.request_batch_size)
        if self.request_batch_size == 0:
            msg = "request_batch_size must be positive, or -1 to send all pages in a task concurrently"
            raise ValueError(msg)
        if self.client_num_workers is not None:
            self.client_num_workers = max(1, int(self.client_num_workers))
        self._image_dir = Path(self.image_dir).absolute()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def num_workers(self) -> int | None:
        return self.client_num_workers

    def ray_stage_spec(self) -> dict[str, Any]:
        return {"is_actor_stage": False}

    def _image_ref_for_row(
        self, *, task_id: str, row_idx: int, row: pd.Series, image_bytes: bytes
    ) -> tuple[str, Path | None]:
        if self.image_request_mode == "base64":
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

    async def _query_one_ray_serve_handle(self, handle: Any, messages: list[dict[str, Any]]) -> _ServerPageResult:
        from starlette.responses import JSONResponse, StreamingResponse
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
            "stream": self.stream_response,
            **(self.extra_body or {}),
        }
        if self.stream_response:
            request_payload["stream_options"] = {"include_usage": True}
        body = ChatCompletionRequest.model_validate(request_payload)
        response = await handle.chat.remote(body, None)

        if isinstance(response, JSONResponse) or hasattr(response, "body"):
            payload = json.loads(bytes(response.body).decode("utf-8"))
            choice = payload.get("choices", [{}])[0]
            message = choice.get("message") or {}
            prompt_tokens, output_tokens = _extract_usage(payload)
            return _ServerPageResult(
                text=message.get("content") or "",
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                finish_reason=choice.get("finish_reason"),
            )

        if isinstance(response, StreamingResponse) or hasattr(response, "body_iterator"):
            text_parts: list[str] = []
            prompt_tokens = 0
            output_tokens = 0
            finish_reason: str | None = None
            async for raw_chunk in response.body_iterator:
                chunk_text = raw_chunk.decode("utf-8") if isinstance(raw_chunk, bytes) else str(raw_chunk)
                for line in chunk_text.splitlines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if not data or data == "[DONE]":
                        continue
                    payload = json.loads(data)
                    usage_prompt, usage_output = _extract_usage(payload)
                    prompt_tokens = usage_prompt or prompt_tokens
                    output_tokens = usage_output or output_tokens
                    choices = payload.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    finish_reason = choice.get("finish_reason") or finish_reason
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        text_parts.append(content)
            return _ServerPageResult(
                text="".join(text_parts),
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            )

        msg = f"Unexpected Ray Serve handle response type: {type(response)!r}"
        raise TypeError(msg)

    async def _query_one(
        self,
        *,
        client: AsyncOpenAI | None,
        ray_serve_handle: Any | None,
        image_ref: str,
    ) -> _ServerPageResult:
        messages = build_multimodal_chat_messages(self.task_prompt or "", image_ref)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.server_transport == "ray_serve_handle":
                    result = await self._query_one_ray_serve_handle(ray_serve_handle, messages)
                elif self.stream_response:
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

    def _effective_request_batch_size(self, num_image_refs: int) -> int:
        if self.request_batch_size < 0:
            return max(1, int(num_image_refs))
        return self.request_batch_size

    async def _infer_async(self, image_refs: list[str]) -> list[_ServerPageResult]:
        client: AsyncOpenAI | None = None
        ray_serve_handle: Any | None = None
        if self.server_transport == "ray_serve_handle":
            from ray import serve

            ray_serve_handle = serve.get_app_handle(self.ray_serve_app_name)
        else:
            client = AsyncOpenAI(
                api_key="unused",  # pragma: allowlist secret
                base_url=self.endpoint.rstrip("/"),
                timeout=self.request_timeout_s,
            )

        results: list[_ServerPageResult] = []
        try:
            request_batch_size = self._effective_request_batch_size(len(image_refs))
            for start in range(0, len(image_refs), request_batch_size):
                batch_refs = image_refs[start : start + request_batch_size]

                async def _query(idx: int, image_ref: str) -> tuple[int, _ServerPageResult]:
                    return idx, await self._query_one(
                        client=client,
                        ray_serve_handle=ray_serve_handle,
                        image_ref=image_ref,
                    )

                pairs = await asyncio.gather(*[_query(idx, ref) for idx, ref in enumerate(batch_refs, start=start)])
                results.extend(result for _, result in sorted(pairs, key=lambda pair: pair[0]))
        finally:
            if client is not None:
                await client.close()

        return results

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
        metadata["inference_mode"] = self.server_transport
        metadata["inference_server_endpoint"] = self.endpoint

        return InterleavedBatch(
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
    pipeline.add_stage(InterleavedParquetWriterStage(path=args.output_dir, materialize_on_write=False))
    return pipeline


def _compute_pdf_parse_metrics(
    output_tasks: list,
    run_time_taken: float,
    *,
    accounting_num_gpus: int,
    stage_normalization_workers: int,
) -> dict[str, float]:
    task_metrics = TaskPerfUtils.aggregate_task_metrics(output_tasks, prefix="task")
    metric_prefix = "task_nemotron_parse_inference_custom"
    accounting_num_gpus = max(1, int(accounting_num_gpus))
    stage_normalization_workers = max(1, int(stage_normalization_workers))

    total_num_pages = task_metrics.get(f"{metric_prefix}.num_valid_pages_sum", 0.0)
    total_input_pages = task_metrics.get(f"{metric_prefix}.num_input_pages_sum", 0.0)
    total_skipped_pages = task_metrics.get(f"{metric_prefix}.num_skipped_pages_sum", 0.0)
    total_output_tokens = task_metrics.get(f"{metric_prefix}.total_output_tokens_sum", 0.0)
    total_output_chars = task_metrics.get(f"{metric_prefix}.total_output_chars_sum", 0.0)
    vllm_inference_time_sum = task_metrics.get(f"{metric_prefix}.vllm_inference_time_sum", 0.0)
    server_request_time_sum = task_metrics.get(f"{metric_prefix}.inference_server_request_time_sum", 0.0)
    num_request_errors = task_metrics.get(f"{metric_prefix}.num_request_errors_sum", 0.0)
    num_empty_outputs = task_metrics.get(f"{metric_prefix}.num_empty_outputs_sum", 0.0)
    num_output_length_truncated = task_metrics.get(f"{metric_prefix}.num_output_length_truncated_sum", 0.0)
    inference_wall_time_s = _safe_div(vllm_inference_time_sum, accounting_num_gpus)
    inference_pages_sec = _safe_div(total_num_pages, inference_wall_time_s)
    pipeline_pages_sec = _safe_div(total_num_pages, run_time_taken)
    stage_process_time_sum = task_metrics.get("task_nemotron_parse_inference_process_time_sum", 0.0)
    normalized_stage_time_s = _safe_div(stage_process_time_sum, stage_normalization_workers)
    normalized_stage_pages_sec = _safe_div(total_num_pages, normalized_stage_time_s)

    return {
        "total_num_pages": total_num_pages,
        "total_input_pages": total_input_pages,
        "total_skipped_pages": total_skipped_pages,
        "total_output_tokens": total_output_tokens,
        "total_output_chars": total_output_chars,
        "vllm_inference_time_sum": vllm_inference_time_sum,
        "inference_server_request_time_sum": server_request_time_sum,
        "num_request_errors": num_request_errors,
        "num_empty_outputs": num_empty_outputs,
        "num_output_length_truncated": num_output_length_truncated,
        "accounting_num_gpus": float(accounting_num_gpus),
        "stage_normalization_workers": float(stage_normalization_workers),
        "normalized_inference_stage_time_s": normalized_stage_time_s,
        "normalized_inference_stage_pages_per_sec": normalized_stage_pages_sec,
        "steady_state_inference_stage_time_s": normalized_stage_time_s,
        "steady_state_inference_stage_pages_per_sec": normalized_stage_pages_sec,
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


def _create_inference_stage(
    args: argparse.Namespace,
    *,
    endpoint: str | None,
    served_model_name: str,
    output_dir: Path,
    accounting_num_gpus: int,
    extra_body: dict[str, Any],
    engine_kwargs: dict[str, Any] | None = None,
    in_process_runtime_env: dict[str, Any] | None = None,
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
        ).with_(
            resources=Resources(cpus=1.0, gpus=1.0),
            num_workers=args.gpu_num_workers,
            runtime_env=in_process_runtime_env,
        )

    if endpoint is None:
        msg = "endpoint is required for inference-server modes"
        raise ValueError(msg)

    image_dir = args.inference_server_image_dir or str(output_dir / "_inference_server_page_images")
    server_stage = NemotronParseInferenceServerStage(
        endpoint=endpoint,
        model_name=served_model_name,
        server_transport=_server_transport_for_mode(args.inference_mode),
        image_request_mode=getattr(args, "server_image_request_mode", "base64"),
        image_dir=image_dir,
        file_url_prefix=args.inference_server_file_url_prefix,
        text_in_pic=args.text_in_pic,
        proc_size=args.inference_server_proc_size,
        request_batch_size=args.request_batch_size,
        request_timeout_s=args.inference_server_request_timeout_s,
        max_retries=args.inference_server_max_retries,
        max_tokens=args.inference_server_max_tokens,
        stream_response=args.inference_server_stream,
        extra_body=extra_body,
        accounting_num_gpus=accounting_num_gpus,
        client_num_workers=args.client_num_workers,
    )
    return server_stage.with_(resources=Resources(cpus=1.0), num_workers=args.client_num_workers)


def run_nemotron_parse_pdf_benchmark(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    executor_config = {"execution_mode": args.execution_mode} if args.executor == "xenna" else None
    executor = setup_executor(args.executor, config=executor_config)

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.server_image_request_mode == "file_url":
        image_dir = Path(args.inference_server_image_dir or output_dir / "_inference_server_page_images").absolute()
        image_dir.mkdir(parents=True, exist_ok=True)
        args.inference_server_image_dir = str(image_dir)

    parsed_engine_kwargs = parse_json_arg(args.engine_kwargs, arg_name="--engine-kwargs")
    autoscaling_config = parse_json_arg(args.autoscaling_config, arg_name="--autoscaling-config")
    autoscaling_config = autoscaling_config or static_autoscaling_config(args.server_num_replicas)
    dynamo_disagg_config = parse_json_arg(args.dynamo_disagg_config, arg_name="--dynamo-disagg-config")
    dynamo_encoder_disagg_config = parse_json_arg(
        args.dynamo_encoder_disagg_config, arg_name="--dynamo-encoder-disagg-config"
    )
    dynamo_model_runtime_env = parse_json_arg(args.dynamo_model_runtime_env, arg_name="--dynamo-model-runtime-env")
    in_process_runtime_env = parse_json_arg(args.in_process_runtime_env, arg_name="--in-process-runtime-env")
    extra_body = parse_json_arg(args.inference_server_extra_body, arg_name="--inference-server-extra-body") or {}

    server_backend = _server_backend_for_mode(args.inference_mode)
    if dynamo_disagg_config and server_backend != "dynamo":
        msg = "--dynamo-disagg-config is only valid with --inference-mode=dynamo_http"
        raise ValueError(msg)
    if dynamo_encoder_disagg_config and server_backend != "dynamo":
        msg = "--dynamo-encoder-disagg-config is only valid with --inference-mode=dynamo_http"
        raise ValueError(msg)
    if dynamo_disagg_config and dynamo_encoder_disagg_config:
        msg = "--dynamo-disagg-config and --dynamo-encoder-disagg-config are mutually exclusive"
        raise ValueError(msg)

    engine_kwargs = parsed_engine_kwargs
    frontend_kwargs: dict[str, Any] | None = None
    started_server = None
    serve_startup_s = 0.0
    server_gpu_total = 0
    served_model_name = args.model_id or DEFAULT_MODEL_PATH

    if server_backend is not None:
        engine_kwargs = default_nemotron_parse_engine_kwargs(parsed_engine_kwargs)
        server_gpu_total = server_gpu_count(
            engine_kwargs, autoscaling_config, dynamo_disagg_config, dynamo_encoder_disagg_config
        )
        accounting_num_gpus = max(1, int(args.accounting_num_gpus or server_gpu_total))
        if server_backend == "dynamo":
            frontend_kwargs = default_dynamo_frontend_kwargs(
                use_vllm_chat_processor=args.dynamo_use_vllm_chat_processor,
                overrides=parse_json_arg(args.dynamo_frontend_kwargs, arg_name="--dynamo-frontend-kwargs"),
            )
        else:
            frontend_kwargs = {}
    elif args.inference_mode == "external_http":
        accounting_num_gpus = max(1, int(args.accounting_num_gpus or args.gpu_num_workers))
        server_gpu_total = accounting_num_gpus
        autoscaling_config = None
    else:
        accounting_num_gpus = max(1, int(args.accounting_num_gpus or args.gpu_num_workers))
        autoscaling_config = None

    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"PDF source: zip_base_dir={args.zip_base_dir}, pdf_dir={args.pdf_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Inference mode: {args.inference_mode}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"PDFs per task: {args.pdfs_per_task}, max PDFs: {args.max_pdfs}")
    logger.info(f"Accounting GPUs: {accounting_num_gpus}")

    try:
        if server_backend is not None:
            logger.info(
                f"Starting {server_backend} server with replicas={args.server_num_replicas}, "
                f"engine_kwargs={engine_kwargs}, frontend_kwargs={frontend_kwargs}, "
                f"autoscaling_config={autoscaling_config}, dynamo_disagg_config={dynamo_disagg_config}, "
                f"dynamo_encoder_disagg_config={dynamo_encoder_disagg_config}, "
                f"server_gpus={server_gpu_total}"
            )
            started_server = start_inference_server(
                backend=server_backend,
                model_id=served_model_name,
                model_path=args.model_path,
                engine_kwargs=engine_kwargs,
                autoscaling_config=autoscaling_config,
                frontend_kwargs=frontend_kwargs,
                request_timeout_s=max(1, int(args.inference_server_request_timeout_s)),
                dynamo_disagg_config=dynamo_disagg_config,
                dynamo_encoder_disagg_config=dynamo_encoder_disagg_config,
                dynamo_model_runtime_env=dynamo_model_runtime_env,
            )
            serve_startup_s = started_server.startup_s
            logger.info(f"InferenceServer ready at {started_server.server.endpoint} (startup: {serve_startup_s:.1f}s)")

        endpoint = _resolve_inference_endpoint(
            args.inference_mode,
            configured_endpoint=args.inference_server_endpoint,
            managed_endpoint=started_server.server.endpoint if started_server is not None else None,
        )
        inference_stage = _create_inference_stage(
            args,
            endpoint=endpoint,
            served_model_name=served_model_name,
            output_dir=output_dir,
            accounting_num_gpus=accounting_num_gpus,
            extra_body=extra_body,
            engine_kwargs=engine_kwargs,
            in_process_runtime_env=in_process_runtime_env,
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
                stage_normalization_workers=args.gpu_num_workers
                if args.inference_mode == "in_process_vllm"
                else args.client_num_workers,
            )
            num_request_errors = pdf_parse_metrics.get("num_request_errors", 0.0)

            logger.info(f"Benchmark completed in {run_time_taken:.2f}s")
            logger.info(f"Processed {num_pdfs_processed} PDFs")
            logger.info(
                "Normalized inference stage throughput: "
                f"{pdf_parse_metrics['normalized_inference_stage_pages_per_sec']:.2f} pages/s"
            )
            logger.info(f"Inference throughput: {pdf_parse_metrics['inference_pages_sec']:.2f} pages/s")
            logger.info(f"Pipeline throughput: {pdf_parse_metrics['pipeline_pages_sec']:.2f} pages/s")
            if num_request_errors > 0:
                logger.error(f"Benchmark produced {num_request_errors:.0f} inference server request error(s)")
            else:
                logger.success("Benchmark completed without inference server request errors")
                success = True

        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Benchmark failed: {e}")
            logger.debug(f"Full traceback:\n{error_traceback}")
            run_time_taken = time.perf_counter() - run_start_time
            num_pdfs_processed = 0
            pdf_parse_metrics = {}

    finally:
        if started_server is not None:
            with contextlib.suppress(Exception):
                started_server.server.stop()

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
            "server_backend": server_backend,
            "server_transport": _server_transport_for_mode(args.inference_mode),
            "server_image_request_mode": args.server_image_request_mode,
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
            "gpu_num_workers": args.gpu_num_workers,
            "client_num_workers": args.client_num_workers,
            "request_batch_size": args.request_batch_size,
            "server_num_replicas": args.server_num_replicas,
            "dynamo_use_vllm_chat_processor": args.dynamo_use_vllm_chat_processor,
            "inference_server_request_timeout_s": args.inference_server_request_timeout_s,
            "inference_server_endpoint": endpoint,
            "inference_server_max_retries": args.inference_server_max_retries,
            "inference_server_max_tokens": args.inference_server_max_tokens,
            "inference_server_stream": args.inference_server_stream,
            "inference_server_proc_size": args.inference_server_proc_size,
            "inference_server_file_url_prefix": args.inference_server_file_url_prefix,
            "engine_kwargs": engine_kwargs,
            "dynamo_frontend_kwargs": frontend_kwargs,
            "dynamo_disagg_config": dynamo_disagg_config,
            "dynamo_encoder_disagg_config": dynamo_encoder_disagg_config,
            "dynamo_model_runtime_env": dynamo_model_runtime_env,
            "in_process_runtime_env": in_process_runtime_env,
            "autoscaling_config": autoscaling_config,
            "server_gpu_count": server_gpu_total,
            "accounting_num_gpus": accounting_num_gpus,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_pdfs_processed": num_pdfs_processed,
            "num_output_tasks": len(output_tasks),
            "throughput_pdfs_per_sec": _safe_div(num_pdfs_processed, run_time_taken),
            "serve_startup_s": serve_startup_s,
            "server_gpu_count": server_gpu_total,
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
        choices=["in_process_vllm", "dynamo_http", "external_http", "ray_serve_http", "ray_serve_grpc"],
        help="Inference path to benchmark. ray_serve_grpc uses Ray Serve's Python deployment handle.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_PATH,
        help="Served model name for server modes.",
    )
    parser.add_argument(
        "--engine-kwargs",
        type=str,
        default=None,
        help="JSON string of vLLM engine kwargs, for example '{\"tensor_parallel_size\": 1}'",
    )
    parser.add_argument(
        "--dynamo-frontend-kwargs",
        type=str,
        default=None,
        help="JSON string of Dynamo frontend kwargs.",
    )
    parser.add_argument(
        "--dynamo-disagg-config",
        type=str,
        default=None,
        help=(
            "Optional JSON role config for Dynamo disaggregated serving. "
            'Example: \'{"prefill":{"num_replicas":3},"decode":{"num_replicas":1}}\'.'
        ),
    )
    parser.add_argument(
        "--dynamo-encoder-disagg-config",
        type=str,
        default=None,
        help=(
            "Optional JSON role config for Dynamo multimodal encoder-disaggregated E/PD serving. "
            'Example: \'{"encode":{"num_replicas":1}}\'. Backend replicas still come from '
            "--server-num-replicas."
        ),
    )
    parser.add_argument(
        "--in-process-runtime-env",
        type=str,
        default=None,
        help=(
            "Optional JSON Ray runtime_env applied to in-process GPU workers. "
            "Useful for model-specific env_vars or packages."
        ),
    )
    parser.add_argument(
        "--dynamo-model-runtime-env",
        type=str,
        default=None,
        help=(
            "Optional JSON Ray runtime_env merged into Dynamo model actors. "
            "Useful for smoke-testing patched actor-side packages via env_vars or py_modules."
        ),
    )
    parser.add_argument(
        "--autoscaling-config",
        type=str,
        default=None,
        help=(
            "Optional static JSON replica config. If omitted, --server-num-replicas is used. "
            "For this benchmark min_replicas must equal max_replicas."
        ),
    )
    parser.add_argument("--gpu-num-workers", type=int, default=4, help="Fixed GPU workers for in-process Ray Data.")
    parser.add_argument(
        "--client-num-workers",
        "--inference-server-stage-workers",
        dest="client_num_workers",
        type=int,
        default=16,
        help="Fixed CPU client workers for inference-server modes.",
    )
    parser.add_argument(
        "--request-batch-size",
        "--inference-server-concurrency",
        dest="request_batch_size",
        type=int,
        default=4,
        help="Number of page requests each client worker keeps in flight at once. Use -1 for all pages in a task.",
    )
    parser.add_argument(
        "--server-num-replicas",
        type=int,
        default=1,
        help="Static server replica count when --autoscaling-config is omitted.",
    )
    parser.add_argument(
        "--dynamo-use-vllm-chat-processor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include dyn_chat_processor='vllm' in Dynamo frontend kwargs.",
    )
    parser.add_argument(
        "--server-image-request-mode",
        choices=["base64", "file_url"],
        default="base64",
        help="How page images are sent to inference servers.",
    )
    parser.add_argument(
        "--inference-server-endpoint",
        default=None,
        help="Existing OpenAI-compatible HTTP endpoint used with external_http.",
    )
    parser.add_argument(
        "--inference-server-request-timeout-s",
        type=float,
        default=180.0,
        help="Per-request timeout for server inference.",
    )
    parser.add_argument(
        "--inference-server-max-retries",
        type=int,
        default=3,
        help="Retries per page request for server inference.",
    )
    parser.add_argument(
        "--inference-server-max-tokens",
        type=int,
        default=9000,
        help="Max generated tokens per page for server inference.",
    )
    parser.add_argument(
        "--inference-server-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use streaming chat completions and aggregate chunks client-side.",
    )
    parser.add_argument(
        "--inference-server-extra-body",
        type=str,
        default='{"top_k": 1, "repetition_penalty": 1.1, "skip_special_tokens": false}',
        help="JSON object passed as OpenAI extra_body for HTTP server inference.",
    )
    parser.add_argument(
        "--inference-server-image-dir",
        default=None,
        help="Directory used to write PNG page images for file_url mode.",
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
        help="Processor size HEIGHT,WIDTH used by postprocess in server modes.",
    )
    parser.add_argument(
        "--accounting-num-gpus",
        type=int,
        default=None,
        help="GPU count used to normalize inference metrics.",
    )

    args = parser.parse_args()

    logger.info("=== Nemotron-Parse PDF Benchmark Starting ===")
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
