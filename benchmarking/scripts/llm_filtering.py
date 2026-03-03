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

# ruff: noqa: PLR0913, ANN401, TRY300

"""LLM-based filtering benchmark.

Sends each row to an OpenAI-compatible LLM endpoint, parses a keep/remove
decision from the response, and writes only the kept rows.

Backends:
  - ray-serve   : vLLM behind Ray Serve via ModelServer (supports multi-replica autoscaling)
  - vllm-direct : standalone vLLM OpenAI-compatible server (tensor parallel)
  - nvidia-nim  : NVIDIA Build cloud API

Usage:
    python llm_filtering.py \
        --benchmark-results-path /tmp/results \
        --input-path ./data/llm_filtering \
        --output-path /tmp/llm_filtering_output \
        --model-type ray-serve \
        --model-id google/gemma-3-27b-it \
        --executor xenna \
        --autoscaling-config '{"min_replicas": 1, "max_replicas": 4}'
"""

import argparse
import asyncio
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from loguru import logger

from nemo_curator.models.client import AsyncOpenAIClient
from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
from nemo_curator.tasks.document import DocumentBatch

from .utils import load_dataset_files, setup_executor, start_inference_server, write_benchmark_results

# ---------------------------------------------------------------------------
# Default prompt - override via --prompt-template or --prompt-template-path
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATE = """\
You are a data quality filter. Decide whether the following text is high \
quality and should be KEPT, or is low quality and should be REMOVED.

TEXT:
{text}

Respond with a JSON object containing your reasoning and decision.
"""

# JSON schema for vLLM guided generation — constrains output tokens so the
# model can *only* produce valid JSON matching this structure.
DECISION_JSON_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "filter_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                },
                "decision": {
                    "type": "string",
                    "enum": ["keep", "remove"],
                },
            },
            "required": ["reasoning", "decision"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# LLM filter stage
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _run_async(coro: "asyncio.Coroutine[Any, Any, T]") -> T:
    """Run a coroutine from sync code. Safe in Jupyter (existing event loop) and in scripts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop (e.g. Jupyter); run in a new thread to avoid nesting
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def get_last_assistant_content(row: Any) -> str | None:
    """Extract the content of the last assistant message that has actual text.

    Skips tool-call-only messages (where content is empty/None but tool_calls is present).
    """
    messages = row["messages"]
    return next(
        (m["content"] for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")),
        None,
    )


def _strip_markdown_fences(text: str) -> str:
    """Strip ```json ... ``` fences that some endpoints add despite response_format."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_decision(response_text: str) -> tuple[str, str]:
    """Extract decision and reasoning from an LLM response.

    Returns (decision, reasoning) where decision is 'keep' or 'remove'
    (defaults to 'keep') and reasoning is the LLM's explanation (may be empty).

    With JSON schema enforcement via response_format, the LLM output should
    always be valid JSON. The markdown stripping and regex fallback are
    defense-in-depth for edge cases.
    """
    cleaned = _strip_markdown_fences(response_text)

    # Primary path: structured output from response_format should parse cleanly
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            decision = data.get("decision", "keep").lower().strip()
            reasoning = data.get("reasoning", "")
            decision = "keep" if decision in ("keep", "kept", "yes") else "remove"
            return decision, reasoning
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: regex extraction if JSON parsing fails
    m = re.search(r'\{[^}]*"decision"\s*:\s*"(\w+)"[^}]*\}', response_text, re.DOTALL)
    if m:
        decision = "keep" if m.group(1).lower().strip() in ("keep", "kept", "yes") else "remove"
        return decision, ""

    logger.warning(f"Could not parse LLM response as JSON, marking as 'fail': {response_text[:200]}")
    return "fail", ""


class LLMFilterStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Calls an OpenAI-compatible endpoint to annotate rows with keep/remove/fail.

    Accepts either ``AsyncLLMClient`` (concurrent via ``asyncio.gather``) or
    ``LLMClient`` (sequential).  Retry and concurrency control are handled by
    the client.  Structured output (``response_format``) is passed via
    ``GenerationConfig.extra_kwargs``.

    All rows are returned with annotations — no rows are dropped.
    """

    name = "LLMFilterStage"

    def __init__(
        self,
        client: AsyncLLMClient | LLMClient,
        model_id: str,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        text_field: str = "text",
        text_extractor: Any | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        use_structured_output: bool = True,
    ):
        self.client = client
        self.model_id = model_id
        self.prompt_template = prompt_template
        self.text_field = text_field
        self.text_extractor = text_extractor
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_structured_output = use_structured_output
        self.is_async = isinstance(client, AsyncLLMClient)

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ARG002
        self.client.setup()

    def _generation_config(self) -> GenerationConfig:
        extra: dict[str, Any] = {}
        if self.use_structured_output:
            extra["response_format"] = DECISION_JSON_SCHEMA
        return GenerationConfig(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_kwargs=extra or None,
        )

    def _score_one_sync(self, text: str | None, gen_config: GenerationConfig) -> tuple[str, str, str]:
        """Score a single row synchronously. Returns (decision, reasoning, raw_response)."""
        if text is None or text == "" or (isinstance(text, float) and pd.isna(text)):
            return "empty_prompt", "text_extractor returned None/empty", ""
        prompt = self.prompt_template.format(text=str(text))
        messages = [{"role": "user", "content": prompt}]
        try:
            result = self.client.query_model(
                messages=messages,
                model=self.model_id,
                generation_config=gen_config,
            )
            content = result[0] if result else ""
            content = content or ""
            decision, reasoning = _parse_decision(content)
            return decision, reasoning, content
        except Exception:
            logger.exception("LLM query failed")
            return "fail", "", ""

    async def _score_one_async(self, text: str | None, gen_config: GenerationConfig) -> tuple[str, str, str]:
        """Score a single row asynchronously. Returns (decision, reasoning, raw_response)."""
        if text is None or text == "" or (isinstance(text, float) and pd.isna(text)):
            return "empty_prompt", "text_extractor returned None/empty", ""
        prompt = self.prompt_template.format(text=str(text))
        messages = [{"role": "user", "content": prompt}]
        try:
            result = await self.client.query_model(
                messages=messages,
                model=self.model_id,
                generation_config=gen_config,
            )
            content = result[0] if result else ""
            content = content or ""
            decision, reasoning = _parse_decision(content)
            return decision, reasoning, content
        except Exception:
            logger.exception("LLM query failed after retries")
            return "fail", "", ""

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df = task.to_pandas()
        if df.empty:
            return task

        if self.text_extractor is not None:
            texts = df.apply(self.text_extractor, axis=1).tolist()
        else:
            texts = df[self.text_field].tolist()

        gen_config = self._generation_config()

        if self.is_async:

            async def _score_batch() -> list[tuple[str, str, str]]:
                return await asyncio.gather(*(self._score_one_async(t, gen_config) for t in texts))

            results = _run_async(_score_batch())
        else:
            results = [self._score_one_sync(t, gen_config) for t in texts]

        decisions = [r[0] for r in results]
        reasonings = [r[1] for r in results]
        responses = [r[2] for r in results]
        df["llm_filter_decision"] = decisions
        df["llm_reasoning"] = reasonings
        df["llm_response"] = responses

        num_kept = (df["llm_filter_decision"] == "keep").sum()
        num_removed = (df["llm_filter_decision"] == "remove").sum()
        num_failed = (df["llm_filter_decision"] == "fail").sum()
        num_empty_prompt = (df["llm_filter_decision"] == "empty_prompt").sum()
        logger.info(
            f"LLM filter: kept {num_kept}/{len(df)}, removed {num_removed}/{len(df)}, "
            f"failed {num_failed}/{len(df)}, empty_prompt {num_empty_prompt}/{len(df)}"
        )

        self._log_metric("num_input", float(len(df)))
        self._log_metric("num_kept", float(num_kept))
        self._log_metric("num_removed", float(num_removed))
        self._log_metric("num_failed", float(num_failed))
        self._log_metric("num_empty_prompt", float(num_empty_prompt))

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    model_type: str,
    model_id: str,
    input_path: str,
    output_path: str,
    executor: str,
    engine_kwargs: dict[str, Any],
    autoscaling_config: dict[str, Any],
    num_files: int | None,
    max_concurrent_requests: int,
    prompt_template: str,
    text_field: str,
    no_structured_output: bool = False,
    extract_last_assistant: bool = False,
    benchmark_results_path: str = "",
    **kwargs: Any,  # noqa: ARG001
) -> dict[str, Any]:
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path).absolute()
    output_path_obj.mkdir(parents=True, exist_ok=True)

    input_files = load_dataset_files(input_path_obj, dataset_ratio=1.0, keep_extensions="jsonl")
    if num_files is not None and num_files > 0:
        input_files = input_files[:num_files]
    logger.info(f"Input files: {len(input_files)}")

    # -- Start inference server ------------------------------------------
    log_dir = Path(benchmark_results_path) / "logs" if benchmark_results_path else None
    server = start_inference_server(model_type, model_id, engine_kwargs, autoscaling_config, log_dir=log_dir)

    # -- Build and run pipeline ------------------------------------------
    executor_obj = setup_executor(executor)

    client = AsyncOpenAIClient(
        max_concurrent_requests=max_concurrent_requests,
        api_key=server.api_key,
        base_url=server.endpoint,
        timeout=120,
    )

    text_extractor = get_last_assistant_content if extract_last_assistant else None

    pipeline = Pipeline(
        name="llm_filtering_benchmark",
        stages=[
            JsonlReader(file_paths=input_files),
            LLMFilterStage(
                client=client,
                model_id=model_id,
                prompt_template=prompt_template,
                text_field=text_field,
                text_extractor=text_extractor,
                use_structured_output=not no_structured_output,
            ),
            JsonlWriter(path=str(output_path_obj)),
        ],
    )

    logger.info("Starting LLM filtering pipeline...")
    run_start = time.perf_counter()
    try:
        output_tasks = pipeline.run(executor_obj)
    finally:
        run_time = time.perf_counter() - run_start
        server.stop()

    logger.success(f"Completed in {run_time:.2f}s")
    return {
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time,
            "serve_startup_s": server.startup_s,
            "model_type": model_type,
            "model_id": model_id,
            "num_files": num_files or "all",
        },
        "tasks": output_tasks,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM filtering benchmark")
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-type", required=True, choices=["ray-serve", "vllm-direct", "nvidia-nim"])
    parser.add_argument("--model-id", default="google/gemma-3-27b-it")
    parser.add_argument("--executor", default="xenna", choices=["ray_data", "xenna", "ray_actors"])
    parser.add_argument(
        "--engine-kwargs",
        type=json.loads,
        default="{}",
        help='JSON dict of vLLM engine kwargs (e.g. \'{"tensor_parallel_size": 4, "max_model_len": 4096}\')',
    )
    parser.add_argument(
        "--autoscaling-config",
        type=json.loads,
        default='{"min_replicas": 1, "max_replicas": 1}',
        help='JSON dict for Ray Serve autoscaling (e.g. \'{"min_replicas": 0, "max_replicas": 4}\')',
    )
    parser.add_argument("--max-concurrent-requests", type=int, default=512, help="Sync requests in-flight")
    parser.add_argument("--num-files", type=int, default=None)
    parser.add_argument("--text-field", default="text", help="JSONL field containing the text to filter")
    parser.add_argument("--prompt-template", default=None, help="Inline prompt template (use {text} placeholder)")
    parser.add_argument("--prompt-template-path", default=None, help="Path to file containing prompt template")
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help="Disable JSON schema enforcement (response_format). Use for endpoints that don't support guided generation.",
    )
    parser.add_argument(
        "--extract-last-assistant",
        action="store_true",
        help="Extract the last assistant message from a 'messages' column instead of using --text-field directly.",
    )

    args = parser.parse_args()

    # Resolve prompt template
    if args.prompt_template_path:
        prompt_template = Path(args.prompt_template_path).read_text()
    elif args.prompt_template:
        prompt_template = args.prompt_template
    else:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    logger.info(f"=== LLM Filtering Benchmark Starting ===\n{vars(args)}")

    # Serialize JSON dicts as strings for params.json
    params = vars(args).copy()
    params["engine_kwargs"] = json.dumps(params["engine_kwargs"])
    params["autoscaling_config"] = json.dumps(params["autoscaling_config"])

    result_dict: dict[str, Any] = {"params": params, "metrics": {"is_success": False}, "tasks": []}
    success = 1
    try:
        result_dict.update(run_benchmark(**{**vars(args), "prompt_template": prompt_template}))
        success = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success


if __name__ == "__main__":
    raise SystemExit(main())
