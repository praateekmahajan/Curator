# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise request routing, concurrency, token accounting and Parquet output."""

import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking" / "scripts"))
from native_vllm_benchmark import NativeVLLMClientStage, _compute_native_vllm_metrics


def test_http_routing_concurrency_and_metrics(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = {"qwen": 0, "deepseek": 0}
    peak = {"qwen": 0, "deepseek": 0}
    seen: list[tuple[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = body["model"]
            with lock:
                active[model] += 1
                peak[model] = max(peak[model], active[model])
                seen.append((model, body["chat_template_kwargs"]))
            time.sleep(0.02)
            response = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                }
            ).encode()
            with lock:
                active[model] -= 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        stage = NativeVLLMClientStage(
            models={
                alias: {
                    "model": alias,
                    "endpoint": f"http://127.0.0.1:{server.server_port}/v1",
                    "replicas": 1,
                    "gpus_per_replica": 8,
                    "chat_template_kwargs": {"thinking": False},
                }
                for alias in active
            },
            prompt_field="curator_question",
            generation={"max_tokens": 2048, "temperature": 0.0, "top_p": 1.0, "seed": 42},
            concurrency={"qwen": 2, "deepseek": 1},
        )
        records = asyncio.run(stage.query([{"curator_int_id": i, "curator_question": "Question?"} for i in range(4)]))
        assert len(records) == 8
        assert peak == {"qwen": 2, "deepseek": 1}
        assert all(kwargs == {"thinking": False} for _, kwargs in seen)
        pq.write_table(pa.Table.from_pylist(records), tmp_path / "results.parquet")
        metrics = _compute_native_vllm_metrics([tmp_path], 2.0, {"qwen": 4, "deepseek": 8})
        assert metrics["qwen_requests"] == 4
        assert metrics["deepseek_mean_input_tokens"] == 7
        assert metrics["qwen_output_tokens_per_s"] == 6
        assert metrics["deepseek_mean_output_tokens"] == 3
        assert metrics["deepseek_gpu_hours_per_million_requests"] == 2 * metrics["qwen_gpu_hours_per_million_requests"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
