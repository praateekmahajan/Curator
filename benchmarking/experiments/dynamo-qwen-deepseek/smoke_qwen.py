# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001
"""Exercise the preinstalled Qwen environment through Curator's real Ray actors."""

import json
import os
import time
import urllib.request
from pathlib import Path

import ray

from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer


def main() -> None:
    model = DynamoVLLMModelConfig(
        model_identifier="Qwen/Qwen3.8-27B",
        runtime_env={"py_executable": os.environ.get("SERVING_PYTHON", "/usr/bin/python3")},
        engine_kwargs={
            "max_model_len": 8192,
            "max_num_seqs": 4,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "disable_hybrid_kv_cache_manager": False,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
        },
        dynamo_kwargs={"enable_multimodal": True, "dyn_reasoning_parser": "qwen3"},
    )
    backend = DynamoServerConfig(
        request_plane="tcp",
        subprocess_env={
            "HF_HOME": "/hf",
            "HF_HUB_OFFLINE": "1",
            "CUDA_CACHE_PATH": "/cache/cuda",
            "TRITON_CACHE_DIR": "/cache/triton",
            "VLLM_CACHE_ROOT": "/cache/vllm",
        },
    )
    ray.init(num_cpus=12, num_gpus=1, include_dashboard=False, _temp_dir="/results/ray", object_store_memory=1024**3)
    started = time.monotonic()
    try:
        with InferenceServer(models=[model], backend=backend, health_check_timeout_s=1200) as server:
            ready_s = time.monotonic() - started
            payload = {
                "model": model.model_identifier,
                "messages": [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
                "max_tokens": 128,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request = urllib.request.Request(  # noqa: S310
                f"{server.endpoint}/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                result = json.load(response)
            if result["choices"][0]["message"]["content"].strip() != "4":
                msg = f"Unexpected response: {result}"
                raise RuntimeError(msg)
            report = {"ready_s": ready_s, "response": result}
            Path("/results/result.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
