# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001
"""Read, infer through the isolated serving venv, and write in Curator's base env."""

import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import ray

from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
from nemo_curator.models.client.llm_client import GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemotron_cc.base import BaseSyntheticStage
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter


def environment() -> dict:
    return {"python": sys.executable, "python_version": sys.version, "ray": version("ray"), "torch": version("torch")}


def main() -> None:
    results = Path("/results")
    data = pd.DataFrame({"id": [1, 2], "text": ["2 + 2", "3 + 5"], "expected": ["4", "8"]})
    data.to_json(results / "input.jsonl", orient="records", lines=True)
    data.to_parquet(results / "input.parquet", index=False)
    serving_env = {"py_executable": os.environ.get("SERVING_PYTHON", sys.executable)}
    model = DynamoVLLMModelConfig(
        model_identifier="Qwen/Qwen3.8-27B",
        runtime_env=serving_env,
        engine_kwargs={
            "load_format": os.environ.get("LOAD_FORMAT", "auto"),
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
        report = {
            "driver": environment(),
            "serving_actor": ray.get(ray.remote(environment).options(runtime_env=serving_env).remote()),
            "load_format": model.engine_kwargs["load_format"],
            "pipelines": {},
        }
        with InferenceServer(models=[model], backend=backend, health_check_timeout_s=1200) as server:
            report["ready_s"] = time.monotonic() - started
            for input_format, reader in (("jsonl", JsonlReader), ("parquet", ParquetReader)):
                for output_format, writer in (("jsonl", JsonlWriter), ("parquet", ParquetWriter)):
                    name = f"{input_format}-to-{output_format}"
                    output = results / name
                    pipeline = Pipeline(name=name)
                    pipeline.add_stage(reader(file_paths=str(results / f"input.{input_format}")))
                    pipeline.add_stage(
                        BaseSyntheticStage(
                            prompt="Calculate {document}. Return only the integer.",
                            input_field="text",
                            output_field="answer",
                            client=AsyncOpenAIClient(api_key="unused", base_url=server.endpoint),
                            model_name=model.model_identifier,
                            generation_config=GenerationConfig(
                                temperature=0,
                                max_tokens=64,
                                extra_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
                            ),
                        ).with_(num_workers=1)
                    )
                    pipeline.add_stage(writer(path=str(output)))
                    pipeline.build()
                    pipeline.stages = [stage.with_(num_workers=1) for stage in pipeline.stages]
                    pipeline.run()
                    paths = sorted(output.rglob(f"*.{output_format}"))
                    read = (
                        (lambda path: pd.read_json(path, lines=True, dtype=False))
                        if output_format == "jsonl"
                        else pd.read_parquet
                    )
                    actual = pd.concat([read(path) for path in paths]).sort_values("id")
                    if actual["id"].tolist() != [1, 2] or actual["answer"].astype(str).str.strip().tolist() != [
                        "4",
                        "8",
                    ]:
                        msg = f"Unexpected pipeline output: {actual.to_dict(orient='records')}"
                        raise RuntimeError(msg)
                    report["pipelines"][name] = actual.to_dict(orient="records")
                    print(f"PASS {name}: {report['pipelines'][name]}", flush=True)
            (results / "result.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
