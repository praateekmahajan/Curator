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

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def _import_kiran_benchmark(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import the benchmark script without requiring Data Designer test deps."""
    data_designer = types.ModuleType("data_designer")
    dd_config = types.ModuleType("data_designer.config")
    dd_config.DataDesignerConfigBuilder = object
    data_designer.config = dd_config

    benchmark_utils = types.ModuleType("utils")
    benchmark_utils.setup_executor = object
    benchmark_utils.write_benchmark_results = object

    data_designer_stage_mod = types.ModuleType("nemo_curator.stages.synthetic.nemo_data_designer.data_designer")
    data_designer_stage_mod.DataDesignerStage = object

    monkeypatch.setitem(sys.modules, "data_designer", data_designer)
    monkeypatch.setitem(sys.modules, "data_designer.config", dd_config)
    monkeypatch.setitem(sys.modules, "utils", benchmark_utils)
    monkeypatch.setitem(
        sys.modules,
        "nemo_curator.stages.synthetic.nemo_data_designer.data_designer",
        data_designer_stage_mod,
    )

    script_dir = Path(__file__).resolve().parents[2] / "benchmarking" / "scripts"
    monkeypatch.syspath_prepend(str(script_dir))
    return importlib.import_module("kiran_sdg_translation_eval_benchmark")


def test_kiran_config_loader_drops_legacy_column_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    benchmark = _import_kiran_benchmark(monkeypatch)
    captured_config = {}

    class FakeConfigBuilder:
        @staticmethod
        def from_config(config: dict) -> str:
            captured_config.update(config)
            return "builder"

    benchmark.dd.DataDesignerConfigBuilder = FakeConfigBuilder
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_designer": {
                    "seed_config": {"path": "/old/seed"},
                    "columns": [
                        {
                            "name": "translated_text",
                            "column_type": "llm-text",
                            "allow_resize": False,
                            "skip": None,
                            "propagate_skip": True,
                        }
                    ],
                    "model_configs": [
                        {
                            "alias": benchmark.TRANSLATION_ALIAS,
                            "model": "translation",
                            "inference_parameters": {},
                        },
                        {
                            "alias": benchmark.EVALUATION_ALIAS,
                            "model": "evaluation",
                            "inference_parameters": {},
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        benchmark._load_config_builder(config_path, translation_concurrency=11, evaluation_concurrency=22) == "builder"
    )

    assert "seed_config" not in captured_config
    column_config = captured_config["columns"][0]
    assert "allow_resize" not in column_config
    assert "skip" not in column_config
    assert "propagate_skip" not in column_config
    translation_config, evaluation_config = captured_config["model_configs"]
    assert translation_config["provider"] == benchmark.PROVIDER_NAME
    assert translation_config["skip_health_check"] is True
    assert translation_config["inference_parameters"]["max_parallel_requests"] == 11
    assert evaluation_config["provider"] == benchmark.PROVIDER_NAME
    assert evaluation_config["skip_health_check"] is True
    assert evaluation_config["inference_parameters"]["max_parallel_requests"] == 22


def test_kiran_gemma4_models_request_transformers5_actor_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = _import_kiran_benchmark(monkeypatch)
    args = argparse.Namespace(
        translation_model_identifier="RedHatAI/gemma-4-12B-it-FP8-Dynamic",
        translation_model_path="/model_weights/models--RedHatAI--gemma-4-12B-it-FP8-Dynamic/snapshots/abc",
        translation_served_model_name="google/gemma-4-12B-it",
        evaluation_model_identifier="RedHatAI/gemma-4-31B-it-FP8-block",
        evaluation_model_path="/model_weights/models--RedHatAI--gemma-4-31B-it-FP8-block/snapshots/def",
        evaluation_served_model_name="google/gemma-4-31B-it",
        max_num_seqs=256,
        max_model_len=32768,
        disable_speculative=True,
        translation_speculative_model="google/gemma-4-12B-it-assistant",
        evaluation_speculative_model="google/gemma-4-31B-it-assistant",
        num_speculative_tokens=4,
        translation_replicas=2,
        evaluation_replicas=1,
        translation_linear_backend=None,
        evaluation_linear_backend=None,
        translation_disable_deep_gemm=False,
        evaluation_disable_deep_gemm=True,
        hf_home=None,
    )

    models = benchmark._build_model_configs(args)

    assert len(models) == 2
    assert models[0].model_identifier == args.translation_model_path
    assert models[0].model_name == args.translation_served_model_name
    assert models[1].model_identifier == args.evaluation_model_path
    assert models[1].model_name == args.evaluation_served_model_name
    assert "linear_backend" not in models[0].engine_kwargs
    assert "linear_backend" not in models[1].engine_kwargs
    assert models[0].runtime_env["uv"]["packages"] == ["transformers>=5.10.1"]
    assert "env_vars" not in models[0].runtime_env
    assert models[1].runtime_env["uv"]["packages"] == ["transformers>=5.10.1"]
    assert models[1].runtime_env["env_vars"] == {"VLLM_USE_DEEP_GEMM": "0"}
