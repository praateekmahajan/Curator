# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking" / "scripts"))
import manifest_inference as benchmark
import native_vllm_metrics

from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.performance_utils import StagePerfStats


def arguments(tmp_path: Path) -> argparse.Namespace:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"manifest_inference": {"models": {"qwen": {"replicas": 4, "gpus_per_replica": 1}}}}))
    return argparse.Namespace(workload_config=config, model_key="qwen", client_workers=64)


@pytest.mark.parametrize("success", [True, False])
def test_aggregation_failure_preserves_pipeline_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, success: bool
):
    def fail(*_args, **_kwargs) -> None:
        msg = "aggregation failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(native_vllm_metrics, "_compute_native_vllm_metrics", fail)
    basic = {"is_success": success, "time_taken_s": 2.0, "rows_written_this_attempt": 4}
    results = {"metrics": dict(basic), "tasks": []}
    benchmark.add_benchmark_metrics(results, arguments(tmp_path), 100, 102)
    assert results["metrics"] == {**basic, "benchmark_metrics_aggregation_success": False}
    assert results["tasks"] == []


@pytest.mark.parametrize("success", [True, False])
def test_jsonl_metrics_and_failed_pipeline_status(tmp_path: Path, success: bool):
    task = FileGroupTask(dataset_name="test", data=["part.jsonl.zst"])
    task.add_stage_perf(
        StagePerfStats(
            stage_name="native_vllm_client",
            custom_metrics={
                "num_requests": 4,
                "num_successful_completions": 4,
                "num_api_attempts": 5,
                "num_input_tokens": 28,
                "num_output_tokens": 12,
                "request_retries": 1,
                "requests_retried": 1,
                "first_request_started_unix_s": 100,
                "last_response_finished_unix_s": 102,
            },
        )
    )
    task.add_stage_perf(StagePerfStats(stage_name="named_jsonl_writer", num_items_processed=4))
    results = {
        "metrics": {"is_success": success, "time_taken_s": 2.0, "rows_written_this_attempt": 4},
        "tasks": [task],
    }
    benchmark.add_benchmark_metrics(results, arguments(tmp_path), 100, 102)
    metrics = results["metrics"]
    assert metrics["benchmark_metrics_aggregation_success"] is True
    assert metrics["is_success"] is success
    assert metrics["is_complete"] is success
    assert metrics["num_rows_written"] == 4
    assert metrics["rows_per_sec"] == 2
    assert metrics["output_tokens_per_sec_per_gpu"] == 1.5
    assert metrics["retry_rate"] == 0.25
    assert metrics["service_window_time_s"] == 2
    assert metrics["service_window_rows_per_sec"] == 2
    assert "mean_request_latency_s" not in metrics


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize("failure", ["raises", "missing_helper"])
def test_entrypoint_writes_results_even_when_aggregation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, success: bool, failure: str
):
    args = arguments(tmp_path)
    output = tmp_path / "results"
    basic = {"is_success": success, "time_taken_s": 2.0, "rows_written_this_attempt": 4}
    results = {"metrics": dict(basic), "tasks": [], "params": {"model_key": "qwen"}}
    monkeypatch.setattr(benchmark, "run", lambda _args: results)
    if failure == "missing_helper":
        monkeypatch.setitem(sys.modules, "native_vllm_metrics", None)
    else:

        def fail(*_args, **_kwargs) -> None:
            msg = "deliberate aggregation failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(native_vllm_metrics, "_compute_native_vllm_metrics", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manifest_inference.py",
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
            "--input-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "data"),
            "--checkpoint-path",
            str(tmp_path / "checkpoint"),
            "--workload-config",
            str(args.workload_config),
            "--model-key",
            "qwen",
            "--max-concurrent-requests",
            "128",
            "--benchmark-results-path",
            str(output),
        ],
    )
    assert benchmark.main() == (0 if success else 1)
    assert json.loads((output / "metrics.json").read_text()) == {
        **basic,
        "benchmark_metrics_aggregation_success": False,
    }
    assert (output / "tasks.pkl").is_file()
    assert json.loads((output / "params.json").read_text()) == {"model_key": "qwen"}
