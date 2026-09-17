# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify viewer schema, time-window filtering and GPU identity across nodes."""

import csv
import json
from pathlib import Path

import pytest

from benchmarking.runner.gpu_stats_recorder import GPUStatsRecorder
from benchmarking.scripts.collect_native_vllm_gpustats import collect_entry


def test_multi_node_export(tmp_path: Path) -> None:
    entry = tmp_path / "entry"
    entry.mkdir()
    metrics = {
        "is_success": 1,
        "pipeline_started_unix_s": 1,
        "pipeline_finished_unix_s": 2,
        "qwen_requests": 10,
        "deepseek_requests": 10,
    }
    (entry / "metrics.json").write_text(json.dumps(metrics))
    for node in ("node-a", "node-b"):
        folder = tmp_path / "telemetry" / node
        folder.mkdir(parents=True)
        (folder / "gpustats.csv").write_text(
            "timestamp,index,uuid,power.draw [W],power.limit [W],utilization.gpu [%],"
            "utilization.memory [%],memory.used [MiB],memory.total [MiB],temperature.gpu\n"
            + "".join(
                f"1970/01/01 00:00:0{second}.000,0,GPU-{node},400 W,700 W,99 %,30 %,50 MiB,100 MiB,55\n"
                for second in range(4)
            )
            + "1970/01/01 00:00:04.000,0,"  # Incomplete last record from a live writer.
        )
    nodes = {"qwen": "node-a", "deepseek": "node-b"}
    assert collect_entry(entry, tmp_path / "telemetry", nodes) == 4
    with (entry / "gpustats.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames[: len(GPUStatsRecorder.HEADER)] == GPUStatsRecorder.HEADER
        rows = list(reader)
    assert {row["gpu_id"] for row in rows} == {"0", "1"}
    assert {row["node"] for row in rows} == set(nodes.values())
    assert all(row["utilization_memory_pct"] == "50.0" for row in rows)
    assert all(row["memory_access_busy_pct"] == "30" for row in rows)
    assert all(json.loads(row["processes"]) == [] for row in rows)
    exported = json.loads((entry / "metrics.json").read_text())
    assert exported["node_a_mean_gpu_power_w"] == 400
    assert exported["node_b_measured_gpu_energy_kwh"] == pytest.approx(400 / 3_600_000)
    # A single-model entry must exclude the other node, including its idle power.
    del metrics["qwen_requests"]
    (entry / "metrics.json").write_text(json.dumps(metrics))
    assert collect_entry(entry, tmp_path / "telemetry", nodes) == 2
    with (entry / "gpustats.csv").open() as handle:
        assert {row["node"] for row in csv.DictReader(handle)} == {"node-b"}
    metrics["pipeline_started_unix_s"] = 20
    metrics["pipeline_finished_unix_s"] = 21
    (entry / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="No GPU samples"):
        collect_entry(entry, tmp_path / "telemetry", nodes)
