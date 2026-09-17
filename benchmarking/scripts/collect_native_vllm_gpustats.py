# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge serving-node telemetry into each completed entry's viewer-compatible CSV."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from benchmarking.runner.gpu_stats_recorder import GPUStatsRecorder

EXTRA_COLUMNS = ["node", "local_gpu_id", "gpu_uuid", "memory_access_busy_pct"]


def _number(value: str) -> str:
    """Strip nvidia-smi units; unsupported NVML measurements remain blank."""
    token = value.split()[0]
    try:
        float(token)
    except ValueError:
        return ""
    return token


def _read_node(telemetry: Path, node: str, start: float, end: float) -> tuple[list[dict[str, str]], dict[str, float]]:
    samples: list[dict[str, str]] = []
    metrics: dict[str, float] = {}
    powers: list[float] = []
    utilization: list[float] = []
    previous: dict[str, tuple[float, float]] = {}
    energy_ws = 0.0
    count = 0
    with (telemetry / node / "gpustats.csv").open() as handle:
        for row in csv.DictReader(handle, skipinitialspace=True):
            if not row.get("temperature.gpu"):
                continue  # A live recorder may have written only part of its final row.
            timestamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC).timestamp()
            if not start <= timestamp <= end:
                continue
            used, total = float(_number(row["memory.used [MiB]"])), float(_number(row["memory.total [MiB]"]))
            power = _number(row["power.draw [W]"])
            busy = _number(row["utilization.gpu [%]"])
            samples.append(
                {
                    "timestamp_utc": datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z"),
                    "utilization_gpu_pct": busy,
                    "utilization_memory_pct": str(round(100 * used / total, 2)),
                    "temperature_c": _number(row["temperature.gpu"]),
                    "power_draw_w": power,
                    "power_limit_w": _number(row["power.limit [W]"]),
                    "fan_speed_pct": "",
                    "processes": "[]",
                    "node": node,
                    "local_gpu_id": row["index"],
                    "gpu_uuid": row["uuid"],
                    "memory_access_busy_pct": _number(row["utilization.memory [%]"]),
                }
            )
            count += 1
            if busy:
                utilization.append(float(busy))
            if power:
                watts = float(power)
                powers.append(watts)
                if row["uuid"] in previous:
                    last_time, last_power = previous[row["uuid"]]
                    energy_ws += (timestamp - last_time) * (watts + last_power) / 2
                previous[row["uuid"]] = (timestamp, watts)
    if not count:
        msg = f"No GPU samples for {node} during [{start}, {end}]; check telemetry and clock synchronization"
        raise ValueError(msg)
    prefix = node.replace("-", "_")
    metrics[f"{prefix}_gpu_samples"] = count
    if powers:
        metrics[f"{prefix}_mean_gpu_power_w"] = sum(powers) / len(powers)
        metrics[f"{prefix}_measured_gpu_energy_kwh"] = energy_ws / 3_600_000
    if utilization:
        metrics[f"{prefix}_mean_gpu_utilization_pct"] = sum(utilization) / len(utilization)
    return samples, metrics


def collect_entry(entry: Path, telemetry: Path, model_nodes: dict[str, str]) -> int:
    metrics_path = entry / "metrics.json"
    metrics = cast("dict[str, float]", json.loads(metrics_path.read_text()))
    if not metrics.get("is_success"):
        return 0
    start, end = metrics["pipeline_started_unix_s"], metrics["pipeline_finished_unix_s"]
    nodes = list(dict.fromkeys(node for alias, node in model_nodes.items() if f"{alias}_requests" in metrics))
    samples: list[dict[str, str]] = []
    for node in nodes:
        rows, node_metrics = _read_node(telemetry, node, start, end)
        samples.extend(rows)
        metrics.update(node_metrics)
    if not samples:
        msg = f"No model telemetry matched {entry}"
        raise ValueError(msg)
    identities = {(row["node"], int(row["local_gpu_id"]), row["gpu_uuid"]) for row in samples}
    gpu_ids = {identity: str(index) for index, identity in enumerate(sorted(identities))}
    for row in samples:
        row["gpu_id"] = gpu_ids[(row["node"], int(row["local_gpu_id"]), row["gpu_uuid"])]
    destination = entry / "gpustats.csv"
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GPUStatsRecorder.HEADER + EXTRA_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(samples, key=lambda row: (row["timestamp_utc"], int(row["gpu_id"]))))
    temporary.replace(destination)
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metrics, indent=2) + "\n")
    temporary.replace(metrics_path)
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--qwen-node", required=True, help="Recorder's short hostname")
    parser.add_argument("--deepseek-node", required=True, help="Recorder's short hostname")
    args = parser.parse_args()
    nodes = {"qwen": args.qwen_node, "deepseek": args.deepseek_node}
    for path in sorted(args.session.glob("*/metrics.json")):
        count = collect_entry(path.parent, args.telemetry_dir, nodes)
        if count:
            print(f"{path.parent / 'gpustats.csv'}: {count} samples", flush=True)


if __name__ == "__main__":
    main()
