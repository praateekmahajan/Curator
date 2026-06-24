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

import csv
import json
import socket
import threading
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, ClassVar, Self

from loguru import logger

if TYPE_CHECKING:
    from types import ModuleType


class GPUStatsRecorder:
    """Background thread that polls all GPUs via gpustat and appends rows to a CSV.

    Polls every ``interval_s`` seconds and writes one row per (timestamp, GPU)
    pair to ``output_path``. Designed to wrap a benchmark subprocess:

        with GPUStatsRecorder(session_entry_path / "gpustats.csv", interval_s=1.0):
            run_benchmark_subprocess(...)

    The recorder polls all visible GPUs via NVML, independent of
    ``CUDA_VISIBLE_DEVICES`` — useful for verifying that workloads honor the
    visible-device mask.

    Failures during a single poll iteration are logged at WARNING and the
    thread keeps running; polling is best-effort and must not crash the
    benchmark.
    """

    HEADER: ClassVar[list[str]] = [
        "timestamp_utc",
        "gpu_id",
        "utilization_gpu_pct",
        "utilization_memory_pct",
        "temperature_c",
        "power_draw_w",
        "power_limit_w",
        "fan_speed_pct",
        "processes",
    ]

    def __init__(self, output_path: Path, interval_s: float = 1.0) -> None:
        self.output_path = Path(output_path)
        self.interval_s = float(interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._csv_file: IO[str] | None = None
        self._csv_writer: csv._writer | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            msg = "GPUStatsRecorder already started"
            raise RuntimeError(msg)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.output_path.open("w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.HEADER)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="gpu-stats-recorder",
            daemon=True,
        )
        self._thread.start()
        logger.debug(f"GPUStatsRecorder started: {self.output_path} (interval={self.interval_s}s)")

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=10.0)
        self._thread = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        logger.debug(f"GPUStatsRecorder stopped: {self.output_path}")

    def _poll_loop(self) -> None:
        # Import lazily so importing this module never requires gpustat at module load time.
        import gpustat

        while not self._stop_event.is_set():
            try:
                self._poll_once(gpustat)
            except Exception as e:
                # Best-effort poller — must not crash the benchmark.
                logger.warning(f"GPUStatsRecorder poll failed: {e}")
            # Wait returns True if the event was set during the wait, allowing prompt shutdown.
            self._stop_event.wait(self.interval_s)

    def _poll_once(self, gpustat_mod: ModuleType) -> None:
        query = gpustat_mod.new_query()
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for gpu in query:
            mem_total = gpu.memory_total or 0
            mem_pct = (gpu.memory_used / mem_total * 100.0) if mem_total else 0.0
            procs = [{k: p.get(k) for k in ("pid", "username", "command", "gpu_memory_usage")} for p in gpu.processes]
            # power_draw / power_limit / fan_speed are part of the same NVML query
            # gpustat already issued — reading them is free. Datacenter SKUs often
            # return None for fan_speed (no controllable per-card fan); render
            # None as empty string for CSV cleanliness.
            self._csv_writer.writerow(
                [
                    ts,
                    gpu.index,
                    gpu.utilization,
                    round(mem_pct, 2),
                    gpu.temperature,
                    "" if gpu.power_draw is None else round(gpu.power_draw, 1),
                    "" if gpu.power_limit is None else int(gpu.power_limit),
                    "" if gpu.fan_speed is None else int(gpu.fan_speed),
                    json.dumps(procs, separators=(",", ":")),
                ]
            )
        if self._csv_file is not None:
            self._csv_file.flush()


def _gpu_stats_actor_class() -> Any:
    import ray

    @ray.remote(num_cpus=0)
    class GPUStatsActor:
        def __init__(self, output_path: str, interval_s: float, node_index: int, gpu_id_stride: int) -> None:
            self.output_path = Path(output_path)
            self.interval_s = interval_s
            self.node_index = node_index
            self.gpu_id_offset = node_index * gpu_id_stride
            self.hostname = socket.gethostname()
            self.node_id = ray.get_runtime_context().get_node_id()
            self._recorder: GPUStatsRecorder | None = None

        def start(self) -> dict[str, str]:
            if self._recorder is None:
                self._recorder = GPUStatsRecorder(self.output_path, interval_s=self.interval_s)
                self._recorder.start()
            return self.metadata()

        def stop(self) -> dict[str, str]:
            if self._recorder is not None:
                self._recorder.stop()
                self._recorder = None
            return self.metadata()

        def metadata(self) -> dict[str, str]:
            return {
                "hostname": self.hostname,
                "node_id": self.node_id,
                "node_index": str(self.node_index),
                "gpu_id_offset": str(self.gpu_id_offset),
                "output_path": str(self.output_path),
            }

    return GPUStatsActor


class RayGPUStatsRecorder:
    """Launch one lightweight GPU stats recorder actor on every alive Ray node."""

    def __init__(
        self,
        output_path: Path,
        per_node_dir: Path,
        interval_s: float = 1.0,
        gpu_id_stride: int = 8,
    ) -> None:
        self.output_path = Path(output_path)
        self.per_node_dir = Path(per_node_dir)
        self.interval_s = float(interval_s)
        self.gpu_id_stride = int(gpu_id_stride)
        self._actors: list[Any] = []
        self._metadata: list[dict[str, str]] = []
        self._ray_initialized_here = False

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()

    def start(self) -> None:
        try:
            import ray
            from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
                self._ray_initialized_here = True

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.per_node_dir.mkdir(parents=True, exist_ok=True)
            actor_cls = _gpu_stats_actor_class()

            alive_nodes = sorted(
                [node for node in ray.nodes() if node.get("Alive")],
                key=lambda node: (
                    node.get("NodeManagerHostname") or node.get("NodeManagerAddress") or "",
                    node["NodeID"],
                ),
            )
            for node_index, node in enumerate(alive_nodes):
                node_id = node["NodeID"]
                hostname = node.get("NodeManagerHostname") or node.get("NodeManagerAddress") or f"node{node_index}"
                node_output_path = self.per_node_dir / f"{hostname}_{node_id[:8]}.csv"
                actor = actor_cls.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                ).remote(str(node_output_path), self.interval_s, node_index, self.gpu_id_stride)
                self._actors.append(actor)

            if not self._actors:
                logger.warning("RayGPUStatsRecorder found no alive Ray nodes; GPU stats recording disabled.")
                return

            self._metadata = ray.get([actor.start.remote() for actor in self._actors], timeout=30)
            logger.debug(
                f"RayGPUStatsRecorder started {len(self._actors)} node recorder(s): {self.per_node_dir} "
                f"(interval={self.interval_s}s)"
            )
        except Exception as e:
            logger.warning(f"RayGPUStatsRecorder failed to start; GPU stats recording disabled: {e}")
            self._actors = []
            self._metadata = []

    def stop(self) -> None:
        try:
            import ray

            if self._actors and ray.is_initialized():
                try:
                    stopped_metadata = ray.get([actor.stop.remote() for actor in self._actors], timeout=30)
                    if stopped_metadata:
                        self._metadata = stopped_metadata
                except Exception as e:
                    logger.warning(f"RayGPUStatsRecorder failed to stop one or more actors cleanly: {e}")
            self._merge_per_node_csvs()
        except Exception as e:
            logger.warning(f"RayGPUStatsRecorder cleanup failed: {e}")
        finally:
            self._actors = []
            if self._ray_initialized_here:
                try:
                    import ray

                    ray.shutdown()
                except Exception:
                    logger.debug("RayGPUStatsRecorder could not shut down Ray cleanly.")
                self._ray_initialized_here = False

    def _merge_per_node_csvs(self) -> None:
        rows: list[dict[str, Any]] = []
        for metadata in self._metadata:
            node_path = Path(metadata["output_path"])
            if not node_path.exists():
                logger.warning(f"RayGPUStatsRecorder missing per-node stats file: {node_path}")
                continue
            gpu_id_offset = int(metadata["gpu_id_offset"])
            with node_path.open(newline="") as node_file:
                reader = csv.DictReader(node_file)
                for row in reader:
                    try:
                        row["gpu_id"] = str(int(row["gpu_id"]) + gpu_id_offset)
                    except (KeyError, TypeError, ValueError):
                        logger.warning(f"RayGPUStatsRecorder could not offset gpu_id in {node_path}: {row}")
                        continue
                    rows.append(row)

        rows.sort(key=lambda row: (row.get("timestamp_utc", ""), int(row.get("gpu_id", 0))))
        with self.output_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=GPUStatsRecorder.HEADER)
            writer.writeheader()
            writer.writerows(rows)
        logger.debug(f"RayGPUStatsRecorder merged {len(rows)} rows into {self.output_path}")
