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

import http
import json
import os
import pickle
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.backends.experimental.ray_actor_pool.executor import RayActorPoolExecutor
from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.serve import ModelConfig, ModelServer
from nemo_curator.core.utils import get_free_port
from nemo_curator.utils.file_utils import get_all_file_paths_and_size_under

_executor_map = {"ray_data": RayDataExecutor, "xenna": XennaExecutor, "ray_actors": RayActorPoolExecutor}


def setup_executor(executor_name: str) -> RayDataExecutor | XennaExecutor | RayActorPoolExecutor:
    """Setup the executor for the given name."""
    try:
        executor = _executor_map[executor_name]()
    except KeyError:
        msg = f"Executor {executor_name} not supported"
        raise ValueError(msg) from None
    return executor


def load_dataset_files(
    dataset_path: Path,
    dataset_size_gb: float | None = None,
    dataset_ratio: float | None = None,
    keep_extensions: str = "parquet",
) -> list[str]:
    """Load the dataset files at the given path and return a subset of the files whose combined size is approximately the given size in GB."""
    input_files = get_all_file_paths_and_size_under(
        dataset_path, recurse_subdirectories=True, keep_extensions=keep_extensions
    )
    if (not dataset_size_gb and not dataset_ratio) or (dataset_size_gb and dataset_ratio):
        msg = "Either dataset_size_gb or dataset_ratio must be provided, but not both"
        raise ValueError(msg)
    if dataset_size_gb:
        desired_size_bytes = (1024**3) * dataset_size_gb
    else:
        total_file_size_bytes = sum(size for _, size in input_files)
        desired_size_bytes = total_file_size_bytes * dataset_ratio

    total_size = 0
    subset_files = []
    for file, size in input_files:
        if size + total_size > desired_size_bytes:
            break
        else:
            subset_files.append(file)
            total_size += size

    return subset_files


_VLLM_PORT = 8000
_HEALTH_TIMEOUT_S = 300


@dataclass
class BenchmarkingInferenceServer:
    """Handle returned by ``start_inference_server``.

    Use ``endpoint`` and ``api_key`` to connect to the server.
    Call ``stop()`` when done (no-op for nvidia-nim).
    """

    endpoint: str
    api_key: str
    startup_s: float = 0.0
    _stop_fn: Any = field(default=None, repr=False)

    def stop(self) -> None:
        if self._stop_fn is not None:
            self._stop_fn()
            self._stop_fn = None


def _wait_for_model_ready(port: int = _VLLM_PORT, timeout_s: int = _HEALTH_TIMEOUT_S) -> None:
    """Poll /v1/models until the server is ready."""
    models_url = f"http://localhost:{port}/v1/models"
    for attempt in range(timeout_s):
        try:
            resp = urllib.request.urlopen(models_url, timeout=5)  # noqa: S310
            if resp.status == http.HTTPStatus.OK:
                logger.info(f"Model server ready after {attempt + 1}s")
                return
        except Exception:  # noqa: S110
            pass
        time.sleep(1)
    msg = f"Model server did not become ready within {timeout_s}s"
    raise TimeoutError(msg)


def _engine_kwargs_to_vllm_args(engine_kwargs: dict[str, Any]) -> list[str]:
    """Convert engine_kwargs dict to vLLM CLI arguments."""
    args = []
    for key, value in engine_kwargs.items():
        cli_key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(cli_key)
        elif isinstance(value, (dict, list)):
            args.extend([cli_key, json.dumps(value)])
        else:
            args.extend([cli_key, str(value)])
    return args


def start_inference_server(
    model_type: str,
    model_id: str,
    engine_kwargs: dict[str, Any] | None = None,
    autoscaling_config: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
) -> BenchmarkingInferenceServer:
    """Start an inference server and return a handle with endpoint, api_key, and startup time.

    Supports three model types:
      - ``ray-serve``: vLLM behind Ray Serve via ModelServer.
      - ``vllm-direct``: standalone vLLM OpenAI-compatible server as a subprocess.
      - ``nvidia-nim``: NVIDIA Build cloud API (no server started, just resolves endpoint).

    Args:
        log_dir: Directory to write vLLM server logs into.  Falls back to
            ``/tmp`` if not provided.

    Call ``handle.stop()`` when done to clean up (no-op for nvidia-nim).
    """
    engine_kwargs = engine_kwargs or {}
    autoscaling_config = autoscaling_config or {"min_replicas": 1, "max_replicas": 1}

    if model_type == "ray-serve":
        os.environ.setdefault("NVIDIA_API_KEY", "none")

        model_config = ModelConfig(
            model_identifier=model_id,
            deployment_config={"autoscaling_config": autoscaling_config},
            engine_kwargs=engine_kwargs,
        )
        server = ModelServer(models=[model_config])

        logger.info(
            f"Starting ModelServer: model={model_id}, engine_kwargs={engine_kwargs}, autoscaling={autoscaling_config}"
        )
        t0 = time.perf_counter()
        server.start()
        startup_s = time.perf_counter() - t0
        logger.info(f"ModelServer started in {startup_s:.2f}s at {server.endpoint}")

        return BenchmarkingInferenceServer(
            endpoint=server.endpoint, api_key="none", startup_s=startup_s, _stop_fn=server.stop
        )

    if model_type == "vllm-direct":
        os.environ.setdefault("NVIDIA_API_KEY", "none")
        port = get_free_port(_VLLM_PORT)

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_id,
            "--host",
            "localhost",
            "--port",
            str(port),
            "--disable-log-requests",
            *_engine_kwargs_to_vllm_args(engine_kwargs),
        ]
        # Strip RAY_ env vars so vLLM starts its own process group for TP.
        env = {k: v for k, v in os.environ.items() if not k.startswith("RAY_")}

        log_parent = Path(log_dir) if log_dir else Path("/tmp")  # noqa: S108
        log_parent.mkdir(parents=True, exist_ok=True)
        log_file = log_parent / "vllm_server.log"
        logger.info(f"Starting vLLM server on port {port}: {' '.join(cmd)}")
        logger.info(f"vLLM server log: {log_file}")
        log_fh = log_file.open("w")
        process = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True, env=env)  # noqa: S603

        t0 = time.perf_counter()
        _wait_for_model_ready(port)
        startup_s = time.perf_counter() - t0
        logger.info(f"vLLM server started in {startup_s:.2f}s on port {port}")

        def _stop_vllm() -> None:
            logger.info("Stopping vLLM server...")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            log_fh.close()

        return BenchmarkingInferenceServer(
            endpoint=f"http://localhost:{port}/v1",
            api_key="none",
            startup_s=startup_s,
            _stop_fn=_stop_vllm,
        )

    if model_type == "nvidia-nim":
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            msg = "NVIDIA_API_KEY must be set for nvidia-nim model type"
            raise OSError(msg)
        return BenchmarkingInferenceServer(endpoint="https://integrate.api.nvidia.com/v1", api_key=api_key)

    msg = f"Unknown model_type: {model_type}"
    raise ValueError(msg)


def write_benchmark_results(results: dict, output_path: str | Path) -> None:
    """Write benchmark results (params, metrics, tasks) to the appropriate files in the output directory.

    - Writes 'params.json' and 'metrics.json' (merging with existing file contents if present and updating values).
    - Writes 'tasks.pkl' as a pickle file if present in results.
    - The output directory is created if it does not exist.

    Typically used by benchmark scripts to persist results in the format expected by the benchmarking framework.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if "params" in results:
        params_path = output_path / "params.json"
        params_data = {}
        if params_path.exists():
            params_data = json.loads(params_path.read_text())
        params_data.update(results["params"])
        params_path.write_text(json.dumps(params_data, default=convert_paths_to_strings, indent=2))
    if "metrics" in results:
        metrics_path = output_path / "metrics.json"
        metrics_data = {}
        if metrics_path.exists():
            metrics_data = json.loads(metrics_path.read_text())
        metrics_data.update(results["metrics"])
        metrics_path.write_text(json.dumps(metrics_data, default=convert_paths_to_strings, indent=2))
    if "tasks" in results:
        (output_path / "tasks.pkl").write_bytes(pickle.dumps(results["tasks"]))


def convert_paths_to_strings(obj: object) -> object:
    """
    Convert Path objects to strings, support conversions in container types in a recursive manner.
    """
    if isinstance(obj, dict):
        retval = {convert_paths_to_strings(k): convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        retval = [convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, Path):
        retval = str(obj)
    else:
        retval = obj
    return retval
