# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare CPU/GPU ordered decisions on real-vector similarity tiles.

Measures the decision block only, including the CPU implementation's transfers.
This is a diagnostic, not an end-to-end throughput claim. Run on an idle GPU.
"""

# Benchmark closures are invoked before advancing the surrounding loop.
# ruff: noqa: B023

import argparse
import json
import statistics
import time
from functools import lru_cache

import cudf
import cupy as cp
import numpy as np
import torch

from nemo_curator.stages.deduplication.semantic.weighted import (
    WeightedPruningConfig,
    _gpu_decide_block,
    _weight_block,
)


@lru_cache(maxsize=32)
def _band_kernel(rules: tuple[tuple[float, int], ...]) -> cp.ElementwiseKernel:
    expression = "(W)0"
    for threshold, weight in reversed(rules):
        expression = f"(similarities > (T)({float(threshold)!r}) ? (W){weight} : {expression})"
    return cp.ElementwiseKernel("T similarities", "W weights", f"weights = {expression};", "semantic_similarity_bands")


def _fused_weight_block(x: torch.Tensor, start: int, end: int, config: WeightedPruningConfig) -> torch.Tensor:
    """Benchmark-only candidate: inconsistent small-tile gains do not justify a library branch."""
    similarities = x[start:end] @ x.T
    with cp.cuda.Device(x.device.index), cp.cuda.Stream.from_external(torch.cuda.current_stream(x.device)):
        values = cp.from_dlpack(similarities)
        dtype = cp.uint8 if max(weight for _, weight in config.rules) <= np.iinfo(np.uint8).max else cp.int32
        weights = cp.empty(values.shape, dtype=dtype)
        _band_kernel(config.rules)(values, weights)
        result = torch.from_dlpack(weights)
    result[:, start:end].fill_diagonal_(0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    frame = cudf.read_parquet(args.input, columns=["embedding"]).iloc[:50_000]
    x = torch.as_tensor(frame.embedding.list.leaves.values.reshape(len(frame), -1), device="cuda")
    x = torch.nn.functional.normalize(x, dim=1)
    for rules in (((0.98, 10), (0.95, 7), (0.92, 3)), ((0.5, 10), (0.2, 3))):
        config = WeightedPruningConfig(rules=rules)
        for requested_rows in (256, 1024, 4096):
            rows = min(requested_rows, len(x))
            weights = _weight_block(x, 0, rows, config)
            original_scores = weights.sum(dim=1, dtype=torch.int32)

            def cpu_decisions() -> torch.Tensor:
                scores = original_scores.cpu().numpy()
                local = weights[:, :rows].cpu().numpy()
                removed = np.zeros(rows, dtype=bool)
                for i in range(rows):
                    if scores[i] >= config.drop_score:
                        removed[i] = True
                        scores[i + 1 :] -= local[i, i + 1 :]
                return torch.as_tensor(removed, device="cuda")

            def gpu_decisions() -> torch.Tensor:
                return _gpu_decide_block(original_scores.clone(), weights[:, :rows], config.drop_score)

            torch.testing.assert_close(cpu_decisions(), gpu_decisions())
            torch.testing.assert_close(weights, _fused_weight_block(x, 0, rows, config))
            result = {"rows": rows, "shard_rows": len(x), "rules": rules, "removed": int(gpu_decisions().sum())}
            for name, operation in (
                ("cpu", cpu_decisions),
                ("gpu", gpu_decisions),
                ("bands_reference", lambda: _weight_block(x, 0, rows, config)),
                ("bands_fused", lambda: _fused_weight_block(x, 0, rows, config)),
            ):
                times = []
                warmup = 3
                for iteration in range(warmup + 20):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    operation()
                    torch.cuda.synchronize()
                    if iteration >= warmup:
                        times.append(time.perf_counter() - start)
                result[f"{name}_median_seconds"] = statistics.median(times)
            result["speedup"] = result["cpu_median_seconds"] / result["gpu_median_seconds"]
            result["bands_speedup"] = result["bands_reference_median_seconds"] / result["bands_fused_median_seconds"]
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
