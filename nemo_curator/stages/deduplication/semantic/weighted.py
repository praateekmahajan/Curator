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

"""Order-dependent weighted pruning, with bounded pairwise workspace."""

import math
from dataclasses import dataclass
from functools import lru_cache

import cupy as cp
import numpy as np
import torch


@lru_cache(maxsize=4)
def _decision_kernel(weight_type: str, score_type: str) -> cp.RawKernel:
    return cp.RawKernel(
        f"""
        extern "C" __global__ void decide(
            {score_type}* scores, const {weight_type}* weights, bool* removed,
            long long rows, long long stride, long long threshold) {{
            __shared__ bool drop;
            for (long long i = 0; i < rows; ++i) {{
                if (threadIdx.x == 0) {{
                    drop = scores[i] >= threshold;
                    removed[i] = drop;
                }}
                __syncthreads();
                if (drop)
                    for (long long j = i + 1 + threadIdx.x; j < rows; j += blockDim.x)
                        scores[j] -= weights[i * stride + j];
                __syncthreads();
            }}
        }}
        """,
        "decide",
    )


def _gpu_decide_block(scores: torch.Tensor, weights: torch.Tensor, threshold: int) -> torch.Tensor:
    """Exact ordered decisions in one CUDA block; update local scores in place.

    Each decision is serial, but its later-neighbor updates are parallel. This
    avoids a B² GPU-to-CPU copy and Python iteration without changing semantics.
    Accepts a strided diagonal tile of the existing B-by-shard weight buffer.
    """
    removed = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
    if threshold > torch.iinfo(scores.dtype).max:
        return removed
    if not len(scores):
        return removed
    kernel = _decision_kernel(
        "unsigned char" if weights.dtype == torch.uint8 else "int",
        "int" if scores.dtype == torch.int32 else "long long",
    )
    with (
        cp.cuda.Device(scores.device.index),
        cp.cuda.Stream.from_external(torch.cuda.current_stream(scores.device)),
    ):
        kernel(
            (1,),
            (256,),
            (
                cp.from_dlpack(scores),
                cp.from_dlpack(weights),
                cp.from_dlpack(removed),
                np.int64(len(scores)),
                np.int64(weights.stride(0)),
                np.int64(threshold),
            ),
        )
    return removed


@dataclass(frozen=True)
class WeightedPruningConfig:
    """Opt into Forge-style pruning instead of maximum-similarity thresholding.

    Rules are descending (strict cosine threshold, integer weight) bands.
    Each pair receives only the highest matching band's weight. Documents are
    visited in removal-priority order, not preservation-priority order.
    ``shard_size=None`` compares the entire cluster; otherwise seeded random
    shards bound work, deliberately excluding cross-shard comparisons.
    ``spill_embeddings`` stages ranked embeddings on local disk and transfers
    one shard at a time. Ranking metadata must still fit on the GPU. Scratch
    uses Python's temporary-file location unless ``scratch_path`` is supplied;
    the directory must exist and should be backed by SSD rather than tmpfs.
    ``read_memory_bytes`` bounds projected decoding up to Parquet page granularity.
    """

    rules: tuple[tuple[float, int], ...] = ((0.98, 10), (0.95, 7), (0.92, 3))
    drop_score: int = 10
    shard_size: int | None = 50_000
    seed: int = 0
    spill_embeddings: bool = False
    scratch_path: str | None = None
    read_memory_bytes: int = 256 * 1024**2

    def __post_init__(self) -> None:  # noqa: C901
        previous = math.inf
        if not self.rules:
            msg = "weighted rules must not be empty"
            raise ValueError(msg)
        for threshold, weight in self.rules:
            if not math.isfinite(threshold) or not -1 <= threshold <= 1 or threshold >= previous:
                msg = "weighted thresholds must be finite, in [-1, 1], and strictly descending"
                raise ValueError(msg)
            if isinstance(weight, bool) or not isinstance(weight, int) or not 0 < weight <= np.iinfo(np.int32).max:
                msg = "weighted weights must be positive int32 integers"
                raise ValueError(msg)
            previous = threshold
        for name, value in (
            ("drop_score", self.drop_score),
            ("shard_size", self.shard_size),
            ("read_memory_bytes", self.read_memory_bytes),
        ):
            if value is None and name == "shard_size":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                msg = f"{name} must be a positive integer"
                raise ValueError(msg)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            msg = "seed must be a non-negative integer"
            raise ValueError(msg)
        if not isinstance(self.spill_embeddings, bool):
            msg = "spill_embeddings must be a boolean"
            raise TypeError(msg)
        if self.spill_embeddings and self.shard_size is None:
            msg = "spill_embeddings requires a finite shard_size to bound GPU embedding memory"
            raise ValueError(msg)


def _weight_block(x: torch.Tensor, start: int, end: int, config: WeightedPruningConfig) -> torch.Tensor:
    similarities = x[start:end] @ x.T
    weight_dtype = torch.uint8 if max(weight for _, weight in config.rules) <= np.iinfo(np.uint8).max else torch.int32
    weights = torch.zeros(similarities.shape, dtype=weight_dtype, device=x.device)
    for threshold, weight in reversed(config.rules):
        weights.masked_fill_(similarities > threshold, weight)
    weights[:, start:end].fill_diagonal_(0)
    return weights


def _prune_shard(x: torch.Tensor, config: WeightedPruningConfig, batch_size: int) -> np.ndarray:
    if not torch.isfinite(x).all().item():
        msg = "embeddings must be finite"
        raise ValueError(msg)
    # Normalize in FP32 even when storage/compute is FP16, as in Forge's worker.
    dtype = x.dtype
    x = torch.nn.functional.normalize(x.float(), dim=1, eps=1e-8).to(dtype)
    n = len(x)
    max_score = max(weight for _, weight in config.rules) * n
    if config.drop_score > max_score:
        return np.zeros(n, dtype=bool)
    score_dtype = torch.int32 if max_score <= np.iinfo(np.int32).max else torch.int64
    scores = torch.empty(n, dtype=score_dtype, device=x.device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        weights = _weight_block(x, start, end, config)
        scores[start:end] = weights.sum(dim=1, dtype=score_dtype)
        del weights

    removed = torch.zeros(n, dtype=torch.bool, device=x.device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Positive weights make scores monotone decreasing: this whole block
        # cannot delete anything, so its second similarity pass is unnecessary.
        if not (scores[start:end] >= config.drop_score).any().item():
            continue
        weights = _weight_block(x, start, end, config)
        removed[start:end] = _gpu_decide_block(scores[start:end], weights[:, start:end], config.drop_score)
        # Propagate this block's deletions to all later rows in one GPU reduction.
        keep = ~removed[start:end]
        weights[keep] = 0
        scores[end:] -= weights[:, end:].sum(dim=0, dtype=score_dtype)
        del weights
    return removed.cpu().numpy()


def sequential_weighted_pruning(
    embeddings: torch.Tensor,
    config: WeightedPruningConfig,
    batch_size: int = 1024,
    cluster_id: int = 0,
    *,
    compute_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Return a removal mask, preserving Forge's sequential decisions per shard.

    Two similarity passes trade recomputation for O(N*B + N*D) GPU memory and
    O(N) host mask/shard-index memory, instead of retaining a dense N² matrix or
    an unbounded edge list. Ordered decisions and score updates stay on GPU;
    one scalar check per block skips tiles that cannot remove anything.
    CPU tensors (including disk-backed arrays) transfer only one shard at a time.
    Precision, row order, rules, and shard membership all affect the result.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        msg = "batch_size must be a positive integer"
        raise ValueError(msg)
    if embeddings.ndim != 2 or embeddings.dtype not in {torch.float16, torch.float32}:  # noqa: PLR2004
        msg = "embeddings must be a two-dimensional FP16/FP32 tensor"
        raise ValueError(msg)
    n = len(embeddings)
    device = embeddings.device if embeddings.is_cuda else "cuda"
    dtype = embeddings.dtype if compute_dtype is None else compute_dtype
    if dtype not in {torch.float16, torch.float32}:
        msg = "compute_dtype must be FP16 or FP32"
        raise ValueError(msg)
    if config.shard_size is None or n <= config.shard_size:
        return _prune_shard(embeddings.to(device=device, dtype=dtype), config, batch_size)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, cluster_id]))
    removed = np.zeros(n, dtype=bool)
    for indices in np.array_split(rng.permutation(n), math.ceil(n / config.shard_size)):
        indices.sort()
        shard = embeddings[torch.as_tensor(indices, device=embeddings.device)].to(device=device, dtype=dtype)
        removed[indices] = _prune_shard(shard, config, batch_size)
        del shard
    return removed
