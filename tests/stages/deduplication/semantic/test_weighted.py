# modality: text
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

from contextlib import suppress

import numpy as np
import pytest
import torch

with suppress(ImportError):
    from nemo_curator.stages.deduplication.semantic.weighted import (
        WeightedPruningConfig,
        _gpu_decide_block,
        sequential_weighted_pruning,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [0, 1, 7, 1024, 4097])
@pytest.mark.parametrize("large_weights", [False, True])
def test_gpu_ordered_decisions_with_strided_tile(rows: int, large_weights: bool) -> None:
    rng = np.random.default_rng(17)
    weights = rng.integers(0, 4, size=(rows, rows + 9), dtype=np.int64)
    if large_weights:
        weights *= 2**29
    scores = weights.sum(axis=1)
    threshold = 10 if not large_weights else 2**31
    expected_scores = scores.copy()
    expected_removed = np.zeros(rows, dtype=bool)
    for i in range(rows):
        if expected_scores[i] >= threshold:
            expected_removed[i] = True
            expected_scores[i + 1 :] -= weights[i, i + 1 : rows]
    gpu_weights = torch.as_tensor(weights, device="cuda", dtype=torch.int32 if large_weights else torch.uint8)
    gpu_scores = torch.as_tensor(scores, device="cuda", dtype=torch.int64 if large_weights else torch.int32)
    actual = _gpu_decide_block(gpu_scores, gpu_weights[:, :rows], threshold)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected_removed)
    np.testing.assert_array_equal(gpu_scores.cpu().numpy(), expected_scores)


def forge_reference(x: torch.Tensor, config: WeightedPruningConfig) -> np.ndarray:
    """Dense independent reference: update the full incident graph after each deletion."""
    x = torch.nn.functional.normalize(x.float(), dim=1, eps=1e-8).to(x.dtype)
    similarities = (x @ x.T).cpu().float().numpy()
    weights = np.zeros(similarities.shape, dtype=np.int64)
    for threshold, weight in reversed(config.rules):
        weights[similarities > threshold] = weight
    np.fill_diagonal(weights, 0)
    removed = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        if weights[i].sum() >= config.drop_score:
            removed[i] = True
            weights[i, :] = 0
            weights[:, i] = 0
    return removed


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 7, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_weighted_matches_dense_reference(batch_size: int, dtype: torch.dtype, device: str) -> None:
    generator = torch.Generator(device="cuda").manual_seed(72)
    x = torch.randn(127, 32, generator=generator, device="cuda", dtype=dtype)
    x[20:40] = x[0]
    config = WeightedPruningConfig(rules=((0.98, 10), (0.5, 7), (0.1, 3)), shard_size=None)
    np.testing.assert_array_equal(
        sequential_weighted_pruning(x.to(device), config, batch_size), forge_reference(x, config)
    )


@pytest.mark.gpu
def test_shards_match_forge_membership() -> None:
    x = torch.ones(23, 4, device="cuda")
    config = WeightedPruningConfig(shard_size=8)
    expected = np.zeros(len(x), dtype=bool)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 17]))
    for indices in np.array_split(rng.permutation(len(x)), 3):
        indices.sort()
        expected[indices] = forge_reference(x[indices], config)
    actual = sequential_weighted_pruning(x, config, batch_size=3, cluster_id=17)
    np.testing.assert_array_equal(actual, expected)
    assert (~actual).sum() == 3


@pytest.mark.gpu
def test_weighted_boundaries_and_empty() -> None:
    config = WeightedPruningConfig(rules=((1.0, 10), (0.0, 3)), drop_score=4)
    x = torch.tensor([[1.0, 0], [1.0, 0], [0.0, 1]], device="cuda")
    assert not sequential_weighted_pruning(x, config, 2).any()
    assert not sequential_weighted_pruning(x[:1], config, 2).any()
    assert sequential_weighted_pruning(x[:0], config, 2).size == 0
    assert not sequential_weighted_pruning(x, WeightedPruningConfig(drop_score=2**100), 2).any()
    with pytest.raises(ValueError, match="finite"):
        sequential_weighted_pruning(x * float("nan"), config)


@pytest.mark.gpu
def test_removed_neighbors_stop_contributing() -> None:
    x = torch.ones(4, 3, device="cuda")
    np.testing.assert_array_equal(
        sequential_weighted_pruning(x, WeightedPruningConfig(), 2), [True, True, True, False]
    )


@pytest.mark.gpu
def test_large_weights_do_not_overflow_scores() -> None:
    x = torch.ones(5, 3, device="cuda")
    config = WeightedPruningConfig(rules=((0.9, 2**30),), drop_score=2**31, shard_size=None)
    np.testing.assert_array_equal(sequential_weighted_pruning(x, config, 2), forge_reference(x, config))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "kwargs",
    [
        {"rules": ()},
        {"rules": ((0.5, 3), (0.9, 10))},
        {"rules": ((0.5, -1),)},
        {"drop_score": 0},
        {"shard_size": 0},
        {"read_memory_bytes": 0},
        {"spill_embeddings": True, "shard_size": None},
    ],
)
def test_invalid_config(kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"weighted|drop_score|shard_size|read_memory_bytes"):
        WeightedPruningConfig(**kwargs)
