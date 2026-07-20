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

import pytest

from benchmarking.semdedup.kmeans import estimate_kmeans_memory


def test_estimate_kmeans_memory_exposes_components() -> None:
    estimate = estimate_kmeans_memory(
        num_rows=1_000,
        embedding_dim=16,
        n_clusters=10,
        num_actors=2,
        fit_data_fraction=0.1,
        oversampling_factor=2,
        max_samples_per_batch=32,
        safety_factor=1.5,
    )

    assert estimate["fit_rows"] == 100
    assert estimate["fit_arrays_cluster_bytes"] == 12_800
    assert estimate["fit_arrays_per_actor_bytes"] == 6_400
    assert estimate["distance_batch_per_actor_bytes"] == 1_280
    assert estimate["candidate_centroids_per_actor_bytes"] == 10_240
    assert estimate["centroids_per_actor_bytes"] == 640
    assert estimate["lower_bound_per_actor_bytes"] == 18_560
    assert estimate["projected_per_actor_bytes"] == 27_840
    assert estimate["projected_cluster_bytes"] == 55_680


@pytest.mark.parametrize("fit_data_fraction", [0, 1, -0.1, 1.1])
def test_estimate_kmeans_memory_rejects_invalid_fraction(fit_data_fraction: float) -> None:
    with pytest.raises(ValueError, match="fit_data_fraction"):
        estimate_kmeans_memory(
            num_rows=1,
            embedding_dim=1,
            n_clusters=1,
            num_actors=1,
            fit_data_fraction=fit_data_fraction,
            oversampling_factor=2,
            max_samples_per_batch=1,
            safety_factor=1,
        )
