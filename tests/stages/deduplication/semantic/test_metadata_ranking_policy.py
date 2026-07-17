# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import suppress

import pytest
import torch

with suppress(ImportError):
    import cudf

with suppress(ImportError):
    from nemo_curator.stages.deduplication.semantic.pairwise import pairwise_cosine_similarity_batched
    from nemo_curator.stages.deduplication.semantic.ranking import RankingStrategy


@pytest.mark.gpu
def test_source_quality_recency_policy() -> None:
    rows = cudf.DataFrame(
        {
            "id": [10, 20, 30, 40, 50],
            "source_priority": [0, 0, 0, 0, 1],
            "quality_rank": [12, 12, 13, 17, 0],
            "recency_rank": [0, 1, 0, 2, 0],
        }
    )
    strategy = RankingStrategy.metadata_based(
        metadata_cols=["source_priority", "quality_rank", "recency_rank", "id"],
        ascending=[False, False, False, True],
    )

    ranked = strategy.rank_cluster(rows)

    # Source policy wins globally; then quality; then recency within quality.
    assert ranked["id"].to_arrow().to_pylist() == [50, 40, 30, 20, 10]


@pytest.mark.gpu
def test_cosine_score_points_to_best_earlier_ranked_row() -> None:
    rows = cudf.DataFrame(
        {
            "id": [30, 40, 50],
            "source_priority": [0, 0, 1],
            "quality_rank": [13, 17, 0],
            "recency_rank": [0, 2, 0],
        }
    )
    embeddings_by_id = {
        30: [0.0, 1.0],
        40: [0.8, 0.6],
        50: [1.0, 0.0],
    }
    strategy = RankingStrategy.metadata_based(
        metadata_cols=["source_priority", "quality_rank", "recency_rank", "id"],
        ascending=[False, False, False, True],
    )
    ranked = strategy.rank_cluster(rows)
    ranked_ids = ranked["id"].to_arrow().to_pylist()
    ranked_embeddings = torch.tensor([embeddings_by_id[row_id] for row_id in ranked_ids])

    scores, parent_indices = pairwise_cosine_similarity_batched(ranked_embeddings)
    parent_ids = [ranked_ids[index] for index in parent_indices.get().tolist()]

    assert ranked_ids == [50, 40, 30]
    assert parent_ids == [50, 50, 40]
    assert scores.get().tolist() == pytest.approx([0.0, 0.8, 0.6])
