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

"""Compose math-18's stages on prepared Parquet input; no Spark/Bondi dependency.

Input: unique id, embeddings (float array), text, math_topic, ms/rd/cr (0..5).
Provide pre-dedup counts from the successfully quality-tagged input pool.
Start Ray before calling. output_path must be a new directory shared by workers.
This example uses local/shared paths; configure storage credentials on stages for S3.
"""

from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic.diversity import ClusterDiversityStage
from nemo_curator.stages.deduplication.semantic.kmeans import KMeansStage
from nemo_curator.stages.deduplication.semantic.pairwise_io import ClusterWiseFilePartitioningStage
from nemo_curator.stages.deduplication.semantic.sampling import (
    FusionSamplingStage,
    RepeatParquetStage,
    plan_sampling_budgets,
)
from nemo_curator.stages.deduplication.semantic.weighted import WeightedPruningConfig
from nemo_curator.stages.deduplication.semantic.workflow import SemanticDeduplicationWorkflow
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow

# Forge nl_web_18src_pipeline.py, mix_930 FULL Layer A recipe (not library defaults).
MATH18_TOPIC_MULTIPLIERS = {
    "algebra_number_theory": 2.5,
    "contest_problem_solving": 3.0,
    "applied_physics": 1.8,
    "calculus_analysis": 2.0,
    "probability_statistics": 2.0,
    "applied_math": 1.0,
    "geometry_topology": 1.8,
    "discrete_math_logic": 2.0,
    "computer_science_theory": 1.2,
    "general_mixed": 0.8,
    "linear_algebra": 0.8,
    "non_math_or_unclear": 1.0,
    "optimization": 1.0,
}


def run_math18(  # noqa: PLR0913
    input_path: str,
    output_path: str,
    pre_dedup_topic_counts: dict[str, int],
    coarse_clusters: int = 500,
    fine_clusters: int = 3000,
    sampling_shards: int = 128,
    target_epochs: float = 2.5,
    topic_multipliers: dict[str, float] | None = None,
    sampling_memory_limit: str | None = None,
    temperature: float = 2.0,
    diversity_weight: float = 1.0,
    quality_weights: tuple[float, float, float] = (0.45, 0.30, 0.25),
    gate_threshold: float = 3.0,
    max_repeat: int = 8,
    metadata_fields: list[str] | None = None,
    weighted_pruning: WeightedPruningConfig | None = None,
    repeat_batch_rows: int = 1024,
    spherical: bool = True,
) -> str:
    """Weighted dedup → removal → fine k-means/relocation → diversity → sampling → repeats.

    Input Parquet files must share a schema. By default preserve all survivor
    metadata except embeddings; metadata_fields restricts optional carried columns.
    Text, topic and quality labels are always retained.
    spherical=True (default) selects unit-center Lloyd updates for both clustering passes;
    initialization and convergence are not a reproduction of Forge's private build.
    """
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=False)
    executor = RayActorPoolExecutor()
    dedup = SemanticDeduplicationWorkflow(
        input_path=input_path,
        output_path=str(root / "coarse"),
        n_clusters=coarse_clusters,
        weighted_pruning=weighted_pruning if weighted_pruning is not None else WeightedPruningConfig(),
        pairwise_compute_dtype="float32",
        kmeans_embedding_output_dtype="float32",
        spherical=spherical,
        init="random" if spherical else "k-means||",
    )
    dedup.run(kmeans_executor=executor, pairwise_executor=executor)
    survivors = str(root / "survivors")
    TextDuplicatesRemovalWorkflow(
        input_path=input_path,
        ids_to_remove_path=dedup.duplicates_output_path,
        output_path=survivors,
        id_field="id",
    ).run(executor=executor)

    fields = ds.dataset(survivors, format="parquet").schema.names if metadata_fields is None else metadata_fields
    metadata = list(
        dict.fromkeys(
            ["text", "math_topic", "ms", "rd", "cr", *(name for name in fields if name not in {"id", "embeddings"})]
        )
    )

    fine = Pipeline(name="fine_clustering")
    fine.add_stage(
        KMeansStage(
            n_clusters=fine_clusters,
            id_field="id",
            embedding_field="embeddings",
            input_path=survivors,
            output_path=str(root / "fine"),
            metadata_fields=metadata,
            cache_path=str(root / "centroids"),
            write_embeddings=False,
            spherical=spherical,
            init="random" if spherical else "k-means||",
        )
    )
    fine.run(executor)

    diversity = Pipeline(name="cluster_diversity")
    diversity.add_stage(ClusterWiseFilePartitioningStage(str(root / "fine")))
    diversity.add_stage(
        ClusterDiversityStage(
            centroids_path=str(root / "centroids/kmeans_centroids.npy"),
            output_path=str(root / "diversity"),
            metadata_fields=metadata,
            topic_field="math_topic",
            sampling_shards=sampling_shards,
        )
    )
    tasks = diversity.run(executor)
    group_counts = Counter()
    for task in tasks:
        group_counts.update(task._metadata["sampling_group_counts"])
    multipliers = MATH18_TOPIC_MULTIPLIERS if topic_multipliers is None else topic_multipliers
    budgets = plan_sampling_budgets(dict(group_counts), pre_dedup_topic_counts, target_epochs, multipliers, max_repeat)
    fusion_kwargs = {
        "temperature": temperature,
        "diversity_weight": diversity_weight,
        "quality_weights": quality_weights,
        "gate_threshold": gate_threshold,
        "max_repeat": max_repeat,
    }
    if sampling_memory_limit is not None:
        from nemo_curator.stages.deduplication.semantic.sampling_disk import sample_fusion_disk

        sampled_tasks = sample_fusion_disk(
            [path for task in tasks for path in task.data],
            str(root / "sample_counts"),
            budgets,
            scratch_path=str(root / "sampling_scratch"),
            memory_limit=sampling_memory_limit,
            **fusion_kwargs,
        )
    else:
        sampling = Pipeline(name="fusion_sampling")
        sampling.add_stage(
            FusionSamplingStage(
                budgets=budgets,
                output_path=str(root / "sample_counts"),
                total_nparts=max(1, len(budgets)),
                rmm_pool_size=None,
                **fusion_kwargs,
            )
        )
        sampled_tasks = sampling.run(executor, initial_tasks=tasks)
    final_path = str(root / "sampled")
    if not sampled_tasks:
        Path(final_path).mkdir()
        return final_path

    repeats = Pipeline(name="repeat_documents")
    repeats.add_stage(
        FilePartitioningStage(
            file_paths=str(root / "sample_counts"), files_per_partition=1, file_extensions=[".parquet"]
        )
    )
    repeats.add_stage(
        RepeatParquetStage(output_path=final_path, max_repeat=max_repeat, input_batch_rows=repeat_batch_rows)
    )
    repeats.run(executor)
    return final_path
