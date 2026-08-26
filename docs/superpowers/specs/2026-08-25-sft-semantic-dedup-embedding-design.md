# SFT Semantic Dedup Embedding Benchmark Design

## Goal

Add a Slurm-array-compatible semantic-dedup embedding benchmark for the SFT QA dataset, producing embeddings for `question`, `combined_question_answer`, and `original_text` with one shared session-level checkpoint and shared output dataset.

## Data flow

The benchmark runner invokes `01_embedding.py`. The script reads Parquet with `files_per_partition=1`, selects the three source text columns plus `int_id`, runs one vLLM embedding stage configured with three text/embedding mappings, and writes one Arrow-preserving Parquet output dataset under the configured shared output root. The pipeline calls `pipeline.run(executor, checkpoint_path=session_checkpoint_path)`, where the checkpoint path is the session directory rather than the entry directory.

Slurm-array filtering is delegated to Curator's existing `Pipeline.run` and `SlurmArrayConfig` behavior. All shards use the same checkpoint directory and output root; each shard writes only its assigned source partitions, so no shard-specific embedding directory is created.

## Interfaces

- `VLLMEmbeddingModelStage` gains an optional `embedding_fields` mapping of source text column to output embedding column. The existing single-field arguments remain backward compatible.
- The stage exposes a configured `num_workers()` value equal to `num_vllm_replicas_per_gpu * ray_utils.get_num_gpus()` (with the benchmark resolving the cluster GPU count after Ray is available). Each worker retains one GPU resource.
- `ParquetWriter.write_data` passes an existing `pyarrow.Table` directly to `pyarrow.parquet.write_table`, applying field selection with Arrow APIs; non-Arrow batches retain the existing pandas path.

## Naming and paths

The YAML entry uses the effective name `harrier_oss_270m_${SLURM_ARRAY_TASK_ID:-0}_${SLURM_JOB_ID:-local}`. Because YAML entry names are not shell-expanded by the runner, the launcher resolves the name before invoking `benchmarking/run.py` and passes it through the runner's session/entry selection mechanism. The first tuning session is `embedding-bench-harrier-270m-YYYY-MM-DD`.

Benchmark metadata and `gpustats.csv` remain under `/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/benchmarking/results`; embedding output is shared under `/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/benchmarking/output`.

## Validation

Focused CPU tests cover mapping validation, Arrow writer schema preservation, worker-count calculation, and name construction. Full imports/tests and benchmark smoke runs execute only on the existing worker allocation. Initial tuning uses 5–10% of the SFT data, starts at four vLLM replicas per GPU, keeps `files_per_partition=1`, and compares `gpustats.csv` power draw and throughput across attempts.

