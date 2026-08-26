# SFT Semantic Dedup Embedding Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-column vLLM SFT embedding benchmark with Arrow-native output, shared session checkpointing, and Slurm-array-safe naming/output.

**Architecture:** Extend the existing vLLM text stage without breaking its single-column API. Add a focused benchmark script/config in the existing SFT worktree; use Curator's native Slurm-array filtering and session-level checkpointing, and write shard outputs into one shared output root.

**Tech Stack:** Python, PyArrow, Parquet, Ray, vLLM, Curator benchmarking runner, Slurm.

**Spec:** `docs/superpowers/specs/2026-08-25-sft-semantic-dedup-embedding-design.md`

## Global Constraints

- Work only in the existing `benchmarking/sft_fuzzy_dedup/worktree` checkout.
- Use `benchmarking/run.py` for benchmark launches.
- Use `files_per_partition=1`.
- Use session-level checkpointing, not entry-level checkpointing.
- Keep benchmark results under `/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/benchmarking/results` and shared embeddings under `/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/benchmarking/output`.
- Do not run project imports, tests, or workloads on the login node; execute them through the existing worker allocation.
- Commit each coherent change incrementally.

---

### Task 1: Add failing tests for multi-column vLLM configuration

**Files:**
- Create: `tests/stages/text/embedders/test_vllm_multi_column.py`
- Modify: none

**Interfaces:**
- Tests target `VLLMEmbeddingModelStage(embedding_fields=...)`, `outputs()`, and `num_workers()`.

- [ ] Write tests asserting three mappings produce three embedding output fields and reject duplicate output names or missing source fields.
- [ ] Write a test asserting `num_workers()` returns the configured replica count multiplied by the discovered GPU count through the chosen Ray utility seam.
- [ ] Run the focused test file on the worker allocation and confirm the failures are due to missing behavior.
- [ ] Commit the red tests: `test: specify multi-column vllm embedding behavior`.

### Task 2: Implement multi-column vLLM embedding and worker sizing

**Files:**
- Modify: `nemo_curator/stages/text/embedders/vllm.py`
- Modify: `nemo_curator/utils/ray_utils.py` only if the existing GPU-count helper is absent
- Test: `tests/stages/text/embedders/test_vllm_multi_column.py`

**Interfaces:**
- `VLLMEmbeddingModelStage(..., embedding_fields: dict[str, str] | None = None, num_workers: int | None = None)` preserves `text_field`/`embedding_field` behavior when the mapping is omitted.
- The stage processes each configured source column and adds its corresponding embedding column to the same output table.

- [ ] Implement mapping normalization and validation while preserving the legacy single-field path.
- [ ] Implement Arrow-table processing so each source column is embedded independently and output columns are appended without converting the whole batch to pandas.
- [ ] Implement the configured worker-count override and retain one GPU resource per vLLM actor.
- [ ] Run the focused tests on the worker and confirm green output.
- [ ] Commit: `feat: support multi-column vllm embeddings`.

### Task 3: Preserve Arrow-native Parquet schemas

**Files:**
- Create or modify: `tests/stages/text/io/writer/test_parquet.py`
- Modify: `nemo_curator/stages/text/io/writer/parquet.py`

**Interfaces:**
- `ParquetWriter.write_data` accepts a `DocumentBatch` backed by `pyarrow.Table`; selected fields retain Arrow types, including list-valued float embeddings.

- [ ] Write a failing test with an Arrow table containing an embedding list column and assert the written Parquet Arrow schema matches.
- [ ] Run only that test on the worker and confirm it fails because the current path converts through pandas.
- [ ] Add the Arrow path using `pyarrow.parquet.write_table`; keep existing pandas fallback and `write_kwargs` semantics.
- [ ] Run the focused writer tests on the worker and confirm green output.
- [ ] Commit: `fix: preserve arrow schemas in parquet writer`.

### Task 4: Add the benchmark script and Slurm-array config

**Files:**
- Create: `benchmarking/sft_semantic_deudp/__init__.py`
- Create: `benchmarking/sft_semantic_deudp/01_embedding.py`
- Create: `benchmarking/sft_semantic_deudp/semantic_embedding.yaml`
- Create: `benchmarking/sft_semantic_deudp/README.md`

**Interfaces:**
- Script CLI includes `--num-vllm-replicas-per-gpu`, `--dataset-size-gb` or ratio control, `--session-checkpoint-path`, shared `--output-path`, and the three source-column mappings.
- Script calls `Pipeline.run(executor_obj, checkpoint_path=session_checkpoint_path)`.

- [ ] Add tests for name construction and argument defaults before implementation.
- [ ] Implement the script by adapting the existing embedding benchmark, loading only `int_id` and the three text fields, selecting `files_per_partition=1`, and logging resolved GPU/worker counts.
- [ ] Resolve the effective Slurm name from `SLURM_ARRAY_TASK_ID` and `SLURM_JOB_ID`; keep the first tuning session format `embedding-bench-harrier-270m-%Y-%m-%d`.
- [ ] Configure the YAML entry with the requested `harrier_oss_270m_...` naming, shared output root, session checkpoint path, Slurm-array environment settings, and 5–10% initial subset.
- [ ] Run YAML/config parsing and script help on the worker without launching the full dataset.
- [ ] Commit: `feat: add sft semantic embedding benchmark`.

### Task 5: Worker smoke run and tuning evidence

**Files:**
- Modify: `benchmarking/sft_semantic_deudp/README.md` with exact launch and cleanup commands
- Create: task-root logs/exports only, outside the repository

- [ ] Inspect active jobs, allocation state, exact scripts, and output collisions before launch.
- [ ] Run one 5–10% smoke/tuning attempt with four replicas per GPU through `benchmarking/run.py` on the existing allocation.
- [ ] Verify interpreter, Ray initialization, vLLM initialization, first output, checkpoint metadata, shared output location, and `gpustats.csv`.
- [ ] If the entry fails, confirm no live process uses the entry result directory, remove only that failed entry result, and retry with one diagnosed change.
- [ ] Compare power draw and throughput, record findings in the README or task-root export, and commit any code/config fix separately.

### Task 6: Final verification and handoff

- [ ] Run focused tests and static checks on the worker allocation.
- [ ] Inspect `git diff`, `git status`, and commit history.
- [ ] Verify every requirement against the spec, including shared session checkpointing and no shard-specific embedding output directories.
- [ ] Commit any final documentation-only change: `docs: document sft semantic embedding benchmark`.

