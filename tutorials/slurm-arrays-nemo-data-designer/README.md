# SLURM Arrays For NeMo Data Designer

This tutorial-local prototype runs a seeded NeMo Data Designer pipeline as a
SLURM array. Each array element owns one deterministic shard of the input files,
can reserve multiple nodes, starts one or more model endpoints through Curator's
`InferenceServer`, and writes an independent completion marker for retry.

## What It Demonstrates

| Requirement | Tutorial implementation |
| --- | --- |
| Horizontal fanout across `H` array jobs | `SlurmArrayFilePartitioningStage` assigns file groups by stable hash modulo `CURATOR_NUM_SHARDS`. |
| Vertical scale per array job | `submit_array.sh` sets `NODES`, `GPUS_PER_NODE`, and starts Ray through `SlurmRayClient`. |
| Multiple model endpoints per shard | `MODELS_JSON` or `MODELS_JSON_FILE` expands into multiple Curator `InferenceServer` model configs. |
| Resumability | Output writers create `<output>.done` sidecars; the partitioner skips only sidecar-complete outputs. |
| Retry only unfinished shards | `submit_array.sh retry-missing` resubmits shards that do not have `_SUCCESS/shard_*.json`. |
| Empty shard fast path | The pipeline computes the partition plan before Ray or model startup and exits immediately for empty or complete shards. |
| Per-shard metrics | `_SUCCESS/shard_*.json` includes partition counts, served models, output files, bytes, timings, and task performance data. |

The helpers intentionally live under `tutorials/slurm-arrays-nemo-data-designer`
so the workflow can be reviewed without changing Curator core APIs.

## Files

- `slurm_array_support.py`: `SlurmArrayFilePartitioningStage`, sidecar-aware writers, and `SlurmArrayPipeline`.
- `seeded_sdg_pipeline.py`: seeded JSONL or Parquet Data Designer pipeline with preflight partition planning.
- `submit_array.sh`: submit, status, retry, and worker entrypoint for SLURM arrays.
- `prepare_seed_dataset.py`: prepares sharded JSONL and Parquet seed datasets.
- `configs/gretel_medical_two_model.sdg.json`: Data Designer config for the Gretel symptoms dataset.
- `configs/cached_hf_gretel_models.json`: two cached local HF model specs for endpoint testing.
- `configs/wiki_summarizer_two_model.sdg.json`: Data Designer config for a Wikipedia summarization seed dataset.
- `configs/wiki_gpt_oss_models.json`: two-model GPT-OSS-style endpoint config.

## Dataset Layout

The Gretel seed dataset is prepared outside the tutorial tree:

```text
datasets/ndd_gretel_symptoms/
  jsonl/part_00000.jsonl
  parquet/part_00000.parquet
  manifest.json
```

Regenerate it from the workspace root:

```bash
uv run --with pandas --with pyarrow \
  python Curator/tutorials/slurm-arrays-nemo-data-designer/prepare_seed_dataset.py \
  --dataset ndd_gretel_symptoms \
  --rows-per-file 25 \
  --force
```

## Output Layout

Use a run root under `datasets/slurm-arrays-nemo-data-designer/runs`:

```text
datasets/slurm-arrays-nemo-data-designer/runs/<run-name>/
  data/*.jsonl
  data/*.jsonl.done
  _SUCCESS/shard_00000.json
  _logs/*.out
  _logs/*.err
```

`OUTPUT_LAYOUT=flat` writes all completed files under `data/`. Use
`OUTPUT_LAYOUT=by_shard` to write under `data/shard_00000/`,
`data/shard_00001/`, and so on.

## Submit A Single-Node Run

`MODEL=...` is a single-model shorthand. For multiple endpoints, use
`MODELS_JSON` or `MODELS_JSON_FILE`.

```bash
INPUT_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/praateekm/datasets/ndd_gretel_symptoms/jsonl \
OUTPUT_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/praateekm/datasets/slurm-arrays-nemo-data-designer/runs/ndd_gretel_v1_h1 \
DD_CONFIG=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/praateekm/Curator/tutorials/slurm-arrays-nemo-data-designer/configs/gretel_medical_two_model.sdg.json \
MODELS_JSON_FILE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/praateekm/Curator/tutorials/slurm-arrays-nemo-data-designer/configs/cached_hf_gretel_models.json \
INPUT_FORMAT=jsonl \
OUTPUT_FORMAT=jsonl \
ARRAY_SIZE=1 \
MAX_CONCURRENT=1 \
NODES=1 \
GPUS_PER_NODE=8 \
PARTITION=batch_short \
TIME_LIMIT=02:00:00 \
Curator/tutorials/slurm-arrays-nemo-data-designer/submit_array.sh submit
```

Recommended scale checks:

```bash
# H=1, V=1
ARRAY_SIZE=1 MAX_CONCURRENT=1 NODES=1 ...

# H=1, V=2
ARRAY_SIZE=1 MAX_CONCURRENT=1 NODES=2 ...

# H=2, V=1
ARRAY_SIZE=2 MAX_CONCURRENT=2 NODES=1 ...

# H=2, V=2
ARRAY_SIZE=2 MAX_CONCURRENT=2 NODES=2 ...
```

## Multi-Model Config

`MODELS_JSON_FILE` points at JSON shaped like:

```json
[
  {
    "model": "/path/to/model-a",
    "served_model_name": "model-a",
    "alias": "diagnosis_model",
    "tp": 1,
    "replicas": "auto"
  },
  {
    "model": "/path/to/model-b",
    "served_model_name": "model-b",
    "alias": "review_model",
    "tp": 1,
    "replicas": "auto"
  }
]
```

Aliases are used to retarget matching model names in the Data Designer config.
With `replicas=auto`, the pipeline divides available GPUs across the configured
models after accounting for tensor parallelism.

## Status And Retry

```bash
INPUT_PATH=... OUTPUT_PATH=... DD_CONFIG=... MODELS_JSON_FILE=... ARRAY_SIZE=2 \
  Curator/tutorials/slurm-arrays-nemo-data-designer/submit_array.sh status

INPUT_PATH=... OUTPUT_PATH=... DD_CONFIG=... MODELS_JSON_FILE=... ARRAY_SIZE=2 \
  Curator/tutorials/slurm-arrays-nemo-data-designer/submit_array.sh retry-missing
```

`status` checks `_SUCCESS/shard_*.json`. `retry-missing` submits only the shard
indices that do not have success markers.

## Core Follow-Ups

- Promote sidecar-complete output detection into Curator writer APIs instead of
  tutorial-only writer subclasses.
- Promote `SlurmArrayFilePartitioningStage` or a generic indexed-shard
  partitioning stage into Curator core.
- Add a core success-marker hook to `Pipeline` so per-shard metrics and retry
  metadata are emitted consistently.
- Add a record-range equivalent for pure non-seed generation.
