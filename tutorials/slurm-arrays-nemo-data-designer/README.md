# SLURM Arrays For NeMo Data Designer (advanced reference)

> **Just want a quickstart?** See `tutorials/slurm-datadesigner/` — single
> model, JSONL only, ~600 LOC total. This tutorial is the advanced reference
> that adds multi-model serving, Dynamo backend, Parquet I/O, by-shard
> output, and rich per-shard metrics.

Two independent concepts:

1. **SLURM-array support for Curator** (`slurm_array_support.py`). Four
   pieces, reusable in any pipeline:
   - `Shard` — namespace of static helpers: `env()`, `data_path()`,
     `marker_path()`, `has_marker()`, `completed()`, `missing()`,
     `write_marker()`. All static methods; no instance state.
   - `SlurmArrayFilePartitioningStage` — `FilePartitioningStage` subclass.
     Filters parent-class output through two checks: shard ownership by
     hash, and deterministic output already on disk. Runs as a normal
     pipeline stage on the worker — no driver-side filesystem walks, so
     the host that submits the pipeline doesn't need lustre access or
     curator-container permissions.
   - `enable_slurm_array_partitioning(reader, ...)` — one-line swap of the
     first `FilePartitioningStage` inside any reader composite (JsonlReader,
     ParquetReader, …).
   - `SlurmArrayPipeline` — `Pipeline` subclass. Short-circuits on an
     existing `_SUCCESS/shard_*.json` marker and writes that marker on
     clean completion (pipeline name + output file list + run time;
     anything else goes through the caller-supplied `success_payload`).

2. **The demo SDG pipeline** (`seeded_sdg_pipeline.py`). Loads a Data Designer
   config from JSON/YAML, brings up one or more model endpoints through
   `InferenceServer` (Dynamo or Ray Serve), and runs the seeded pipeline
   over this shard's slice of input files.

| File | What it adds |
| --- | --- |
| `slurm_array_support.py` | The three pieces above. |
| `seeded_sdg_pipeline.py` | The SDG pipeline + multi-model serving. |
| `submit_array.sh` | `submit` / `status` / `retry-missing` / `worker`. |
| `configs/*.json` | Data Designer configs and example models specs. |

## Env-var contract

| Var | Meaning |
| --- | --- |
| `CURATOR_SHARD_INDEX` | This shard's index. Defaults to `SLURM_ARRAY_TASK_ID - CURATOR_SHARD_OFFSET`. |
| `CURATOR_NUM_SHARDS` | Original shard count. Preferred over `SLURM_ARRAY_TASK_COUNT` so sparse retries (e.g. `--array=3,5,9`) keep the original count. |
| `CURATOR_ORIGINAL_ARRAY_SIZE` | Fallback set by `submit_array.sh` on the first submission. |

## Output layout

```text
<OUTPUT_PATH>/
  data/                            # OUTPUT_LAYOUT=flat (default)
    <hash>.jsonl                   # filename = get_deterministic_hash(source_files, task_id)
    ...
  data/shard_00000/                # OUTPUT_LAYOUT=by_shard
    <hash>.jsonl
  _SUCCESS/
    shard_00000.json               # per-shard marker w/ metrics
    shard_00002.json
  _logs/*.out
  _logs/*.err
```

Filenames are content-hashed, so flat layout never collides across shards.

## Submit a single-node run

Model serving is configured by `MODELS_JSON_FILE` (path) or `MODELS_JSON`
(inline JSON). Always a JSON list, even for one model — see
`configs/cached_hf_gretel_models.json` for the schema.

```bash
INPUT_PATH=/path/to/seed/jsonl_dir \
OUTPUT_PATH=/path/to/output_root \
DD_CONFIG=Curator/tutorials/slurm-arrays-nemo-data-designer/configs/gretel_medical_two_model.sdg.json \
MODELS_JSON_FILE=Curator/tutorials/slurm-arrays-nemo-data-designer/configs/cached_hf_gretel_models.json \
INPUT_FORMAT=jsonl OUTPUT_FORMAT=jsonl \
ARRAY_SIZE=1 MAX_CONCURRENT=1 NODES=1 GPUS_PER_NODE=8 \
PARTITION=batch_short TIME_LIMIT=02:00:00 \
Curator/tutorials/slurm-arrays-nemo-data-designer/submit_array.sh submit
```

Recommended scale checks: H=1×V=1 → H=1×V=2 → H=2×V=1 → H=2×V=2.

## Model spec JSON

```json
[
  {"model": "/path/to/a", "served_model_name": "model-a", "tp": 1, "replicas": "auto"},
  {"model": "/path/to/b", "served_model_name": "model-b", "tp": 1, "replicas": "auto"}
]
```

Required per entry: `model`, `tp`. Optional: `served_model_name` (defaults
to `model`), `replicas` (defaults to `"auto"`), `engine_kwargs`,
`max_model_len`.

With `replicas=auto`, the pipeline divides the GPUs Ray sees across
auto-sized models, using as many as possible.

**Wiring to your Data Designer config:** the DD config's `model_configs`
entries must set `model: "<served_model_name>"` and `provider: "<your
provider name>"` (default `"local"`) so each LLM column hits the local
inference server. See `configs/gretel_medical_two_model.sdg.json` for an
example.

## Status and retry

```bash
INPUT_PATH=... OUTPUT_PATH=... DD_CONFIG=... ARRAY_SIZE=2 \
  submit_array.sh status

INPUT_PATH=... OUTPUT_PATH=... DD_CONFIG=... ARRAY_SIZE=2 \
  submit_array.sh retry-missing
```

`status` reads `_SUCCESS/shard_*.json`. `retry-missing` resubmits only the
shard indices missing a marker, preserving `CURATOR_ORIGINAL_ARRAY_SIZE`
so the hash-based partitioning stays consistent across retries.

## Seed data

This tutorial assumes you already have seeded data (JSONL or Parquet) at
`INPUT_PATH`. For a one-line download + chunk into JSONL, see the
quickstart at `tutorials/slurm-datadesigner/pipeline.py`, which reuses
`download_and_convert_seed_data` from
`tutorials/synthetic/nemo_data_designer/`. For Parquet, convert with
`pandas`/`pyarrow`.

## Promoting upstream

1. Move `SlurmArrayFilePartitioningStage` into `nemo_curator/stages/file_partitioning.py`.
2. Add `output_path` + shard params to `FilePartitioningStage` directly — the
   monkey-patch in `enable_slurm_array_partitioning` then goes away.
3. Move `SlurmArrayPipeline` into `nemo_curator/pipeline/slurm.py`.
4. Port `submit_array.sh` to a Python entry point.

## Limitations

- **Dynamo backend** doesn't handle simultaneous HF model pulls; pre-stage
  weights to a local path (Ray Serve does handle it).
- **Image/video/audio readers** aren't covered by `enable_slurm_array_partitioning`'s
  detection — it walks the composite's stages until it finds a
  `FilePartitioningStage`, so any reader that decomposes that way works.
- **Adding files between runs:** if a shard already has its marker,
  `SlurmArrayPipeline.run()` short-circuits and won't pick up new files
  for that shard. Delete the marker to force a re-run (data is preserved
  because filenames are deterministic).
- **Pure-`num_records` SDG (no input files):** needs a record-range
  partitioning stage; not yet written.
