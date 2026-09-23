# Resumable inference over large JSONL files with SLURM arrays

Generate answers for a large collection of questions without physically splitting
the input files. A small JSONL manifest describes byte ranges containing at most
512 rows. Each range is one Curator source task, one checkpoint unit, and one
output file. Independent SLURM array elements process disjoint sets of these tasks.

The example preserves input columns and appends `updated_<model-key>_*` answer,
usage, and diagnostic columns. It uses the same user-message answer-generation
behavior as the native vLLM benchmark; it does **not** instruct the model to
rewrite questions. Change the prompt implementation and start a new logical run
if your application requires that behavior.

## Files and data flow

| File | Responsibility |
| --- | --- |
| `manifest.py` | Stream inputs once, index byte ranges, estimate actual shard loads |
| `stages.py` | Emit deterministic source tasks; seek and read only one range |
| `inference.py` | Bounded async requests, row diagnostics, and systemic-failure retries |
| `writer.py` | Reuse JsonlWriter with exact manifest filenames and optional Zstandard compression |
| `pipeline.py` | Compose stages and enable source checkpointing |
| `recovery.py` | Bundle selected original logical shards into a standalone recovery manifest |
| `verification.py` | Validate selected shard outputs and optionally publish completion manifests |
| `worker-env.sh` | Activate the worker environment and preserve array or bundle settings |
| `serve.sh` | Reference Qwen/DeepSeek vLLM server flags for four GB300 GPUs |
| `step.sh` | Start vLLM, wait for readiness, run the benchmark wrapper |
| `array.sbatch` | Execute the same validated step in each array element |
| `../../../benchmarking/scripts/manifest_inference.py` | Thin benchmark metrics wrapper |
| `../../../benchmarking/manifest-inference.yaml` | One parameterized benchmark entry |

```text
ManifestFilePartitioningStage
  -> Curator array assignment + completed-source filtering
  -> SpecificJsonlReader
  -> ResumableInferenceStage
  -> NamedJsonlWriter
```

Array assignment is owned by Curator's adapter, not the reader. Each source emits
the same deterministic list of manifest tasks. The adapter hashes each source
task's lineage ID to select its owner, then skips completed sources on resume.
The manifest metadata is loaded in memory by the source stage; the large input
files are not. This is suitable for hundreds of thousands of manifest records.

## Step 1: Prepare a worker environment and paths

Use an existing checkout environment with Curator's text dependencies, Ray,
PyArrow, PyYAML, and the OpenAI client installed. The benchmark runner also needs
its usual dependencies. Prepare the environment on a worker using your project's
locked setup command. Do not install dependencies or run data processing on a
login node. No DuckDB or additional JSON reader dependency is required.

Run the following from the checkout, substituting your own shared-filesystem paths:

```bash
export WORKTREE=/path/to/Curator
export INPUT_DIR=/shared/data/question_only
export MANIFEST_PATH=/shared/data/question_only_manifest_512.jsonl
export OUTPUT_DIR=/shared/results/question_only_qwen
export BENCHMARK_ROOT=/shared/benchmarking
export SESSION_NAME=qwen-manifest-run-001
export MODEL_KEY=qwen
```

Inputs must be immutable, uncompressed UTF-8 `.jsonl` files with one JSON object
per physical line and compatible column types. Blank lines are rejected. A final
line without a newline is supported. Compression needs a different indexed
reader and is not supported by this tutorial.

Keep the manifest outside `INPUT_DIR`. Relative paths in the manifest are resolved
against `INPUT_DIR`; generated manifests contain no absolute dataset paths.
The logs record resolved paths for recovery.
Never put site-specific paths, manifests, or generated outputs into Git.

## Step 2: Generate the manifest once

Inside a CPU worker allocation, activate the checkout environment and run:

```bash
source "$WORKTREE/.venv/bin/activate"
cd "$WORKTREE"
python -m tutorials.text.manifest_inference.manifest generate \
  --input-dir "$INPUT_DIR" --manifest "$MANIFEST_PATH" --max-num-rows 512
```

This makes one sequential pass through the input bytes. Memory use is bounded by
one input line plus the sorted list of filenames; it does not load an entire input
file or parse all documents. The manifest is published only when complete, and an
existing manifest is never overwritten. JSON parsing happens in the reader.

Each record contains `input_file`, `output_file`, zero-based `[start_line,end_line)`
and `[start_byte,end_byte)` ranges, row count, file size/mtime, and a range SHA-256.
For example, input `part-001.jsonl` produces outputs named
`part-001.jsonl/part-001_task_0.jsonl`, `part-001.jsonl/part-001_task_1.jsonl`, etc.
Including the relative input path avoids basename collisions across subdirectories.

Readers seek directly to the stored byte offset, retain at most one task's rows,
and validate the range length and row count before inference. This
avoids the repeated prefix scans required by line-number-only slicing.

## Step 3: Size the array using its heaviest shard

Task size and array size are different controls. Keep `--max-num-rows=512` for
short replay units; choose enough array elements to meet your runtime budget.
For 69 million rows, expect about 135,000 output files per model, with additional
partial tasks at source-file boundaries. There are no concurrent shared-file writes.

The successful four-GPU benchmark reference points are:

| Profile | Rows/s | Time for 131,072 rows | Client workers | Requests per worker |
| --- | ---: | ---: | ---: | ---: |
| Qwen NVFP4, DP4/TP1 | 38.02 | 57.5 min | 64 | 128 |
| DeepSeek default, DP2/TP2 + expert parallelism | 50.90 | 42.9 min | 64 | 80 |

These rates exclude model-server startup and depend on hardware, input/output
length distributions, and server configuration. They are estimates, not runtime
guarantees. A useful starting budget is 30 minutes of setup, 80% of reference
throughput, a three-hour target, and a 3:30 SLURM wall limit.

On a CPU worker, inspect an initial candidate:

```bash
export TOTAL_SHARDS=300
python -m tutorials.text.manifest_inference.manifest plan \
  --manifest "$MANIFEST_PATH" --shards "$TOTAL_SHARDS" \
  --rows-per-second 38.016 --setup-minutes 30 \
  --throughput-fraction 0.8 --target-minutes 180
```

Use `50.900` for the DeepSeek profile; 230 shards is an initial candidate for a
69-million-row corpus. The planner uses the same source-ID hashing as Curator,
reports minimum/maximum assigned rows, and estimates the heaviest shard's time.
Increase the shard count if `within_target` is false. Validate throughput on the
new corpus before accepting this estimate. The concurrency cap in `--array=...%N`
controls simultaneous jobs, not the number of logical shards.

## Step 4: Prepare and validate the model server

Each array element needs its own server and Ray cluster. By default, `step.sh`
uses the bundled `serve.sh` with the activated environment's `vllm` executable.
Validate that environment and profile in a held allocation first. To use a separate
server environment or custom flags, set `SERVER_SCRIPT` to an absolute path to a
tested script that stays in the foreground and uses `exec` to launch vLLM. It must listen at `MODEL_ENDPOINT` (default
`http://127.0.0.1:8000/v1`) and expose `/health`. It may activate its separately
validated server environment. It must not background or daemonize the server.

The YAML selects model IDs and generation settings, **not server flags**. To
reproduce the reference results, preserve the profiles in the bundled `serve.sh`:

- Qwen: `MODEL_KEY=qwen`, DP4/TP1, 1024 sequences/replica, 8192 batched tokens,
  8192 maximum graph capture, FP8 KV cache, context 32768, memory utilization 0.95.
- DeepSeek: `MODEL_KEY=deepseek`, DP2/TP2, expert parallelism,
  2048 sequences/replica, 4096 batched tokens and graph capture, context 32768,
  memory utilization 0.95. Preserve that profile's loader, offload, parser,
  compilation sizes, and environment flags as well.

Both generate at most 8192 tokens with thinking disabled and request seed 42.
Qwen uses temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5,
and repetition penalty 1.0. DeepSeek uses temperature 1.0 and top-p 0.95.
These profiles were
measured on four GB300 GPUs; they are not portable memory defaults for smaller GPUs.
Reuse your validated server environment and shared caches throughout the run.

First test a small, separately generated manifest in a held allocation matching
the final topology. Use a separate smoke-test session/output directory. Export
`SLURM_ARRAY_TASK_ID=0` and `TOTAL_SHARDS=1` for this manual step, then run:

```bash
srun --jobid="$JOB_ID" --overlap --ntasks=1 \
  bash "$WORKTREE/tutorials/text/manifest_inference/step.sh"
```

Check interpreter, server readiness, assigned task count, completed output, and
throughput. Interrupt a test and rerun with the same manifest/session/output and
a new attempt identifier; completed tasks must be skipped. Restore the planned
array count and unset manually exported `SLURM_ARRAY_*` variables before submission.

## Step 5: Submit the array

Choose a distinct session and output directory per model. Every retry of that
model's logical run must reuse its original session, manifest, output directory,
and **original `TOTAL_SHARDS`**. Do not derive the total from the size of a retry
array. The worker exports Curator's explicit shard variables for this reason.

```bash
# Optional when using the bundled serve.sh:
# export SERVER_SCRIPT=/path/to/validated-server.sh
export ARRAY_CONCURRENCY=8
mkdir -p "$BENCHMARK_ROOT/$SESSION_NAME/slurm"
sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
  --qos="$SLURM_QOS" --cpus-per-task=144 --mem=0 \
  --array="0-$((TOTAL_SHARDS - 1))%${ARRAY_CONCURRENCY}" \
  --output="$BENCHMARK_ROOT/$SESSION_NAME/slurm/%A_%a.out" \
  --error="$BENCHMARK_ROOT/$SESSION_NAME/slurm/%A_%a.err" \
  "$WORKTREE/tutorials/text/manifest_inference/array.sbatch"
```

Supply your cluster's account, partition, and non-interactive QOS. The example
requests one exclusive four-GPU node per element for 3:30; adjust resources for
your cluster and keep full-node exclusivity paired with full-GPU use. Inspect
active jobs, the submission script, and output collisions before submitting.
Record the returned job ID. Do not launch the full array before the smoke test.

For DeepSeek, switch `MODEL_KEY=deepseek`, session, output, and
planned shard count (and `SERVER_SCRIPT` if using a custom launcher). The same manifest can be shared. The worker selects 80
requests/client for DeepSeek and 128 for Qwen. The YAML always contains one entry.

```text
BENCHMARK_ROOT/SESSION_NAME/
  qwen_<SLURM_ARRAY_TASK_ID>_<SLURM_JOB_ID>/
    logs/                         # benchmark log and per-restart server logs
    metrics.json
    params.json
  array_environments/             # distinct environment capture per attempt
  .nemo_curator_checkpoint_dir/
    .nemo_curator_metadata/        # LMDB, shard completion, failure records

OUTPUT_DIR/
  part-001.jsonl/part-001_task_0.jsonl.zst
  part-001.jsonl/part-001_task_1.jsonl.zst
```

The entry name is `<model>_<physical-array-index>_<SLURM_JOB_ID>`. An automatic
SLURM requeue keeps the same child job ID, so restart `N` uses a distinct
`_restart_N` suffix. The YAML reads this complete name from `ENTRY`.
Benchmark metrics count work written in the current attempt; resumed tasks are
not counted again. `manifest_rows` describes the whole manifest, not one shard.

## Step 6: Inspect failures and resume

Request-specific terminal failures do not invalidate the other rows in a source
task. An over-context prompt, invalid prompt or response, exhausted empty
completion, zero-token completion, reasoning-budget exhaustion, or unexpected
finish reason is written in place with `is_success=false` and a nonempty
`failed_reason`. Its answer may be null or empty. The complete task is published
and checkpointed, so a later attempt does not regenerate its successful rows.

Exhausted transport errors, HTTP 5xx responses, and unknown systemic failures
still return `FailedTask()`. If a task contains both a terminal row failure and
a systemic failure, no output is published for that task. It remains pending and
the shard remains incomplete; other tasks can finish and checkpoint normally.
The wrapper exits nonzero when the attempt has any `FailedTask` records.

Diagnostics live under the checkpoint's `.nemo_curator_metadata/.failed_tasks/`
attempt directories, in `tasks/<source-id>.jsonl`. Each includes
the original manifest record, task lineage, model, zero-based failed row numbers,
reasons, and request counts. These files describe tasks held for systemic retry;
terminal row diagnostics live in the generated row metadata. Historical task
diagnostics remain after successful retries; use current shard-completion state
to determine whether work is pending.
Reader/writer exceptions abort the pipeline and remain visible in its logs;
their incomplete sources are also replayed.

After all attempts have stopped, use the existing retry helper in your worker
environment to discover incomplete logical shard IDs:

```bash
python tutorials/slurm/retry_array.py \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --format fields
```

For example, `1,4-6 0 0 240` means retry physical IDs `1,4-6`, with
`SHARD_INDEX_OFFSET=0`, minimum index zero, and the original `TOTAL_SHARDS=240`.
Repeat Step 5's submission with `--array="1,4-6%${ARRAY_CONCURRENCY}"` and the same
session/output/model settings. Empty output means there are no incomplete shards.
See the [SLURM array retry workflow](../../slurm/README.md) for windowed submissions
when scheduler array limits require nonzero `SHARD_INDEX_OFFSET`.

Resubmit those IDs with the same `TOTAL_SHARDS` and all other run settings.
Curator reruns only the incomplete source tasks inside each selected shard.
Never run two attempts of the same logical shard concurrently.

For a small or fragmented recovery set, build one standalone bundle instead of
starting one array element per original shard:

```bash
export RECOVERY_SHARDS=1,4,5,6
export RECOVERY_MANIFEST=/shared/recovery/manifest.jsonl
export RECOVERY_PLAN=/shared/recovery/plan.json

python -m tutorials.text.manifest_inference.recovery \
  --manifest "$MANIFEST_PATH" \
  --input-dir "$INPUT_DIR" \
  --shards "$RECOVERY_SHARDS" \
  --total-shards "$TOTAL_SHARDS" \
  --output "$RECOVERY_MANIFEST" \
  --plan-file "$RECOVERY_PLAN"
```

The bundle contains every canonical source task assigned to the selected original
logical shards. Completed sources are skipped only because the recovery run uses
the original checkpoint. For the standalone job, keep `INPUT_DIR`, `OUTPUT_DIR`,
`CHECKPOINT_PATH`, model settings, and the original `TOTAL_SHARDS`; set
`MANIFEST_PATH=$RECOVERY_MANIFEST` and explicitly export
`NEMO_CURATOR_SLURM_ARRAY_ENABLED=0`. The worker preserves that disabled value
and names the entry `<model>_bundle_<SLURM_JOB_ID>` (plus `_restart_N` after an
automatic requeue). Run at most one bundle GPU job per model, and do not overlap
it with an array attempt that can touch any selected shard.

The recovery plan records original shard counts and an optional disjoint
`--exclude-shards` assertion. Bundle execution does not replace the original
logical shard count or checkpoint identity.

After recovery stops, validate the selected outputs before publishing their
original logical-shard completion manifests:

```bash
python -m tutorials.text.manifest_inference.verification \
  --original-manifest "$ORIGINAL_MANIFEST_PATH" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --model-key "$MODEL_KEY" \
  --total-shards "$TOTAL_SHARDS" \
  --shards "$RECOVERY_SHARDS" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --report "$RECOVERY_REPORT"
```

The verifier checks every expected `.zst` file, row count, nested metadata,
terminal-failure classification, lineage, and the 8192-token request cap. It
writes completion manifests only after every selected output passes. Omit
`--checkpoint-path` for a read-only verification. Use `--expected-max-tokens`
when the logical run intentionally used a different cap.

Checkpointing is at-least-once: a task can replay after a crash. The writer reuses
Curator's `JsonlWriter.write_data()` and overwrites the exact manifest output path
on retry. An interrupted write can leave a partial file; a file's existence alone
is not proof of completion. Use Curator's checkpoints and shard verification.
Keep checkpoint files and their outputs together.

The launcher supplies the original logical shard count even for a one-element
pilot or a subset retry. Curator owns task assignment and completion tracking.
Manifest range IDs distinguish multiple
tasks from the same input file; they are not output filenames.

Inputs must remain unchanged while using the byte-offset manifest. The reader
uses Arrow's JSON parser on each bounded range, checks its row count, and does
not recompute content checksums. Malformed JSON or incompatible field types fail
that task rather than silently dropping or moving rows.

Use a new checkpoint and output directory when changing inputs, models, or
sampling settings. There is no custom pinned run contract.

Output retains `updated_<model>_answer` and `updated_<model>_reasoning`. The
`updated_<model>_metadata` object contains `finish_reason`, `prompt_tokens`,
`completion_tokens`, `attempt_count`, `retry_count`, `is_success`, and
`failed_reason`. No per-row request latency is written. `generation_lineage`
records the model, captured server arguments, and request settings. Compressed
files keep the manifest filename with `.zst` appended.
