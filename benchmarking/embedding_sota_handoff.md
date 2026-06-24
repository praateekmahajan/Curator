# Embedding SOTA Handoff

Date: 2026-06-24

This is the canonical restart context for the embedding generation SOTA investigation. If the conversation compacts, read this file first, then `benchmarking/embedding_sota_tracking.md`, then the two historical note files listed below.

## Goal

Produce a conclusive, evidence-backed ranking of the fastest correct embedding generation path in Curator, and explain why the ranking is what it is.

The paths under comparison are:

- Curator in-process vLLM embedding with pretokenization.
- Curator Xenna in-process vLLM embedding with pretokenization.
- Curator Ray Serve endpoint embedding.
- Curator Dynamo endpoint embedding.

The old notes are directional, not ground truth. Do not trust old-vs-new throughput deltas because subtle differences such as max characters, dataset slice, worker geometry, batching, output format, and endpoint correctness can dominate. The only historical signal to treat as reasonably credible is that in-process pretokenized vLLM has tended to beat endpoint paths unless endpoint transport and batching are optimized.

## Current Hypothesis

The expected baseline ranking is:

1. In-process pretokenized vLLM should be fastest because it avoids HTTP/JSON serialization, request routing, endpoint scheduler overhead, and response decoding.
2. Xenna may beat or lose to Ray Data depending on whether its stage overlap and scheduling feed the vLLM actors better than Ray Data does.
3. Dynamo/Ray Serve endpoints can only compete if the endpoint is fed with enough concurrency and uses efficient payload/response formats. Prior one-GPU work suggests pretokenized inputs, large request batches, and base64 responses matter.
4. Ray Serve direct handle may be a future optimization because it can bypass HTTP while preserving service-style separation.

The benchmark must validate correctness and speed together. A fast endpoint result is not useful if it returns missing rows, duplicate indexes, token-level embeddings instead of pooled embeddings, or a different text cap than the in-process path.

## Current Branch State

Worktree:

```bash
/raid/praateekm/NeMo-Curator/.worktrees/embedding-benchmarks
```

Branch:

```bash
praateek/embedding-benchmarks
```

Local commits made for this investigation:

- `ccd6bac9` - benchmark script correctness fixes: endpoint response validation, shared `--max-chars`, endpoint/in-process text-cap parity, Dynamo pooling patch wiring.
- `c7264f1c` - initial tracking file.
- `a2418640` - single-YAML corrected baseline entries for run `i=2`.

Previous agent diffs were not discarded. They were preserved in:

```bash
stash@{0}: preexisting embedding benchmark diffs before single-yaml run 2026-06-24
```

The restored historical note files are intentionally left untracked unless the user asks otherwise:

- `benchmarking/embedding-endpoint-benchmark-notes.md`
- `benchmarking/embedding_sota_investigation_notes.md`

## Benchmark Files

Primary benchmark script:

```bash
benchmarking/scripts/embedding_generation_benchmark.py
```

Single working YAML:

```bash
benchmarking/local-embedding-endpoint.yaml
```

Do not create a new YAML for each experiment. Add, remove, or rename entries in this single working YAML.

Tracking file:

```bash
benchmarking/embedding_sota_tracking.md
```

Tmux log for the current run:

```bash
benchmarking/embedding_sota_investigation_tmux.log
```

Result session directory:

```bash
/raid/praateekm/curator-nightly/results/embedding-sota-investigation
```

## Current Run

The active benchmark session name is:

```bash
embedding-sota-investigation
```

The stable tmux session name is:

```bash
embedding_sota_investigation
```

Always reuse this tmux session name. Before starting a new run, kill the old session with this exact name if it exists. Do not create many new tmux sessions.

Current run commit/reason:

```bash
a2418640 / single-yaml-corrected-baseline-a2418640-i2
```

Current exact entries:

```text
embedding_generation_raydata_c7264f1c_i2
embedding_generation_xenna_c7264f1c_i2
embedding_generation_ray_serve_endpoint_c7264f1c_i2
embedding_generation_dynamo_endpoint_c7264f1c_i2
```

Shared baseline shape:

- Dataset: `/raid/praateekm/datasets/fineweb-edu-fortified-5m-1280`
- Source parquet files counted before launch: 1280.
- Dataset ratio: `0.2`, expected to select at least 256 files, satisfying `64 CPUs * 4 GPU workers`.
- Model: `google/embeddinggemma-300m`
- GPUs: `device=3,4,5,6`
- CPUs: 64
- In-process workers/replicas: 4
- In-process batch size: 32
- Endpoint replicas: 4
- Endpoint client workers: 16
- Endpoint max concurrent requests per client: 64
- Endpoint request batch size: 8
- Text cap: `--max-chars=1500` for in-process, `--endpoint-max-chars=1500` for endpoint.

## How To Run

Use Docker through `benchmarking/tools/run.sh`. Do not run the benchmark script directly on bare metal.

This image does not have the benchmark runner as its default Docker command, so use `run.sh --shell` and invoke `python benchmarking/run.py ...` inside the container.

Current launch shape:

```bash
tmux has-session -t embedding_sota_investigation 2>/dev/null && tmux kill-session -t embedding_sota_investigation || true

cd /raid/praateekm/NeMo-Curator/.worktrees/embedding-benchmarks
source /raid/praateekm/NeMo-Curator/.venv/bin/activate

export CURATOR_BENCHMARKING_IMAGE=nemo_curator_nightly_ray_256_dynamo_130_20260615:20260623
export HOST_CURATOR_DIR=/raid/praateekm/NeMo-Curator/.worktrees/embedding-benchmarks
export GPUS='"device=3,4,5,6"'

benchmarking/tools/run.sh \
  --use-host-curator \
  --config benchmarking/nightly-benchmark.yaml \
  --config benchmarking/local-embedding-endpoint.yaml \
  --shell "cd /opt/Curator && export USER=root LOGNAME=root RAY_SERVE_EXPERIMENTAL_PIP_HAPROXY=1 && python benchmarking/run.py --config benchmarking/nightly-benchmark.yaml --config benchmarking/local-embedding-endpoint.yaml --session-name embedding-sota-investigation --entries-exact embedding_generation_raydata_c7264f1c_i2,embedding_generation_xenna_c7264f1c_i2,embedding_generation_ray_serve_endpoint_c7264f1c_i2,embedding_generation_dynamo_endpoint_c7264f1c_i2 --reason single-yaml-corrected-baseline-a2418640-i2"
```

If running through tmux, pipe output to:

```bash
benchmarking/embedding_sota_investigation_tmux.log
```

## Operational Rules

- Use Docker for benchmark/script execution.
- Docker runs as root in the current workflow. Files created by Docker root may need Docker root for cleanup.
- Do not delete the restored historical note files.
- Do not run unit tests while focusing on benchmark execution unless the user explicitly asks.
- Do not push.
- Make incremental commits for source/config/tracking changes.
- Keep appending to `benchmarking/embedding_sota_tracking.md`.
- Use the same benchmark session name so all results land under one result session.
- Use the same tmux session name and clean it up before relaunching.

## Correctness Requirements

The benchmark script should enforce:

- Endpoint response count equals request input count.
- Endpoint response indexes are present, unique, and contiguous for each request.
- In-process and endpoint paths use the same text cap.
- Dynamo returns fixed-size pooled embeddings, not token-level variable-length embeddings.
- Metrics include enough context to explain throughput: number of input files, number of documents, text cap, endpoint concurrency, request batch size, response encoding, and startup time when applicable.

## Next Experiment Motivation

After the corrected baseline finishes, rank by `throughput_docs_per_sec` and verify `num_documents_processed`, `num_input_files`, and endpoint correctness metrics.

If endpoint paths are slower, next optimize one variable at a time in the single YAML:

- Endpoint pretokenized inputs.
- Base64 embedding responses instead of float JSON.
- Larger endpoint request batches, especially for Dynamo.
- Ray Serve handle/direct mode if implemented in Curator.
- Endpoint concurrency sweep around enough total HTTP requests to keep four engines fed.

If in-process wins but GPU utilization has gaps, investigate:

- More vLLM actors per GPU, such as fractional-GPU actor overlap.
- Xenna versus Ray Data scheduling behavior.
- Tokenization/embedding overlap and backpressure.

The final answer should explain the ranking in terms of actual bottlenecks: GPU occupancy, tokenization placement, endpoint transport overhead, request/response serialization, scheduler overhead, and batching/concurrency.
