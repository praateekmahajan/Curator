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

The benchmark must validate correctness and speed together. A fast endpoint result is not useful if it returns missing rows, duplicate indexes, token-level embeddings instead of pooled embeddings, raw in-process vLLM text input, or benchmark-side character truncation.

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
- `c6af206b` - durable handoff context.
- `517cdd10` - recorded directional capped Ray Data result.
- `f5eaed3c` - uncapped, pretokenized baseline enforcement.
- `2fc98b70` - recorded uncapped Ray Data in-process result.
- `48c6b49f` - recorded uncapped Xenna in-process result.
- `1157f44f` - endpoint text requests default to model-context token truncation.
- `918bd744` - recorded endpoint truncation rerun failure and next endpoint motivation.
- `22efeee8` - prepared tokenized endpoint rerun with base64 responses.

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

The stopped capped run was:

```bash
a2418640 / single-yaml-corrected-baseline-a2418640-i2
```

It produced directional, non-final results only:

- Ray Data in-process: 1,023,449 docs, 262 input files, 3571.74 docs/s, `max_chars=1500`.
- Xenna in-process: 1,023,449 docs, 262 input files, 4150.21 docs/s, `max_chars=1500`.

It was stopped before endpoint entries because character caps are not representative of real embedding workloads.

The first uncapped run was:

```bash
uncapped-pretokenized-baseline-f5eaed3c-i3
```

It produced:

- Ray Data in-process pretokenized: 1,023,449 docs, 262 input files, no char caps, 2150.33 docs/s.
- Xenna in-process pretokenized: 1,023,449 docs, 262 input files, no char caps, 2394.53 docs/s.
- Ray Serve endpoint failed because vLLM rejected prompts over the 2048-token model context when `truncate_prompt_tokens` was omitted.
- Dynamo endpoint failed with `Failed to fold embeddings stream`, consistent with endpoint text mode not safely handling over-context uncapped prompts.

The benchmark script has since been changed so endpoint `truncate_prompt_tokens <= 0` resolves to the model context length by default. This keeps no character cap while matching in-process pretokenized model-context token truncation.

The endpoint text rerun was:

```bash
uncapped-endpoint-model-context-truncation-1157f44f-i4
```

It failed:

- Ray Serve text input avoided the over-context validation error, but failed with HTTP `ReadError` / OpenAI `APIConnectionError` under 16 client tasks.
- Dynamo text input still failed with `Failed to fold embeddings stream`.

The next endpoint experiment used client-side pretokenized `token_ids` plus base64 responses, with no character cap.

Completed i5 run/reason:

```bash
uncapped-endpoint-tokenids-base64-i5
```

Completed i5 exact entries:

```text
embedding_generation_ray_serve_endpoint_918bd744_i5
embedding_generation_dynamo_endpoint_918bd744_i5
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
- In-process model variation: `vllm_text_pretokenized`.
- Endpoint replicas: 4
- Endpoint client workers: 16
- Endpoint max concurrent requests per client: 64
- Endpoint request batch size: 8
- Endpoint input format for i5: `token_ids`.
- Endpoint response encoding for i5: `base64`.
- No benchmark-side character cap. Do not set `--max-chars` or `--endpoint-max-chars` unless the entry name and tracking row explicitly say the experiment is char-capped.
- Endpoint token truncation defaults to the model context length. This is not a character cap; it is required because Ray Serve's vLLM OpenAI text path otherwise rejects over-context prompts instead of truncating them automatically.

The i5 endpoint rerun is complete and the tmux session was killed for hygiene:

- Ray Serve token_ids/base64 failed before producing throughput with `httpcore.ReadError` / OpenAI `APIConnectionError`. Its results confirmed tokenized input, base64 response encoding, no character cap, and `pretokenized=true`; this is an endpoint transport/serving failure, not a raw-text or over-context failure.
- Dynamo token_ids/base64 succeeded on 1,023,449 docs and 262 input files with `pretokenized=true`, `endpoint_pretokenized=true`, no character cap, and effective endpoint truncation of 2048 tokens.
- Dynamo end-to-end metric: 504.61s total, 92.19s service startup, 2028.21 docs/s.
- Dynamo steady-state pipeline rate excluding service startup: about 2481.61 docs/s.

Current successful uncapped ranking by end-to-end benchmark throughput:

1. Xenna in-process pretokenized vLLM: 2394.53 docs/s.
2. Xenna in-process pretokenized vLLM, batch size 64: 2339.28 docs/s.
3. Ray Data in-process pretokenized vLLM: 2150.33 docs/s.
4. Dynamo endpoint token_ids/base64, request batch size 16: 2061.27 docs/s, including service startup.
5. Dynamo endpoint token_ids/base64, request batch size 32: 2041.99 docs/s, including service startup.
6. Dynamo endpoint token_ids/base64, request batch size 8: 2028.21 docs/s, including service startup.
7. Ray Serve endpoint token_ids/base64 at aggregate concurrency 128: 913.45 docs/s, including service startup.
8. Ray Serve endpoint token_ids/base64 at aggregate concurrency 512: 870.98 docs/s, including service startup.

Current successful uncapped ranking by persistent-service steady-state throughput:

1. Dynamo endpoint token_ids/base64, request batch size 16: about 2515.39 docs/s after excluding service startup.
2. Dynamo endpoint token_ids/base64, request batch size 32: about 2506.17 docs/s after excluding service startup.
3. Dynamo endpoint token_ids/base64, request batch size 8: about 2481.61 docs/s after excluding service startup.
4. Xenna in-process pretokenized vLLM: 2394.53 docs/s.
5. Xenna in-process pretokenized vLLM, batch size 64: 2339.28 docs/s.
6. Ray Data in-process pretokenized vLLM: 2150.33 docs/s.
7. Ray Serve endpoint token_ids/base64 at aggregate concurrency 128: about 968.72 docs/s after excluding service startup.
8. Ray Serve endpoint token_ids/base64 at aggregate concurrency 512: about 920.58 docs/s after excluding service startup.

Do not collapse these two rankings into one claim. Batch jobs pay startup; already-running services do not. Current endpoint tuning says Dynamo request batch size 16 is the best tested endpoint point; the next higher-value experiment is tuning the in-process winner.

Current intended next run/reason:

```bash
xenna-inprocess-pretokenized-batch16-fa3fea19-i11
```

Current exact entry:

```text
embedding_generation_xenna_fa3fea19_i11
```

This should keep Xenna in-process vLLM on `vllm_text_pretokenized`, no character caps, 4 model workers, the same 262-file dataset slice, and change only `--model-inference-batch-size` from 32 to 16. The purpose is to bracket the current best Xenna batch size 32 from below after batch size 64 regressed.

The i6 Ray Serve lower-concurrency run succeeded:

- 1,023,449 docs, 262 input files.
- End-to-end: 1120.43s, 913.45 docs/s.
- Startup: 63.94s.
- Persistent-service steady state excluding startup: about 968.72 docs/s.
- `pretokenized=true`, `endpoint_pretokenized=true`, no char caps, endpoint token truncation 2048, token_ids/base64.

This confirms Ray Serve is correct at aggregate concurrency 128 but much slower than current in-process and Dynamo runs. The next Ray Serve experiment should sweep concurrency upward, changing only `--endpoint-max-concurrent-requests`.

The i7 Ray Serve aggregate-concurrency 512 run also succeeded, but it was slower than aggregate 128:

- 1,023,449 docs, 262 input files.
- End-to-end: 1175.05s, 870.98 docs/s.
- Startup: 63.31s.
- Persistent-service steady state excluding startup: about 920.58 docs/s.

Current Ray Serve conclusion: aggregate concurrency 128 is the best tested Ray Serve point. Aggregate 512 is stable but slower; aggregate 1024 fails with ingress/client transport errors. If more Ray Serve tuning is needed, test aggregate 256 next. Otherwise prioritize Dynamo tuning and in-process/Xenna comparisons.

The i8 Dynamo request-batch-size 16 run succeeded and modestly beat the previous Dynamo batch-size 8 result:

- 1,023,449 docs, 262 input files.
- End-to-end: 496.51s, 2061.27 docs/s.
- Startup: 89.64s.
- Persistent-service steady state excluding startup: about 2515.39 docs/s.
- `pretokenized=true`, `endpoint_pretokenized=true`, no char caps, endpoint token truncation 2048, token_ids/base64.

At this point, request batch size 16 was the best tested Dynamo point and the best steady-state point overall, but still lost to Xenna for startup-inclusive batch-job throughput. The i9 run below tested request batch size 32.

The i9 Dynamo request-batch-size 32 run succeeded but regressed slightly:

- 1,023,449 docs, 262 input files.
- End-to-end: 501.20s, 2041.99 docs/s.
- Startup: 92.83s.
- Persistent-service steady state excluding startup: about 2506.17 docs/s.
- `pretokenized=true`, `endpoint_pretokenized=true`, no char caps, endpoint token truncation 2048, token_ids/base64.

Current Dynamo conclusion: request batch size 16 is the best tested Dynamo point. Batch size 32 stayed correct but was slower than 16, so larger request payload/backpressure appears to offset the reduced request count. Do not try batch size 64 unless the goal is explicitly to map the full Dynamo curve; the higher-value next step is tuning the in-process winner.

The i10 Xenna in-process batch-size 64 run succeeded but regressed:

- 1,023,449 docs, 262 input files.
- End-to-end: 437.51s, 2339.28 docs/s.
- `pretokenized=true`, no char caps, in-process `vllm_text_pretokenized`.

Current Xenna conclusion: batch size 32 is still the best tested startup-inclusive in-process point. Batch size 64 is correct but slower, so the next check is batch size 16 to see whether the optimum is below 32 or whether 32 is the local sweet spot.

## How To Run

Use Docker through `benchmarking/tools/run.sh`. Do not run the benchmark script directly on bare metal.

This image does not have the benchmark runner as its default Docker command, so use `run.sh --shell` and invoke `python benchmarking/run.py ...` inside the container.

Current i11 launch shape:

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
  --shell "cd /opt/Curator && export USER=root LOGNAME=root RAY_SERVE_EXPERIMENTAL_PIP_HAPROXY=1 && python benchmarking/run.py --config benchmarking/nightly-benchmark.yaml --config benchmarking/local-embedding-endpoint.yaml --session-name embedding-sota-investigation --entries-exact embedding_generation_xenna_fa3fea19_i11 --reason xenna-inprocess-pretokenized-batch16-fa3fea19-i11"
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
- Keep in-process entries on `vllm_text_pretokenized`. Raw in-process `vllm_text` requires explicit `--allow-raw-inprocess-vllm` and should only be used for an intentional raw-tokenization regression experiment.
- Do not use benchmark-side character caps unless that is the explicit experiment motivation.
- Latest Docker validation before i9 confirmed the single YAML has pretokenized in-process entries, token_ids endpoint entries, and no character caps; the script still has the pretokenized default/guard and the Curator vLLM stage token-ID path.

## Correctness Requirements

The benchmark script should enforce:

- Endpoint response count equals request input count.
- Endpoint response indexes are present, unique, and contiguous for each request.
- In-process paths use `vllm_text_pretokenized`.
- In-process and endpoint paths avoid benchmark-side character truncation unless the experiment intentionally studies a character cap.
- Endpoint text paths use model-context token truncation by default, because the OpenAI-compatible vLLM endpoint does not automatically truncate over-context raw text.
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
