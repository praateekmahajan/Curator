# Embedding SOTA Handoff

Date: 2026-06-24

This is the canonical restart context for the embedding generation SOTA investigation. If the conversation compacts, read this file first, then `benchmarking/embedding_sota_conclusions.md`, then `benchmarking/embedding_sota_tracking.md`, then the two historical note files listed below.

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
- `f7a9ad75` - added real fractional GPU controls for in-process vLLM embedding stages.
- `9453a1a9` - prepared distinct Xenna fractional rerun entry.
- `7eacfdfe` - recorded fractional Ray Data result and reran Xenna.
- `de4ab6b8` - recorded fractional Xenna result.
- `ca31a35e` - enabled Ray 2.56 `ray-haproxy`, forced Ray Serve vLLM RayExecutorV2, and added Ray Serve direct-handle client mode.
- `f1c2d269` - prepared first Ray Serve HAProxy HTTP and direct-handle entries.
- `233d9e42` - avoided Ray Serve HAProxy metrics port collisions.
- `db6443b4` - moved the HAProxy metrics wildcard bind check into the shared `get_free_port` helper via `bind_host`.
- `1e5553e1` - prepared corrected Ray Serve HAProxy port rerun entries.
- `4fb16baf` - fixed the Ray Serve direct-handle client to request a streaming `DeploymentResponseGenerator`.
- `a487e4da` - prepared the corrected Ray Serve direct-handle i22 rerun entry.
- `f8a7400c` - recorded the successful corrected Ray Serve direct-handle i22 result.
- `d97e3165` - raised the Docker `nofile` limit in `benchmarking/tools/run.sh` to test Ray Serve HTTP without the default fd cap.
- `e307cf23` - prepared the corrected Ray Serve HTTP i23 rerun entry using the raised Docker fd limit.

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

Conclusion file:

```bash
benchmarking/embedding_sota_conclusions.md
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
- In-process `--model-inference-batch-size`: present in some entry args but ignored by `VLLMEmbeddingModelStage`; do not treat it as vLLM batch-size tuning evidence.
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

Current successful uncapped ranking by end-to-end benchmark throughput, including the corrected Ray Serve direct-handle i22 run, the corrected raised-`nofile` Ray Serve HTTP i23 run, and excluding old Ray Serve runs that did not enable HAProxy:

1. Xenna in-process pretokenized vLLM, 16 fractional workers at 0.249 GPU: 2865.16 docs/s.
2. Ray Data in-process pretokenized vLLM, 16 fractional workers at 0.249 GPU: 2732.02 docs/s.
3. Xenna in-process pretokenized vLLM: 2394.53 docs/s.
4. Xenna in-process pretokenized vLLM, inert CLI batch value 16: 2377.21 docs/s.
5. Xenna in-process pretokenized vLLM, inert CLI batch value 64: 2339.28 docs/s.
6. Ray Data in-process pretokenized vLLM, inert CLI batch value 64: 2215.38 docs/s.
7. Ray Data in-process pretokenized vLLM, inert CLI batch value 128: 2205.00 docs/s.
8. Ray Data in-process pretokenized vLLM: 2150.33 docs/s.
9. Dynamo endpoint token_ids/base64, request batch size 16: 2061.27 docs/s, including service startup.
10. Dynamo endpoint token_ids/base64, request batch size 32: 2041.99 docs/s, including service startup.
11. Dynamo endpoint token_ids/base64, request batch size 8: 2028.21 docs/s, including service startup.
12. Ray Serve direct handle with HAProxy enabled, token_ids/base64, request batch size 8: 2011.39 docs/s, including service startup.
13. Ray Serve HTTP with HAProxy enabled, raised Docker `nofile`, token_ids/base64, request batch size 8: 1021.70 docs/s, including service startup.

Current successful uncapped ranking by persistent-service steady-state throughput, including the corrected Ray Serve direct-handle i22 run, the corrected raised-`nofile` Ray Serve HTTP i23 run, and excluding old Ray Serve runs that did not enable HAProxy:

1. Xenna in-process pretokenized vLLM, 16 fractional workers at 0.249 GPU: 2865.16 docs/s.
2. Ray Data in-process pretokenized vLLM, 16 fractional workers at 0.249 GPU: 2732.02 docs/s.
3. Dynamo endpoint token_ids/base64, request batch size 16: about 2515.39 docs/s after excluding service startup.
4. Dynamo endpoint token_ids/base64, request batch size 32: about 2506.17 docs/s after excluding service startup.
5. Dynamo endpoint token_ids/base64, request batch size 8: about 2481.61 docs/s after excluding service startup.
6. Xenna in-process pretokenized vLLM: 2394.53 docs/s.
7. Xenna in-process pretokenized vLLM, inert CLI batch value 16: 2377.21 docs/s.
8. Ray Serve direct handle with HAProxy enabled, token_ids/base64, request batch size 8: about 2358.18 docs/s after excluding service startup.
9. Xenna in-process pretokenized vLLM, inert CLI batch value 64: 2339.28 docs/s.
10. Ray Data in-process pretokenized vLLM, inert CLI batch value 64: 2215.38 docs/s.
11. Ray Data in-process pretokenized vLLM, inert CLI batch value 128: 2205.00 docs/s.
12. Ray Data in-process pretokenized vLLM: 2150.33 docs/s.
13. Ray Serve HTTP with HAProxy enabled, raised Docker `nofile`, token_ids/base64, request batch size 8: about 1100.87 docs/s after excluding service startup.

Do not collapse these two rankings into one claim. Batch jobs pay startup; already-running services do not. Current endpoint tuning says Dynamo request batch size 16 is the best tested endpoint point. For in-process vLLM, do not run more `--model-inference-batch-size` sweeps unless the script first adds a real vLLM-stage batching control. Previous Ray Serve HTTP entries remain diagnostic unless their logs verify HAProxy and RayExecutorV2; i23 is the first valid corrected HTTP datapoint.

Latest fractional GPU status:

- `embedding_generation_raydata_fracgpu_f7a9ad75_i16` succeeded: 1,023,449 docs, 262 files, no char caps, 16 workers, 0.249 GPU per worker, vLLM `gpu_memory_utilization=0.22`, 374.61s, 2732.02 docs/s.
- `embedding_generation_xenna_fracgpu_f7a9ad75_i16` is invalid/incomplete. It was killed before writing metrics; logs reached 94/262 VLLM blocks.
- `embedding_generation_xenna_fracgpu_9453a1a9_i17` succeeded: 1,023,449 docs, 262 files, no char caps, 16 workers, 0.249 GPU per worker, vLLM `gpu_memory_utilization=0.22`, 357.20s, 2865.16 docs/s. Stage sums: VLLM stage process 3887.66s, embedding 3664.67s, tokenization 144.57s.

Latest corrected Ray Serve HAProxy status:

```bash
embedding_generation_ray_serve_haproxy_http_d97e3165_i23
```

Completed corrected HAProxy entries:

```text
embedding_generation_ray_serve_haproxy_http_db6443b4_i20
embedding_generation_ray_serve_haproxy_handle_db6443b4_i21
embedding_generation_ray_serve_haproxy_handle_4fb16baf_i22
embedding_generation_ray_serve_haproxy_http_d97e3165_i23
```

The i20 HTTP entry failed after startup, before throughput:

- HAProxy was enabled and started successfully (`HAProxy is enabled in ServeController`; packaged `ray-haproxy` binary).
- Endpoint was ready with 65.8s startup.
- All four vLLM replicas used `vllm.v1.executor.ray_executor_v2.RayExecutorV2`.
- The Ray Data client then hit OpenAI `InternalServerError`; the OpenAI ingress logged `Too many open files` from gRPC socket creation.
- This is failure evidence for 16 HTTP clients * 64 concurrent requests = 1024 aggregate in-flight requests. Do not treat it as a throughput datapoint.

The i23 HTTP entry intentionally reran the i20 geometry with only Docker's fd limit raised:

- `benchmarking/tools/run.sh` passes `--ulimit nofile=1048576:1048576` to `docker run`.
- Docker HostConfig and container PID 1 both showed `nofile=1048576`.
- Live fd counts reached about 1552 for HAProxy and 1057 for one Serve replica, which explains why the default fd cap could fail.
- Server/client geometry remains 4 replicas, 16 HTTP clients, 64 concurrent requests per client, 1024 aggregate max in-flight requests, request batch size 8.
- Endpoint payload remains token_ids/base64, no character caps, model-context token truncation 2048.
- Logs proved HAProxy enabled/started and all four vLLM replicas used `vllm.v1.executor.ray_executor_v2.RayExecutorV2`.
- Metrics confirmed `max_chars=null`, `endpoint_max_chars=null`, `endpoint_client_mode=tasks`, `pretokenized=true`, `endpoint_pretokenized=true`, 262 input files, and 1,023,449 documents.
- End-to-end: 1001.71s, 1021.70 docs/s.
- Startup: 72.04s.
- Post-startup service rate: `1023449 / (1001.7103 - 72.0361) = 1100.87 docs/s`.
- Ray Data stage execution time: 893.58s.
- Main embedding client stage sum: 14056.81s process time, 13560.74s endpoint embedding time, 205.87s endpoint tokenization time, 128,118 endpoint requests.
- The run emitted repeated non-fatal `httpx.AsyncClient.aclose()` cleanup warnings on closed TCP transports.

The i21 direct-handle entry also failed after startup, before throughput:

- HAProxy was enabled and endpoint was ready with 62.3s startup.
- All four vLLM replicas used RayExecutorV2.
- The benchmark script incorrectly did `async for` over `handle.embeddings.remote(request)`, which returned a non-streaming `DeploymentResponse`.
- Commit `4fb16baf` fixed this by using `handle.options(method_name="embeddings", stream=True).remote(request)`.

The i22 corrected direct-handle entry succeeded:

- 1,023,449 docs, 262 input files.
- No character caps: `max_chars=null`, `endpoint_max_chars=null`.
- `pretokenized=true`, `endpoint_pretokenized=true`.
- `endpoint_input_format=token_ids`, `endpoint_encoding_format=base64`, `endpoint_client_mode=ray_handle`.
- 4 replicas, 16 handle clients, 64 concurrent requests per client, 1024 aggregate max in-flight requests, request batch size 8.
- Endpoint token truncation resolved to 2048 model-context tokens.
- HAProxy was enabled and started successfully (`HAProxy is enabled in ServeController`; packaged `ray-haproxy` binary).
- All four vLLM replicas logged `vllm.v1.executor.ray_executor_v2.RayExecutorV2`.
- End-to-end: 508.83s, 2011.39 docs/s.
- Startup: 74.83s.
- Post-startup service rate: `1023449 / (508.8279 - 74.8292) = 2358.18 docs/s`.
- Ray Data stage execution time: 397.53s.
- Main embedding client stage sum: 6079.54s process time, 6053.78s endpoint embedding time, 180.03s endpoint tokenization time, 128,118 endpoint requests.
- The run showed repeated Ray Serve queue-length deadline warnings from the direct-handle router. This points to Serve router/scheduler/backpressure overhead, not HTTP ingress overhead.

Current corrected Ray Serve conclusion: direct handle is the faster Ray Serve client path, but it is still slower than fractional Xenna, fractional Ray Data, and Dynamo batch16. Raising Docker `nofile` makes Ray Serve HTTP+HAProxy complete at the i20 pressure point, but HTTP remains much slower than direct handle: about 1100.87 docs/s post-startup for HTTP versus about 2358.18 docs/s post-startup for direct handle.

Latest Dynamo text-input status:

```bash
embedding_generation_dynamo_endpoint_text_45c963d7_i24
```

This entry is prepared to rerun Dynamo HTTP with text input under the corrected benchmark setup:

- No character caps: do not set `--max-chars` or `--endpoint-max-chars`.
- `endpoint_input_format=text`.
- Base64 responses, matching the efficient token-input endpoint response path.
- Request batch size 16, matching the best tested Dynamo token-input run.
- 4 replicas, 16 HTTP clients, 64 concurrent requests per client.
- Model-context token truncation resolves to 2048.
- Same dataset slice: 262 input files, expected 1,023,449 documents.

Why rerun: the old Dynamo text run `embedding_generation_dynamo_endpoint_48c6b49f_i4` failed with `500 Failed to fold embeddings stream`. It used text input, no char caps, and model-context token truncation, but it used float responses and request batch size 8. That failure is diagnostic, not a conclusive proof that Dynamo text input is impossible under the current corrected transport shape. If i24 fails with the same error, inspect Dynamo worker logs around embedding output shape and the pooling patch.

The previous corrected Ray Serve attempts failed before throughput:

- `embedding_generation_ray_serve_haproxy_http_ca31a35e_i18`
- `embedding_generation_ray_serve_haproxy_handle_ca31a35e_i19`

Both logs proved HAProxy enablement (`HAProxy is enabled in ServeController`, `Using HAProxy binary`) but HAProxy crashed because its metrics/stats frontend could not bind `0.0.0.0:9102`. No vLLM RayExecutorV2 class evidence was possible because replicas never became healthy.

Ray Serve requirements before any new Ray Serve result is trusted:

- Curator must enable Ray 2.56 HAProxy using `ray-haproxy`, not the old `haproxy` plus `socat` PATH check.
- Logs must show `HAProxy is enabled in ServeController, replacing Serve proxy with HAProxy.`
- Logs should show the packaged HAProxy binary path or `HAProxyManager` evidence.
- Ray Serve vLLM engine must force/verify `distributed_executor_backend="ray"` and `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`.
- Previous Ray Serve endpoint results are diagnostic only if their logs showed the default Python proxy path or failed before throughput. The i22 direct-handle result is valid because logs verified HAProxy and RayExecutorV2, but direct-handle traffic bypasses HTTP/HAProxy on the client path. The i23 HTTP result is the valid corrected HTTP datapoint because logs verified HAProxy, RayExecutorV2, and successful completion under the raised Docker fd limit.

The fractional memory-utilization cap was intentional because each physical GPU hosted four vLLM engines. Without it, every vLLM worker would use the default memory reservation and likely over-reserve GPU memory.

The synthesis artifact is now:

```bash
benchmarking/embedding_sota_conclusions.md
```

The conclusion file records the two valid rankings: Xenna in-process pretokenized vLLM with 16 fractional workers at 0.249 GPU each is the fastest tested startup-inclusive batch-job path and also the fastest tested persistent-service steady-state path. The old `--model-inference-batch-size` sweeps remain invalid as vLLM batch-size evidence because that argument is ignored by `VLLMEmbeddingModelStage`.

The i6 Ray Serve lower-concurrency run succeeded:

- 1,023,449 docs, 262 input files.
- End-to-end: 1120.43s, 913.45 docs/s.
- Startup: 63.94s.
- Persistent-service steady state excluding startup: about 968.72 docs/s.
- `pretokenized=true`, `endpoint_pretokenized=true`, no char caps, endpoint token truncation 2048, token_ids/base64.

This confirms the old non-HAProxy Ray Serve path was correct at aggregate concurrency 128 but much slower than current in-process and Dynamo runs. Treat it as diagnostic only now that HAProxy is a hard requirement.

The i7 Ray Serve aggregate-concurrency 512 run also succeeded, but it was slower than aggregate 128:

- 1,023,449 docs, 262 input files.
- End-to-end: 1175.05s, 870.98 docs/s.
- Startup: 63.31s.
- Persistent-service steady state excluding startup: about 920.58 docs/s.

Current legacy Ray Serve conclusion: aggregate concurrency 128 was the best tested non-HAProxy point. It is superseded by the corrected i20/i22 HAProxy evidence for ranking decisions.

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

The i10 Xenna in-process run with inert CLI batch value 64 succeeded:

- 1,023,449 docs, 262 input files.
- End-to-end: 437.51s, 2339.28 docs/s.
- `pretokenized=true`, no char caps, in-process `vllm_text_pretokenized`.

Current corrected Xenna conclusion: the run was correct, but it does not show that vLLM batch size 64 is slower because `--model-inference-batch-size` is ignored by `VLLMEmbeddingModelStage`.

The i11 Xenna in-process run with inert CLI batch value 16 succeeded:

- 1,023,449 docs, 262 input files.
- End-to-end: 430.52s, 2377.21 docs/s.
- `pretokenized=true`, no char caps, in-process `vllm_text_pretokenized`.

Current corrected Xenna conclusion: the fastest observed Xenna run is still the i3 run at 2394.53 docs/s, but the batch-size bracketing claim is invalid. Treat i3/i10/i11 as repeated Xenna in-process measurements under the same effective vLLM stage configuration.

The i12 Ray Data in-process run with inert CLI batch value 64 succeeded and remained behind Xenna:

- 1,023,449 docs, 262 input files.
- End-to-end: 461.97s, 2215.38 docs/s.
- `pretokenized=true`, no char caps, in-process `vllm_text_pretokenized`.

Current corrected Ray Data conclusion: the run does not prove that vLLM batch size 64 improves Ray Data. It is another Ray Data in-process measurement with the same effective vLLM stage configuration.

The i13 Ray Data in-process run with inert CLI batch value 128 succeeded:

- 1,023,449 docs, 262 input files.
- End-to-end: 464.15s, 2205.00 docs/s.
- `pretokenized=true`, no char caps, in-process `vllm_text_pretokenized`.

Current corrected Ray Data conclusion: the tested `--model-inference-batch-size` values are inert for vLLM, so do not infer a Ray Data batch-size curve. The best observed Ray Data run still trails the best observed Xenna run by about 7.5%, and the remaining gap is more likely executor scheduling, overlap, or backpressure than local vLLM batch size.

## How To Run

Use Docker through `benchmarking/tools/run.sh`. Do not run the benchmark script directly on bare metal.

This image does not have the benchmark runner as its default Docker command, so use `run.sh --shell` and invoke `python benchmarking/run.py ...` inside the container.

Last i23 launch shape:

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
  --shell "cd /opt/Curator && export USER=root LOGNAME=root RAY_SERVE_EXPERIMENTAL_PIP_HAPROXY=1 && python benchmarking/run.py --config benchmarking/nightly-benchmark.yaml --config benchmarking/local-embedding-endpoint.yaml --session-name embedding-sota-investigation --entries-exact embedding_generation_ray_serve_haproxy_http_d97e3165_i23 --reason ray-serve-haproxy-http-ulimit-d97e3165-i23"
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
- Ray Serve endpoint numbers are not valid unless the logs show HAProxy was actually enabled. In Ray 2.56 this branch must use the `ray-haproxy` package path, not only the old Ray 2.55 system `haproxy`/`socat` check.
- Ray Serve endpoint reruns must also verify that the vLLM engine uses RayExecutorV2.
- Latest Docker validation before i9 confirmed the single YAML has pretokenized in-process entries, token_ids endpoint entries, and no character caps; the script still has the pretokenized default/guard and the Curator vLLM stage token-ID path.
- Latest Docker validation after i13 used `benchmarking/tools/run.sh --shell` with `GPUS=none` and confirmed both active in-process entries are `vllm_text_pretokenized`, no active YAML entry sets `--max-chars` or `--endpoint-max-chars`, and the script still defaults to `vllm_text_pretokenized` with raw `vllm_text` guarded behind `--allow-raw-inprocess-vllm`.
- Latest Docker validation after adding the conclusion file used `benchmarking/tools/run.sh --shell` with `GPUS=none` and confirmed the 11 ranked results match raw `metrics.json`/`results.json`, all ranked runs have 1,023,449 docs and 262 input files, no ranked run has character caps, and the active YAML still has no character caps with pretokenized in-process entries.

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
