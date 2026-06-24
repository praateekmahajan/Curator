# Embedding SOTA Conclusions

Date: 2026-06-24

This is the evidence-backed conclusion for the embedding generation SOTA investigation. Treat the old historical notes as motivation only. The ranking below is based on the corrected, uncapped runs in:

```text
/raid/praateekm/curator-nightly/results/embedding-sota-investigation
```

## Bottom Line

For offline/batch embedding generation where the benchmark owns the full pipeline lifecycle, the fastest tested correct path is:

```text
Xenna in-process vLLM, pretokenized, 16 fractional workers at 0.249 GPU each
2865.16 docs/s end-to-end
```

For an already-running embedding service where startup is amortized outside the measured job, the fastest tested correct steady-state path is:

```text
Xenna in-process vLLM, pretokenized, 16 fractional workers at 0.249 GPU each
2865.16 docs/s
```

Do not collapse startup-inclusive and service-amortized rankings. The endpoint path has a real service startup cost in the benchmark, and the in-process path does not. Previous Ray Serve HTTP results are excluded from ranking evidence because those runs did not enable HAProxy. The corrected Ray Serve direct-handle run is now included; corrected Ray Serve HTTP+HAProxy at 1024 aggregate in-flight requests still failed with file-descriptor pressure and remains unresolved.

## Methodology Gates

The successful ranking evidence below satisfies these gates:

- Dataset slice: `/raid/praateekm/datasets/fineweb-edu-fortified-5m-1280`, `--load-dataset-ratio=0.2`.
- Input-file count: 262 files, satisfying the working rule of at least `64 CPUs * 4 GPU workers = 256` files.
- Documents processed: 1,023,449 for every ranked successful run.
- Model: `google/embeddinggemma-300m`.
- GPUs: 4 GPUs, `device=3,4,5,6`.
- No benchmark-side character cap: `max_chars=null` and `endpoint_max_chars=null` for every ranked run.
- In-process vLLM uses `vllm_text_pretokenized`; raw in-process `vllm_text` is not part of SOTA ranking evidence.
- `--model-inference-batch-size` is ignored by `VLLMEmbeddingModelStage`; the i10-i13 in-process "batch-size" runs are repeated in-process measurements with different inert CLI values, not valid vLLM batch-size tuning evidence.
- Fractional in-process runs use real vLLM-stage controls: `--model-num-workers=16`, `--model-worker-gpus=0.249`, and `--model-gpu-memory-utilization=0.22`.
- Endpoint ranking evidence uses `endpoint_input_format=token_ids` and `endpoint_encoding_format=base64`.
- Endpoint token truncation is model-context token truncation, not character truncation. For these runs, `endpoint_truncate_prompt_tokens=2048`.
- Legacy Ray Serve endpoint runs are excluded until HAProxy is actually enabled and verified in logs. The corrected i22 Ray Serve direct-handle run is ranked because logs verified HAProxy startup and all four vLLM replicas using RayExecutorV2.

Script-level checks that make the metrics trustworthy:

- Endpoint response count and indexes are validated in `benchmarking/scripts/embedding_generation_benchmark.py`.
- Raw in-process vLLM text mode raises unless `--allow-raw-inprocess-vllm` is set.
- The parser default is `--model-variation=vllm_text_pretokenized`.
- The benchmark script logs that `--model-inference-batch-size` is ignored for `vllm_text` and `vllm_text_pretokenized`.
- Endpoint tokenized requests use tokenizer output as token IDs and set `add_special_tokens=False` on the OpenAI-compatible endpoint request.
- Dynamo gets a runtime patch so embeddinggemma returns one pooled embedding vector per document rather than variable-length token embeddings.

Latest invariant check:

```text
Docker validation after i13 used benchmarking/tools/run.sh --shell with GPUS=none and confirmed:
- both active in-process YAML entries are vllm_text_pretokenized
- no active YAML entry sets --max-chars or --endpoint-max-chars
- the script still defaults to vllm_text_pretokenized
- raw vllm_text remains guarded behind --allow-raw-inprocess-vllm
```

## Startup-Inclusive Ranking

This is the ranking to use for batch jobs that start the endpoint inside the benchmark/job.

| rank | path | key setting | time_s | startup_s | docs/s |
|---:|---|---|---:|---:|---:|
| 1 | Xenna in-process vLLM | pretokenized, 16 fractional workers, 0.249 GPU/worker | 357.20 | 0.00 | 2865.16 |
| 2 | Ray Data in-process vLLM | pretokenized, 16 fractional workers, 0.249 GPU/worker | 374.61 | 0.00 | 2732.02 |
| 3 | Xenna in-process vLLM | pretokenized, 4 workers; inert CLI batch value 32 | 427.41 | 0.00 | 2394.53 |
| 4 | Xenna in-process vLLM | pretokenized, 4 workers; inert CLI batch value 16 | 430.52 | 0.00 | 2377.21 |
| 5 | Xenna in-process vLLM | pretokenized, 4 workers; inert CLI batch value 64 | 437.51 | 0.00 | 2339.28 |
| 6 | Ray Data in-process vLLM | pretokenized, 4 workers; inert CLI batch value 64 | 461.97 | 0.00 | 2215.38 |
| 7 | Ray Data in-process vLLM | pretokenized, 4 workers; inert CLI batch value 128 | 464.15 | 0.00 | 2205.00 |
| 8 | Ray Data in-process vLLM | pretokenized, 4 workers; inert CLI batch value 32 | 475.95 | 0.00 | 2150.33 |
| 9 | Dynamo endpoint | token_ids/base64, request batch 16 | 496.51 | 89.64 | 2061.27 |
| 10 | Dynamo endpoint | token_ids/base64, request batch 32 | 501.20 | 92.83 | 2041.99 |
| 11 | Dynamo endpoint | token_ids/base64, request batch 8 | 504.61 | 92.19 | 2028.21 |
| 12 | Ray Serve direct handle | HAProxy enabled, RayExecutorV2, token_ids/base64, request batch 8 | 508.83 | 74.83 | 2011.39 |

Conclusion: use Xenna in-process pretokenized vLLM with fractional GPU workers for offline batch embedding in Curator. The speedup comes from real worker geometry, not `--model-inference-batch-size`: four vLLM engines per physical GPU overlap tokenization and embedding work enough to keep the GPU busier.

## Persistent-Service Ranking

This is the ranking to use if endpoint service startup is amortized elsewhere. For endpoints, steady-state docs/s is:

```text
num_documents_processed / (time_taken_s - serve_startup_s)
```

For in-process runs, this equals the startup-inclusive value because `serve_startup_s=0`.

| rank | path | key setting | steady-state docs/s |
|---:|---|---|---:|
| 1 | Xenna in-process vLLM | pretokenized, 16 fractional workers, 0.249 GPU/worker | 2865.16 |
| 2 | Ray Data in-process vLLM | pretokenized, 16 fractional workers, 0.249 GPU/worker | 2732.02 |
| 3 | Dynamo endpoint | token_ids/base64, request batch 16 | about 2515.39 |
| 4 | Dynamo endpoint | token_ids/base64, request batch 32 | about 2506.17 |
| 5 | Dynamo endpoint | token_ids/base64, request batch 8 | about 2481.61 |
| 6 | Xenna in-process vLLM | pretokenized, 4 workers; inert CLI batch value 32 | 2394.53 |
| 7 | Xenna in-process vLLM | pretokenized, 4 workers; inert CLI batch value 16 | 2377.21 |
| 8 | Ray Serve direct handle | HAProxy enabled, RayExecutorV2, token_ids/base64, request batch 8 | about 2358.18 |
| 9 | Xenna in-process vLLM | pretokenized, 4 workers; inert CLI batch value 64 | 2339.28 |
| 10 | Ray Data in-process vLLM | pretokenized, 4 workers; inert CLI batch value 64 | 2215.38 |
| 11 | Ray Data in-process vLLM | pretokenized, 4 workers; inert CLI batch value 128 | 2205.00 |
| 12 | Ray Data in-process vLLM | pretokenized, 4 workers; inert CLI batch value 32 | 2150.33 |

Conclusion: even in the service-amortized view, the fastest current measured path is still in-process fractional Xenna. Dynamo token_ids/base64 request batch 16 remains the best tested endpoint point, but it is now behind fractional in-process vLLM.

## Why The Ranking Looks Like This

In-process pretokenized vLLM is the best batch-job path because it avoids endpoint transport overhead: HTTP, OpenAI client serialization, request routing, endpoint scheduler overhead, response decoding, and service startup. Pretokenization is essential because raw vLLM text mode can make tokenizer work dominate throughput and can give misleadingly slow in-process results.

Fractional in-process workers are the biggest confirmed improvement so far. With 16 actors over 4 GPUs, each physical GPU runs four vLLM engines. That raises throughput from the best old Xenna four-worker result at 2394.53 docs/s to 2865.16 docs/s. The VLLM stage stayed busy enough that the summed actor idle time was effectively zero, while summed embedding time dominated summed tokenization time by about 25x.

Xenna beats Ray Data for the in-process path under the tested geometry. Both use the same model, same 262-file slice, same four model workers, no character cap, and pretokenized vLLM. The apparent Xenna and Ray Data "batch-size" sweeps do not prove anything about vLLM batch size because `--model-inference-batch-size` is not passed to `VLLMEmbeddingModelStage`. Treat those runs as repeated measurements with different inert CLI values. The remaining Xenna/Ray Data gap is therefore more likely executor scheduling, overlap, or backpressure than local vLLM batch-size tuning.

Dynamo was previously able to beat the four-worker in-process path only in the persistent-service view. The corrected Dynamo path uses token IDs, base64 responses, high aggregate concurrency, request batching, model-context token truncation, and the pooling patch. With fractional in-process workers, Dynamo no longer leads either current ranking, but it remains the best validated endpoint path.

Ray Serve direct handle is now valid evidence but not a leader. The corrected i22 run used token IDs, base64 responses, no character caps, HAProxy enabled, and RayExecutorV2 on all four replicas. It reached 2011.39 docs/s startup-inclusive and about 2358.18 docs/s after excluding 74.83s of startup. That places it behind Dynamo batch16 by about 6.3% in the persistent-service view and behind fractional Xenna by about 17.7%. Because the client bypassed HTTP but still showed repeated Ray Serve queue-length deadline warnings, the remaining overhead is likely Serve router/scheduler/backpressure plus per-request service boundaries, not OpenAI HTTP ingress alone.

Ray Serve HTTP is still unresolved, not ranked. Corrected HTTP+HAProxy at 1024 aggregate in-flight requests failed after startup with OpenAI `InternalServerError` and ingress `Too many open files`, even though HAProxy and RayExecutorV2 were verified. The next HTTP rerun should change exactly one pressure variable, probably aggregate concurrency or fd limits.

## Evidence Excluded From Ranking

- The capped i2 in-process results are directional only because `max_chars=1500` is not representative of real documents.
- Raw-text endpoint failures are not ranked because they were not stable/correct for uncapped documents.
- Failed endpoint runs with no successful throughput are useful diagnostics but not ranking evidence.
- Prior Ray Serve HTTP successes are excluded because the logs did not show HAProxy enabled. The corrected i22 direct-handle run is included; corrected HTTP+HAProxy high-concurrency failure remains diagnostic only.
- Historical standalone and one-GPU notes are used for hypotheses only. The ranking above comes from the corrected four-GPU Curator benchmark session.
- The in-process vLLM batch-size interpretation from the first conclusion revision is excluded. Code inspection showed the argument only affects `EmbeddingCreatorStage`, not `VLLMEmbeddingModelStage`.

## Next Useful Experiments

The current fastest tested path is fractional Xenna in-process vLLM. Corrected Ray Serve direct handle has been tested and is slower. The remaining Ray Serve gap is HTTP+HAProxy at a pressure point that does not hit file-descriptor failure.

If we want to push beyond the current result:

- Rerun Ray Serve HTTP with HAProxy verified while changing only one pressure variable from the failed 1024 aggregate in-flight run.
- Test more or fractional in-process vLLM actors per GPU to keep engines fed while other actors tokenize.
- Do not run more `--model-inference-batch-size` sweeps for in-process vLLM unless the script first adds a real vLLM-stage batching control.
- Treat Dynamo request batch 16 as the best tested endpoint point unless a new transport or payload shape changes the bottleneck.
