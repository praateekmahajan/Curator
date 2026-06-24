# Embedding SOTA Conclusions

Date: 2026-06-24

This is the evidence-backed conclusion for the embedding generation SOTA investigation. Treat the old historical notes as motivation only. The ranking below is based on the corrected, uncapped runs in:

```text
/raid/praateekm/curator-nightly/results/embedding-sota-investigation
```

## Bottom Line

For offline/batch embedding generation where the benchmark owns the full pipeline lifecycle, the fastest tested correct path is:

```text
Xenna in-process vLLM, pretokenized, model_inference_batch_size=32
2394.53 docs/s end-to-end
```

For an already-running embedding service where startup is amortized outside the measured job, the fastest tested correct steady-state path is:

```text
Dynamo endpoint, token_ids input, base64 response, endpoint_request_batch_size=16
about 2515.39 docs/s after excluding service startup
```

Do not collapse these into one ranking. The endpoint path has a real service startup cost in the benchmark, and the in-process path does not.

## Methodology Gates

The successful ranking evidence below satisfies these gates:

- Dataset slice: `/raid/praateekm/datasets/fineweb-edu-fortified-5m-1280`, `--load-dataset-ratio=0.2`.
- Input-file count: 262 files, satisfying the working rule of at least `64 CPUs * 4 GPU workers = 256` files.
- Documents processed: 1,023,449 for every ranked successful run.
- Model: `google/embeddinggemma-300m`.
- GPUs: 4 GPUs, `device=3,4,5,6`.
- No benchmark-side character cap: `max_chars=null` and `endpoint_max_chars=null` for every ranked run.
- In-process vLLM uses `vllm_text_pretokenized`; raw in-process `vllm_text` is not part of SOTA ranking evidence.
- Endpoint ranking evidence uses `endpoint_input_format=token_ids` and `endpoint_encoding_format=base64`.
- Endpoint token truncation is model-context token truncation, not character truncation. For these runs, `endpoint_truncate_prompt_tokens=2048`.

Script-level checks that make the metrics trustworthy:

- Endpoint response count and indexes are validated in `benchmarking/scripts/embedding_generation_benchmark.py`.
- Raw in-process vLLM text mode raises unless `--allow-raw-inprocess-vllm` is set.
- The parser default is `--model-variation=vllm_text_pretokenized`.
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
| 1 | Xenna in-process vLLM | pretokenized, batch 32 | 427.41 | 0.00 | 2394.53 |
| 2 | Xenna in-process vLLM | pretokenized, batch 16 | 430.52 | 0.00 | 2377.21 |
| 3 | Xenna in-process vLLM | pretokenized, batch 64 | 437.51 | 0.00 | 2339.28 |
| 4 | Ray Data in-process vLLM | pretokenized, batch 64 | 461.97 | 0.00 | 2215.38 |
| 5 | Ray Data in-process vLLM | pretokenized, batch 128 | 464.15 | 0.00 | 2205.00 |
| 6 | Ray Data in-process vLLM | pretokenized, batch 32 | 475.95 | 0.00 | 2150.33 |
| 7 | Dynamo endpoint | token_ids/base64, request batch 16 | 496.51 | 89.64 | 2061.27 |
| 8 | Dynamo endpoint | token_ids/base64, request batch 32 | 501.20 | 92.83 | 2041.99 |
| 9 | Dynamo endpoint | token_ids/base64, request batch 8 | 504.61 | 92.19 | 2028.21 |
| 10 | Ray Serve endpoint | token_ids/base64, aggregate concurrency 128 | 1120.43 | 63.94 | 913.45 |
| 11 | Ray Serve endpoint | token_ids/base64, aggregate concurrency 512 | 1175.05 | 63.31 | 870.98 |

Conclusion: use Xenna in-process pretokenized vLLM batch 32 for offline batch embedding in Curator.

## Persistent-Service Ranking

This is the ranking to use if endpoint service startup is amortized elsewhere. For endpoints, steady-state docs/s is:

```text
num_documents_processed / (time_taken_s - serve_startup_s)
```

For in-process runs, this equals the startup-inclusive value because `serve_startup_s=0`.

| rank | path | key setting | steady-state docs/s |
|---:|---|---|---:|
| 1 | Dynamo endpoint | token_ids/base64, request batch 16 | about 2515.39 |
| 2 | Dynamo endpoint | token_ids/base64, request batch 32 | about 2506.17 |
| 3 | Dynamo endpoint | token_ids/base64, request batch 8 | about 2481.61 |
| 4 | Xenna in-process vLLM | pretokenized, batch 32 | 2394.53 |
| 5 | Xenna in-process vLLM | pretokenized, batch 16 | 2377.21 |
| 6 | Xenna in-process vLLM | pretokenized, batch 64 | 2339.28 |
| 7 | Ray Data in-process vLLM | pretokenized, batch 64 | 2215.38 |
| 8 | Ray Data in-process vLLM | pretokenized, batch 128 | 2205.00 |
| 9 | Ray Data in-process vLLM | pretokenized, batch 32 | 2150.33 |
| 10 | Ray Serve endpoint | token_ids/base64, aggregate concurrency 128 | about 968.72 |
| 11 | Ray Serve endpoint | token_ids/base64, aggregate concurrency 512 | about 920.58 |

Conclusion: if the embedding service is already warm and reused across jobs, Dynamo token_ids/base64 request batch 16 is the fastest tested steady-state path.

## Why The Ranking Looks Like This

In-process pretokenized vLLM is the best batch-job path because it avoids endpoint transport overhead: HTTP, OpenAI client serialization, request routing, endpoint scheduler overhead, response decoding, and service startup. Pretokenization is essential because raw vLLM text mode can make tokenizer work dominate throughput and can give misleadingly slow in-process results.

Xenna beats Ray Data for the in-process path under the tested geometry. Both use the same model, same 262-file slice, same four model workers, no character cap, and pretokenized vLLM. Ray Data improved when batch size moved from 32 to 64, but batch 128 regressed and its best point still trailed Xenna batch 32. That makes the remaining gap more likely executor scheduling, overlap, or backpressure than local vLLM batch size.

Dynamo can beat in-process only in the persistent-service view. The corrected Dynamo path uses token IDs, base64 responses, high aggregate concurrency, request batching, model-context token truncation, and the pooling patch. With those fixes it reaches the best steady-state rate, but about 90 seconds of startup makes it slower for benchmark-owned batch jobs.

Ray Serve HTTP is not SOTA in the tested form. Aggregate concurrency 1024 failed under transport pressure, aggregate 512 was slower than 128, and the best stable Ray Serve point was still less than half the Dynamo and in-process rates. The likely bottleneck is HTTP ingress/client pressure and request/response overhead rather than embedding math. Ray Serve direct-handle/no-HTTP remains the plausible future Ray Serve experiment.

## Evidence Excluded From Ranking

- The capped i2 in-process results are directional only because `max_chars=1500` is not representative of real documents.
- Raw-text endpoint failures are not ranked because they were not stable/correct for uncapped documents.
- Failed endpoint runs with no successful throughput are useful diagnostics but not ranking evidence.
- Historical standalone and one-GPU notes are used for hypotheses only. The ranking above comes from the corrected four-GPU Curator benchmark session.

## Next Useful Experiments

No further experiment is required to choose the fastest tested path for the current benchmark scope.

If we want to push beyond the current result:

- Test Ray Serve direct-handle/no-HTTP endpoint path.
- Test more or fractional in-process vLLM actors per GPU to keep engines fed while other actors tokenize.
- Only map a Ray Data batch midpoint such as 96 if we need a more precise Ray Data curve; current evidence is already enough to rank it below Xenna.
- Treat Dynamo request batch 16 as the best tested endpoint point unless a new transport or payload shape changes the bottleneck.
