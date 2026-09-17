# Curator pipeline with a local Dynamo endpoint

The tested pipeline is `JsonlReader` / `ParquetReader` → `AsyncOpenAIClient` →
`JsonlWriter` / `ParquetWriter`, executed by Xenna. The same container hosts
Dynamo and the model. It needs Curator's base and client dependencies, not every
Curator extra or a particular CUDA version in the pipeline environment.

## Images

| Dockerfile | Layout | Local tag | Uncompressed size |
|---|---|---|---|
| `Dockerfile.nightly` | vLLM CUDA 13 base + Dynamo + Curator without extras; one compatible preinstalled Python environment | `nemo-curator:dynamo-vllm-nightly-20260917` | 26.39 GB |
| `Dockerfile.curator` | Full Curator image; preserved `/opt/venv` plus separate `/opt/dynamo-models` serving venv | `nemo-curator:full-dynamo-models-20260917` | 52.10 GB |

Both passed the complete pipeline. Prefer the smaller image for this CPU/API
workflow. The full-image variant is for pipelines needing the existing Curator
image's other dependencies. Its driver uses Python 3.13, Ray 2.57 and CUDA 12.9
Torch; its serving actors use the same Python/Ray versions with CUDA 13 Torch.
The smaller image uses Python 3.12, Ray 2.58 and CUDA 13 Torch throughout.
Sizes exclude model weights and external caches.

`runtime_env={"py_executable": ...}` selects the baked interpreter for Dynamo
workers and the shared frontend. No Ray package install or venv clone is needed.
A separate venv is useful when pipeline and serving dependencies conflict; it is
not necessary merely because the client talks to an HTTP endpoint. The serving
venv still needs Curator without extras for actor bootstrap (including Xenna),
and must match the driver's Python minor version and Ray version.

## Build and run

Build the smaller image from the repository root:

```bash
docker build -f benchmarking/experiments/dynamo-qwen-deepseek/Dockerfile.nightly \
  -t nemo-curator:dynamo-vllm-nightly-20260917 .
```

Set `HF_HOME` to the existing model cache and `SERVING_CACHE` to writable
persistent storage. Use an empty results directory for each run and a free GPU:

```bash
RESULTS_DIR=/raid/praateekm/tmp_ai_agent/qwen-results
mkdir -p "$SERVING_CACHE"/{cuda,triton,vllm} "$RESULTS_DIR"
docker run --rm --init --gpus '"device=0"' --shm-size=4g \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v "$HF_HOME:/hf:ro" -v "$SERVING_CACHE:/cache" \
  -v "$RESULTS_DIR:/results" \
  -v "$PWD/benchmarking/experiments/dynamo-qwen-deepseek/smoke_qwen.py:/smoke_qwen.py:ro" \
  --entrypoint python3 nemo-curator:dynamo-vllm-nightly-20260917 /smoke_qwen.py
```

For the full-image variant, build `Dockerfile.curator` with its tag above, then
use that image with `--entrypoint /opt/venv/bin/python` and
`-e SERVING_PYTHON=/opt/dynamo-models/bin/python`.

The smoke script starts Ray and Dynamo, runs all four JSONL/Parquet input/output
combinations using Curator's existing `BaseSyntheticStage` and `AsyncOpenAIClient`,
and checks the saved answers. `result.json` records interpreter versions,
readiness time and outputs. Ray logs remain in the results directory. Weights
are mounted read-only and downloads are disabled. Stage worker counts are
bounded for this tiny test. These are 8K-context, eager-mode text tests, not
full-context, multimodal or throughput measurements.

## Backend selection

| Model | Cached revision | Requirement |
|---|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `Qwen3_5ForConditionalGeneration`; [recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) specifies Transformers >=5.8. |
| `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | `DeepseekV41ForCausalLM`; [recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) needs the newer nightly architecture implementation and tokenizer/parsers. |

The pinned vLLM image is
`vllm/vllm-openai@sha256:c4392d76e3eec8983fa152651365158cb062e348fd40398963f499d5867b9e28`.
Its build revision is `0bfc7a15d095fe83ecc82b50561a93c177fece2d`; its unusual
reported version is `0.3.1.dev3+g0bfc7a15d`. The full-image variant reads package
versions from this image, then installs the corresponding wheels into a new
venv using the driver's Python. It does not copy a Python 3.12 environment
into a Python 3.13 Ray cluster.

Dynamo `1.5.0.dev20260914` pins an older vLLM, so the experiment overrides that
pin and excludes CUDA 12 NIXL in favor of the CUDA 13 stack. `--no-sources`
prevents Curator's development indexes from changing the serving resolution.
This is an experimental combination: Dynamo logs a missing
`get_kv_cache_group_metadata` method and falls back to its configured block size.
The simple request path passed; KV-aware routing/disaggregated serving is not
validated. Version changes stay here; PR #2422 contains only generic runtime
support and setup guidance.

## Results and hardware limits

Both images passed all four pipeline combinations, preserving both input rows
and producing the expected answers `4` and `8`. Observed server readiness was
109.0 seconds for the lean image and 115.8 seconds for the full image. The tests
ran concurrently on separate GPUs, so these are smoke-test observations rather
than a controlled performance comparison.

The earlier native DeepSeek test used four RTX PRO 6000 Blackwell GPUs (SM120)
with `--engram-config '{"cpu_offload":true}'`. Qwen remained resident on a fifth
GPU. Real DeepSeek weights loaded at 78.72 GiB per rank, with 4.33 GiB available
for KV cache, then attention warmup failed before readiness or generation:

```text
ValueError: SM120 sparse-MLA has no decode kernel for this shape:
num_tokens=16, num_heads=16, topk=128, d_qk=512,
page_block_size=32, model_type=1, extra_topk=0.
```

The model hardcodes SWA page size 32; FlashInfer's SM120 decode dispatch requires
64. CLI block size does not override that constructor. See [issue #56461](https://github.com/vllm-project/vllm/issues/56461)
and the still-open [geometry fix #56509](https://github.com/vllm-project/vllm/pull/56509).
[Issue #56837](https://github.com/vllm-project/vllm/issues/56837) reports another
unsupported top-k shape. Changing the image base or weight loader does not
supply a missing attention kernel. DeepSeek and the combined two-model endpoint
remain unvalidated; this was not an observed GPU out-of-memory failure.

`run_native.sh qwen|deepseek` retains the original native-vLLM reproducer.
Defaults use GPU 0 for Qwen and GPUs 1–4 for DeepSeek. It is diagnostic; the
pipeline example above demonstrates Curator hosting and consuming its endpoint.

## Weight-loading options

ModelExpress's local InstantTensor strategy delegates to vLLM's existing
`instanttensor` loader. Test it without a ModelExpress service by adding
`-e LOAD_FORMAT=instanttensor` to the pipeline command; this sets
`engine_kwargs={"load_format": "instanttensor", ...}`. The experiment's runtime
already contains the package. Loader timing and inference correctness must both
be checked before changing the default.

The InstantTensor run passed all four pipeline combinations on Qwen:

| Lean-image run | Weight-loading step | Model construction + loading | Endpoint ready |
|---|---|---|---|
| Default loader | 23.41 s | 27.99 s | 108.97 s |
| InstantTensor | 5.89 s | 10.16 s | 96.11 s |

Both used the existing local HF cache and GPU 0. The default run overlapped the
full-image test, while InstantTensor ran afterward; cache state and host load
were not controlled. Treat these as observations, not a guaranteed speedup.
InstantTensor does not eliminate Python startup, model initialization, profiling
or warmup (engine initialization still took 29.24 seconds). DeepSeek loading
with InstantTensor has not been validated.

ModelExpress P2P is useful when another compatible replica already holds the
same model: later replicas can receive weights through NIXL. One Qwen replica
and one DeepSeek replica cannot seed each other. For a single cold replica with
weights in `HF_HOME`, first compare native local loaders. Existing host-mounted
compilation caches also avoid needing a cache-distribution service on this node.
See [ModelExpress's path guide](https://github.com/ai-dynamo/modelexpress/blob/main/docs/guides/choose-a-path.md).

## One endpoint

Pass multiple configs to `InferenceServer(models=[...], backend=...)`. Each
model has separate workers/GPU allocations; the request's `model` field selects
one through the shared endpoint. All configs must select the same preinstalled
interpreter because the frontend merges their runtime environments. That merge
does not combine incompatible venvs. Pipeline stages communicate with the
frontend over HTTP, whether their interpreter is shared or separate.
