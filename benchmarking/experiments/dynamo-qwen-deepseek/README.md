# Qwen / DeepSeek serving environment experiment

This branch builds on PR #2422. Model versions, images and hardware-specific
settings belong here; the base PR only contains generic runtime support and
agent guidance.

## Model requirements

| Model | Cached revision | Serving requirements |
|---|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `Qwen3_5ForConditionalGeneration`; the [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) specifies Transformers >=5.8. |
| `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | `DeepseekV41ForCausalLM`; the [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) requires the post-September-10 nightly implementation, including its tokenizer and reasoning/tool parsers. |

The DeepSeek Dynamo preview release documents SGLang, but that is not a
restriction on native vLLM support. Check both projects' current recipes.
Neither stable Dynamo's vLLM pin nor a model config's `transformers_version`
alone establishes support for an architecture.

## Shared runtime candidate

The pinned vLLM image is
`vllm/vllm-openai@sha256:c4392d76e3eec8983fa152651365158cb062e348fd40398963f499d5867b9e28`.
Its build revision is `0bfc7a15d095fe83ecc82b50561a93c177fece2d`, Python is
3.12, Torch is `2.13.0+cu130`, and Transformers is 5.17.0. The wheel reports
`0.3.1.dev3+g0bfc7a15d`; use the image digest/build revision to identify this
nightly, rather than inferring feature support from its version string.

`Dockerfile.nightly` adds Curator without extras and Dynamo
`1.5.0.dev20260914`. It overrides Dynamo's older vLLM pin with the installed
nightly and preserves the image's Torch stack. This is an experimental
combination, not a published Dynamo compatibility guarantee. `--no-sources`
prevents Curator's development wheel sources from selecting another CUDA stack.
It reuses the image's baked `/usr/bin/python3` environment, selected with
`runtime_env={"py_executable": "/usr/bin/python3"}`; no Ray package install or
environment clone is needed. Driver and actors run in that same image/Python.
The CUDA 13 base supplies NIXL; the override excludes Dynamo's CUDA 12 NIXL
dependency. The resulting local image is 26.39 GB (24.57 GiB), versus 21.57 GB
(20.09 GiB) for the vLLM base: an added 4.81 GB (4.48 GiB), excluding model
weights and persistent caches. These are Docker's uncompressed image sizes.

Build from the repository root:

```bash
docker build -f benchmarking/experiments/dynamo-qwen-deepseek/Dockerfile.nightly \
  -t nemo-curator:dynamo-vllm-nightly-20260917 .
```

## Native vLLM bring-up

Qwen passes with this image. The DeepSeek command currently reproduces an
upstream SM120 attention failure; see validation below.

Set `HF_HOME` to the existing cache and `SERVING_CACHE` to a writable persistent
directory. No model downloads are performed. Run each command in its own shell:

```bash
bash benchmarking/experiments/dynamo-qwen-deepseek/run_native.sh qwen
bash benchmarking/experiments/dynamo-qwen-deepseek/run_native.sh deepseek
```

Defaults use GPU 0 / port 18101 for Qwen and GPUs 1–4 / port 18102 for DeepSeek.
`GPUS`, `PORT`, and `SERVING_IMAGE` override these defaults. The tests start with
8K context, eager execution and text-only requests; they do not establish
full-context, multimodal or production-throughput performance.

The cached weights are 51.75 GiB for Qwen and approximately 475 GiB for
DeepSeek. Seven GPUs are visible, each with about 95.6 GiB usable VRAM.
Total memory alone does not determine a valid tensor-parallel size. DeepSeek's
Engram CPU offload allows a TP4 attempt while leaving room for Qwen; host RAM
is also required for the offloaded tables and loading buffers.
The launcher caps OpenMP/MKL threads to avoid CPU oversubscription across
ranks during weight conversion. An uncapped attempt spent prolonged time in
OpenMP tensor copies; the capped run completed model loading in 132–143 seconds
per rank. The second run also benefited from the OS file cache, so these runs
do not isolate the effect of thread count.

## Validation

- Native Qwen on the pinned image returned `4` for `2 + 2`, with thinking disabled.
- Curator + Dynamo Qwen on the derived image also returned `4`; readiness took
  138.4 seconds in this run. Ray workers and the frontend used the baked Python.
- Native DeepSeek loaded its real weights on GPUs 1–4 with Engram CPU offload
  while native Qwen remained resident on GPU 0. vLLM reported 78.72 GiB of model
  memory per rank and 4.33 GiB available for KV cache. It then failed during
  attention warmup, before readiness or generation:

  ```text
  ValueError: SM120 sparse-MLA has no decode kernel for this shape:
  num_tokens=16, num_heads=16, topk=128, d_qk=512,
  page_block_size=32, model_type=1, extra_topk=0.
  ```

  The installed model code hardcodes SWA `block_size=32`; FlashInfer's SM120
  decode dispatch requires 64. The generic CLI block-size setting does not
  override that constructor. This matches [vLLM issue #56461](https://github.com/vllm-project/vllm/issues/56461).
  [Upstream PR #56509](https://github.com/vllm-project/vllm/pull/56509) proposes
  geometry changes but was still open at testing time and is not applied here.
  [Issue #56837](https://github.com/vllm-project/vllm/issues/56837) also reports
  an unsupported top-k shape and that the FlashMLA alternative requires other
  GPU architectures. Fixing the first error alone does not establish serving
  compatibility. Disabling warmup would not supply the missing kernel.

Thus one installed runtime is a candidate for both architectures, but only
Qwen has passed generation. DeepSeek and the combined two-model Dynamo
endpoint remain unvalidated. This is a kernel compatibility blocker on the
tested RTX PRO 6000 Blackwell (SM120), not an observed GPU out-of-memory error.

To repeat the Curator/Ray integration test from the repository root:

```bash
mkdir -p "$PWD/qwen-results"
docker run --rm --init --gpus '"device=0"' --shm-size=4g \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v "$HF_HOME:/hf:ro" -v "$SERVING_CACHE:/cache" \
  -v "$PWD/qwen-results:/results" \
  -v "$PWD/benchmarking/experiments/dynamo-qwen-deepseek/smoke_qwen.py:/smoke_qwen.py:ro" \
  --entrypoint python3 nemo-curator:dynamo-vllm-nightly-20260917 /smoke_qwen.py
```

Stop any other server on the selected GPU first. `result.json` contains the
readiness measurement and checked completion; the Ray logs remain in the
mounted results directory.

An initial attempt to install Dynamo 1.4.2 / vLLM 0.26 into the older CUDA 12
Curator image resolved successfully but selected CUDA 12 Torch alongside CUDA
13 vLLM, failing GPU import with `libcudart.so.13`. A resolver dry run with
`--no-sources` selected the requested CUDA 13 Torch correctly. Starting from
the serving backend's image preserves its tested CUDA stack; CPU-only import
checks missed the initial problem.

## One endpoint

Curator supports `InferenceServer(models=[...], backend=DynamoServerConfig(...))`.
Each config gets its own model workers and GPU allocation; clients choose the
model through the request's `model` field on the same HTTP endpoint. The shared
frontend's runtime environment is merged from the models. Under PR #2422,
preinstalled configs must select the same interpreter. Separate incompatible
venvs are not combined by that merge; they require separate endpoints or a
separate design for selecting the frontend's environment.
