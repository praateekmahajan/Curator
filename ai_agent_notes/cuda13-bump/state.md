# Curator cu13 / RAPIDS 26.04 / vLLM 0.20 / torch 2.11 bump — state

Resumption notes for the CUDA 13 migration work. Last updated 2026-05-07.

## Goal

Migrate NeMo Curator from cu12.9 to cu13. Coordinated dep bump:
- CUDA toolkit 12.9.1 → **13.1.2** (host base image; wheels target CTK 13.0)
- RAPIDS cu12 25.10.* → **cu13 26.04**
- vLLM 0.18 / 0.14.1 / 0.13 → **0.20**
- torch 2.10 → **2.11.0+cu130**
- transformers 4.57 → **5.x**, huggingface-hub 0.34 → **1.x**

EPIC backing this work: [rapidsai/build-infra#346](https://github.com/rapidsai/build-infra/issues/346) (nvJitLink + JIT-LTO + PyTorch wheel compatibility).

Slack thread context: Ayush Dattagupta (UBDSVEQPL) Apr 29 in #swrapids-build (excluded by ECS, content was pasted in conversation).

## Branch + commits

- Branch: **`praateek/cuda13`** off `praateek/dynamo-110`
- Commit pushed: `ae90d2a8 Bump cu12 stack to cu13: RAPIDS 26.04, vLLM 0.20, torch 2.11`
- Pushed to: `git@github.com:praateekmahajan/Curator.git` (origin)
- Uncommitted on top of `ae90d2a8` (as of pause): ffmpeg fix, scale_npp→scale_cuda, PR #1889 ports (4 files + 3 tests), PR #1625 ports (3 files + 1 test). All applied to working tree but not yet committed.

## Phase 1 — Empirical proof (synthetic pyproject)

Two test repros under `/raid/praateekm/tmp_claude/cuda13-repro/` (GA) and `cuda13-repro-nightly/` (RAPIDS 26.06 nightly).

**Single load-bearing finding**: numba pin clash is the only metadata blocker.
- `cudf/cuml/rmm` `dependencies.yaml` cap `numba<0.65.0`
- vLLM 0.20 `requirements/cuda.txt` pins `numba==0.65.0` exactly
- Empirically `numba==0.65.1` works fine in both stacks

Verified:
- Both `cuda13-repro:latest` (RAPIDS 26.04 GA) and `cuda13-repro-nightly:latest` (RAPIDS 26.06.0a304) install + import cleanly inside `nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04`.
- EPIC #346 Option 2a (RAPIDS builds against CTK 13.0, nvjitlink floor `>=13.0,<14`, torch 2.11+cu130 brings nvjitlink 13.0.88) is delivering as documented.

## Phase 2 — Curator pyproject changes

Single sed-pass renames (cu12→cu13, cuda120→cuda130 via substring matching):
- All `*_cuda12` extras → `*_cuda13`
- `nemo_curator[cuda12]` → `nemo_curator[cuda13]`
- `cudf-cu12 → cudf-cu13`, etc. for all RAPIDS packages
- `nvidia-dali-cuda120` → `nvidia-dali-cuda130`
- `cvcuda_cu12` → `cvcuda_cu13>=0.16.0`
- `nixl-cu12` → `nixl-cu13>=1.0`

Surgical version bumps:
- RAPIDS `25.10.*` → `>=26.4,<26.5`
- vllm `>=0.14.1` / `>=0.13` → `>=0.20` (and dropped `vllm<0.19` line in `inference_server`)
- `cuda-python>=12.3` → `>=13.0`
- pytorch index URL `cu129` → `cu130`

Caps dropped (incompatible with target stack):
- `vllm<0.19` (inference_server)
- `flash-attn<=2.8.3` → `flash-attn` (kept, cap dropped — Curator code does not actually `import flash_attn` but vLLM picks it up at runtime)
- `torch<=2.10.0` (video extra + build group)
- `scikit-learn<1.8.0` cuml-25.10 cap → `>=1.5`
- `transformers>=4.56.0,<5.0` (constraint dep)

Override dependencies — net change:
- DROPPED: `torch==2.10.0`, `torchaudio==2.10.0`, `torchvision==0.25.0`, `torchcodec~=0.10.0`, `huggingface-hub>=0.34,<1.0`
- ADDED: `numba>=0.65,<1`, `transformers>=5.0,<6`, `huggingface-hub>=1.0,<2`, `torch>=2.11,<2.12` + matching torchaudio/torchvision

Dockerfile: `CUDA_VER=12.9.1` → `13.1.2`.

CVE pins audit: 5 of 11 still load-bearing (resolved exactly at floor) — kept all.

## Phase 3 — Source ports

### From PR #1889 (transformers 5 / hf-hub 1)
- `nemo_curator/stages/text/models/tokenizer.py` — `batch_encode_plus` → `tokenizer(...)`
- `nemo_curator/stages/text/embedders/vllm.py` — same (the vLLM `tokenization_kwargs` part was already on this branch via prior dynamo work)
- `nemo_curator/stages/text/io/writer/megatron_tokenizer.py` — same
- `nemo_curator/stages/synthetic/nemo_data_designer/data_designer.py` — added `__deepcopy__` (works around hf-hub>=1.0's `DuckDBPyConnection` caching breaking Xenna pipeline_spec deepcopy)
- Matching test fixture updates: `tokenizer.batch_encode_plus = mock` → `tokenizer.side_effect = mock`; `hasattr(..., "batch_encode_plus")` → `callable(...)` in 3 test files

### From PR #1625 (RAPIDS 26.04 API)
- `nemo_curator/stages/deduplication/semantic/kmeans.py` — `cumlKMeans` → `KMeansMG` (cuml dropped public `handle=` kwarg in PR rapidsai/cuml#7751; MG class retains it for multi-gpu comms); `self.kmeans._fit(...multigpu=True)` → `self.kmeans.fit(...)`
- `nemo_curator/stages/deduplication/fuzzy/connected_components.py` — `symmetrize=False` → `symmetrize=True` (cugraph 26.04 graph-build flag)
- `nemo_curator/stages/deduplication/shuffle_utils/rapidsmpf_shuffler.py` — full rewrite to rapidsmpf 26.04 API: `rapidsmpf.buffer.*` → `rapidsmpf.memory.*`, `BaseShufflingActor` → `RapidsMPFActor`, statistics now owned by communicator/actor (not Shuffler), `mr/br` setup moved into `__init__` from `setup_worker`
- `tests/stages/deduplication/semantic/test_kmeans.py` — `stage.kmeans._fit` → `stage.kmeans.fit` mock rename

### Dockerfile/install fixes (not from any PR)
- `docker/common/install_ffmpeg.sh` — added `--nvccflags="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -O2"` (CUDA 13 fully removed compute_30 default that ffmpeg's configure used); dropped `--enable-libnpp` (CUDA 13 removed legacy non-`_Ctx` NPP functions; ffmpeg maintainer declared NPP filters deprecated, recommended `scale_cuda`)
- `nemo_curator/stages/video/clipping/video_frame_extraction.py` — `scale_npp=W:H` → `scale_cuda=W:H`
- `tests/stages/video/clipping/test_video_frame_extraction.py` — corresponding test expectation update

## Phase 4 — Test results (in Docker container `curator-cuda13:latest`)

Container needed because host driver is cu12.2 (per `~/.claude/.../memory/reference_cuda_driver.md`); container has `cuda-compat` libs that let cu13 wheels JIT against cu12.2 driver on data-center GPUs.

Run pattern: `docker run --rm --gpus '"device=X,Y"' -v /raid/praateekm/NeMo-Curator:/opt/Curator -w /opt/Curator curator-cuda13:latest bash -c "source /opt/venv/env.sh && pytest tests/stages/<modality>/ --durations=0"`

Logs: `/raid/praateekm/tmp_claude/cuda13-repro/test-logs/*-docker.log`

| Suite | Passed | Failed | Errors | Notes |
|---|---|---|---|---|
| audio | 330 | 0 | 0 | green incl. whisperx_vad despite torch>=2.11 override |
| image | 71 | 0 | 0 | green |
| math | 85 | 0 | 0 | green |
| translate | 156 | 0 | 0 | green (`tests/stages/text/experimental/translation/`) |
| video | 426 | 0 | 0 | green (incl. fixed scale_cuda test) |
| text | 765 | 6 | 8 | 6× Aegis/instruction_data_guard PEFT classifier; 8× semantic dedup `kmeans handle=` (fixed in PR1625 port — re-run pending) |
| core | 156 | 2 | 12 | see open issues #2 + #3 below |
| dedup | 122 | 0 | 0 | green after PR #1625 ports (was 55F/11E pre-fix) |

## Open follow-ups (priority order)

### 1. Port PR #1889's `DYNAMO_VLLM_RUNTIME_ENV` changes
Source: PR #1889 description, table row 4. **Not** in the user's "4 critical files" list, but causes 6 dynamo integration test failures.

Symptom: `nemo_curator.core.serve.subprocess_mgr.SubprocessError: Dynamo Dynamo_DP0_SmolLM2-135M-Instruct subprocess exited unexpectedly`. Log tail truncated at `vllm/model_executor/layers/rotary_embedding/common.py" line ....<3 lines>...`.

PR #1889 rationale (verbatim): "Dynamo's bundled vLLM pins `flashinfer-python==0.6.3` but not the cubin package → pin `flashinfer-cubin==0.6.3`; `--reinstall-package flash-attn` + `--no-build-isolation-package` for ABI-correct rebuild against actor-local torch; `uvloop<0.22`".

Files to look at on PR #1889 head SHA `bump-rapids-2602` from praateekmahajan/praateek/ray-255 (PR 1889 head):
- `nemo_curator/core/serve/dynamo/backend.py`
- `nemo_curator/core/serve/dynamo/vllm.py`
- `nemo_curator/core/constants.py`

For vLLM 0.20 specifically: verify which flashinfer version vLLM 0.20 actually bundles (may differ from 0.6.3). Adapt the cubin pin accordingly.

Also: Curator's subprocess manager truncates log_tail to last 50 lines. Bump that limit (or capture full log to disk) to get root-cause visibility before guessing.

### 2. `vllm.entrypoints.pooling.score` import path change
4× test errors in `tests/core/serve/ray_serve/test_integration.py` from `ModuleNotFoundError: No module named 'vllm.entrypoints.pooling.score'`. The module was moved/renamed in vLLM 0.20.

Find new path in `/raid/praateekm/vllm/vllm/entrypoints/` (vllm tag `v0.20.1` is already locally fetched). Update Curator import, or guard with version-aware fallback.

Affects `nemo_curator/core/serve/ray_serve/...` (need to grep).

### 3. RAPIDS team feedback — numba ceiling
Action: file PR / issue against `rapidsai/cudf`, `rapidsai/cuml`, `rapidsai/rmm` to bump numba ceiling from `<0.65.0` → `<0.66.0` in `dependencies.yaml`. User has direct access to RAPIDS team. Empirical evidence: numba 0.65.1 imports/runs fine in our test image with all RAPIDS modules.

### 4. Aegis / instruction_data_guard classifier failures
6× text test failures in `tests/stages/text/classifiers/test_classifiers.py`:
- `test_aegis_classifier[*-LlamaGuard-Defensive-1.0]`
- `test_aegis_classifier[*-LlamaGuard-Permissive-1.0]`
- `test_instruction_data_guard_classifier[*]`

Both wrap PEFT/LoRA models on top of LlamaGuard. Likely PEFT API change with transformers 5. Not investigated. Lower priority than the runtime-env work.

### 5. Subprocess log truncation
`nemo_curator/core/serve/subprocess_mgr.py` `_raise_subprocess_error` keeps only last 50 lines of log. For dynamo debugging this needs to be either bumped or replaced with full-log-to-disk. The actual root cause of `Engine core initialization failed` is invisible right now.

### 6. Cosmetic lockfile cleanup (optional)
The lockfile has darwin/non-linux entries (e.g. vllm 0.18.1 for darwin) because Curator doesn't set `tool.uv.environments`. Adding:
```toml
[tool.uv]
environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'linux' and platform_machine == 'aarch64'",
]
```
would lean down the lockfile. Match Curator's per-dep markers (already linux-only). Skip if maintenance preference is to keep current pattern.

## Key file paths to know

### Local working tree
- Curator: `/raid/praateekm/NeMo-Curator` on branch `praateek/cuda13`
- Synthetic test repros: `/raid/praateekm/tmp_claude/cuda13-repro/` and `cuda13-repro-nightly/`
- Test logs: `/raid/praateekm/tmp_claude/cuda13-repro/test-logs/*.log`
- PR #1625 file dump: `/raid/praateekm/tmp_claude/cuda13-repro/pr1625-files/`
- PR #1889 file dump: `/raid/praateekm/tmp_claude/cuda13-repro/pr1889-diffs/`

### Related local clones
- `/raid/praateekm/vllm` — on `main`, has `v0.20.0/.1/.2rc0` tags
- `/raid/praateekm/cudf` `/raid/praateekm/cuml` `/raid/praateekm/raft` `/raid/praateekm/cuvs` `/raid/praateekm/rmm` — all on `main`, have `v26.04.00` and `v26.06.00a` tags
- `/raid/praateekm/pytorch` — on `main` (partial clone)
- `/raid/praateekm/ray` — on `master`

### Built artifacts
- Docker image: `curator-cuda13:latest` (25.2 GB)
- Synthetic test images: `cuda13-repro:latest` (17.6 GB), `cuda13-repro-nightly:latest` (17.6 GB)

## Quick resume commands

Get the current state of the branch:
```bash
cd /raid/praateekm/NeMo-Curator
git status
git log --oneline ae90d2a8^..HEAD
```

Re-run a specific test suite in docker (substitute modality):
```bash
docker run --rm --gpus '"device=0,1"' \
  -v /raid/praateekm/NeMo-Curator:/opt/Curator \
  -w /opt/Curator \
  curator-cuda13:latest \
  bash -c "source /opt/venv/env.sh && pytest tests/stages/audio/ --durations=0"
```

Re-resolve uv after pyproject changes (host):
```bash
cd /raid/praateekm/NeMo-Curator
source .venv/bin/activate
uv lock
```

Rebuild docker image after source changes (full rebuild ~15 min):
```bash
cd /raid/praateekm/NeMo-Curator
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t curator-cuda13:latest .
```

## Conversation reference

Prior conversation persisted via Claude Code session. Key user decisions captured:
- Move HF Hub to >=1.0 (don't respect nemo-toolkit's <1.0 cap)
- Drop flash-attn version cap (kept dep, runtime acceleration only)
- Drop scale_npp (replaced with scale_cuda; ffmpeg deprecated NPP filters)
- Keep whisperx with torch>=2.11 override (works in container despite metadata cap)
- CVE pins kept (5 still load-bearing)
- No `tool.uv.environments` constraint (Curator's per-dep markers handle it)
- Use `KMeansMG` over thread-local hack for cuml handle binding (matches PR #1625's approach)
