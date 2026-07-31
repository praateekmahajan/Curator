# Semantic-deduplication benchmarks on Slurm

This directory contains the KMeans experiments that produce centroid-partitioned
embeddings for semantic deduplication. Pairwise experiments should live in
`benchmarking/pairwise/`, but they must follow the same runner, session, entry,
environment, and Slurm conventions documented here.

## Important paths

Use this worktree and its shared uv environment:

```bash
export WORKTREE=/lustre/fs1/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/Curator/.worktree/ray-256-embedding-benchmark
export SANDIA_ROOT=/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron-v4/sandia/users/praateekm
```

All data-bearing outputs, logs, scratch space, caches, and temporary files must
remain under `SANDIA_ROOT`. The benchmark runner owns this layout:

```text
$SANDIA_ROOT/
└── benchmarking/<session-name>/<entry-name>/
    ├── logs/
    │   ├── stdouterr.log
    │   ├── launcher-node-*.log
    │   └── ray-node-*.log
    ├── gpustats-node-*.csv
    ├── params.json
    ├── metrics.json
    ├── results.json
    ├── tasks.pkl
    ├── scratch/
    ├── ray_cluster/
    └── <benchmark outputs>
```

Do not create separate per-entry logs under `$SANDIA_ROOT/logs`. Older
embedding-generation sessions used that location for Slurm-level logs; the
current runner's entry logs belong inside the entry directory.

## Session and entry hygiene

One session represents one experiment series. Reuse that session and add a new
entry for every variation; do not create a new session for every retry.

For Pairwise, create one YAML under `benchmarking/pairwise/` and use monotonically
versioned entry names:

```yaml
entries:
  - name: v1_ray_data_baseline
    # ...
  - name: v2_<short-change-slug>
    # ...
```

Keep the same session name while comparing `v1_$SLUG`, `v2_$SLUG`, and later
entries. Keep `smoke` in the session name until the launcher and workload have
been validated at small scale. Run only the intended entry:

```bash
python benchmarking/run.py \
  --config benchmarking/pairwise/<session-config>.yaml \
  --session-name <session-name> \
  --entries-exact v2_<short-change-slug>
```

Never launch a new attempt over an existing entry directory. Preserve successful
entries. Diagnose a failed entry before retrying and change only the diagnosed
variable. A failed entry may be removed only after explicit user approval and
after confirming that no live Slurm step, Ray process, or writer is using it.
Never delete the whole session when it already contains a successful entry.

## Keep benchmark scripts trivial

An end-user benchmark script should:

1. Parse benchmark-specific arguments.
2. Construct the Curator stages or pipeline.
3. Assume the runner has already made a Ray cluster available.
4. Run the pipeline with the requested Curator executor.
5. Write benchmark-specific `params.json`, `metrics.json`, and `tasks.pkl`.

It must not:

- contain Slurm allocation, `srun`, Ray-head, or Ray-worker logic;
- call `ray.init()` as its normal setup path;
- start its own GPU recorder;
- introduce Docker or SQSH when the worktree `.venv` is sufficient;
- duplicate task-level metrics that `benchmarking/run.py` aggregates from
  `tasks.pkl`.

Use existing resource helpers such as `get_available_cpu_gpu_resources` when
resource discovery is needed. The runner records GPU state before and after the
entry and runs one `GPUStatsRecorder` per node for the full entry lifetime.

For the current Pairwise experiment series, use `RayDataExecutor`; do not use
Xenna. Pairwise receives one task per `centroid=<id>` partition and recursively
reads the Parquet files in that partition.

## Ray and Slurm responsibilities

Use only `benchmarking/run.py`. It auto-detects Slurm, starts one Ray cluster for
each selected entry, starts a GPU recorder on every node, runs the benchmark
script, and tears down Ray before moving to the next entry. Do not invoke a
separate `slurm_run.py` or start Ray manually.

Ray resources in YAML are per node:

```yaml
ray:
  num_cpus: 32
  num_gpus: 4
  enable_object_spilling: false
```

The Slurm allocation and `srun` step may still request all 144 CPUs on a node.
`num_cpus: 32` controls the logical CPUs advertised by each Ray node; it does
not constrain native library threads. The current launcher does not set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, or similar variables. Do not add them
without evidence from a worker-node experiment.

For multi-node runs, launch one Slurm task per allocated node across the full
allocation. Entering a one-node `srun` shell first narrows the Slurm node list
and can make the runner start only a single-node Ray cluster.

## Environment

The login node has no compute. Never run Curator imports, tests, CUDA commands,
benchmarks, or nontrivial data processing there.

Create or update the locked environment only on an unused worker node:

```bash
cd "$WORKTREE"
uv sync --frozen \
  --extra vllm \
  --extra text_cuda12 \
  --extra deduplication_cuda12 \
  --extra linting
```

Do not mutate `.venv` while another job is using it. At workload launch time,
activate the existing environment and run plain `python`, not `uv run`:

```bash
cd "$WORKTREE"
source .venv/bin/activate
case "$(command -v python)" in
  "$WORKTREE"/.venv/bin/python) ;;
  *) echo "Wrong Python environment" >&2; exit 1 ;;
esac
```

The worker step must also expose the CUDA libraries installed in the uv
environment. Derive the site-packages directory from the active interpreter;
do not hard-code a Python minor version:

```bash
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export CUDA_HOME="$SITE_PACKAGES/nvidia/cu13"
export PATH="$WORKTREE/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$SITE_PACKAGES/nvidia/cublas/include:$SITE_PACKAGES/nvidia/cuda_runtime/include:$CUDA_HOME/include"
export LIBRARY_PATH="$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$CUDA_HOME/lib"
```

For a single-node held-allocation step, give Ray a short socket path backed by
the protected task runtime directory. Ray's Unix-domain sockets cannot exceed
107 bytes:

```bash
PROTECTED_TMP="$SANDIA_ROOT/tmp_ai_agent/<task-slug>/runtime/ray-$SLURM_JOB_ID-$SLURM_STEP_ID"
SHORT_TMP="/tmp/pw-$SLURM_JOB_ID-$SLURM_STEP_ID"
mkdir -p "$PROTECTED_TMP"
ln -s "$PROTECTED_TMP" "$SHORT_TMP"
export TMPDIR="$SHORT_TMP"
export RAY_TMPDIR="$SHORT_TMP"
```

Only the short symlink lives in worker-local `/tmp`; its target and all runtime
contents remain in the protected Sandia subtree. Multi-node launches do not
repeat this setup: `benchmarking/runner/slurm_entrypoint.py` creates per-node
short `/tmp/b...` and `/tmp/r...` symlinks into the runner-owned entry scratch
and Ray directories.

Pre-commit and Ruff may run from the login node after the worker-created
environment includes the linting extra. Do not run unit tests for these
benchmark utilities; validate the benchmark directly inside the held Slurm
allocation. Commit only after that worker-node validation succeeds. Keep each
commit granular, scoped, conventional, and exclude unrelated user changes.

## Iteration workflow

Use one held allocation for smoke tests and retries. Its allocation script must
only wait, so a failed step does not destroy the allocation. The current
single-node Pairwise allocation script is:

```text
$SANDIA_ROOT/tmp_ai_agent/semdedup-pairwise-benchmark/scripts/pairwise-single-node-allocation.sbatch
```

Before every submission:

1. State partition, account, QOS, nodes, GPUs, CPUs, memory, wall time, and
   whether the job is held or standalone.
2. Inspect active jobs once.
3. Inspect the exact script being submitted.
4. Confirm the entry output path does not already exist.
5. Record the returned job ID immediately.

Launch a smoke entry as a targeted step in the held allocation:

```bash
srun --jobid="$ALLOCATION_JOB_ID" --overlap \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task=144 --gpus-per-node=4 \
  /usr/bin/env \
  CONFIG_PATH=benchmarking/pairwise/<session-config>.yaml \
  SESSION_NAME=<session-name> \
  ENTRY_NAME=v1_<short-slug> \
  bash "$SANDIA_ROOT/tmp_ai_agent/<task-slug>/scripts/run-pairwise-step.sh"
```

The canonical step script should activate `$WORKTREE/.venv`, verify
`command -v python`, and invoke `benchmarking/run.py`. Keep that script stable
between retries. A disconnected or rejected `srun` does not imply that the
allocation died; check the allocation by job ID and retry only the step.

After the exact command succeeds in the held allocation, use a standalone
`sbatch` script for an unattended run. The batch script should run the same
reviewed step through `srun` and then exit, which releases the nodes
automatically. Do not silently change the environment, container mode, runtime
paths, resource shape, or launch method between smoke and scale runs.

## Pairwise handoff from the validated KMeans runs

The two validated KMeans entries contain the same 3,788,277,347 input rows:

```text
$SANDIA_ROOT/benchmarking/kmeans-mistral-nvidia-category-20260722/
├── nvidia-vs-mistral-en-humanities-fit-auto-3-nodes-v02/
└── nvidia-vs-mistral-en-humanities-fit-auto-6-nodes-v03/
```

The six-node entry is the preferred Pairwise input because its centroids were
fitted on more data:

```text
input_path:
$SANDIA_ROOT/benchmarking/kmeans-mistral-nvidia-category-20260722/nvidia-vs-mistral-en-humanities-fit-auto-6-nodes-v03/kmeans

centroids:
$SANDIA_ROOT/benchmarking/kmeans-mistral-nvidia-category-20260722/nvidia-vs-mistral-en-humanities-fit-auto-6-nodes-v03/centroids/kmeans_centroids.npy
```

Both centroid arrays are valid `(2048, 1024)` `float32` arrays. KMeans output is
partitioned as `centroid=<id>/`, preserves `_curator_dedup_id`,
`source_family_id`, `quality_rank`, and `recency_rank`, and adds
`l2_dist_to_cent` and `cosine_dist_to_cent`. Embeddings are stored as
bit-preserving FP16 in a `uint16` list column; Pairwise should use
`input_embedding_dtype="auto"` unless an experiment intentionally changes it.

Do not modify or delete either KMeans entry. Pairwise output, cache, and scratch
paths must be inside the new Pairwise session entry.

Start Pairwise with a small, representative smoke workload on the held node.
Before scaling, inspect cluster row counts and skew: Pairwise cost is quadratic
in the rows within a centroid, so the largest centroid—not the average
centroid—determines feasibility. Preserve the ranking metadata and use the
intended ranking strategy rather than dropping columns during reads.

## Verification and comparison

A healthy entry must show:

- the intended node and GPU counts in `params.json`;
- one `gpustats-node-*.csv` per node;
- all allocated Ray workers connected before the pipeline starts;
- GPUs idle in the runner's before/after checks;
- the expected input and output paths in `logs/stdouterr.log`;
- `metrics.json`, `results.json`, and `tasks.pkl`;
- a successful subprocess exit and Ray teardown.

Use the run viewer to compare entries inside the same session. Report
human-readable wall times for the benchmark phases. Concurrent lane metrics are
not additive: distinguish union wall time from summed work-seconds. Normalize
scaling comparisons using both wall-clock speedup and node-time, and inspect
individual GPUs and nodes rather than relying only on cluster averages.
