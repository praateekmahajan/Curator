# Multi-node KMeans benchmark

This benchmark fits KMeans over the existing `nvidia/crawl-n5.5-apr-2025`
embeddings. `benchmarking/run.py` detects a multi-node Slurm allocation, starts
one Ray cluster and one GPU recorder per node for each entry, runs the entry,
and tears the cluster down. Do not start Ray manually or invoke `slurm_run.py`.

## Run the next two-node smoke experiment

Run from the Curator worktree that contains the shared `.venv`:

```bash
export WORKTREE=/lustre/fs1/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/Curator/.worktree/ray-256-embedding-benchmark
cd "$WORKTREE"
```

Before requesting resources, edit `benchmarking/semdedup/kmeans-smoke.yaml`:

1. Copy the latest successful entry and give it a new, unique name.
2. Keep only the new entry enabled.
3. For the next two-node run, use
   `name: crawl-n55-smoke-1792-files` and `--input-file-limit=1792`.
4. Keep `--fit-data-fraction=1`, `--embedding-dim=1024`,
   `--n-clusters=128`, and `--max-samples-per-batch=4096` unchanged.
5. Keep `smoke` in the entry and session names until the full-scale run.

Request an interactive, exclusive allocation with the same per-node resources
declared in the YAML:

```bash
salloc \
  --account=nemotron_n4_pre \
  --partition=batch \
  --qos=interactive \
  --nodes=2 \
  --gpus-per-node=4 \
  --cpus-per-task=144 \
  --mem=0 \
  --time=04:00:00 \
  --exclusive
```

After the allocation is running, enter one allocated worker. Do not run the
benchmarking Python process on the login node:

```bash
srun --jobid="$SLURM_JOB_ID" --overlap \
  --nodes=1 --ntasks=1 --cpus-per-task=1 \
  --pty bash -l
```

On that worker, use the worktree environment directly. Do not use Docker or
`uv run` to launch the workload:

```bash
cd "$WORKTREE"
source .venv/bin/activate
case "$(command -v python)" in
  "$WORKTREE"/.venv/bin/python) ;;
  *) echo "Wrong Python environment" >&2; exit 1 ;;
esac

python benchmarking/run.py \
  --config benchmarking/semdedup/kmeans-smoke.yaml \
  --session-name semdedup-kmeans-2node-smoke-20260720-01 \
  --entries-exact crawl-n55-smoke-1792-files
```

Reuse `semdedup-kmeans-2node-smoke-20260720-01` for this experiment series.
Each attempt must use a new entry name; never run a new attempt over an existing
entry directory. The runner automatically launches one task per allocated node,
starts all Ray workers and GPU recorders, and runs entries sequentially.

## Verify the run

Artifacts are written under the protected Sandia user root:

```text
/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/nemotron-v4/sandia/users/praateekm/
└── benchmarking/<session-name>/<entry-name>/
    ├── logs/stdouterr.log
    ├── logs/ray*.log
    ├── gpustats-node-*.csv
    ├── params.json
    ├── metrics.json
    ├── tasks.pkl
    ├── kmeans/
    ├── centroids/
    └── ray_cluster/
```

A successful run must show all allocated CPUs and GPUs in
`params.json`, preserve `source_family_id`, `quality_rank`, and `recency_rank`
in the KMeans output, and end successfully in `logs/stdouterr.log`. Inspect
`gpustats-node-*.csv` in the run viewer for both aggregate and per-GPU memory;
the hottest GPU, rather than only the cluster sum, determines whether the next
scale increase is safe.

If an entry fails, diagnose it from its logs and preserve the entry. Add a new
uniquely named entry containing only the diagnosed change, then launch that
entry with `--entries-exact`. Do not delete or recreate the whole session.

## Scale guidance

The validated 1,536-file run processed 95,217,984 rows. It peaked at 64.4% of
aggregate memory and 73.3% on the hottest GPU. The measured memory model is:

```text
peak GPU memory ~= 2 * rows * 1024 * 4 bytes + 225 GiB per 8 GPUs
```

Use 1,792 files as the next two-node smoke test. A 2,048-file run is a limit
test: aggregate use is projected near 81%, but the hottest GPU may approach
93%. For the approximately 315-million-row dataset, start with six four-GPU
nodes; five nodes is only a tight lower bound. Update both `--nodes` in the
allocation and the session name when changing node count. The YAML Ray values
remain per-node values (`144` CPUs and `4` GPUs).
