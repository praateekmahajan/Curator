# Native vLLM serving benchmark

Two GPU nodes serve independently: Qwen3.8-27B with DP8/TP1 on one, and
DeepSeek-V4.1-Flash with TP8 and CPU Engram offload on the other. Curator runs
on CPUs on the Qwen node, using `benchmarking/run.py` and the Ray Data executor:

```text
JSONL (curator_question) -> AsyncOpenAIClient -> Parquet
                              |       |
                         Qwen node  DeepSeek node
```

One timestamped session contains three entries: Qwen only, DeepSeek only, and
both models in the same pipeline. Run them sequentially against the same
servers. The shared entry sends a separate HTTP request to each model; each
CPU task waits for both models before taking another batch.

| YAML in `benchmarking/` | Batch size | Clients per model replica | Output cap |
| --- | ---: | ---: | ---: |
| `native-vllm-sweep.yaml` | 16 | 4 | 2048 |
| `native-vllm-throughput-4096.yaml` | 16 | 8 | 4096 |

Reasoning is disabled. Only `curator_question` is sent, never the original
answer. Parquet preserves the source ID, prompt, generated answer, reasoning,
finish reason, token counts and request latency. The sample used here has
8,192 proportionally sampled questions in 128 JSONL files of 64 rows each.
Use already granular files; there is no batch-splitting stage.

## Prepare your cluster

The commands below use Slurm with Pyxis/Enroot, shared storage mounted at the
same paths, and **two nodes with eight GPUs each**. Change the partition,
account, CPU counts and wall time for your site. Leave site NCCL/UCX defaults
alone. All inference, imports, tests and analysis run on workers.

Use an architecture-compatible vLLM nightly SQSH and a Curator image containing
this checkout's dependencies. Qwen requires `transformers>=5.8.0`; provision
that in the serving environment before starting, if necessary. Pin the image
digest/date for reproducibility. Import images on a worker and put the final
SQSH in your persistent Enroot cache, not local `/raid` scratch.

**GB200/GB300:** use ARM64/aarch64-compatible images for Grace-based nodes;
do not copy our x86 H100 SQSH. See [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).
Check how many GPUs Slurm exposes per node. For a different count, change
the allocation and `srun --gres`, export `QWEN_DP`/`DEEPSEEK_TP`, and update
the YAML model `replicas`/`gpus_per_replica` and descriptive entry names to
match. Ensure DeepSeek fits that topology. GPU-hour metrics use the YAML
counts; telemetry discovers GPU IDs rather than assuming eight per node.
This launch bundle has been exercised on H100; GB200/GB300 performance and
the 4096-token configuration still need validation on the target cluster.

Set these once in your shell (use a persistent terminal and share the exported
values with the other terminals below):

```bash
export WORKTREE=/shared/Curator
export TASK_ROOT=/shared/experiments/native-vllm
export RESULTS_PATH=/shared/benchmarking/results
export INPUT_PATH=/shared/question_only_stratified_8k
export TOOLS_CACHE=/shared/tools_cache
export HF_CACHE=$TOOLS_CACHE/hf_cache
export VLLM_IMAGE=/shared/enroot-cache/vllm-nightly.sqsh
export CURATOR_IMAGE=/shared/enroot-cache/curator-nightly.sqsh
export CONFIG=$WORKTREE/benchmarking/native-vllm-sweep.yaml
export SESSION=native-vllm-b16-c4-max2048-$(date +%Y%m%d-%H%M%S)
export BUNDLE=$WORKTREE/benchmarking/native_vllm
mkdir -p "$TASK_ROOT/logs" "$TASK_ROOT/runtime" "$RESULTS_PATH"
export MOUNTS="$WORKTREE:$WORKTREE,$TASK_ROOT:$TASK_ROOT,$RESULTS_PATH:$RESULTS_PATH,$INPUT_PATH:$INPUT_PATH,$TOOLS_CACHE:$TOOLS_CACHE"
```

If `HF_CACHE` lives outside `TOOLS_CACHE`, add it to `MOUNTS` too. Cache roots
for CUDA, Triton and vLLM are set by `worker-env.sh`. Only short Unix socket
paths live under worker-local `/tmp`; outputs and model caches are shared.

## Allocate and start servers

Inspect `sinfo -s`, your partition's access limits, and `squeue -u "$USER"`
before submitting; reuse an existing matching allocation when possible.
This example holds two exclusive eight-GPU nodes for four hours, with
128 CPUs and all node memory each. No explicit QOS is added.

```bash
sbatch --parsable --partition=YOUR_INTERACTIVE_PARTITION --account=YOUR_ACCOUNT \
  --job-name=native-vllm --nodes=2 --ntasks-per-node=1 --gres=gpu:8 \
  --cpus-per-task=128 --mem=0 --exclusive --time=04:00:00 \
  --output="$TASK_ROOT/logs/allocation-%j.out" "$BUNDLE/allocation.sbatch"
# Record the returned ID; wait for RUNNING before starting steps.
export JOB_ID=RETURNED_JOB_ID
scontrol show hostnames "$(squeue -h -j "$JOB_ID" -o %N)"
export QWEN_HOST=FIRST_SHORT_HOSTNAME
export DEEPSEEK_HOST=SECOND_SHORT_HOSTNAME
export TELEMETRY_DIR=$TASK_ROOT/logs/gpu-telemetry-$JOB_ID
```

In separate persistent terminals, keep each `srun` attached. Do not append `&`
in an ephemeral shell. All terminals need the same exported settings.

```bash
# Terminal 1: Qwen; add --load-format instanttensor if installed and supported.
srun --jobid="$JOB_ID" --overlap -N1 -n1 -w "$QWEN_HOST" -c64 --gres=gpu:8 \
  --container-image="$VLLM_IMAGE" --container-mounts="$MOUNTS" \
  --output="$TASK_ROOT/logs/qwen-$JOB_ID.out" --error="$TASK_ROOT/logs/qwen-$JOB_ID.err" \
  bash "$BUNDLE/step.sh" qwen

# Terminal 2: DeepSeek.
srun --jobid="$JOB_ID" --overlap -N1 -n1 -w "$DEEPSEEK_HOST" -c128 --gres=gpu:8 \
  --container-image="$VLLM_IMAGE" --container-mounts="$MOUNTS" \
  --output="$TASK_ROOT/logs/deepseek-$JOB_ID.out" --error="$TASK_ROOT/logs/deepseek-$JOB_ID.err" \
  bash "$BUNDLE/step.sh" deepseek

# Terminal 3: continuous host-side telemetry, one task per serving node.
srun --jobid="$JOB_ID" --overlap -N2 -n2 --ntasks-per-node=1 \
  -w "$QWEN_HOST,$DEEPSEEK_HOST" -c1 --gres=gpu:8 \
  bash "$BUNDLE/record-gpustats.sh"
```

Our H100 nightly needed `DEEPSEEK_H100_WORKAROUND=1` exported before launching
DeepSeek: it disables custom all-reduce and symmetric-memory all-reduce to
avoid an initialization crash. It is **off by default** for other clusters.
There are no vLLM source patches or cluster communication overrides.
The starting sequence limits are Qwen 256 and DeepSeek 64; neither uses
`--enforce-eager`. Check server logs for completed CUDA graph capture and
readiness before running the pipeline; model weight loading alone is not readiness.

For the higher-concurrency series, after finishing the baseline, restart the
servers with `QWEN_MAX_NUM_SEQS=128` and `DEEPSEEK_MAX_NUM_SEQS=128`, and append
`--max-model-len 32768 --max-cudagraph-capture-size 128` to **both** serve steps.
Select `native-vllm-throughput-4096.yaml` and a **new** descriptive session name.
Keep all three entries in that new session. Use new server log names on retries.

## Run the three entries

From a fourth terminal, run this CPU-only driver step. It uses the same session
for every entry, checks both servers, resets prefix caches between entries,
and exports telemetry after each completed entry. Do not run concurrent
benchmarks against these same servers.

```bash
for entry in \
  qwen_dp8_batch16_clients4_per_replica_no_thinking_max2048 \
  deepseek_tp8_batch16_clients4_per_replica_no_thinking_max2048 \
  both_shared_pipeline_batch16_clients4_per_replica_no_thinking_max2048; do
  srun --jobid="$JOB_ID" --overlap -N1 -n1 -w "$QWEN_HOST" -c64 --gres=none \
    --container-image="$CURATOR_IMAGE" --container-mounts="$MOUNTS" \
    --output="$TASK_ROOT/logs/$SESSION-$entry.out" --error="$TASK_ROOT/logs/$SESSION-$entry.err" \
    bash "$BUNDLE/step.sh" benchmark "$entry" || break
done
```

For the 4096-token YAML, use its three `clients8...max4096` entry names instead.
The runner starts the Ray cluster; do not launch a separate inference server
through Curator. If using an existing host environment instead of containers,
activate the worktree's `.venv` on the worker and verify `command -v python`
before the driver step. Keep that launch mode consistent throughout the series.

## Read the Rust frontend logs

With this launch bundle, vLLM's frontend statistics appear in
`$TASK_ROOT/logs/qwen-$JOB_ID.err` and `deepseek-$JOB_ID.err`; check `.out` too
if a different image routes logs there. Follow either server from the login
node without running inference:

```bash
tail -F "$TASK_ROOT/logs/deepseek-$JOB_ID.err" | \
  rg --line-buffered 'RustFrontend.*(Avg prompt tput|Preemptions|ERROR)'
# Inspect initialization separately; confirm graph capture actually finished.
rg 'enforce_eager|cudagraph|Captur|ready|ERROR' "$TASK_ROOT/logs/deepseek-$JOB_ID."{out,err}
```

Example from our running nightly (active statistics arrived about every 10 s):

```text
(RustFrontend pid=3226278) INFO 09-17 15:57:17 [log_stats.rs:273] Avg prompt tput: 672.5 toks/s, Avg generation tput: 1931.6 toks/s, Reqs Running: 57, Waiting: 0, GPU KV cache used: 1.8%, Prefix cache hit rate: 0.0%
```

| Field | Interpretation |
| --- | --- |
| `Avg prompt tput` / `Avg generation tput` | Input/output tokens per second over the last logging interval, not the entire benchmark. This Rust frontend aggregates its engines: Qwen's DP8 value is already the total, not a per-GPU value to multiply by eight. |
| `Reqs Running` / `Waiting` | Current scheduler counts. Repeated drops in running requests with no queue suggest the clients may not keep the server fed. A persistent queue means requests are arriving faster than the server admits them. |
| `GPU KV cache used` | Occupancy of the allocated KV cache, not total GPU memory usage. A small value does not mean model weights or CUDA graphs leave most GPU memory free. |
| `Prefix cache hit rate` | Prefix-cache hits over the interval. Reset caches between entries to avoid giving repeated prompts an unfair advantage. |
| `Preemptions` (when present) | Preemptions during the interval; rising counts warrant checking cache pressure before increasing concurrency. |

For example, our DeepSeek run repeatedly dropped from 64 active requests to
roughly 20–30 with `Waiting: 0`, and output throughput fell alongside it.
Tasks wait for their remaining responses before taking the next file. More
client replicas can overlap these task tails; increasing `max-num-seqs` alone
cannot supply missing requests. Conversely, a queue plus high cache occupancy
is not a reason to blindly increase concurrency.

Compare steady serving intervals with the same time window in `gpustats.csv`.
Frontend timestamps here use the worker's local timezone; the CSV uses UTC.
Exclude loading and idle periods when diagnosing serving, but use `metrics.json`
for end-to-end benchmark throughput, including the final request drain.
GPU busy time near 100% and power below the limit do not by themselves establish
peak throughput or quantify available speedup. Idle frontend statistics may
drop to DEBUG level, so a quiet INFO log alone does not indicate a failed server.

## Results and viewer-compatible GPU stats

Each completed entry is at `$RESULTS_PATH/$SESSION/<entry>/`:

- `parquet/<layout>/`: preserved answers and request metadata; shared output
  contains both models, distinguished by `model_alias`.
- `params.json`, `metrics.json`: settings, elapsed time, per-model throughput,
  average tokens, empty/truncated responses, GPU-hours per million requests,
  and measured per-node GPU power/energy.
- **`gpustats.csv`**: a single CSV in the entry root, with the standard
  `GPUStatsRecorder.HEADER` columns first. Both-model entries combine both nodes
  with unique numeric `gpu_id`s, plus node, local GPU ID and UUID columns.

`record-gpustats.sh` records at 1 Hz on both nodes. The collector clips samples
to the pipeline's start/end timestamps, excluding model loading and excluding
the unused server from single-model entries. Nodes must have synchronized
clocks. `utilization_memory_pct` means allocated memory / total memory, as in
the viewer; `memory_access_busy_pct` is the separate nvidia-smi activity metric.
Fan/process details are not collected (blank fan value and empty process list).
Energy integrates sampled GPU power; it excludes CPU, networking and idle nodes.

To regenerate CSVs after a completed run, use the same CPU driver `srun` above
with `bash "$BUNDLE/step.sh" collect`. It reads the preserved raw files at
`$TELEMETRY_DIR/<short-hostname>/gpustats.csv` and atomically replaces entry CSVs.
Failed/in-progress entries retain raw telemetry but do not get a completed-run
CSV. Keep raw logs until exports are verified. Release the allocation with
`scancel "$JOB_ID"` when finished.

Estimate full-corpus time as `69,100,683 / requests_per_s`. For both answers,
separate jobs cost `Gq*Tq + Gd*Td` GPU-seconds; a shared run costs
`(Gq+Gd)*Tshared`. Compare the same sample and output cap, and inspect truncation
rates. Power and GPU-busy percentages alone do not establish peak throughput.
