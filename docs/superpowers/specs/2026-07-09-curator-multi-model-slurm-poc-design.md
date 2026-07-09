# Curator-native multi-model Slurm POC

## Problem

Kiran's translation/evaluation workload currently uses a Big-Iron configuration
to run a 12B translation model and a 31B evaluation model. The desired public
workflow is Curator-native: users should be able to keep multiple inference
models in one Slurm allocation, shard work with a Slurm array, resume from a
checkpoint, and retry incomplete shards without adopting a second execution
engine.

The Big-Iron configuration is useful as an empirical source of model, serving,
prompt, schema, and concurrency settings. It is not the runtime configuration
format for this POC and there will be no generic Big-Iron-to-Curator converter.

## Goals

1. Provide one approachable Curator Python example for the Kiran-shaped
   translation/evaluation workflow.
2. Keep both models in one aggregated Curator `InferenceServer` inside each
   Slurm array task.
3. Reuse Curator's existing `SlurmRayClient`, array sharding, checkpointing,
   completion manifests, and retry helpers.
4. Make client fanout explicit through `C` (stage workers) and per-model `B`
   (Data Designer request concurrency).
5. Provide a public-facing agent skill that teaches an agent to generate or
   adapt this pattern for other Curator pipelines.
6. Keep the implementation confined to tutorial/example and skill files; do
   not add a Curator Slurm orchestration subsystem.

## Non-goals

- Parsing or executing the Big-Iron configuration envelope.
- Adding a `curator slurm` command or a second Curator execution engine.
- Automatically discovering an optimal model-throughput ratio.
- Cross-array shared inference endpoints.
- Disaggregated prefill/decode serving or advanced routing in the first POC.

## Source settings to preserve

The native example will carry over the meaningful values from
`/datasets/praateekm/bigiron_datasets/kiran/en_hi_native_nostruct_g412b.bigiron.json`:

- Translation model: `RedHatAI/gemma-4-12B-it-FP8-Dynamic`, served as
  `google/gemma-4-12B-it`, tensor parallelism 1.
- Evaluation model: `RedHatAI/gemma-4-31B-it-FP8-block`, served as
  `google/gemma-4-31B-it`, tensor parallelism 2.
- `max_num_seqs=256`, `max_model_len=32768`, and the configured speculative
  decoding models.
- The translation/evaluation column definitions, language fields, structured
  `QualityReport` schema, generation parameters, and model aliases.

The Big-Iron source config says `max_parallel_requests=8192`, but Big-Iron's
client patching path replaces that value with
`max_global_concurrent_inference_requests`, whose default is 64. The POC will
use 64 as its initial empirical client limit and make the value configurable.

## Runtime topology

Each array task is an independent serving island:

```text
Slurm array task
  └── SlurmRayClient
       ├── aggregated Dynamo InferenceServer
       │    ├── 12B translation deployment (TP=1)
       │    └── 31B evaluation deployment (TP=2)
       └── Curator pipeline
            JsonlReader → DataDesignerStage → JsonlWriter
```

The inference server owns the GPUs. The Data Designer client stage remains a
CPU stage and sends requests to both model providers through the shared
OpenAI-compatible endpoint. The stage's dependency graph still determines
when evaluation requests can be issued; the POC will measure throughput rather
than assume that both models are fully saturated at every instant.

The initial POC uses one replica of each model by default so that resource
requirements are obvious. Replica counts are exposed as arguments and are
validated against the allocation using the existing inference-server GPU
planning rules:

```text
required GPUs = translation replicas × 1 + evaluation replicas × 2
```

No 1:7 replica ratio is hard-coded because the supplied material describes
that ratio as a hypothesis rather than a measured result. An 8-GPU experiment
can later express a 6:1 TP-weighted layout if measurements justify it.

## C and B defaults

`C` is the number of Curator workers assigned to `DataDesignerStage`, set with
the existing `stage.with_(num_workers=C)` API.

`B` is each model's Data Designer `max_parallel_requests` value. It is a
per-model, per-worker setting, so the approximate request fanout is:

```text
translation fanout ≈ C × B_translation
evaluation fanout  ≈ C × B_evaluation
```

The first run uses:

```text
C = 1
B_translation = 64
B_evaluation = 64
```

This preserves the effective Big-Iron baseline while remaining below the
configured vLLM sequence limit for a single replica. The example will expose
separate values for the two models, even though both default to 64. A compact
tuning matrix in the README will compare `C=1, B=64` with `C=2, B=32` before
testing model-specific ratios. The sequence limit is a serving capacity
constraint, not a throughput guarantee.

## User experience

The primary artifact will be a tutorial entry point with a small `argparse`
interface. Its local shape will be:

```bash
python tutorials/synthetic/multi_model_slurm/kiran_translation_eval.py \
  --input /shared/input \
  --output /shared/output/run-42 \
  --checkpoint /shared/checkpoints/run-42 \
  --client-workers 1 \
  --translation-concurrency 64 \
  --evaluation-concurrency 64
```

The same file will select `SlurmRayClient` when running inside an allocation,
build the native Data Designer configuration, start both model deployments,
and call `pipeline.run(checkpoint_path=...)`. A short submission example will
show how to invoke it under `sbatch --array` with a per-array concurrency
throttle. The existing Curator retry tutorial/helper will be used to discover
and resubmit incomplete shards.

The agent skill will be named for the capability rather than Big-Iron. It will
teach agents to:

- inspect an existing pipeline and extract model/serving settings;
- translate those settings into Curator `InferenceServer` and Data Designer
  providers;
- calculate GPU requirements and `C × B` request fanout;
- generate the existing array/checkpoint/retry command pattern; and
- stop for confirmation before submitting expensive GPU work.

It will explicitly discourage introducing a Big-Iron config converter or a new
Curator launcher.

## Compatibility and validation

The worktree is based on Curator `upstream/main`, which contains the existing
Slurm-array and resumability implementation. The example will use the
repository's supported Data Designer dependency and fail early with a useful
message if the runtime lacks the inference-server extras or has an incompatible
Data Designer version.

Validation will cover:

1. Native configuration construction and schema/provider validation without a
   Slurm allocation.
2. A mocked or CPU-only pipeline path that verifies `C` and `B` are propagated
   to the stage and model configs.
3. Array environment detection and checkpoint path wiring.
4. A dry-run resource description showing tensor-parallel GPU accounting and
   estimated `C × B` fanout.
5. Documentation checks for local execution, Slurm submission, resumability,
   and retry.

No test will require a live cluster or paid GPU submission.
