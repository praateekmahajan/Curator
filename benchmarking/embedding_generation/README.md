# Manifest embedding array

Copy `paths.yaml.example` to ignored `paths.yaml`, fill in the paths, and set a protected runtime/log root.

Smoke test two logical shards:

```bash
export ARRAY_RUNTIME_ROOT=/path/to/runtime
export ARRAY_LOG_DIR=/path/to/logs
mkdir -p "$ARRAY_RUNTIME_ROOT" "$ARRAY_LOG_DIR"
TOTAL_SHARDS=2 sbatch --array=0-1 \
  --output="$ARRAY_LOG_DIR/%A_%a.out" --error="$ARRAY_LOG_DIR/%A_%a.err" \
  benchmarking/embedding_generation/submit_embedding_try.sbatch
```

For production, remove `--manifest-max-rows`, set new output/checkpoint paths, then use `TOTAL_SHARDS=348` and `--array=0-347`. `--require-min-files-per-shard=16` only validates that fixed partition; it does not choose `S`.

Retry only unfinished shards, reusing the same checkpoint path:

```bash
python tutorials/slurm/retry_array.py --checkpoint-path "$CHECKPOINT_PATH" --format fields |
while read -r ARRAY OFFSET MINIMUM TOTAL; do
  TOTAL_SHARDS="$TOTAL" SHARD_INDEX_OFFSET="$OFFSET" MINIMUM_SHARD_INDEX="$MINIMUM" \
    sbatch --array="$ARRAY" \
      --output="$ARRAY_LOG_DIR/%A_%a.out" --error="$ARRAY_LOG_DIR/%A_%a.err" \
      benchmarking/embedding_generation/submit_embedding_try.sbatch
done
```
