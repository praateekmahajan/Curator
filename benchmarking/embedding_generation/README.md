# Manifest embedding array

Copy `paths.yaml.example` to ignored `paths.yaml`, fill in the paths, and create a private metadata mapping using `metadata_mapping.json.example` as its schema. Keep the real mapping outside the repository and set `metadata_mapping_path` to it.

Prepare the runtime manifest once. The source-path mapping is used only here; array jobs read `dedup_path` and deterministic row-ID ranges directly from the enriched manifest.

```bash
python -m benchmarking.embedding_generation.prepare_manifest \
  --input-manifest=/path/to/inventory.jsonl \
  --path-mapping=/path/to/dataset_path_mapping.json \
  --output-manifest=/path/to/runtime-manifest.jsonl \
  --output-id-registry=/path/to/id_generator.json
```

The compatible ID registry also contains `id_lookup` records with the physical file, manifest index, ID range, and row count. Resolve an ID to its file by range and calculate `row_offset = id - id_start`. Runtime jobs do not need the historical ID registry, container mount aliases, or path mapping.

`MetadataExtractor` resolves one `mapping_names` entry per input file and broadcasts its configured integer provenance/ranking values onto every row. Stable source IDs and mutable policy ranks remain in the private mapping, while the generic stage and example schema contain no dataset-specific policy. Pairwise ranking can sort `source_priority`, `quality_rank`, and `recency_rank` descending, followed by the dedup ID ascending.

When `text_extraction` is configured, an existing scalar `text` column is preserved. If it is absent, the stage keeps string blocks from `<content_field>.content`, ignores non-text items, and creates `text` using either a fixed separator or whitespace-aware smart merging. The extractor preserves other input columns until the writer selects the requested output schema.

Production Parquet contains only `_curator_dedup_id`, `embeddings`, and configured integer metadata. Source-specific IDs and payload columns are intentionally excluded.

Build the private smoke manifest from the enriched runtime manifest. It selects the smallest files and writes explicit shard assignments: 16 files from the first family, 16 from the second, and an 8+8 mixed shard.

```bash
python -m benchmarking.embedding_generation.prepare_smoke_manifest \
  --input-manifest=/path/to/runtime-manifest.jsonl \
  --metadata-mapping=/path/to/metadata_mapping.json \
  --output-manifest=/path/to/smoke-manifest.jsonl \
  --first-family-id=0 --second-family-id=1 --files-per-shard=16
```

Smoke test three logical shards. The smoke YAML enables `--keep-text`, so output contains the generated ID, text used by the embedder, embedding, and integer metadata.

```bash
export ARRAY_RUNTIME_ROOT=/path/to/runtime
export ARRAY_LOG_DIR=/path/to/logs
mkdir -p "$ARRAY_RUNTIME_ROOT" "$ARRAY_LOG_DIR"
TOTAL_SHARDS=3 sbatch --array=0-2 \
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
