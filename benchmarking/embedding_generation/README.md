# Manifest embedding array

Copy `paths.yaml.example` to ignored `paths.yaml`, fill in the manifest, path mapping, current ID registry, and output paths, then create a private metadata mapping using `metadata_mapping.json.example` as its schema. Keep the real mapping outside the repository.

The current smoke workflow intentionally uses the existing ID registry and runtime-to-registry path mapping. This is temporary until a new manifest and ID registry are generated directly from the fuzzy-deduplicated files.

`MetadataExtractor` resolves one `mapping_names` entry per input file and broadcasts its configured integer provenance/ranking values onto every row. Stable source IDs and mutable policy ranks remain in the private mapping, while the generic stage and example schema contain no dataset-specific policy. Pairwise ranking can sort `source_priority`, `quality_rank`, and `recency_rank` descending, followed by the dedup ID ascending.

When `text_extraction` is configured, an existing scalar `text` column is preserved. If it is absent, the stage keeps string blocks from `<content_field>.content`, ignores non-text items, and creates `text` using either a fixed separator or whitespace-aware smart merging. Configure `retained_input_fields` as `[_curator_dedup_id, text]` for embedding generation so heterogeneous nested payloads are removed before conversion to Arrow and only model inputs/provenance continue through the GPU stages.

Production Parquet contains only `_curator_dedup_id`, `embeddings`, and configured integer metadata. Source-specific IDs and payload columns are intentionally excluded.

Build the private smoke manifest from the inventory manifest. It selects the smallest files and writes explicit shard assignments: 16 files from the first family, 16 from the second, and an 8+8 mixed shard.

```bash
python -m benchmarking.embedding_generation.prepare_smoke_manifest \
  --input-manifest=/path/to/inventory-manifest.json \
  --metadata-mapping=/path/to/metadata_mapping.json \
  --output-manifest=/path/to/smoke-manifest.jsonl \
  --first-family-id=0 --second-family-id=1 --files-per-shard=16 \
  --target-rows-per-shard=1632000
```

At 680 rows/second/GPU on a four-GPU node, 1,632,000 rows is approximately ten minutes of embedding work per shard. The row target is applied in addition to the 16-file minimum. The mixed shard receives approximately half its rows from each family, and no file is reused across shards.

Smoke test three logical shards. The smoke YAML enables `--keep-text`, so output contains the generated ID, text used by the embedder, embedding, and integer metadata.

The launcher requests all four GPUs and uses `--exclusive`, guaranteeing one array task per node. Keep those settings together: an exclusive job must use every GPU on its allocated node. A launcher that intentionally requests only one GPU must omit `--exclusive` and first verify on a small run that Slurm correctly coallocates it on a shared node.

```bash
export ARRAY_RUNTIME_ROOT=/path/to/runtime
export ARRAY_LOG_DIR=/path/to/logs
mkdir -p "$ARRAY_RUNTIME_ROOT" "$ARRAY_LOG_DIR"
TOTAL_SHARDS=3 sbatch --array=0-2 \
  --output="$ARRAY_LOG_DIR/%A_%a.out" --error="$ARRAY_LOG_DIR/%A_%a.err" \
  benchmarking/embedding_generation/submit_embedding_try.sbatch
```

For production, use the full manifest, set new output/checkpoint paths, then use `TOTAL_SHARDS=348` and `--array=0-347`. `--require-min-files-per-shard=16` only validates that fixed partition; it does not choose `S`.

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
