# Manifest embedding array

Copy `paths.yaml.example` to ignored `paths.yaml` and fill in the manifest, matching ID registry, metadata mapping, and output paths. The array pipeline has no source-path mapping or container-mount translation.

Export one `SESSION_NAME` for the logical run. Benchmark metadata is written under `benchmarking/<SESSION_NAME>`, embeddings under `results/<SESSION_NAME>`, Slurm logs under `logs/<SESSION_NAME>`, and checkpoints/runtime under `embedding-generation/<SESSION_NAME>`. Keep the canonical launch scripts versioned in this directory rather than copying mutable per-session variants.

Benchmark entry names include both the logical shard index and `SLURM_JOB_ID`, so a retry keeps its own benchmark history. New array submissions already receive distinct `%A_%a` log files and runtime directories. Embeddings and checkpoints intentionally remain shared across attempts so retries resume the same logical run.

The manifest `path` is the authoritative absolute fuzzy-deduplicated JSONL path. The ID registry key for one FPP=1 file is `str(uuid.uuid5(uuid.NAMESPACE_URL, path))`, using that exact string. The partitioner validates every manifest `id_start`/`id_end` against the matching registry entry before emitting tasks; a changed prefix, mount alias, or range fails instead of silently allocating new IDs.

The metadata mapping remains separate because it configures text extraction and integer ranking columns; it is unrelated to ID assignment.

`MetadataExtractor` resolves one `mapping_names` entry per input file and broadcasts three configured integer ranking values onto every row. Pairwise ranking sorts `source_family_id`, `quality_rank`, and `recency_rank` descending, followed by the dedup ID ascending. Family rank takes precedence globally; spaced quality ranks preserve special-source placement without overloading recency; recency breaks ties only within a quality bucket.

The private policy uses family `1` above family `0`. Family `1` quality levels are `super_low=-1`, `low=0`, `mid=1`, and `high=2`. Family `0` score folders use `score * 10`, leaving integer gaps for special sources: `score17=170`, `MQ=175`, `Reddit=176`, `score18=180`, `MHQ=185`, `OpenWebText=186`, `score19=190`, `HQ=195`, `CC-NEWS=196`, and `BigScience=197`. Within matching score folders, recency is `CC99=0`, `CC8=1`, and the newest crawl family is `2`.

When `text_extraction` is configured, an existing scalar `text` column is preserved. If it is absent, the stage keeps string blocks from `<content_field>.content`, ignores non-text items, and creates `text` using either a fixed separator or whitespace-aware smart merging. Configure `retained_input_fields` as `[_curator_dedup_id, text]` for embedding generation so heterogeneous nested payloads are removed before conversion to Arrow and only model inputs/provenance continue through the GPU stages.

Parquet contains `_curator_dedup_id`, `embeddings`, and configured integer metadata. With `--keep-text`, it also retains the exact text sent to the embedder. Source-specific IDs and other payload columns are intentionally excluded.

Build a private smoke manifest from the full runtime manifest. It preserves the absolute paths and ID ranges while selecting the smallest files and writing explicit shard assignments: 16 files from the first family, 16 from the second, and an 8+8 mixed shard.

```bash
python -m benchmarking.embedding_generation.prepare_smoke_manifest \
  --input-manifest=/path/to/manifest.jsonl \
  --metadata-mapping=/path/to/metadata_mapping.json \
  --output-manifest=/path/to/smoke-manifest.jsonl \
  --first-family-id=0 --second-family-id=1 --files-per-shard=16 \
  --target-rows-per-shard=1632000
```

At 680 rows/second/GPU on a four-GPU node, 1,632,000 rows is approximately ten minutes of embedding work per shard. The row target is applied in addition to the 16-file minimum. The mixed shard receives approximately half its rows from each family, and no file is reused across shards.

The production YAML omits `--keep-text`: vLLM emits only the generated ID, embedding, and integer metadata, so text is released before its output block enters Ray's object store. A diagnostic smoke configuration may explicitly add `--keep-text`; in that case vLLM and Parquet retain the exact text sent to the embedder.

The production run starts Ray with a 96 GiB object store and sets Ray Data's `override_object_store_memory_limit_fraction` to `0.7` through the pipeline CLI.

The launcher requests all four GPUs and uses `--exclusive`, guaranteeing one array task per node. Keep those settings together: an exclusive job must use every GPU on its allocated node. A launcher that intentionally requests only one GPU must omit `--exclusive` and first verify on a small run that Slurm correctly coallocates it on a shared node.

```bash
export SESSION_NAME=embedding-slurm-array-smoke-test-20260717
export USER_RUN_ROOT=/path/to/user/root
export ARRAY_RUNTIME_ROOT="$USER_RUN_ROOT/embedding-generation/$SESSION_NAME/runtime"
export ARRAY_LOG_DIR="$USER_RUN_ROOT/logs/$SESSION_NAME"
mkdir -p "$ARRAY_RUNTIME_ROOT" "$ARRAY_LOG_DIR"
TOTAL_SHARDS=3 sbatch --account=nemotron_n4_pre --partition=batch --qos=normal --array=0-2 \
  --output="$ARRAY_LOG_DIR/%A_%a.out" --error="$ARRAY_LOG_DIR/%A_%a.err" \
  benchmarking/embedding_generation/submit_embedding_try.sbatch
```

For production, use the full manifest and matching ID registry, then set new output/checkpoint paths. With `TOTAL_SHARDS=350` and `--array=0-349`, the 9,887,000,445 rows produce shards between 28,065,145 and 28,452,317 rows. `--require-min-files-per-shard=16` only validates that fixed partition; it does not choose `S`.

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
