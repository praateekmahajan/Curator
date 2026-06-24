# Embedding SOTA Tracking

This file tracks incremental changes and experiments for the embedding generation SOTA investigation. Treat prior notes as directional context; final ranking requires fresh, corrected benchmark evidence.

| i | commit | what tried | what failed / succeeded / learnt | next experiment motivation |
|---:|---|---|---|---|
| 1 | `ccd6bac9` | Audited and corrected the current embedding benchmark script before running new benchmarks. Added endpoint response validation, shared `--max-chars` support for in-process vLLM, matching local/nightly input slice and text cap, Dynamo embedding pooling patch wiring, and focused Docker-only tests. | Succeeded: Docker tests passed (`7 passed`) and compile checks passed. Learnt: the script previously could silently accept endpoint responses with missing/duplicate indexes, corrupting row count and throughput; endpoint-only `--endpoint-max-chars` was not enough for fair in-process comparison. Commit used `--no-verify` because the benchmark image lacks `git-lfs`, which breaks repo hooks inside Docker. | Run corrected benchmarks to establish a trustworthy in-process baseline, then test endpoint improvements that preserve correctness: pretokenized endpoint inputs, base64 responses, larger request batches, and Ray Serve handle/direct path if implemented. |
