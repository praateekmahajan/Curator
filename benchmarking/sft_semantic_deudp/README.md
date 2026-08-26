# SFT semantic embedding benchmark

Run with `benchmarking/run.py` using session name
`embedding-bench-harrier-270m-YYYY-MM-DD` and set `EMBEDDING_MODEL` to the
Harrier OSS 270M embedding model. The entry uses Curator's native Slurm-array
filtering and session-level checkpointing. Embeddings from all shards are
written under the shared `benchmarking/output` path; metrics and `gpustats.csv`
remain under the session entry in `benchmarking/results`.

Start with four vLLM replicas per GPU and inspect mean GPU power in
`gpustats.csv` before changing the replica count.
