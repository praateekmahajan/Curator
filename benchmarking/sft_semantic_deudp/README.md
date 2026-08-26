# SFT semantic embedding benchmark

Run with `benchmarking/run.py` using session name
`embedding-bench-harrier-270m-YYYY-MM-DD`. The YAML resolves the local
Harrier OSS 270M snapshot through `model_weights_path` and the `harrier_model`
file dataset. The entry uses Curator's native Slurm-array
filtering and session-level checkpointing. Embeddings from all shards are
written under the shared `benchmarking/output` path; metrics and `gpustats.csv`
remain under the session entry in `benchmarking/results`.

Start with four vLLM replicas per GPU and inspect mean GPU power in
`gpustats.csv` before changing the replica count.
