# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = Path("tutorials/text/manifest_inference/worker-env.sh").resolve()
CONFIG = Path("benchmarking/manifest-inference.yaml").resolve()


def _source_worker_env(tmp_path: Path, overrides: dict[str, str]) -> dict[str, str]:
    worktree = tmp_path / "worktree"
    fake_bin = worktree / ".venv/bin"
    fake_bin.mkdir(parents=True)
    (fake_bin / "activate").write_text(f'export PATH="{fake_bin}:$PATH"\n')
    python = fake_bin / "python"
    python.write_text("#!/bin/sh\nprintf 'Python 3.test\\n'\n")
    python.chmod(0o755)
    environment = {
        **os.environ,
        "WORKTREE": str(worktree),
        "BENCHMARK_ROOT": str(tmp_path / "benchmarks"),
        "SESSION_NAME": "session",
        "MODEL_KEY": "qwen",
        "INPUT_DIR": str(tmp_path / "input"),
        "MANIFEST_PATH": str(tmp_path / "manifest.jsonl"),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "TOTAL_SHARDS": "300",
        "SLURM_JOB_ID": "1234",
        "SLURM_TMPDIR": str(tmp_path / "slurm-tmp"),
        **overrides,
    }
    for name in (
        "CHECKPOINT_PATH",
        "ENTRY",
        "ENTRY_INDEX",
        "NEMO_CURATOR_SLURM_ARRAY_ENABLED",
        "NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX",
        "NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX",
        "NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_RESTART_COUNT",
    ):
        if name not in overrides:
            environment.pop(name, None)
    command = f"""
source {SCRIPT}
printf '%s\n' \
  "ENTRY=$ENTRY" \
  "ENTRY_INDEX=$ENTRY_INDEX" \
  "CHECKPOINT_PATH=$CHECKPOINT_PATH" \
  "TOTAL_SHARDS=$TOTAL_SHARDS" \
  "NEMO_CURATOR_SLURM_ARRAY_ENABLED=$NEMO_CURATOR_SLURM_ARRAY_ENABLED" \
  "NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX=${{NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX-unset}}" \
  "NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS=${{NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS-unset}}" \
  "NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX=${{NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX-unset}}"
"""
    completed = subprocess.run(  # noqa: S603 — fixed test script with temporary fixture paths
        ["/usr/bin/bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)


def test_array_entry_is_unique_after_automatic_requeue(tmp_path: Path):
    initial = _source_worker_env(tmp_path / "initial", {"SLURM_ARRAY_TASK_ID": "7"})
    restarted = _source_worker_env(
        tmp_path / "restarted",
        {"SLURM_ARRAY_TASK_ID": "7", "SLURM_RESTART_COUNT": "1"},
    )

    assert initial["ENTRY"] == "qwen_7_1234"
    assert restarted["ENTRY"] == "qwen_7_1234_restart_1"
    assert initial["NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX"] == "7"
    assert restarted["NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS"] == "300"


def test_bundle_preserves_disabled_filtering_and_original_run_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original_checkpoint = tmp_path / "original-checkpoint"
    bundle = _source_worker_env(
        tmp_path,
        {
            "CHECKPOINT_PATH": str(original_checkpoint),
            "ENTRY": "stale-entry",
            "NEMO_CURATOR_SLURM_ARRAY_ENABLED": "0",
            "NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX": "stale",
            "NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS": "stale",
            "NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX": "stale",
        },
    )

    assert bundle["ENTRY"] == "qwen_bundle_1234"
    assert bundle["NEMO_CURATOR_SLURM_ARRAY_ENABLED"] == "0"
    assert bundle["NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX"] == "unset"
    assert bundle["NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS"] == "unset"
    assert bundle["NEMO_CURATOR_SLURM_ARRAY_MINIMUM_SHARD_INDEX"] == "unset"
    assert bundle["TOTAL_SHARDS"] == "300"
    assert bundle["CHECKPOINT_PATH"] == str(original_checkpoint)

    monkeypatch.setenv("ENTRY", bundle["ENTRY"])
    config = yaml.safe_load(os.path.expandvars(CONFIG.read_text()))
    assert config["entries"][0]["name"] == bundle["ENTRY"]
