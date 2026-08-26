#!/usr/bin/env bash
set -eo pipefail

source /etc/profile
set -u

WORKTREE=/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/users/praateekm/benchmarking/sft_fuzzy_dedup/worktree

module load cuda12.5/toolkit/12.5.1
cd "$WORKTREE"

activate_worktree_venv() {
  export VIRTUAL_ENV="$WORKTREE/.venv"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  hash -r
}

case "${1:-}" in
  setup)
    echo "host=$(hostname)"
    echo "worktree=$WORKTREE"
    echo "ref=$(git rev-parse --short HEAD)"
    echo "cuda_home=$CUDA_HOME"
    uv sync --all-extras --all-groups
    activate_worktree_venv
    echo "python=$(command -v python)"
    python --version
    python -c 'import nemo_curator; print("nemo_curator=" + getattr(nemo_curator, "__version__", "unknown"))'
    ;;
  run)
    shift
    activate_worktree_venv
    test "$(command -v python)" = "$WORKTREE/.venv/bin/python"
    exec "$@"
    ;;
  *)
    echo "usage: $0 {setup|run COMMAND...}" >&2
    exit 2
    ;;
esac
