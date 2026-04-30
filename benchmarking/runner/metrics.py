# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for capturing per-entry slices of a session-wide Prometheus TSDB."""

import contextlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger


def slice_session_tsdb_to_entry(
    session_tsdb_path: Path,
    entry_metrics_dir: Path,
    start_ms: int,
    end_ms: int,
) -> bool:
    """Extract samples in [start_ms, end_ms] from a Prometheus TSDB into a self-contained TSDB.

    Uses promtool's round-trip pair: ``tsdb dump-openmetrics`` for time-windowed export and
    ``tsdb create-blocks-from openmetrics`` to materialize importable blocks. The intermediate
    OpenMetrics text is written to a tempfile and discarded — the persisted output is the
    block layout under ``entry_metrics_dir/prometheus_data``, which can be re-mounted by any
    Prometheus instance via ``--storage.tsdb.path``.

    Returns True on success, False if promtool is unavailable or either step fails. Failures
    are logged but never raised; benchmark results must remain durable even when metrics
    capture is best-effort.
    """
    promtool = shutil.which("promtool")
    if promtool is None:
        logger.warning("promtool not found on PATH; skipping per-entry TSDB slice")
        return False

    if not session_tsdb_path.is_dir():
        logger.warning(f"Session TSDB not found at {session_tsdb_path}; skipping slice")
        return False

    dst = entry_metrics_dir / "prometheus_data"
    entry_metrics_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".openmetrics", delete=False) as tmp:
        openmetrics_path = tmp.name
    try:
        with open(openmetrics_path, "w") as out:
            dump_proc = subprocess.run(  # noqa: S603
                [
                    promtool,
                    "tsdb",
                    "dump-openmetrics",
                    f"--min-time={start_ms}",
                    f"--max-time={end_ms}",
                    str(session_tsdb_path),
                ],
                stdout=out,
                stderr=subprocess.PIPE,
                check=False,
            )
        if dump_proc.returncode != 0:
            logger.warning(f"promtool tsdb dump-openmetrics failed: {dump_proc.stderr.decode(errors='replace')}")
            return False

        if os.path.getsize(openmetrics_path) == 0:
            logger.info(f"No samples in [{start_ms}, {end_ms}] for {session_tsdb_path}; nothing to slice")
            return False

        create_proc = subprocess.run(  # noqa: S603
            [promtool, "tsdb", "create-blocks-from", "openmetrics", openmetrics_path, str(dst)],
            capture_output=True,
            check=False,
        )
        if create_proc.returncode != 0:
            logger.warning(
                f"promtool tsdb create-blocks-from openmetrics failed: {create_proc.stderr.decode(errors='replace')}"
            )
            return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(openmetrics_path)

    logger.info(f"Sliced TSDB for [{start_ms}, {end_ms}] into {dst}")
    return True
