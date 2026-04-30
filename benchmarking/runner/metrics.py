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

import json
import shutil
import subprocess
from pathlib import Path

from loguru import logger


def slice_session_tsdb_to_entry(  # noqa: PLR0911
    session_tsdb_path: Path,
    entry_metrics_dir: Path,
    start_ms: int,
    end_ms: int,
) -> bool:
    """Extract samples in [start_ms, end_ms] from a Prometheus TSDB into ``metrics.parquet``.

    The slice is dumped via ``promtool tsdb dump-openmetrics``, parsed with
    ``prometheus_client.parser``, and written as a long-format zstd-compressed Parquet file
    with the schema documented in :func:`_records_from_openmetrics`. Failures are logged but
    never raised so benchmark results stay durable when metrics capture is best-effort.
    """
    promtool = shutil.which("promtool")
    if promtool is None:
        logger.warning("promtool not found on PATH; skipping per-entry TSDB slice")
        return False
    if not session_tsdb_path.is_dir():
        logger.warning(f"Session TSDB not found at {session_tsdb_path}; skipping slice")
        return False

    dump_proc = subprocess.run(  # noqa: S603
        [
            promtool,
            "tsdb",
            "dump-openmetrics",
            f"--min-time={start_ms}",
            f"--max-time={end_ms}",
            str(session_tsdb_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if dump_proc.returncode != 0:
        logger.warning(f"promtool tsdb dump-openmetrics failed: {dump_proc.stderr}")
        return False
    if not dump_proc.stdout.strip():
        logger.info(f"No samples in [{start_ms}, {end_ms}] for {session_tsdb_path}; nothing to slice")
        return False

    try:
        import pandas as pd
    except ImportError as exc:
        logger.warning(f"parquet conversion skipped (pandas missing): {exc}")
        return False

    try:
        records = _records_from_openmetrics(dump_proc.stdout)
    except Exception as exc:
        logger.warning(f"parquet conversion: failed to parse openmetrics output: {exc}")
        return False

    if not records:
        return False

    entry_metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = entry_metrics_dir / "metrics.parquet"
    pd.DataFrame(records).to_parquet(out_path, compression="zstd")
    logger.info(f"Wrote {len(records):,} samples for [{start_ms}, {end_ms}] to {out_path}")
    return True


def _records_from_openmetrics(text: str) -> list[dict]:
    """Parse OpenMetrics text into a list of dicts ready for ``pd.DataFrame``.

    Each dict has the columns:
      - ``metric`` (str): metric name (e.g. ``ray_node_cpu_utilization``)
      - ``value`` (float): sample value, may be NaN/Inf
      - ``ts_ms`` (int | None): unix timestamp in milliseconds
      - ``labels`` (str): JSON-encoded ``{label: value}`` dict with sorted keys
    """
    from prometheus_client.parser import text_string_to_metric_families

    return [
        {
            "metric": sample.name,
            "value": sample.value,
            "ts_ms": int(sample.timestamp * 1000) if sample.timestamp is not None else None,
            "labels": json.dumps(dict(sample.labels), sort_keys=True),
        }
        for family in text_string_to_metric_families(text)
        for sample in family.samples
    ]
