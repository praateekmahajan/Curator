# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local/shared-filesystem spillable alternative to the GPU fusion sampler.

Requires optional DuckDB. Only numeric scoring columns participate in sorts;
text is joined back at output. Group boundaries and budgets never change to
accommodate memory. Use a separate scratch directory per invocation.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa


def sample_fusion_disk(  # noqa: PLR0913, PLR0915
    files: list[str],
    output_path: str,
    budgets: dict[tuple[str, int], int],
    *,
    scratch_path: str,
    memory_limit: str = "4GB",
    threads: int = 8,
    max_repeat: int = 8,
    temperature: float = 2.0,
    diversity_weight: float = 1.0,
    quality_weights: tuple[float, float, float] = (0.45, 0.30, 0.25),
    gate_threshold: float = 3.0,
) -> list[str]:
    """Exact capped fusion allocation with external sorts, using standard math-18 columns.

    IDs must be globally unique. Output retains input columns and adds total_copies
    and q (replacing stale input values of both generated columns).
    Ties use ascending ID, as in the GPU sampler.
    The memory limit bounds DuckDB's buffer manager, not total process RSS.
    """
    # ponytail: single-node spill; distribute independent groups if disk throughput limits scaling.
    import duckdb

    if not np.isfinite([temperature, diversity_weight]).all() or temperature < 0 or diversity_weight < 0:
        msg = "temperature and diversity_weight must be finite and non-negative"
        raise ValueError(msg)
    if len(quality_weights) != 3 or not np.isfinite([*quality_weights, gate_threshold]).all():  # noqa: PLR2004
        msg = "quality_weights must contain three finite values and gate_threshold must be finite"
        raise ValueError(msg)

    if (
        not files
        or not budgets
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (threads, max_repeat))
    ):
        msg = "files, budgets, threads and max_repeat must be positive/nonempty"
        raise ValueError(msg)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in budgets.values()):
        msg = "budgets must be non-negative integers"
        raise ValueError(msg)
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=False)
    Path(scratch_path).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fusion-", dir=scratch_path) as scratch:  # noqa: SIM117
        with duckdb.connect(str(Path(scratch) / "sampling.db")) as db:
            db.execute("SET memory_limit = ?", [memory_limit])
            db.execute("SET threads = ?", [threads])
            db.execute("SET temp_directory = ?", [str(Path(scratch) / "spill")])
            db.execute("SET preserve_insertion_order = false")
            source = db.from_parquet(files)
            replaced = [name for name in ("q", "total_copies") if name in source.columns]
            projection = f"source.* EXCLUDE ({','.join(replaced)})" if replaced else "source.*"
            source.create_view("source")
            quality_sql = "least(1,greatest(0,$q_ms*ms::DOUBLE/5.0+$q_rd*rd::DOUBLE/5.0+$q_cr*cr::DOUBLE/5.0))"
            quality_params = dict(zip(("q_ms", "q_rd", "q_cr"), quality_weights, strict=True))
            db.register(
                "budget_input",
                pa.table(
                    {
                        "math_topic": [k[0] for k in budgets],
                        "sampling_shard": [k[1] for k in budgets],
                        "target": list(budgets.values()),
                    }
                ),
            )
            db.execute("CREATE TABLE budgets AS SELECT * FROM budget_input")
            invalid = db.execute("""
                SELECT count(*) FROM source WHERE id IS NULL OR math_topic IS NULL
                OR sampling_shard IS NULL OR ms IS NULL OR rd IS NULL OR cr IS NULL
                OR diversity_score IS NULL OR NOT isfinite(ms) OR NOT isfinite(rd)
                OR NOT isfinite(cr) OR NOT isfinite(diversity_score)
                OR ms NOT BETWEEN 0 AND 5 OR rd NOT BETWEEN 0 AND 5 OR cr NOT BETWEEN 0 AND 5
            """).fetchone()[0]
            duplicates = db.execute(
                "SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) AS n FROM source GROUP BY id)"
            ).fetchone()[0]
            if invalid or duplicates:
                msg = "sampling requires unique non-null IDs and finite, valid scoring metadata"
                raise ValueError(msg)
            db.execute(
                f"""  -- Only static SQL fragments are interpolated; values are bound.
                CREATE TABLE scores AS SELECT id, math_topic, sampling_shard,
                {quality_sql} AS quality,
                greatest(cr::DOUBLE-$gate_threshold, 0) AS gate, diversity_score::DOUBLE AS diversity
                FROM source
                """,  # noqa: S608
                {**quality_params, "gate_threshold": gate_threshold},
            )
            db.execute("""
                CREATE TABLE stats AS SELECT math_topic, sampling_shard, count(*) AS n,
                avg(quality) AS qm, stddev_pop(quality) AS qs,
                avg(gate) AS gm, avg(diversity) AS dm, stddev_pop(diversity) AS ds
                FROM scores GROUP BY math_topic, sampling_shard
            """)
            invalid_groups = db.execute(
                """
                SELECT count(*) FROM stats FULL JOIN budgets USING(math_topic,sampling_shard)
                WHERE target IS NULL OR n IS NULL OR target > n * ?
            """,
                [max_repeat],
            ).fetchone()[0]
            if invalid_groups:
                msg = "group budgets must match input groups and fit max_repeat capacity"
                raise ValueError(msg)
            db.execute(
                """
                CREATE TABLE composites AS SELECT id, math_topic, sampling_shard, n,
                (quality-qm)/CASE WHEN qs>0 THEN qs ELSE 1 END
                + ?*gate/greatest(gm, 1e-9)*greatest((diversity-dm)/CASE WHEN ds>0 THEN ds ELSE 1 END,0)
                AS composite FROM scores JOIN stats USING(math_topic,sampling_shard)
            """,
                [diversity_weight],
            )
            db.execute(
                """
                CREATE TABLE weights AS WITH ranked AS (
                SELECT id, math_topic, sampling_shard,
                (rank() OVER w + (count(*) OVER ties-1)/2.0)/n AS percentile FROM composites
                WINDOW w AS (PARTITION BY math_topic,sampling_shard ORDER BY composite),
                ties AS (PARTITION BY math_topic,sampling_shard,composite)
                ) SELECT id, math_topic, sampling_shard,
                exp(?*(percentile-max(percentile) OVER (PARTITION BY math_topic,sampling_shard))) AS weight,
                0::BIGINT AS copies FROM ranked
            """,
                [temperature],
            )
            while True:
                db.execute(
                    """
                    CREATE OR REPLACE TABLE remaining AS
                    SELECT math_topic, sampling_shard, target-sum(copies) AS remaining,
                    sum(CASE WHEN copies < ? THEN weight ELSE 0 END) AS active_weight
                    FROM weights JOIN budgets USING(math_topic,sampling_shard)
                    GROUP BY math_topic,sampling_shard,target
                """,
                    [max_repeat],
                )
                if db.execute("SELECT coalesce(sum(remaining),0) FROM remaining").fetchone()[0] == 0:
                    break
                if db.execute(
                    "SELECT count(*) FROM remaining WHERE remaining>0 AND (active_weight<=0 OR active_weight IS NULL)"
                ).fetchone()[0]:
                    msg = "sampling weights underflowed before exhausting the budget; reduce temperature"
                    raise ValueError(msg)
                db.execute(
                    """
                    CREATE OR REPLACE TABLE raw AS SELECT id, math_topic,sampling_shard,copies,weight,remaining,
                    CASE WHEN copies < ? AND remaining>0 THEN weight/active_weight*remaining ELSE 0 END AS raw
                    FROM weights JOIN remaining USING(math_topic,sampling_shard)
                """,
                    [max_repeat],
                )
                db.execute(
                    """
                    CREATE OR REPLACE TABLE weights AS SELECT id, math_topic,sampling_shard,weight,
                    least(?, copies+floor(raw)::BIGINT + CASE WHEN
                        row_number() OVER (PARTITION BY math_topic,sampling_shard
                            ORDER BY CASE WHEN copies < ? THEN raw-floor(raw) ELSE -1 END DESC,id)
                        <= remaining-sum(floor(raw)::BIGINT) OVER (PARTITION BY math_topic,sampling_shard)
                        THEN 1 ELSE 0 END)::BIGINT AS copies FROM raw
                """,
                    [max_repeat, max_repeat],
                )
            db.execute(
                f"""
                COPY (SELECT {projection}, {quality_sql} AS q, weights.copies AS total_copies
                      FROM source JOIN weights USING(id) WHERE weights.copies>0)
                TO $destination (FORMAT PARQUET, PER_THREAD_OUTPUT, ROW_GROUP_SIZE_BYTES '16MB', FILE_SIZE_BYTES '64MB')
            """,  # noqa: S608
                {**quality_params, "destination": str(root)},
            )
    return sorted(str(p) for p in root.glob("*.parquet"))
