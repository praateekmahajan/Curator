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

"""Execute Forge's actual math-18 pandas sampler against the GPU implementation.

Pass a trusted local nl_web_18src_pipeline.py with --forge-file. Only the nested
_sample_topic function is executed, without importing Spark/Dagster. This tests
fixed-order numeric parity; it cannot reproduce Spark's unspecified row order.
"""

import argparse
import ast
import hashlib
import json
import runpy
from pathlib import Path

import cudf
import numpy as np
import pandas as pd

from nemo_curator.stages.deduplication.semantic.sampling import fusion_copies, plan_sampling_budgets


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--forge-file", type=Path, required=True)
    parser.add_argument("--forge-weighted-file", type=Path)
    args = parser.parse_args()
    source = args.forge_file.read_text()
    tree = ast.parse(source)
    sampler = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_sample_topic")
    constants = next(n.value for n in tree.body if isinstance(n, ast.AnnAssign) and n.target.id == "TOPIC_MULTS")
    multipliers = ast.literal_eval(constants)
    recipe = runpy.run_path(str(Path(__file__).with_name("semdedup_math18.py")))
    if recipe["MATH18_TOPIC_MULTIPLIERS"] != multipliers:
        raise AssertionError("Math-18 topic multipliers differ from Forge")  # noqa: EM101
    budgets = plan_sampling_budgets(
        {(t, 0): 100 for t in multipliers}, dict.fromkeys(multipliers, 100), topic_multipliers=multipliers
    )
    if budgets[("contest_problem_solving", 0)] <= budgets[("general_mixed", 0)]:
        raise AssertionError("Recipe topic multipliers did not affect budgets")  # noqa: EM101

    rng = np.random.default_rng(713)
    frame = pd.DataFrame(rng.uniform(0, 5, (257, 4)), columns=["ms", "rd", "cr", "diversity_score"])
    frame["hash"] = [f"{i:04x}" for i in range(len(frame))]
    frame["source"] = "reference"
    cases = 0
    for weights, temperature, diversity_weight, gate in [
        ((0.45, 0.30, 0.25), 2.0, 1.0, 3.0),
        ((0.7, 0.2, 0.1), 1.3, 1.7, 2.5),
        ((0.2, 0.3, 0.5), 5.0, 0.0, 5.0),
    ]:
        scope = {
            "q_ms": weights[0],
            "q_rd": weights[1],
            "q_cr": weights[2],
            "temp": temperature,
            "w_d": diversity_weight,
            "gate_thr": gate,
            "cap": 8,
        }
        exec(compile(ast.Module(body=[sampler], type_ignores=[]), str(args.forge_file), "exec"), scope)  # noqa: S102
        for target in [0, 1, 643, 2047, 2056]:
            frame["target_copies"] = target
            reference = scope["_sample_topic"](frame).set_index("hash")["copies"]
            expected = frame["hash"].map(reference).fillna(0).to_numpy(dtype=np.int64)
            actual = fusion_copies(
                cudf.from_pandas(frame),
                target,
                temperature=temperature,
                diversity_weight=diversity_weight,
                quality_weights=weights,
                gate_threshold=gate,
            ).get()
            np.testing.assert_array_equal(actual, expected)
            cases += 1
    discrete_diagnostics = []
    scope.update(q_ms=0.45, q_rd=0.30, q_cr=0.25, temp=2.0, w_d=1.0, gate_thr=3.0)
    for kind in ["integer", "constant", "mixed"]:
        discrete_rng = np.random.default_rng(713)
        values = discrete_rng.integers(0, 6, (1000, 4)).astype(float)
        if kind == "constant":
            values[:] = 3
        elif kind == "mixed":
            values[:, 3] = discrete_rng.random(1000)
        discrete = pd.DataFrame(values, columns=["ms", "rd", "cr", "diversity_score"])
        discrete["hash"] = [str(i) for i in range(len(discrete))]
        discrete["source"] = "reference"
        for target in [1, 2500, 7990]:
            discrete["target_copies"] = target
            reference = scope["_sample_topic"](discrete).set_index("hash")["copies"]
            expected = discrete.hash.map(reference).fillna(0).to_numpy(dtype=np.int64)
            actual = fusion_copies(cudf.from_pandas(discrete), target).get()
            discrete_diagnostics.append(
                {"kind": kind, "target": target, "mismatched_documents": int((actual != expected).sum())}
            )
            if kind != "mixed":
                np.testing.assert_array_equal(actual, expected)
    weighted_cases = 0
    if args.forge_weighted_file:
        import torch

        from nemo_curator.stages.deduplication.semantic.weighted import (
            WeightedPruningConfig,
            sequential_weighted_pruning,
        )

        weighted_tree = ast.parse(args.forge_weighted_file.read_text())
        names = {"compute_sim_matrix", "weighted_upper_edges", "removals_sequential_weighted", "balanced_splits"}
        functions = [n for n in weighted_tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if len(functions) != len(names):
            raise AssertionError("Forge weighted reference functions are missing")  # noqa: EM101
        weighted_scope = {"np": np, "torch": torch}
        exec(  # noqa: S102
            compile(ast.Module(body=functions, type_ignores=[]), str(args.forge_weighted_file), "exec"), weighted_scope
        )
        prototypes = rng.normal(size=(8, 64)).astype(np.float32)
        embeddings = prototypes[np.arange(257) % 8] + rng.normal(0, 0.025, (257, 64)).astype(np.float32)
        for shard_size in [None, 64]:
            for cluster_id in [0, 19]:
                for batch_size in [32, 128]:
                    config = WeightedPruningConfig(shard_size=shard_size)
                    splits = weighted_scope["balanced_splits"](
                        len(embeddings),
                        shard_size or 0,
                        np.random.default_rng(np.random.SeedSequence([0, cluster_id])),
                    ) or [np.arange(len(embeddings))]
                    expected = np.zeros(len(embeddings), dtype=bool)
                    for indices in splits:
                        expected[indices] = weighted_scope["removals_sequential_weighted"](
                            embeddings[indices], torch.device("cuda"), batch_size, config.rules, config.drop_score
                        )
                    actual = sequential_weighted_pruning(
                        torch.as_tensor(embeddings, device="cuda"),
                        config,
                        batch_size=batch_size,
                        cluster_id=cluster_id,
                    )
                    np.testing.assert_array_equal(actual, expected)
                    weighted_cases += 1
    print(
        json.dumps(
            {
                "forge_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "sampler_cases_equal": cases,
                "topic_multipliers_equal": True,
                "weighted_cases_equal": weighted_cases,
                "discrete_diagnostics": discrete_diagnostics,
                "scope": "fixed input order, continuous random scores; not a tie-order or clustering parity proof",
            }
        )
    )


if __name__ == "__main__":
    main()
