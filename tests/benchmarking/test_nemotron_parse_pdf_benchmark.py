# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT_PATH = _REPO_ROOT / "benchmarking" / "scripts" / "nemotron_parse_pdf_benchmark.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("nemotron_parse_pdf_benchmark", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark
_SPEC.loader.exec_module(benchmark)


def test_parse_json_arg_requires_an_object() -> None:
    assert benchmark._parse_json_arg('{"tensor_parallel_size": 1}', arg_name="--engine-kwargs") == {
        "tensor_parallel_size": 1
    }
    with pytest.raises(TypeError, match="must decode to a JSON object"):
        benchmark._parse_json_arg("[]", arg_name="--engine-kwargs")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2048,1664", (2048, 1664)), ("1024, 768", (1024, 768))],
)
def test_parse_proc_size(raw: str, expected: tuple[int, int]) -> None:
    assert benchmark._parse_proc_size(raw) == expected


def test_parse_proc_size_rejects_wrong_dimensions() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="HEIGHT,WIDTH"):
        benchmark._parse_proc_size("2048")


def test_dynamo_replicas_must_be_static() -> None:
    assert benchmark._dynamo_num_replicas({"min_replicas": 4, "max_replicas": 4}) == 4
    with pytest.raises(ValueError, match="does not support autoscaling"):
        benchmark._dynamo_num_replicas({"min_replicas": 2, "max_replicas": 4})


def test_dynamo_gpu_count_includes_parallelism() -> None:
    assert (
        benchmark._dynamo_gpu_count(
            {"tensor_parallel_size": 2, "pipeline_parallel_size": 2},
            {"min_replicas": 3, "max_replicas": 3},
        )
        == 12
    )
