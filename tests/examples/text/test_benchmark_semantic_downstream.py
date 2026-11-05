# modality: text
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

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/text"))
    import benchmark_semantic_downstream

    return benchmark_semantic_downstream


@pytest.mark.parametrize(("free_gib", "allowed"), [(1, False), (64, True)])
def test_repeat_capacity_guard(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_gib: int, allowed: bool
) -> None:
    monkeypatch.setattr(benchmark.shutil, "disk_usage", lambda _: SimpleNamespace(free=free_gib * 1024**3))
    task = benchmark.FileGroupTask(data=["unused.parquet"], dataset_name="guard-check")
    parent = Mock(return_value=task)
    monkeypatch.setattr(benchmark.RepeatParquetStage, "process", parent)
    stage = benchmark.CapacityCheckedRepeatStage(str(tmp_path / "output"))
    if allowed:
        assert stage.process(task) is task
        parent.assert_called_once_with(task)
    else:
        with pytest.raises(OSError, match="32 GiB"):
            stage.process(task)
        parent.assert_not_called()


@pytest.mark.parametrize(("free_gib", "allowed"), [(33, False), (34, True)])
def test_cached_probe_rechecks_available_space(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, free_gib: int, allowed: bool
) -> None:
    report = {"projected_output_bytes": 1024**3}
    (tmp_path / "repeat_storage_probe.json").write_text(json.dumps(report))
    monkeypatch.setattr(benchmark.shutil, "disk_usage", lambda _: SimpleNamespace(free=free_gib * 1024**3))
    if allowed:
        assert benchmark.probe_repeat_storage(tmp_path, []) == report
    else:
        with pytest.raises(OSError, match=r"1\.5x estimated output"):
            benchmark.probe_repeat_storage(tmp_path, [])
