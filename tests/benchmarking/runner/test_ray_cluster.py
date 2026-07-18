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

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner import ray_cluster


class _Connection:
    def __init__(self, *, ready: bool, payload: tuple[bool, object] | None = None):
        self.ready = ready
        self.payload = payload
        self.closed = False

    def poll(self, timeout_s: int) -> bool:
        return self.ready

    def recv(self) -> tuple[bool, object]:
        assert self.payload is not None
        return self.payload

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self) -> None:
        self.started = False
        self.terminated = False
        self.killed = False
        self.alive = True

    def start(self) -> None:
        self.started = True

    def join(self, timeout: int) -> None:
        if self.terminated:
            self.alive = False

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.alive = False


class _Context:
    def __init__(self, parent: _Connection, child: _Connection, process: _Process):
        self.parent = parent
        self.child = child
        self.process = process

    def Pipe(self, *, duplex: bool) -> tuple[_Connection, _Connection]:  # noqa: N802
        assert not duplex
        return self.parent, self.child

    def Process(self, *, target: object, args: tuple[object, ...]) -> _Process:  # noqa: N802
        return self.process


def _patch_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ready: bool,
    payload: tuple[bool, object] | None = None,
) -> tuple[_Connection, _Connection, _Process]:
    parent = _Connection(ready=ready, payload=payload)
    child = _Connection(ready=False)
    process = _Process()
    context = _Context(parent, child, process)
    monkeypatch.setattr(ray_cluster, "check_ray_responsive", lambda: True)
    monkeypatch.setattr(ray_cluster.multiprocessing, "get_context", lambda _method: context)
    return parent, child, process


def test_get_ray_cluster_data_returns_child_result(monkeypatch: pytest.MonkeyPatch) -> None:
    parent, child, process = _patch_context(
        monkeypatch,
        ready=True,
        payload=(True, {"CPU": 144.0, "GPU": 4.0}),
    )
    process.alive = False

    assert ray_cluster.get_ray_cluster_data(timeout_s=1) == {"CPU": 144.0, "GPU": 4.0}
    assert process.started
    assert parent.closed
    assert child.closed


def test_get_ray_cluster_data_hard_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    parent, child, process = _patch_context(monkeypatch, ready=False)

    with pytest.raises(TimeoutError, match="Timed out after 1s"):
        ray_cluster.get_ray_cluster_data(timeout_s=1)

    assert process.terminated
    assert not process.is_alive()
    assert parent.closed
    assert child.closed


def test_get_ray_cluster_data_propagates_child_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _parent, _child, process = _patch_context(
        monkeypatch,
        ready=True,
        payload=(False, "ray.init failed"),
    )
    process.alive = False

    with pytest.raises(RuntimeError, match=r"ray\.init failed"):
        ray_cluster.get_ray_cluster_data(timeout_s=1)
