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

import contextlib
import os

import pytest

from nemo_curator.core.serve.internal.subprocess_mgr import (
    ManagedSubprocess,
    NodeAllocation,
    ReplicaPlan,
    _define_subprocess_actor,
    _engine_kwargs_to_cli_flags,
    _ignore_head_node,
    _resolve_node_ip,
    build_worker_actor_name,
    plan_replica_placement,
)

# ---------------------------------------------------------------------------
# engine_kwargs -> CLI flags
# ---------------------------------------------------------------------------


class TestEngineKwargsToCliFlags:
    def test_basic_conversion(self):
        flags = _engine_kwargs_to_cli_flags({"tensor_parallel_size": 4, "max_model_len": 8192})
        assert flags == ["--tensor-parallel-size", "4", "--max-model-len", "8192"]

    def test_bool_true_becomes_flag(self):
        assert _engine_kwargs_to_cli_flags({"enforce_eager": True}) == ["--enforce-eager"]

    def test_bool_false_omitted(self):
        assert _engine_kwargs_to_cli_flags({"enforce_eager": False}) == []

    def test_empty_dict(self):
        assert _engine_kwargs_to_cli_flags({}) == []


# ---------------------------------------------------------------------------
# Worker actor naming
# ---------------------------------------------------------------------------


class TestBuildWorkerActorName:
    def test_single_gpu_no_tp(self):
        assert build_worker_actor_name("Qwen3-0.6B", 0, 0, 1) == "Dynamo_DP0_Qwen3-0.6B"

    def test_tp_suffix_added_when_tp_gt_1(self):
        assert build_worker_actor_name("Qwen3-0.6B", 1, 0, 4) == "Dynamo_DP1_TP0_Qwen3-0.6B"
        assert build_worker_actor_name("Qwen3-0.6B", 0, 2, 4) == "Dynamo_DP0_TP2_Qwen3-0.6B"

    def test_hf_path_stripped(self):
        assert build_worker_actor_name("Qwen/Qwen3-0.6B", 0, 0, 1) == "Dynamo_DP0_Qwen3-0.6B"

    def test_disagg_role(self):
        assert build_worker_actor_name("Qwen3-0.6B", 0, 0, 2, role="decode") == "Dynamo_decode_DP0_TP0_Qwen3-0.6B"
        assert build_worker_actor_name("Qwen3-0.6B", 1, 0, 2, role="prefill") == "Dynamo_prefill_DP1_TP0_Qwen3-0.6B"


# ---------------------------------------------------------------------------
# NodeAllocation / ReplicaPlan dataclasses
# ---------------------------------------------------------------------------


class TestNodeAllocation:
    def test_fields(self):
        a = NodeAllocation(node_id="n1", node_ip="10.0.0.1", num_gpus=2, node_rank=0)
        assert a.node_id == "n1"
        assert a.node_ip == "10.0.0.1"
        assert a.num_gpus == 2
        assert a.node_rank == 0


class TestReplicaPlan:
    def test_single_node(self):
        plan = ReplicaPlan(
            replica_index=0,
            ranks=[NodeAllocation(node_id="n1", node_ip="10.0.0.1", num_gpus=4, node_rank=0)],
        )
        assert not plan.is_multi_node
        assert plan.nnodes == 1
        assert plan.total_gpus == 4
        assert plan.master_addr == "10.0.0.1"

    def test_multi_node(self):
        plan = ReplicaPlan(
            replica_index=0,
            ranks=[
                NodeAllocation(node_id="n1", node_ip="10.0.0.1", num_gpus=4, node_rank=0),
                NodeAllocation(node_id="n2", node_ip="10.0.0.2", num_gpus=4, node_rank=1),
            ],
        )
        assert plan.is_multi_node
        assert plan.nnodes == 2
        assert plan.total_gpus == 8
        assert plan.master_addr == "10.0.0.1"


# ---------------------------------------------------------------------------
# GPU placement planner -- mocked inventory (no Ray needed)
# ---------------------------------------------------------------------------


class TestPlanReplicaPlacementMocked:
    """Tests with pre-built inventory (no Ray cluster required)."""

    def _inventory_1x8(self) -> list[dict]:
        return [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 8, "is_head": False}]

    def _inventory_2x4(self) -> list[dict]:
        return [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
        ]

    def _inventory_3x4(self) -> list[dict]:
        return [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
            {"node_id": "n3", "node_ip": "10.0.0.3", "num_gpus": 4, "is_head": False},
        ]

    def test_single_node_single_replica(self):
        plans = plan_replica_placement(num_replicas=1, tp_size=4, _inventory=self._inventory_1x8())
        assert len(plans) == 1
        assert not plans[0].is_multi_node
        assert plans[0].ranks[0].num_gpus == 4

    def test_single_node_multiple_replicas(self):
        plans = plan_replica_placement(num_replicas=2, tp_size=4, _inventory=self._inventory_1x8())
        assert len(plans) == 2
        for plan in plans:
            assert not plan.is_multi_node
            assert plan.ranks[0].num_gpus == 4

    def test_multi_node_tp(self):
        """TP=8 across 2 nodes with 4 GPUs each."""
        plans = plan_replica_placement(num_replicas=1, tp_size=8, _inventory=self._inventory_2x4())
        assert len(plans) == 1
        plan = plans[0]
        assert plan.is_multi_node
        assert plan.nnodes == 2
        assert plan.ranks[0].node_rank == 0
        assert plan.ranks[0].num_gpus == 4
        assert plan.ranks[0].node_ip == "10.0.0.1"
        assert plan.ranks[1].node_rank == 1
        assert plan.ranks[1].num_gpus == 4
        assert plan.ranks[1].node_ip == "10.0.0.2"
        assert plan.master_addr == "10.0.0.1"

    def test_multi_node_tp_three_nodes(self):
        """TP=12 across 3 nodes with 4 GPUs each."""
        plans = plan_replica_placement(num_replicas=1, tp_size=12, _inventory=self._inventory_3x4())
        assert len(plans) == 1
        plan = plans[0]
        assert plan.nnodes == 3
        assert plan.total_gpus == 12

    def test_multiple_replicas_across_nodes(self):
        """2 replicas x TP=4 on 2 nodes with 4 GPUs each."""
        plans = plan_replica_placement(num_replicas=2, tp_size=4, _inventory=self._inventory_2x4())
        assert len(plans) == 2
        # Each replica fits on one node
        assert not plans[0].is_multi_node
        assert not plans[1].is_multi_node
        # Different nodes
        assert plans[0].ranks[0].node_id != plans[1].ranks[0].node_id

    def test_partial_leftover_is_skipped_when_full_node_is_available(self):
        """A leftover GPU must not force an invalid 1+3 split for TP=4."""
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 5, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 8, "is_head": False},
        ]

        plans = plan_replica_placement(num_replicas=3, tp_size=4, _inventory=inventory)

        assert [[(r.node_id, r.num_gpus) for r in plan.ranks] for plan in plans] == [
            [("n1", 4)],
            [("n2", 4)],
            [("n2", 4)],
        ]

    def test_even_multi_node_split_uses_equal_per_node_share(self):
        """TP=4 on two 3-GPU nodes should use a valid 2+2 split."""
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 3, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 3, "is_head": False},
        ]

        plans = plan_replica_placement(num_replicas=1, tp_size=4, _inventory=inventory)

        assert len(plans) == 1
        assert plans[0].is_multi_node
        assert [(rank.node_id, rank.num_gpus) for rank in plans[0].ranks] == [("n1", 2), ("n2", 2)]

    def test_invalid_asymmetric_multi_node_split_raises(self):
        """TP=4 cannot be launched from a 1+3 split even though total GPUs match."""
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 1, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 3, "is_head": False},
        ]

        with pytest.raises(RuntimeError, match="even split"):
            plan_replica_placement(num_replicas=1, tp_size=4, _inventory=inventory)

    def test_insufficient_gpus_raises(self):
        with pytest.raises(RuntimeError, match="Need"):
            plan_replica_placement(num_replicas=1, tp_size=16, _inventory=self._inventory_2x4())

    def test_empty_inventory_raises(self):
        with pytest.raises(RuntimeError, match="No GPU nodes"):
            plan_replica_placement(num_replicas=1, tp_size=1, _inventory=[])

    def test_partial_placement_raises(self):
        """Not enough GPUs for all replicas."""
        with pytest.raises(RuntimeError):
            plan_replica_placement(num_replicas=3, tp_size=4, _inventory=self._inventory_2x4())


# ---------------------------------------------------------------------------
# Deterministic rank-0 placement + head-node exclusion
# ---------------------------------------------------------------------------


class TestHeadNodePolicy:
    """Tests for CURATOR_IGNORE_RAY_HEAD_NODE and deterministic ordering."""

    def _inventory_head_and_workers(self) -> list[dict]:
        """Head node (n1, 4 GPUs) + two worker nodes (n2, n3, 4 GPUs each)."""
        return [
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": True},
            {"node_id": "n3", "node_ip": "10.0.0.3", "num_gpus": 4, "is_head": False},
        ]

    def test_rank0_on_head_node_when_allowed(self):
        """Without the flag, head node gets rank 0."""
        inv = self._inventory_head_and_workers()
        plans = plan_replica_placement(num_replicas=1, tp_size=4, _inventory=inv)
        assert plans[0].ranks[0].node_id == "n1"
        assert plans[0].ranks[0].node_ip == "10.0.0.1"

    def test_head_node_excluded_from_workers(self, monkeypatch: pytest.MonkeyPatch):
        """With the flag set, no worker plans land on the head node."""
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        inv = self._inventory_head_and_workers()
        plans = plan_replica_placement(num_replicas=2, tp_size=4, _inventory=inv)
        for plan in plans:
            for rank in plan.ranks:
                assert rank.node_id != "n1", "Worker should not be on head node"

    def test_no_eligible_nodes_raises(self, monkeypatch: pytest.MonkeyPatch):
        """Single head-only node + flag -> clear error."""
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        inv = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 8, "is_head": True}]
        with pytest.raises(RuntimeError, match="CURATOR_IGNORE_RAY_HEAD_NODE"):
            plan_replica_placement(num_replicas=1, tp_size=1, _inventory=inv)

    def test_deterministic_ordering(self):
        """Same inventory -> same plan every time, head first."""
        inv = self._inventory_head_and_workers()
        plans_a = plan_replica_placement(num_replicas=3, tp_size=4, _inventory=inv)
        plans_b = plan_replica_placement(num_replicas=3, tp_size=4, _inventory=inv)
        for a, b in zip(plans_a, plans_b, strict=True):
            assert a.ranks[0].node_id == b.ranks[0].node_id

    def test_deterministic_ordering_head_first(self):
        """Head node sorts first regardless of input order."""
        inv = self._inventory_head_and_workers()
        plans = plan_replica_placement(num_replicas=3, tp_size=4, _inventory=inv)
        # First replica on head (n1), then n2, then n3
        assert plans[0].ranks[0].node_id == "n1"
        assert plans[1].ranks[0].node_id == "n2"
        assert plans[2].ranks[0].node_id == "n3"

    def test_multi_replica_stable_plans(self, monkeypatch: pytest.MonkeyPatch):
        """Multiple replicas with head excluded -> stable assignment by node_id."""
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "true")
        inv = self._inventory_head_and_workers()
        plans = plan_replica_placement(num_replicas=2, tp_size=4, _inventory=inv)
        # n2 and n3 are the only options, sorted by node_id
        assert plans[0].ranks[0].node_id == "n2"
        assert plans[1].ranks[0].node_id == "n3"

    def test_ignore_head_node_helper(self, monkeypatch: pytest.MonkeyPatch):
        """_ignore_head_node reads the env var correctly."""
        monkeypatch.delenv("CURATOR_IGNORE_RAY_HEAD_NODE", raising=False)
        assert not _ignore_head_node()
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        assert _ignore_head_node()
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "yes")
        assert _ignore_head_node()
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "TRUE")
        assert _ignore_head_node()
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "0")
        assert not _ignore_head_node()
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "")
        assert not _ignore_head_node()


# ---------------------------------------------------------------------------
# GPU placement planner -- real Ray cluster
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestGpuPlacement:
    """Tests against the real shared Ray cluster (session fixture provides 2 GPUs)."""

    def test_placement_uses_available_gpus(self):
        plans = plan_replica_placement(num_replicas=2, tp_size=1)
        assert len(plans) == 2
        assert plans[0].ranks[0].num_gpus == 1
        assert plans[1].ranks[0].num_gpus == 1

    def test_placement_tp2(self):
        """TP=2 on 2 GPUs -> 1 replica, 1 rank, 2 GPUs."""
        plans = plan_replica_placement(num_replicas=1, tp_size=2)
        assert len(plans) == 1
        assert plans[0].ranks[0].num_gpus == 2
        assert not plans[0].is_multi_node

    def test_insufficient_gpus_raises(self):
        with pytest.raises(RuntimeError, match="Need"):
            plan_replica_placement(num_replicas=10, tp_size=4)

    def test_placement_has_valid_node_info(self):
        plans = plan_replica_placement(num_replicas=1, tp_size=1)
        assert plans[0].ranks[0].node_id
        assert plans[0].ranks[0].node_ip


# ---------------------------------------------------------------------------
# Network addressing -- _resolve_node_ip (mocked ray.nodes())
# ---------------------------------------------------------------------------

_RAY_NODES = [
    {"NodeID": "head1", "NodeManagerAddress": "10.0.0.1", "Alive": True, "Resources": {"GPU": 4, "CPU": 8}},
    {"NodeID": "worker1", "NodeManagerAddress": "10.0.0.2", "Alive": True, "Resources": {"GPU": 4, "CPU": 8}},
    {"NodeID": "worker2", "NodeManagerAddress": "10.0.0.3", "Alive": True, "Resources": {"GPU": 4, "CPU": 8}},
]


class TestResolveNodeIp:
    """Tests for _resolve_node_ip with mocked ray.nodes()."""

    def test_returns_routable_ip(self):
        ip = _resolve_node_ip("worker1", nodes=_RAY_NODES)
        assert ip == "10.0.0.2"

    def test_raises_for_unknown_node(self):
        with pytest.raises(RuntimeError, match="Could not resolve IP"):
            _resolve_node_ip("nonexistent", nodes=_RAY_NODES)

    def test_skips_dead_nodes(self):
        nodes = [
            {"NodeID": "n1", "NodeManagerAddress": "10.0.0.1", "Alive": False},
            {"NodeID": "n1", "NodeManagerAddress": "10.0.0.99", "Alive": True},
        ]
        assert _resolve_node_ip("n1", nodes=nodes) == "10.0.0.99"


class TestMultiNodePlacement:
    """Tests for multi-node TP addressing via plan_replica_placement."""

    def test_multi_node_tp_uses_rank0_ip_as_master_addr(self):
        """Multi-node TP: master_addr is the rank-0 node's IP."""
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
        ]
        plans = plan_replica_placement(num_replicas=1, tp_size=8, _inventory=inventory)
        assert plans[0].master_addr == "10.0.0.1"
        assert plans[0].ranks[0].node_ip == "10.0.0.1"
        assert plans[0].ranks[1].node_ip == "10.0.0.2"


# ---------------------------------------------------------------------------
# Subprocess actor lifecycle (real subprocesses, real Ray cluster)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestSubprocessActorLifecycle:
    """Test subprocess actor creation and run-ref based exit detection."""

    def test_worker_death_detected_via_run_ref(self):
        """Killing an actor makes its run ref ready in ray.wait()."""
        import ray

        actor_cls = _define_subprocess_actor()
        actor_name = f"test_liveness_death_{os.getpid()}"
        actor = actor_cls.options(name=actor_name, lifetime="detached").remote()
        status = ray.get(actor.initialize.remote(["sleep", "3600"], {}, None), timeout=30)
        assert status["pid"] > 0
        run_ref = actor.run.remote()
        proc = ManagedSubprocess(label="death", actor=actor, run_ref=run_ref)

        try:
            assert ray.get(proc.actor.is_alive.remote(), timeout=30)
            ray.kill(proc.actor, no_restart=True)
            ready, _ = ray.wait([proc.run_ref], timeout=30)
            assert len(ready) == 1
        except Exception:
            with contextlib.suppress(Exception):
                ray.kill(proc.actor, no_restart=True)
            raise


# ---------------------------------------------------------------------------
# Subprocess env propagation (real subprocesses, real Ray cluster)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestSubprocessEnvPropagation:
    """Validate the simplified single-dict subprocess_env model.

    Actors inherit ``os.environ`` from the raylet (which shares the
    driver's base environment).  Only runtime-computed vars (e.g.
    ``ETCD_ENDPOINTS``) need explicit passing via ``subprocess_env``.
    """

    def _run_subprocess(
        self,
        command: list[str],
        subprocess_env: dict[str, str] | None = None,
        *,
        tmp_path: os.PathLike,
        label: str = "env_test",
    ) -> str:
        """Spawn a subprocess actor, run *command*, return log output."""
        import ray

        actor_cls = _define_subprocess_actor()
        actor_name = f"test_{label}_{os.getpid()}"
        actor = actor_cls.options(name=actor_name, lifetime="detached").remote()
        log_file = str(tmp_path / f"{label}.log")

        try:
            ray.get(
                actor.initialize.remote(command, subprocess_env or {}, log_file),
                timeout=30,
            )
            run_ref = actor.run.remote()
            ray.get(run_ref, timeout=10)
            return ray.get(actor.read_log_tail.remote(), timeout=10)
        finally:
            with contextlib.suppress(Exception):
                ray.kill(actor, no_restart=True)

    def test_pre_existing_env_inherited_without_propagation(self, tmp_path: os.PathLike):
        """Subprocess sees PATH from the raylet without any explicit propagation."""
        log = self._run_subprocess(
            ["bash", "-c", "echo PATH=$PATH"],
            subprocess_env={},
            tmp_path=tmp_path,
            label="inherit",
        )
        assert "PATH=/" in log, f"PATH should be inherited from raylet, got: {log!r}"

    def test_post_init_var_not_inherited(self, tmp_path: os.PathLike):
        """Vars set after ray.init() are NOT in the actor's env -- explicit passing is needed."""
        sentinel = f"CURATOR_SENTINEL_{os.getpid()}"
        os.environ[sentinel] = "hello_from_driver"

        try:
            log = self._run_subprocess(
                ["bash", "-c", f"echo val=${{{sentinel}:-MISSING}}"],
                subprocess_env={},
                tmp_path=tmp_path,
                label="post_init",
            )
            assert "val=MISSING" in log, (
                f"Post-ray.init var should NOT be in actor env without explicit propagation, got: {log!r}"
            )
        finally:
            os.environ.pop(sentinel, None)

    def test_explicit_overrides_reach_subprocess(self, tmp_path: os.PathLike):
        """Explicit vars passed via subprocess_env reach the subprocess."""
        log = self._run_subprocess(
            ["bash", "-c", "echo etcd=$ETCD_ENDPOINTS"],
            subprocess_env={"ETCD_ENDPOINTS": "http://10.0.0.1:2379"},
            tmp_path=tmp_path,
            label="overwrite",
        )
        assert "etcd=http://10.0.0.1:2379" in log

    def test_overrides_do_not_clobber_inherited_vars(self, tmp_path: os.PathLike):
        """Passing targeted overrides doesn't clobber inherited PATH."""
        log = self._run_subprocess(
            ["bash", "-c", "echo PATH=$PATH && echo etcd=$ETCD_ENDPOINTS"],
            subprocess_env={"ETCD_ENDPOINTS": "http://10.0.0.1:2379"},
            tmp_path=tmp_path,
            label="no_clobber",
        )
        assert "PATH=/" in log, f"PATH should survive targeted overwrite, got: {log!r}"
        assert "etcd=http://10.0.0.1:2379" in log
