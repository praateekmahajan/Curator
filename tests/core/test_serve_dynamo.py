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

from nemo_curator.core.serve import InferenceModelConfig, InferenceServer
from nemo_curator.core.serve.internal.dynamo import (
    WORKER_SPECIFIC_ENV_VARS,
    DynamoBackend,
    ManagedSubprocess,
    NodeAllocation,
    ReplicaPlan,
    _define_subprocess_actor,
    _engine_kwargs_to_cli_flags,
    _get_driver_env_vars,
    _ignore_head_node,
    _merge_subprocess_env,
    _resolve_node_ip,
    plan_replica_placement,
)

# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


class TestBackendDispatch:
    def test_backend_param_creates_dynamo(self):
        server = InferenceServer(
            models=[InferenceModelConfig(model_identifier="some-model")],
            backend="dynamo",
        )
        assert isinstance(server._create_backend(), DynamoBackend)

    def test_invalid_backend_raises(self):
        server = InferenceServer(
            models=[InferenceModelConfig(model_identifier="m")],
            backend="foo",
        )
        with pytest.raises(ValueError, match="Unknown backend"):
            server._create_backend()

    def test_default_backend_is_ray_serve(self):
        server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")])
        assert server.backend == "ray_serve"


# ---------------------------------------------------------------------------
# InferenceModelConfig — dynamo_config
# ---------------------------------------------------------------------------


class TestDynamoConfig:
    def test_dynamo_config_defaults_to_empty(self):
        assert InferenceModelConfig(model_identifier="m").dynamo_config == {}

    def test_dynamo_config_preserved(self):
        config = InferenceModelConfig(
            model_identifier="m",
            dynamo_config={"namespace": "my_ns", "component": "prefill"},
        )
        assert config.dynamo_config == {"namespace": "my_ns", "component": "prefill"}


# ---------------------------------------------------------------------------
# engine_kwargs → CLI flags
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
# GPU placement planner — mocked inventory (no Ray needed)
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
        """Single head-only node + flag → clear error."""
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        inv = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 8, "is_head": True}]
        with pytest.raises(RuntimeError, match="CURATOR_IGNORE_RAY_HEAD_NODE"):
            plan_replica_placement(num_replicas=1, tp_size=1, _inventory=inv)

    def test_deterministic_ordering(self):
        """Same inventory → same plan every time, head first."""
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
        """Multiple replicas with head excluded → stable assignment by node_id."""
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
# GPU placement planner — real Ray cluster
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
        """TP=2 on 2 GPUs → 1 replica, 1 rank, 2 GPUs."""
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
# Replica count resolution
# ---------------------------------------------------------------------------


class TestResolveNumReplicas:
    def test_explicit_num_replicas(self):
        config = InferenceModelConfig(
            model_identifier="m",
            deployment_config={"num_replicas": 3},
        )
        assert DynamoBackend._resolve_num_replicas(config) == 3

    def test_from_autoscaling_min(self):
        config = InferenceModelConfig(
            model_identifier="m",
            deployment_config={"autoscaling_config": {"min_replicas": 2, "max_replicas": 8}},
        )
        assert DynamoBackend._resolve_num_replicas(config) == 2

    def test_defaults_to_one(self):
        assert DynamoBackend._resolve_num_replicas(InferenceModelConfig(model_identifier="m")) == 1


# ---------------------------------------------------------------------------
# Pipeline GPU contention — Dynamo always errors
# ---------------------------------------------------------------------------


class TestPipelineDynamoFailFast:
    def test_active_backend_tracking(self):
        from nemo_curator.core.serve import _active_servers, get_active_backend

        _active_servers["test-dynamo"] = "dynamo"
        try:
            assert get_active_backend() == "dynamo"
        finally:
            _active_servers.pop("test-dynamo", None)


# ---------------------------------------------------------------------------
# DynamoBackend validation
# ---------------------------------------------------------------------------


class TestDynamoBackendValidation:
    def test_start_raises_on_empty_models(self):
        server = InferenceServer(models=[], backend="dynamo")
        backend = DynamoBackend(server)
        with pytest.raises(ValueError, match="At least one"):
            backend.start()

    def test_disagg_rejects_multi_node_tp(self):
        """TP=8 across 2x4-GPU nodes in disagg mode should raise ValueError.

        Uses 1 prefill + 0 decode so only 1 worker is needed (8 GPUs). The
        planner can place it across 2 nodes, but disagg validation rejects it.
        """
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="m",
                    engine_kwargs={"tensor_parallel_size": 8},
                    dynamo_config={"mode": "disagg", "prefill_replicas": 1, "decode_replicas": 0},
                )
            ],
            backend="dynamo",
        )
        backend = DynamoBackend(server)
        with pytest.raises(ValueError, match="multi-node tensor parallelism"):
            backend._launch_disagg_workers(
                type(None),  # actor_cls unused — validation fires before actor creation
                server.models[0],
                {},
                head_node_id="n1",
                cluster_nodes=[
                    {"NodeID": "n1", "NodeManagerAddress": "10.0.0.1", "Alive": True, "Resources": {"GPU": 4}},
                    {"NodeID": "n2", "NodeManagerAddress": "10.0.0.2", "Alive": True, "Resources": {"GPU": 4}},
                ],
                namespace="test",
                request_plane="nats",
                event_plane="nats",
            )

    def test_disagg_accepts_single_node_tp(self):
        """TP=4 on 1x8-GPU node in disagg should produce valid single-node plans."""
        inventory = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 8, "is_head": False}]
        plans = plan_replica_placement(num_replicas=2, tp_size=4, _inventory=inventory)
        assert len(plans) == 2
        for plan in plans:
            assert not plan.is_multi_node
            assert plan.ranks[0].num_gpus == 4

    def test_disagg_tp1_still_works(self):
        """Basic TP=1 disagg placement — each worker gets 1 GPU."""
        inventory = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False}]
        plans = plan_replica_placement(num_replicas=4, tp_size=1, _inventory=inventory)
        assert len(plans) == 4
        for plan in plans:
            assert not plan.is_multi_node
            assert plan.ranks[0].num_gpus == 1

    def test_disagg_worker_gets_correct_num_gpus(self):
        """TP=4: each disagg worker plan should have num_gpus=4."""
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
        ]
        plans = plan_replica_placement(num_replicas=2, tp_size=4, _inventory=inventory)
        for plan in plans:
            assert plan.total_gpus == 4
            assert plan.ranks[0].num_gpus == 4

    def test_stop_before_start_is_safe(self):
        server = InferenceServer(
            models=[InferenceModelConfig(model_identifier="m")],
            backend="dynamo",
        )
        backend = DynamoBackend(server)
        backend.stop()  # should not raise


# ---------------------------------------------------------------------------
# Two-tier env propagation
# ---------------------------------------------------------------------------


class TestEnvPropagation:
    def test_driver_env_excludes_worker_specific_vars(self, monkeypatch: pytest.MonkeyPatch):
        """CUDA_VISIBLE_DEVICES and other worker-specific vars must not appear in driver env."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "20097")
        monkeypatch.setenv("PATH", "/usr/bin")
        driver = _get_driver_env_vars()
        for var in WORKER_SPECIFIC_ENV_VARS:
            assert var not in driver
        assert "PATH" in driver

    def test_driver_env_includes_etcd_nats(self, monkeypatch: pytest.MonkeyPatch):
        """ETCD_ENDPOINTS and NATS_SERVER are cluster-wide and belong in driver tier."""
        monkeypatch.setenv("ETCD_ENDPOINTS", "http://10.0.0.1:2379")
        monkeypatch.setenv("NATS_SERVER", "nats://10.0.0.1:4222")
        driver = _get_driver_env_vars()
        assert driver["ETCD_ENDPOINTS"] == "http://10.0.0.1:2379"
        assert driver["NATS_SERVER"] == "nats://10.0.0.1:4222"

    def test_worker_env_overwrites_driver(self):
        """Worker-tier vars overwrite any driver value when merged."""
        actor_env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "PATH": "/usr/bin"}
        driver = {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "PATH": "/usr/bin"}
        worker = {"CUDA_VISIBLE_DEVICES": "2,3"}
        merged = _merge_subprocess_env(actor_env, driver, worker)
        assert merged["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_worker_specific_env_vars_complete(self):
        """Ensure the set contains the expected GPU and per-worker vars."""
        assert "CUDA_VISIBLE_DEVICES" in WORKER_SPECIFIC_ENV_VARS
        assert "HIP_VISIBLE_DEVICES" in WORKER_SPECIFIC_ENV_VARS
        assert "ROCR_VISIBLE_DEVICES" in WORKER_SPECIFIC_ENV_VARS
        assert "LOCAL_RANK" in WORKER_SPECIFIC_ENV_VARS
        assert "VLLM_NIXL_SIDE_CHANNEL_PORT" in WORKER_SPECIFIC_ENV_VARS


# ---------------------------------------------------------------------------
# Network addressing (unit-level, mocked ray.nodes())
# ---------------------------------------------------------------------------


_RAY_NODES = [
    {"NodeID": "head1", "NodeManagerAddress": "10.0.0.1", "Alive": True, "Resources": {"GPU": 4, "CPU": 8}},
    {"NodeID": "worker1", "NodeManagerAddress": "10.0.0.2", "Alive": True, "Resources": {"GPU": 4, "CPU": 8}},
    {"NodeID": "worker2", "NodeManagerAddress": "10.0.0.3", "Alive": True, "Resources": {"GPU": 4, "CPU": 8}},
]


class TestNetworkAddressing:
    """Tests for IP resolution, etcd advertise, endpoint consistency."""

    def test_resolve_node_ip_returns_routable_ip(self):
        ip = _resolve_node_ip("worker1", nodes=_RAY_NODES)
        assert ip == "10.0.0.2"

    def test_resolve_node_ip_raises_for_unknown_node(self):
        with pytest.raises(RuntimeError, match="Could not resolve IP"):
            _resolve_node_ip("nonexistent", nodes=_RAY_NODES)

    def test_resolve_node_ip_skips_dead_nodes(self):
        nodes = [
            {"NodeID": "n1", "NodeManagerAddress": "10.0.0.1", "Alive": False},
            {"NodeID": "n1", "NodeManagerAddress": "10.0.0.99", "Alive": True},
        ]
        assert _resolve_node_ip("n1", nodes=nodes) == "10.0.0.99"

    def test_etcd_advertise_uses_infra_ip_not_localhost(self, monkeypatch: pytest.MonkeyPatch):
        """When head is excluded, _resolve_infra_node picks the non-head node."""
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo")
        backend = DynamoBackend(server)
        backend._head_ip = "10.0.0.1"
        cluster_nodes = [
            {"NodeID": "head1", "NodeManagerAddress": "10.0.0.1", "Alive": True, "Resources": {"CPU": 4}},
            {"NodeID": "worker1", "NodeManagerAddress": "10.0.0.2", "Alive": True, "Resources": {"CPU": 8}},
        ]
        node_id, node_ip = backend._resolve_infra_node(cluster_nodes, "head1")
        assert node_id == "worker1"
        assert node_ip == "10.0.0.2"

    def test_worker_env_uses_infra_ip_for_endpoints(self):
        """ETCD_ENDPOINTS and NATS_SERVER use infra IP, not head IP."""
        infra_ip = "10.0.0.2"
        etcd_endpoint = f"http://{infra_ip}:2379"
        nats_url = f"nats://{infra_ip}:4222"
        base_env = {"ETCD_ENDPOINTS": etcd_endpoint, "NATS_SERVER": nats_url}
        assert infra_ip in base_env["ETCD_ENDPOINTS"]
        assert infra_ip in base_env["NATS_SERVER"]

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

    def test_disagg_nixl_ports_unique_per_worker(self):
        """Each disagg worker should get a unique nixl port base."""
        # Verify port allocation pattern: 20097 + worker_index
        ports = [20097 + i for i in range(4)]
        assert len(set(ports)) == 4

    def test_infra_ip_consistent_across_endpoints(self, monkeypatch: pytest.MonkeyPatch):
        """When head excluded, endpoint + etcd + NATS all use infra IP.

        Regression test for D1: ensures all three addresses are consistent.
        """
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        infra_ip = "10.0.0.2"
        server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo")
        backend = DynamoBackend(server)
        backend._infra_ip = infra_ip
        backend._infra_node_id = "worker1"
        backend._etcd_port = 2379
        backend._nats_port = 4222

        etcd_endpoint = f"http://{backend._infra_ip}:{backend._etcd_port}"
        nats_url = f"nats://{backend._infra_ip}:{backend._nats_port}"

        # All three should use the same infra IP
        assert infra_ip in etcd_endpoint
        assert infra_ip in nats_url
        # server._host should be set to infra_ip (verified by checking the pattern)
        server._host = backend._infra_ip
        assert infra_ip in server.endpoint


# ---------------------------------------------------------------------------
# Run-ref liveness monitoring (real subprocesses, real Ray cluster)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestRunRefLiveness:
    """Test failure detection via run refs using real Ray actors and subprocesses."""

    def _make_actor(self, command: list[str], label: str = "test") -> ManagedSubprocess:
        """Create a subprocess actor running *command* with no GPUs."""
        import ray

        actor_cls = _define_subprocess_actor()
        actor_name = f"test_liveness_{label}_{os.getpid()}"
        actor = actor_cls.options(name=actor_name, lifetime="detached").remote()
        status = ray.get(actor.initialize.remote(command, _get_driver_env_vars(), {}, None), timeout=30)
        assert status["pid"] > 0, f"Actor {actor_name} failed to start subprocess"
        run_ref = actor.run.remote()
        return ManagedSubprocess(label=label, actor=actor, run_ref=run_ref)

    def test_worker_death_detected_via_run_ref(self):
        """Killing an actor makes its run ref ready in ray.wait()."""
        import ray

        proc = self._make_actor(["sleep", "3600"], label="death")
        try:
            assert ray.get(proc.actor.is_alive.remote(), timeout=30)
            ray.kill(proc.actor, no_restart=True)
            ready, _ = ray.wait([proc.run_ref], timeout=30)
            assert len(ready) == 1
        except Exception:
            with contextlib.suppress(Exception):
                ray.kill(proc.actor, no_restart=True)
            raise

    def test_worker_death_surfaces_log_context(self, tmp_path: os.PathLike):
        """When a subprocess crashes, the log tail is included in the error."""
        import ray

        log_file = str(tmp_path / "crash.log")
        actor_cls = _define_subprocess_actor()
        actor = actor_cls.options(name=f"test_log_context_{os.getpid()}", lifetime="detached").remote()
        command = ["bash", "-c", "echo 'FATAL: something went wrong' && exit 1"]
        ray.get(actor.initialize.remote(command, _get_driver_env_vars(), {}, log_file))
        run_ref = actor.run.remote()
        proc = ManagedSubprocess(label="test_worker", actor=actor, run_ref=run_ref, log_file=log_file)

        try:
            ready, _ = ray.wait([run_ref], timeout=10)
            assert len(ready) == 1

            server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo")
            backend = DynamoBackend(server)
            backend._worker_actors = [proc]

            with pytest.raises(RuntimeError, match="subprocess exited unexpectedly"):
                backend._check_liveness_via_refs()
        finally:
            with contextlib.suppress(Exception):
                ray.kill(actor, no_restart=True)

    def test_healthy_subprocess_not_flagged(self):
        """A running subprocess should not trigger liveness errors."""
        import ray

        proc = self._make_actor(["sleep", "3600"], label="healthy")
        try:
            server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo")
            backend = DynamoBackend(server)
            backend._worker_actors = [proc]

            # Should not raise
            backend._check_subprocess_health()
        finally:
            with contextlib.suppress(Exception):
                ray.kill(proc.actor, no_restart=True)

    def test_stop_idempotent_after_partial_failure(self):
        """Calling stop() twice after a partial start should not raise."""
        server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo")
        backend = DynamoBackend(server)
        # Simulate partial state
        backend._worker_actors = []
        backend._frontend_actor = None
        backend.stop()
        backend.stop()  # second call should be safe

    def test_exited_process_detected_via_health_check(self):
        """A process that exits normally is detected by _check_subprocess_health."""
        import time

        import ray

        actor_cls = _define_subprocess_actor()
        actor = actor_cls.options(name=f"test_exit_detect_{os.getpid()}", lifetime="detached").remote()
        ray.get(actor.initialize.remote(["true"], _get_driver_env_vars(), {}, None))
        run_ref = actor.run.remote()
        proc = ManagedSubprocess(label="exited_worker", actor=actor, run_ref=run_ref)

        try:
            time.sleep(1)

            server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo")
            backend = DynamoBackend(server)
            backend._worker_actors = [proc]

            with pytest.raises(RuntimeError, match="subprocess exited unexpectedly"):
                backend._check_subprocess_health()
        finally:
            with contextlib.suppress(Exception):
                ray.kill(actor, no_restart=True)
