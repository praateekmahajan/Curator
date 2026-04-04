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

import pytest

from nemo_curator.core.serve import InferenceModelConfig, InferenceServer
from nemo_curator.core.serve.internal.dynamo import (
    DynamoBackend,
    NodeAllocation,
    ReplicaPlan,
    _engine_kwargs_to_cli_flags,
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

    def test_stop_before_start_is_safe(self):
        server = InferenceServer(
            models=[InferenceModelConfig(model_identifier="m")],
            backend="dynamo",
        )
        backend = DynamoBackend(server)
        backend.stop()  # should not raise
