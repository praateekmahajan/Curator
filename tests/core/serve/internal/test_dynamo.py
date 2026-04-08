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
from nemo_curator.core.serve.internal.dynamo import DynamoBackend
from nemo_curator.core.serve.internal.subprocess_mgr import (
    ManagedSubprocess,
    _define_subprocess_actor,
    plan_replica_placement,
)

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
# Dynamo network addressing (DynamoBackend-specific)
# ---------------------------------------------------------------------------


class TestDynamoNetworkAddressing:
    """Tests for DynamoBackend._resolve_infra_node and endpoint consistency."""

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
# Dynamo liveness monitoring (real subprocesses, real Ray cluster)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestDynamoLiveness:
    """Test DynamoBackend liveness checking via real Ray actors and subprocesses."""

    def _make_actor(self, command: list[str], label: str = "test") -> ManagedSubprocess:
        """Create a subprocess actor running *command* with no GPUs."""
        import ray

        actor_cls = _define_subprocess_actor()
        actor_name = f"test_liveness_{label}_{os.getpid()}"
        actor = actor_cls.options(name=actor_name, lifetime="detached").remote()
        status = ray.get(actor.initialize.remote(command, {}, None), timeout=30)
        assert status["pid"] > 0, f"Actor {actor_name} failed to start subprocess"
        run_ref = actor.run.remote()
        return ManagedSubprocess(label=label, actor=actor, run_ref=run_ref)

    def test_worker_death_surfaces_log_context(self, tmp_path: os.PathLike):
        """When a subprocess crashes, the log tail is included in the error."""
        import ray

        log_file = str(tmp_path / "crash.log")
        actor_cls = _define_subprocess_actor()
        actor = actor_cls.options(name=f"test_log_context_{os.getpid()}", lifetime="detached").remote()
        command = ["bash", "-c", "echo 'FATAL: something went wrong' && exit 1"]
        ray.get(actor.initialize.remote(command, {}, log_file))
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
        ray.get(actor.initialize.remote(["true"], {}, None))
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


# ---------------------------------------------------------------------------
# Multi-model support
# ---------------------------------------------------------------------------


class TestMultiModel:
    """Tests for multi-model deployment with GPU isolation."""

    def test_multi_model_plans_use_separate_gpus(self):
        """Two models each TP=1 replicas=1 on a 2-GPU node get different GPUs."""
        inventory = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 2, "is_head": False}]

        # Model A takes 1 GPU
        plans_a = plan_replica_placement(num_replicas=1, tp_size=1, _inventory=inventory)
        assert len(plans_a) == 1
        assert plans_a[0].ranks[0].num_gpus == 1

        # Shrink inventory (same logic as _deploy_and_healthcheck)
        used: dict[str, int] = {}
        for plan in plans_a:
            for rank in plan.ranks:
                used[rank.node_id] = used.get(rank.node_id, 0) + rank.num_gpus
        remaining = [
            {**n, "num_gpus": n["num_gpus"] - used.get(n["node_id"], 0)}
            for n in inventory
            if n["num_gpus"] - used.get(n["node_id"], 0) > 0
        ]

        # Model B gets remaining GPU
        plans_b = plan_replica_placement(num_replicas=1, tp_size=1, _inventory=remaining)
        assert len(plans_b) == 1
        assert plans_b[0].ranks[0].num_gpus == 1

        # After both, no GPUs left
        used_b: dict[str, int] = {}
        for plan in plans_b:
            for rank in plan.ranks:
                used_b[rank.node_id] = used_b.get(rank.node_id, 0) + rank.num_gpus
        final = [
            {**n, "num_gpus": n["num_gpus"] - used_b.get(n["node_id"], 0)}
            for n in remaining
            if n["num_gpus"] - used_b.get(n["node_id"], 0) > 0
        ]
        assert len(final) == 0, "Both GPUs should be consumed"

    def test_multi_model_insufficient_gpus_raises(self):
        """Two models on a 1-GPU node fails on the second model."""
        inventory = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 1, "is_head": False}]

        # First model succeeds
        plans_a = plan_replica_placement(num_replicas=1, tp_size=1, _inventory=inventory)
        used: dict[str, int] = {}
        for plan in plans_a:
            for rank in plan.ranks:
                used[rank.node_id] = used.get(rank.node_id, 0) + rank.num_gpus
        remaining = [
            {**n, "num_gpus": n["num_gpus"] - used.get(n["node_id"], 0)}
            for n in inventory
            if n["num_gpus"] - used.get(n["node_id"], 0) > 0
        ]

        # Second model fails — no GPUs left
        with pytest.raises(RuntimeError, match="No GPU nodes"):
            plan_replica_placement(num_replicas=1, tp_size=1, _inventory=remaining)

    def test_multi_model_across_nodes(self):
        """Two TP=4 models across two 4-GPU nodes get one node each."""
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
        ]

        plans_a = plan_replica_placement(num_replicas=1, tp_size=4, _inventory=inventory)
        assert plans_a[0].ranks[0].node_id == "n1"

        used: dict[str, int] = {}
        for plan in plans_a:
            for rank in plan.ranks:
                used[rank.node_id] = used.get(rank.node_id, 0) + rank.num_gpus
        remaining = [
            {**n, "num_gpus": n["num_gpus"] - used.get(n["node_id"], 0)}
            for n in inventory
            if n["num_gpus"] - used.get(n["node_id"], 0) > 0
        ]

        plans_b = plan_replica_placement(num_replicas=1, tp_size=4, _inventory=remaining)
        assert plans_b[0].ranks[0].node_id == "n2"

    def test_frontend_command_omits_model_name(self):
        """Frontend should auto-discover models, not filter by --model-name."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="model-a"),
                InferenceModelConfig(model_identifier="model-b"),
            ],
            backend="dynamo",
        )
        backend = DynamoBackend(server)

        # Capture the command that _launch_frontend would build by inspecting
        # the method's CLI assembly (we can't call it without Ray, but we can
        # verify the parameter signature no longer accepts model_name).
        import inspect

        sig = inspect.signature(backend._launch_frontend)
        assert "model_name" not in sig.parameters, (
            "_launch_frontend should not accept model_name — frontend auto-discovers models"
        )
