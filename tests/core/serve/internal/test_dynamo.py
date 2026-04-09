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
from nemo_curator.core.serve.internal.dynamo import DynamoBackend, _dynamo_endpoint, _model_name_to_component
from nemo_curator.core.serve.internal.errors import SubprocessError
from nemo_curator.core.serve.internal.subprocess_mgr import (
    ManagedSubprocess,
    _define_subprocess_actor,
    _kill_actor,
    plan_replica_placement,
)


def _pid_alive(pid: int) -> bool:
    """Check if a process is alive via kill(0)."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


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
# Disagg role config resolution
# ---------------------------------------------------------------------------


class TestResolveDisaggRoleConfig:
    def test_defaults_to_1_replica_and_base_engine_kwargs(self):
        config = InferenceModelConfig(
            model_identifier="m",
            engine_kwargs={"tensor_parallel_size": 4, "max_model_len": 8192},
            dynamo_config={"mode": "disagg"},
        )
        (np, pek), (nd, dek) = DynamoBackend._resolve_disagg_role_config(config)
        assert np == 1
        assert nd == 1
        assert pek == {"tensor_parallel_size": 4, "max_model_len": 8192}
        assert dek == {"tensor_parallel_size": 4, "max_model_len": 8192}

    def test_role_engine_kwargs_override_base(self):
        config = InferenceModelConfig(
            model_identifier="m",
            engine_kwargs={"tensor_parallel_size": 2, "max_model_len": 8192},
            dynamo_config={
                "mode": "disagg",
                "prefill": {"num_replicas": 4, "engine_kwargs": {"tensor_parallel_size": 4}},
                "decode": {"num_replicas": 2},
            },
        )
        (np, pek), (nd, dek) = DynamoBackend._resolve_disagg_role_config(config)
        assert np == 4
        assert nd == 2
        # Prefill overrides TP but inherits max_model_len
        assert pek == {"tensor_parallel_size": 4, "max_model_len": 8192}
        # Decode inherits everything
        assert dek == {"tensor_parallel_size": 2, "max_model_len": 8192}

    def test_both_roles_override(self):
        config = InferenceModelConfig(
            model_identifier="m",
            engine_kwargs={"tensor_parallel_size": 2},
            dynamo_config={
                "mode": "disagg",
                "prefill": {"num_replicas": 3, "engine_kwargs": {"tensor_parallel_size": 4}},
                "decode": {"num_replicas": 1, "engine_kwargs": {"tensor_parallel_size": 1}},
            },
        )
        (np, pek), (nd, dek) = DynamoBackend._resolve_disagg_role_config(config)
        assert np == 3
        assert nd == 1
        assert pek["tensor_parallel_size"] == 4
        assert dek["tensor_parallel_size"] == 1


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
                    dynamo_config={
                        "mode": "disagg",
                        "prefill": {"num_replicas": 1},
                        "decode": {"num_replicas": 0},
                    },
                )
            ],
            backend="dynamo",
        )
        backend = DynamoBackend(server)
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
        ]
        with pytest.raises(ValueError, match="multi-node tensor parallelism"):
            backend._launch_disagg_workers(
                server.models[0],
                {},
                inventory=inventory,
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
# Model name → component slug
# ---------------------------------------------------------------------------


class TestModelNameToComponent:
    def test_hf_org_and_model(self):
        assert _model_name_to_component("Qwen/Qwen3-0.6B") == "qwen_qwen3_0_6b"

    def test_hf_org_with_dots(self):
        assert _model_name_to_component("google/gemma-3-4b-it") == "google_gemma_3_4b_it"

    def test_simple_name(self):
        assert _model_name_to_component("my-custom-model") == "my_custom_model"

    def test_already_clean(self):
        assert _model_name_to_component("model_a") == "model_a"

    def test_leading_trailing_special_chars(self):
        assert _model_name_to_component("/some/model/") == "some_model"

    def test_dots_replaced(self):
        """Dots must be replaced — they are delimiters in dyn:// endpoint strings."""
        result = _model_name_to_component("v1.5-model")
        assert "." not in result
        assert result == "v1_5_model"

    def test_all_special_chars_raises(self):
        """Input that sanitizes to empty should raise, not produce an invalid endpoint."""
        with pytest.raises(ValueError, match="empty component slug"):
            _model_name_to_component("---")


# ---------------------------------------------------------------------------
# Unique model name validation
# ---------------------------------------------------------------------------


class TestUniqueModelNameValidation:
    def test_unique_names_pass(self):
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="org/model-a", model_name="model-a"),
                InferenceModelConfig(model_identifier="org/model-b", model_name="model-b"),
            ],
            backend="dynamo",
        )
        DynamoBackend._validate_unique_model_names(server)  # should not raise

    def test_duplicate_model_name_raises(self):
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="org/model-a", model_name="same-name"),
                InferenceModelConfig(model_identifier="org/model-b", model_name="same-name"),
            ],
            backend="dynamo",
        )
        with pytest.raises(ValueError, match="Duplicate model name"):
            DynamoBackend._validate_unique_model_names(server)

    def test_duplicate_identifier_without_model_name_raises(self):
        """Two configs with same model_identifier and no model_name should collide."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="Qwen/Qwen3-0.6B"),
                InferenceModelConfig(model_identifier="Qwen/Qwen3-0.6B"),
            ],
            backend="dynamo",
        )
        with pytest.raises(ValueError, match="Duplicate model name"):
            DynamoBackend._validate_unique_model_names(server)

    def test_same_identifier_different_model_name_passes(self):
        """Same model_identifier is fine if model_name is unique."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="Qwen/Qwen3-0.6B", model_name="qwen-fast"),
                InferenceModelConfig(model_identifier="Qwen/Qwen3-0.6B", model_name="qwen-throughput"),
            ],
            backend="dynamo",
        )
        DynamoBackend._validate_unique_model_names(server)  # should not raise

    def test_single_model_always_passes(self):
        server = InferenceServer(
            models=[InferenceModelConfig(model_identifier="m")],
            backend="dynamo",
        )
        DynamoBackend._validate_unique_model_names(server)  # should not raise

    def test_component_slug_collision_raises(self):
        """Names that differ but sanitize to the same component should be rejected."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="x", model_name="a.b"),
                InferenceModelConfig(model_identifier="y", model_name="a-b"),
            ],
            backend="dynamo",
        )
        with pytest.raises(ValueError, match="both sanitize to component"):
            DynamoBackend._validate_unique_model_names(server)


# ---------------------------------------------------------------------------
# Endpoint URI builder
# ---------------------------------------------------------------------------


class TestDynamoEndpoint:
    def test_aggregate(self):
        assert _dynamo_endpoint("curator", "qwen") == "dyn://curator.qwen.generate"

    def test_with_role(self):
        assert _dynamo_endpoint("curator", "gemma", role="decode") == "dyn://curator.gemma_decode.generate"

    def test_prefill_role(self):
        assert _dynamo_endpoint("ns", "model_a", role="prefill") == "dyn://ns.model_a_prefill.generate"


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
                backend._check_subprocess_health()
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

    def test_kill_actor_also_kills_child_processes(self, tmp_path: os.PathLike):
        """Killing an actor must also terminate child processes spawned by the subprocess.

        Subprocesses are launched with start_new_session=True so the entire
        process group is signaled, preventing orphaned grandchildren (e.g.
        vLLM torch.distributed workers).
        """
        import signal
        import time

        import ray

        pid_file = str(tmp_path / "child_pids.txt")
        # Bash spawns two child sleep processes then waits.
        command = [
            "bash",
            "-c",
            f"sleep 3600 & echo $! > {pid_file}; sleep 3600 & echo $! >> {pid_file}; wait",
        ]

        actor_cls = _define_subprocess_actor()
        actor_name = f"test_orphan_kill_{os.getpid()}"
        actor = actor_cls.options(name=actor_name, lifetime="detached").remote()
        status = ray.get(actor.initialize.remote(command, {}, None), timeout=10)
        parent_pid = status["pid"]
        actor.run.remote()

        # Poll for PID file to have 2 lines (children spawned).
        deadline = time.monotonic() + 5
        child_pids: list[int] = []
        while time.monotonic() < deadline:
            if os.path.exists(pid_file):
                with open(pid_file) as f:
                    lines = [line.strip() for line in f if line.strip()]
                if len(lines) == 2:
                    child_pids = [int(line) for line in lines]
                    break
            time.sleep(0.05)
        assert len(child_pids) == 2, f"Expected 2 child PIDs, got {child_pids}"

        # All should be alive before kill
        for pid in [parent_pid, *child_pids]:
            os.kill(pid, 0)  # raises if not alive

        _kill_actor(ray, actor_name, actor)
        time.sleep(0.5)

        alive = [pid for pid in child_pids if _pid_alive(pid)]
        try:
            assert not alive, f"Child processes {alive} survived actor kill (orphaned)"
        finally:
            for pid in [parent_pid, *child_pids]:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


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


# ---------------------------------------------------------------------------
# Frontend config validation
# ---------------------------------------------------------------------------


class TestFrontendConfigValidation:
    """Tests for DynamoBackend._validate_frontend_config."""

    def test_rejects_mismatched_namespace(self):
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="model-a", dynamo_config={"namespace": "ns1"}),
                InferenceModelConfig(model_identifier="model-b", dynamo_config={"namespace": "ns2"}),
            ],
            backend="dynamo",
        )
        with pytest.raises(ValueError, match="namespace"):
            DynamoBackend._validate_frontend_config(server)

    def test_rejects_mismatched_request_plane(self):
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="model-a", dynamo_config={"request_plane": "nats"}),
                InferenceModelConfig(model_identifier="model-b", dynamo_config={"request_plane": "grpc"}),
            ],
            backend="dynamo",
        )
        with pytest.raises(ValueError, match="request_plane"):
            DynamoBackend._validate_frontend_config(server)

    def test_accepts_matching_config(self):
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="model-a", dynamo_config={"namespace": "ns1"}),
                InferenceModelConfig(model_identifier="model-b", dynamo_config={"namespace": "ns1"}),
            ],
            backend="dynamo",
        )
        DynamoBackend._validate_frontend_config(server)  # should not raise

    def test_single_model_always_passes(self):
        server = InferenceServer(
            models=[InferenceModelConfig(model_identifier="m", dynamo_config={"namespace": "any"})],
            backend="dynamo",
        )
        DynamoBackend._validate_frontend_config(server)  # should not raise


# ---------------------------------------------------------------------------
# Pre-infra GPU validation
# ---------------------------------------------------------------------------


class TestGpuValidation:
    """Tests for DynamoBackend._validate_gpu_requirements."""

    def test_rejects_disagg_tp_exceeding_max_node_gpus(self):
        """TP=8 in disagg mode on nodes with max 4 GPUs each should fail."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="big-model",
                    engine_kwargs={"tensor_parallel_size": 8},
                    dynamo_config={
                        "mode": "disagg",
                        "prefill": {"num_replicas": 1},
                        "decode": {"num_replicas": 1},
                    },
                )
            ],
            backend="dynamo",
        )
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
        ]
        with pytest.raises(ValueError, match="does not support multi-node TP"):
            DynamoBackend._validate_gpu_requirements(server, inventory)

    def test_rejects_aggregate_gpu_overcommit(self):
        """Two models needing 4 GPUs each on a 4-GPU cluster should fail."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="model-a",
                    engine_kwargs={"tensor_parallel_size": 2},
                    deployment_config={"num_replicas": 2},
                ),
                InferenceModelConfig(
                    model_identifier="model-b",
                    engine_kwargs={"tensor_parallel_size": 2},
                    deployment_config={"num_replicas": 2},
                ),
            ],
            backend="dynamo",
        )
        inventory = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False}]
        with pytest.raises(ValueError, match="require 8 GPUs total but only 4"):
            DynamoBackend._validate_gpu_requirements(server, inventory)

    def test_accepts_valid_config(self):
        """Config that fits should not raise."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="m",
                    engine_kwargs={"tensor_parallel_size": 2},
                    deployment_config={"num_replicas": 1},
                )
            ],
            backend="dynamo",
        )
        inventory = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False}]
        DynamoBackend._validate_gpu_requirements(server, inventory)  # should not raise

    def test_disagg_aggregate_check_counts_prefill_and_decode(self):
        """Disagg model with 2 prefill + 2 decode at TP=2 needs 8 GPUs."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="m",
                    engine_kwargs={"tensor_parallel_size": 2},
                    dynamo_config={
                        "mode": "disagg",
                        "prefill": {"num_replicas": 2},
                        "decode": {"num_replicas": 2},
                    },
                )
            ],
            backend="dynamo",
        )
        inventory = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False}]
        with pytest.raises(ValueError, match="require 8 GPUs total but only 4"):
            DynamoBackend._validate_gpu_requirements(server, inventory)

    def test_asymmetric_tp_gpu_count(self):
        """4 prefill @ TP=2 + 2 decode @ TP=1 = 10 GPUs."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="m",
                    engine_kwargs={"tensor_parallel_size": 1},
                    dynamo_config={
                        "mode": "disagg",
                        "prefill": {"num_replicas": 4, "engine_kwargs": {"tensor_parallel_size": 2}},
                        "decode": {"num_replicas": 2},
                    },
                )
            ],
            backend="dynamo",
        )
        inventory = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 8, "is_head": False}]
        with pytest.raises(ValueError, match="require 10 GPUs total but only 8"):
            DynamoBackend._validate_gpu_requirements(server, inventory)

    def test_asymmetric_tp_accepts_valid(self):
        """4 prefill @ TP=2 + 2 decode @ TP=1 = 10 GPUs on a 12-GPU cluster."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="m",
                    dynamo_config={
                        "mode": "disagg",
                        "prefill": {"num_replicas": 4, "engine_kwargs": {"tensor_parallel_size": 2}},
                        "decode": {"num_replicas": 2, "engine_kwargs": {"tensor_parallel_size": 1}},
                    },
                )
            ],
            backend="dynamo",
        )
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 8, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
        ]
        DynamoBackend._validate_gpu_requirements(server, inventory)  # should not raise

    def test_asymmetric_tp_2node_16gpu_decode_tp8(self):
        """2 prefill TP=4 + 1 decode TP=8 on 2x8-GPU nodes = 16 GPUs."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="m",
                    dynamo_config={
                        "mode": "disagg",
                        "prefill": {"num_replicas": 2, "engine_kwargs": {"tensor_parallel_size": 4}},
                        "decode": {"num_replicas": 1, "engine_kwargs": {"tensor_parallel_size": 8}},
                    },
                )
            ],
            backend="dynamo",
        )
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 8, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 8, "is_head": False},
        ]
        DynamoBackend._validate_gpu_requirements(server, inventory)  # should not raise

    def test_asymmetric_tp_2node_16gpu_decode_tp1(self):
        """2 prefill TP=4 + 8 decode TP=1 on 2x8-GPU nodes = 16 GPUs."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="m",
                    dynamo_config={
                        "mode": "disagg",
                        "prefill": {"num_replicas": 2, "engine_kwargs": {"tensor_parallel_size": 4}},
                        "decode": {"num_replicas": 8, "engine_kwargs": {"tensor_parallel_size": 1}},
                    },
                )
            ],
            backend="dynamo",
        )
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 8, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 8, "is_head": False},
        ]
        DynamoBackend._validate_gpu_requirements(server, inventory)  # should not raise

    def test_disagg_ignores_deployment_config_num_replicas(self):
        """deployment_config.num_replicas is ignored in disagg mode — only role replicas count."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="m",
                    deployment_config={"num_replicas": 99},
                    dynamo_config={
                        "mode": "disagg",
                        "prefill": {"num_replicas": 1, "engine_kwargs": {"tensor_parallel_size": 1}},
                        "decode": {"num_replicas": 1, "engine_kwargs": {"tensor_parallel_size": 1}},
                    },
                )
            ],
            backend="dynamo",
        )
        # 2 GPUs total (1 prefill + 1 decode), NOT 99*something
        inventory = [{"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 2, "is_head": False}]
        DynamoBackend._validate_gpu_requirements(server, inventory)  # should not raise

    def test_asymmetric_tp_rejects_role_exceeding_node(self):
        """Prefill TP=8 on 4-GPU nodes should fail even if decode TP=1."""
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="m",
                    dynamo_config={
                        "mode": "disagg",
                        "prefill": {"num_replicas": 1, "engine_kwargs": {"tensor_parallel_size": 8}},
                        "decode": {"num_replicas": 1, "engine_kwargs": {"tensor_parallel_size": 1}},
                    },
                )
            ],
            backend="dynamo",
        )
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
        ]
        with pytest.raises(ValueError, match="prefill requests TP=8"):
            DynamoBackend._validate_gpu_requirements(server, inventory)


# ---------------------------------------------------------------------------
# Mixed disagg + non-disagg multi-model inventory isolation
# ---------------------------------------------------------------------------


class TestMixedDisaggInventory:
    """Verify that disagg and non-disagg models share the GPU inventory correctly."""

    def test_asymmetric_tp_placement_decode_tp8_prefill_tp4(self):
        """Decode TP=8 takes all of n1, then 2 prefill TP=4 land on n2."""
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 8, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 8, "is_head": False},
        ]

        # Decode planned first (matches _launch_disagg_workers order)
        decode_plans = plan_replica_placement(num_replicas=1, tp_size=8, _inventory=inventory)
        assert len(decode_plans) == 1
        assert decode_plans[0].ranks[0].node_id == "n1"
        assert decode_plans[0].ranks[0].num_gpus == 8
        inventory = DynamoBackend._subtract_placed_gpus(inventory, decode_plans)

        # Prefill planned second on remaining inventory
        prefill_plans = plan_replica_placement(num_replicas=2, tp_size=4, _inventory=inventory)
        assert len(prefill_plans) == 2
        for plan in prefill_plans:
            assert plan.ranks[0].node_id == "n2"
            assert plan.ranks[0].num_gpus == 4
        inventory = DynamoBackend._subtract_placed_gpus(inventory, prefill_plans)

        assert len(inventory) == 0

    def test_disagg_and_non_disagg_use_disjoint_gpus(self):
        """A non-disagg model followed by a disagg model should not double-book GPUs.

        This is a regression test for the bug where _launch_disagg_workers
        rebuilt inventory from scratch via _nodes= instead of using the
        shared _inventory= parameter.
        """
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "node_ip": "10.0.0.2", "num_gpus": 4, "is_head": False},
        ]

        # Non-disagg model: 1 replica, TP=4 — takes all 4 GPUs on n1
        plans_a = plan_replica_placement(num_replicas=1, tp_size=4, _inventory=inventory)
        assert plans_a[0].ranks[0].node_id == "n1"
        inventory = DynamoBackend._subtract_placed_gpus(inventory, plans_a)

        # Disagg model: 1 decode worker, TP=4 — must land on n2, not n1
        plans_b = plan_replica_placement(num_replicas=1, tp_size=4, _inventory=inventory)
        assert plans_b[0].ranks[0].node_id == "n2"
        inventory = DynamoBackend._subtract_placed_gpus(inventory, plans_b)

        # No GPUs left
        assert len(inventory) == 0

    def test_mixed_models_overcommit_detected(self):
        """Non-disagg + disagg models exceeding total GPUs should fail at placement."""
        inventory = [
            {"node_id": "n1", "node_ip": "10.0.0.1", "num_gpus": 4, "is_head": False},
        ]

        # First model takes all 4 GPUs
        plans_a = plan_replica_placement(num_replicas=1, tp_size=4, _inventory=inventory)
        inventory = DynamoBackend._subtract_placed_gpus(inventory, plans_a)

        # Second model has no GPUs left
        with pytest.raises(RuntimeError, match="No GPU nodes"):
            plan_replica_placement(num_replicas=1, tp_size=1, _inventory=inventory)


# ---------------------------------------------------------------------------
# SubprocessError structured context
# ---------------------------------------------------------------------------


class TestSubprocessErrorContext:
    """Verify SubprocessError carries debug_context."""

    def test_raise_subprocess_error_includes_debug_context(self):
        server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo")
        backend = DynamoBackend(server)

        with pytest.raises(SubprocessError) as exc_info:
            backend._raise_subprocess_error("test_worker", "some log output", reason="crashed")

        err = exc_info.value
        assert err.debug_context["label"] == "test_worker"
        assert err.debug_context["reason"] == "crashed"
        assert "some log output" in err.debug_context["log_tail"]
        assert "crashed" in str(err)

    def test_subprocess_error_is_runtime_error(self):
        """SubprocessError should be catchable as RuntimeError for backwards compatibility."""
        err = SubprocessError("test", debug_context={"key": "val"})
        assert isinstance(err, RuntimeError)
        assert err.debug_context == {"key": "val"}
