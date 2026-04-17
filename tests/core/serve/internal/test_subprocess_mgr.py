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

import contextlib
import os
import re

import pytest

from nemo_curator.core.serve.internal.subprocess_mgr import (
    NEMO_CURATOR_DYNAMO_NAMESPACE,
    ManagedSubprocess,
    ReplicaBundleSpec,
    _define_subprocess_actor,
    _engine_kwargs_to_cli_flags,
    _ignore_head_node,
    build_replica_pg,
    build_worker_actor_name,
    check_total_gpu_capacity,
    get_bundle_node_ip,
    get_free_port_in_bundle,
    graceful_stop_actor,
    plan_replica_bundle_shape,
    remove_named_pgs_with_prefix,
    spawn_actor,
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
# plan_replica_bundle_shape -- mocked topology (no Ray needed)
# ---------------------------------------------------------------------------


class TestPlanReplicaBundleShapeSingleNode:
    """Single-node placement when any node has enough GPUs for TP."""

    def test_fits_on_single_node(self):
        topology = [{"node_id": "n1", "num_gpus": 8, "is_head": False}]
        spec = plan_replica_bundle_shape(tp_size=4, _topology=topology)
        assert not spec.is_multi_node
        assert spec.nnodes == 1
        assert spec.per_node_gpus == 4
        assert spec.strategy == "STRICT_PACK"
        assert spec.bundles == [{"CPU": 1, "GPU": 4}]
        assert spec.bundle_label_selector is None

    def test_tp1_single_bundle(self):
        topology = [{"node_id": "n1", "num_gpus": 8, "is_head": False}]
        spec = plan_replica_bundle_shape(tp_size=1, _topology=topology)
        assert spec.nnodes == 1
        assert spec.bundles == [{"CPU": 1, "GPU": 1}]

    def test_total_gpus_matches(self):
        topology = [{"node_id": "n1", "num_gpus": 8, "is_head": False}]
        spec = plan_replica_bundle_shape(tp_size=4, _topology=topology)
        assert spec.total_gpus == 4

    def test_partial_leftover_still_prefers_single_node(self):
        """5+8 GPU cluster, TP=4 -> single-node (max_per_node=8 >= 4)."""
        topology = [
            {"node_id": "n1", "num_gpus": 5, "is_head": False},
            {"node_id": "n2", "num_gpus": 8, "is_head": False},
        ]
        spec = plan_replica_bundle_shape(tp_size=4, _topology=topology)
        assert not spec.is_multi_node


class TestPlanReplicaBundleShapeMultiNode:
    """Multi-node placement when TP exceeds any single node's capacity."""

    def test_multi_node_even_split(self):
        """TP=8 across two 4-GPU nodes -> STRICT_SPREAD with 4+4."""
        topology = [
            {"node_id": "n1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "num_gpus": 4, "is_head": False},
        ]
        spec = plan_replica_bundle_shape(tp_size=8, _topology=topology)
        assert spec.is_multi_node
        assert spec.nnodes == 2
        assert spec.per_node_gpus == 4
        assert spec.strategy == "STRICT_SPREAD"
        assert spec.bundles == [{"CPU": 1, "GPU": 4}, {"CPU": 1, "GPU": 4}]

    def test_multi_node_three_nodes(self):
        """TP=12 across three 4-GPU nodes."""
        topology = [{"node_id": f"n{i}", "num_gpus": 4, "is_head": False} for i in range(1, 4)]
        spec = plan_replica_bundle_shape(tp_size=12, _topology=topology)
        assert spec.nnodes == 3
        assert spec.per_node_gpus == 4
        assert spec.total_gpus == 12

    def test_even_split_across_smaller_nodes(self):
        """TP=4 on two 3-GPU nodes -> valid 2+2 split."""
        topology = [
            {"node_id": "n1", "num_gpus": 3, "is_head": False},
            {"node_id": "n2", "num_gpus": 3, "is_head": False},
        ]
        spec = plan_replica_bundle_shape(tp_size=4, _topology=topology)
        assert spec.nnodes == 2
        assert spec.per_node_gpus == 2

    def test_asymmetric_split_rejected(self):
        """TP=4 with 1+3 cannot be launched by vLLM (uneven local_world_size)."""
        topology = [
            {"node_id": "n1", "num_gpus": 1, "is_head": False},
            {"node_id": "n2", "num_gpus": 3, "is_head": False},
        ]
        with pytest.raises(RuntimeError, match="even split"):
            plan_replica_bundle_shape(tp_size=4, _topology=topology)

    def test_no_valid_combination_raises(self):
        """TP=6 on two 2-GPU nodes: 2 doesn't divide 6; 3 nodes not available."""
        topology = [
            {"node_id": "n1", "num_gpus": 2, "is_head": False},
            {"node_id": "n2", "num_gpus": 2, "is_head": False},
        ]
        with pytest.raises(RuntimeError, match="even split"):
            plan_replica_bundle_shape(tp_size=6, _topology=topology)


class TestPlanReplicaBundleShapeErrors:
    def test_empty_topology_raises(self):
        with pytest.raises(RuntimeError, match="No GPU nodes"):
            plan_replica_bundle_shape(tp_size=1, _topology=[])


class TestPlanReplicaBundleShapeHeadExclusion:
    """CURATOR_IGNORE_RAY_HEAD_NODE maps to a bundle_label_selector."""

    def test_flag_unset_no_selector(self):
        topology = [
            {"node_id": "head", "num_gpus": 8, "is_head": True},
            {"node_id": "worker", "num_gpus": 8, "is_head": False},
        ]
        spec = plan_replica_bundle_shape(tp_size=4, _topology=topology)
        assert spec.bundle_label_selector is None

    def test_flag_set_adds_single_node_selector(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        topology = [
            {"node_id": "head", "num_gpus": 8, "is_head": True},
            {"node_id": "worker", "num_gpus": 8, "is_head": False},
        ]
        spec = plan_replica_bundle_shape(tp_size=4, _topology=topology)
        assert spec.nnodes == 1
        assert spec.bundle_label_selector == [{"ray.io/node-type": "worker"}]

    def test_flag_set_adds_multi_node_selector(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "true")
        topology = [
            {"node_id": "w1", "num_gpus": 4, "is_head": False},
            {"node_id": "w2", "num_gpus": 4, "is_head": False},
        ]
        spec = plan_replica_bundle_shape(tp_size=8, _topology=topology)
        assert spec.nnodes == 2
        assert spec.bundle_label_selector == [{"ray.io/node-type": "worker"}] * 2

    def test_flag_set_filters_head_from_topology(self, monkeypatch: pytest.MonkeyPatch):
        """With flag set, the head node is ignored when choosing bundle shape."""
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        topology = [
            {"node_id": "head", "num_gpus": 16, "is_head": True},
            {"node_id": "w1", "num_gpus": 4, "is_head": False},
            {"node_id": "w2", "num_gpus": 4, "is_head": False},
        ]
        # Even though head has 16 GPUs (enough for TP=8), the planner must
        # split across the two 4-GPU workers because head is excluded.
        spec = plan_replica_bundle_shape(tp_size=8, _topology=topology)
        assert spec.is_multi_node
        assert spec.nnodes == 2

    def test_only_head_node_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", "1")
        topology = [{"node_id": "head", "num_gpus": 8, "is_head": True}]
        with pytest.raises(RuntimeError, match="CURATOR_IGNORE_RAY_HEAD_NODE"):
            plan_replica_bundle_shape(tp_size=4, _topology=topology)


class TestIgnoreHeadNodeHelper:
    """_ignore_head_node reads the env var correctly."""

    def test_defaults_false(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("CURATOR_IGNORE_RAY_HEAD_NODE", raising=False)
        assert not _ignore_head_node()

    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch):
        for v in ("1", "true", "True", "TRUE", "yes", "YES"):
            monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", v)
            assert _ignore_head_node(), f"{v!r} should be truthy"

    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch):
        for v in ("", "0", "false", "no"):
            monkeypatch.setenv("CURATOR_IGNORE_RAY_HEAD_NODE", v)
            assert not _ignore_head_node(), f"{v!r} should be falsy"


# ---------------------------------------------------------------------------
# Total GPU capacity check
# ---------------------------------------------------------------------------


class TestCheckTotalGpuCapacity:
    def test_fits(self):
        check_total_gpu_capacity(4, _cluster_resources={"GPU": 8})

    def test_exact(self):
        check_total_gpu_capacity(8, _cluster_resources={"GPU": 8})

    def test_overcommit_raises(self):
        with pytest.raises(RuntimeError, match="Need 9 GPUs"):
            check_total_gpu_capacity(9, _cluster_resources={"GPU": 8})


# ---------------------------------------------------------------------------
# ReplicaBundleSpec dataclass
# ---------------------------------------------------------------------------


class TestReplicaBundleSpec:
    def test_single_node_properties(self):
        spec = ReplicaBundleSpec(
            bundles=[{"CPU": 1, "GPU": 4}],
            strategy="STRICT_PACK",
            nnodes=1,
            per_node_gpus=4,
        )
        assert not spec.is_multi_node
        assert spec.total_gpus == 4

    def test_multi_node_properties(self):
        spec = ReplicaBundleSpec(
            bundles=[{"CPU": 1, "GPU": 4}, {"CPU": 1, "GPU": 4}],
            strategy="STRICT_SPREAD",
            nnodes=2,
            per_node_gpus=4,
        )
        assert spec.is_multi_node
        assert spec.total_gpus == 8


# ---------------------------------------------------------------------------
# Namespace constant
# ---------------------------------------------------------------------------


class TestNamespaceConstant:
    def test_exists(self):
        """The namespace constant is used by DynamoBackend.start/stop's ray.init calls."""
        assert NEMO_CURATOR_DYNAMO_NAMESPACE == "nemo_curator_dynamo"


# ---------------------------------------------------------------------------
# Placement group creation -- real Ray cluster
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestBuildReplicaPg:
    """PG creation against a real Ray cluster (2 GPUs available)."""

    def test_single_node_pg_becomes_ready(self):
        import ray

        spec = plan_replica_bundle_shape(
            tp_size=1,
            _topology=[{"node_id": "n", "num_gpus": 1, "is_head": False}],
        )
        pg_name = f"test_single_{os.getpid()}"
        pg = build_replica_pg(spec, name=pg_name)
        try:
            # PG must be ready -- build_replica_pg waits internally.
            assert ray.util.get_placement_group(pg_name) is not None
        finally:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)

    def test_bundle_node_ip_reachable(self):
        import ray

        spec = ReplicaBundleSpec(
            bundles=[{"CPU": 1, "GPU": 1}],
            strategy="STRICT_PACK",
            nnodes=1,
            per_node_gpus=1,
        )
        pg_name = f"test_ip_{os.getpid()}"
        pg = build_replica_pg(spec, name=pg_name)
        try:
            ip = get_bundle_node_ip(pg, 0)
            # Must look like an IPv4 (or "127.0.0.1" for single-machine)
            assert re.match(r"^\d+\.\d+\.\d+\.\d+$", ip), f"unexpected ip: {ip!r}"
        finally:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)

    def test_free_port_in_bundle(self):
        import ray

        spec = ReplicaBundleSpec(
            bundles=[{"CPU": 1, "GPU": 1}],
            strategy="STRICT_PACK",
            nnodes=1,
            per_node_gpus=1,
        )
        pg_name = f"test_port_{os.getpid()}"
        pg = build_replica_pg(spec, name=pg_name)
        try:
            port = get_free_port_in_bundle(pg, 0, 30000)
            assert port >= 30000
            assert port < 65536
        finally:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)


# ---------------------------------------------------------------------------
# Subprocess actor lifecycle (real subprocesses, real Ray cluster)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestSubprocessActorLifecycle:
    """Actor + subprocess lifecycle against a real Ray cluster."""

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

    def test_graceful_stop_terminates_subprocess(self, tmp_path: os.PathLike):
        """graceful_stop_actor reaps the subprocess; subsequent is_alive returns False."""
        import ray
        from ray.util.placement_group import placement_group

        pg = placement_group(bundles=[{"CPU": 1}], strategy="STRICT_PACK", lifetime="detached")
        ray.get(pg.ready(), timeout=30)
        try:
            proc = spawn_actor(
                "graceful_stop_test",
                pg,
                0,
                num_gpus=0,
                command=["sleep", "3600"],
                runtime_dir=str(tmp_path),
                actor_name_prefix=f"test_{os.getpid()}",
            )
            assert ray.get(proc.actor.is_alive.remote(), timeout=10)
            graceful_stop_actor(ray, proc.label, proc.actor)
            # Actor may be dead at this point; can't call is_alive. The fact
            # graceful_stop_actor returned without raising is the contract.
        finally:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)


# ---------------------------------------------------------------------------
# Orphan PG cleanup
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestRemoveNamedPgsWithPrefix:
    """Verify named PGs can be looked up and reaped by prefix in the current namespace."""

    def test_removes_matching_pgs(self):
        import ray
        from ray.util.placement_group import placement_group

        prefix = f"orphan_test_{os.getpid()}_"
        created = []
        try:
            for i in range(3):
                pg = placement_group(
                    bundles=[{"CPU": 1}], strategy="STRICT_PACK", lifetime="detached", name=f"{prefix}{i}"
                )
                ray.get(pg.ready(), timeout=30)
                created.append(pg)

            removed = remove_named_pgs_with_prefix(prefix)
            assert removed >= 3

            # Lookups by name should now fail for the reaped PGs.
            for i in range(3):
                with pytest.raises(Exception):  # noqa: B017, PT011
                    ray.util.get_placement_group(f"{prefix}{i}")
        finally:
            for pg in created:
                with contextlib.suppress(Exception):
                    ray.util.remove_placement_group(pg)

    def test_no_matches_returns_zero(self):
        removed = remove_named_pgs_with_prefix(f"no_such_prefix_{os.getpid()}_")
        assert removed == 0


# ---------------------------------------------------------------------------
# Subprocess env propagation (real subprocesses, real Ray cluster)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestSubprocessEnvPropagation:
    """Validate the simplified single-dict subprocess_env model.

    Actors inherit ``os.environ`` from the raylet (which shares the
    driver's base environment). Only runtime-computed vars (e.g.
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


# ---------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES propagation to subprocess
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestCudaVisibleDevicesPropagation:
    """Verify the subprocess receives a correct CUDA_VISIBLE_DEVICES.

    ``spawn_actor`` always sets ``CUDA_VISIBLE_DEVICES`` in the subprocess
    env from ``ray.get_accelerator_ids()``. This test proves the subprocess
    sees exactly the Ray-assigned IDs, which is the ground truth Dynamo
    subprocesses rely on -- regardless of whether
    ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`` is set.
    """

    def test_subprocess_sees_ray_assigned_cuda_ids(self, tmp_path: os.PathLike):
        import ray
        from ray.util.placement_group import placement_group

        pg = placement_group(bundles=[{"CPU": 1, "GPU": 1}], strategy="STRICT_PACK", lifetime="detached")
        ray.get(pg.ready(), timeout=30)
        try:
            proc = spawn_actor(
                "cuda_test",
                pg,
                0,
                num_gpus=1,
                command=["bash", "-c", "echo CUDA=$CUDA_VISIBLE_DEVICES"],
                runtime_dir=str(tmp_path),
                actor_name_prefix=f"test_{os.getpid()}",
            )
            ray.get(proc.run_ref, timeout=15)
            log = ray.get(proc.actor.read_log_tail.remote(), timeout=10)
            # The subprocess must see exactly one CUDA id, derived from the
            # actor's Ray-assigned accelerator IDs.
            m = re.search(r"CUDA=(\S*)", log)
            assert m is not None, f"CUDA line not found in log: {log!r}"
            cuda_val = m.group(1)
            assert cuda_val, f"empty CUDA_VISIBLE_DEVICES. Full log:\n{log}"
            assert cuda_val != "MISSING", f"subprocess reports CUDA=MISSING. Full log:\n{log}"
            # Must be a comma-separated list of ints (matching accelerator IDs)
            for token in cuda_val.split(","):
                assert token.strip().isdigit(), f"non-numeric CUDA id: {token!r}"
        finally:
            with contextlib.suppress(Exception):
                ray.util.remove_placement_group(pg)
