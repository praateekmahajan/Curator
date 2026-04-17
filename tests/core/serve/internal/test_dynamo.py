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
import json
import os

import pytest

from nemo_curator.core.serve import InferenceModelConfig, InferenceServer
from nemo_curator.core.serve.internal.dynamo import DynamoBackend, _dynamo_endpoint, _model_name_to_component
from nemo_curator.core.serve.internal.errors import SubprocessError
from nemo_curator.core.serve.internal.subprocess_mgr import (
    ManagedSubprocess,
    ReplicaBundleSpec,
    _define_subprocess_actor,
    graceful_stop_actor,
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
        assert pek == {"tensor_parallel_size": 4, "max_model_len": 8192}
        assert dek == {"tensor_parallel_size": 2, "max_model_len": 8192}


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

    def test_plan_disagg_shape_rejects_multi_node_tp(self, monkeypatch: pytest.MonkeyPatch):
        """Disagg worker TP must fit on one node."""
        topology = [
            {"node_id": "n1", "num_gpus": 4, "is_head": False},
            {"node_id": "n2", "num_gpus": 4, "is_head": False},
        ]
        monkeypatch.setattr(
            "nemo_curator.core.serve.internal.subprocess_mgr._get_gpu_topology", lambda *_a, **_k: topology
        )
        with pytest.raises(ValueError, match="does not support multi-node TP"):
            DynamoBackend._plan_disagg_shape(tp_size=8, role="prefill", worker_index=0, model_name="m")

    def test_plan_disagg_shape_accepts_single_node(self, monkeypatch: pytest.MonkeyPatch):
        topology = [{"node_id": "n1", "num_gpus": 8, "is_head": False}]
        monkeypatch.setattr(
            "nemo_curator.core.serve.internal.subprocess_mgr._get_gpu_topology", lambda *_a, **_k: topology
        )
        spec = DynamoBackend._plan_disagg_shape(tp_size=4, role="decode", worker_index=0, model_name="m")
        assert not spec.is_multi_node
        assert spec.per_node_gpus == 4

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

    def test_rejects_mismatched_router_mode(self):
        server = InferenceServer(
            models=[
                InferenceModelConfig(model_identifier="model-a", dynamo_config={"router_mode": "kv"}),
                InferenceModelConfig(model_identifier="model-b", dynamo_config={"router_mode": "round-robin"}),
            ],
            backend="dynamo",
        )
        with pytest.raises(ValueError, match="router_mode"):
            DynamoBackend._validate_frontend_config(server)

    def test_rejects_mismatched_router_kv_events(self):
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="model-a",
                    dynamo_config={"router_mode": "kv", "router_kv_events": True},
                ),
                InferenceModelConfig(
                    model_identifier="model-b",
                    dynamo_config={"router_mode": "kv", "router_kv_events": False},
                ),
            ],
            backend="dynamo",
        )
        with pytest.raises(ValueError, match="router_kv_events"):
            DynamoBackend._validate_frontend_config(server)

    def test_rejects_mismatched_router_temperature(self):
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="model-a",
                    dynamo_config={"router_mode": "kv", "router_temperature": 0.0},
                ),
                InferenceModelConfig(
                    model_identifier="model-b",
                    dynamo_config={"router_mode": "kv", "router_temperature": 0.7},
                ),
            ],
            backend="dynamo",
        )
        with pytest.raises(ValueError, match="router_temperature"):
            DynamoBackend._validate_frontend_config(server)

    def test_allows_omitted_router_fields_to_inherit_shared_defaults(self):
        server = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier="model-a",
                    dynamo_config={"router_mode": "kv", "router_kv_events": False},
                ),
                InferenceModelConfig(model_identifier="model-b", dynamo_config={}),
            ],
            backend="dynamo",
        )
        DynamoBackend._validate_frontend_config(server)  # should not raise


# ---------------------------------------------------------------------------
# Pre-infra GPU validation
# ---------------------------------------------------------------------------


def _mock_topology_and_cluster(monkeypatch: pytest.MonkeyPatch, topology: list[dict], total_gpus: int) -> None:
    """Patch the symbols as seen from dynamo.py's module namespace.

    dynamo.py does ``from ...subprocess_mgr import _get_gpu_topology, check_total_gpu_capacity``
    at module top, which binds local aliases. Patching the source module alone
    won't override those aliases.
    """
    monkeypatch.setattr("nemo_curator.core.serve.internal.dynamo._get_gpu_topology", lambda *_a, **_k: topology)

    def _fake_check(needed: int, **_: object) -> None:
        if needed > total_gpus:
            msg = f"Need {needed} GPUs but cluster has {total_gpus} total."
            raise RuntimeError(msg)

    monkeypatch.setattr("nemo_curator.core.serve.internal.dynamo.check_total_gpu_capacity", _fake_check)


class TestGpuValidation:
    """Tests for DynamoBackend._validate_gpu_requirements."""

    def test_rejects_disagg_tp_exceeding_max_node_gpus(self, monkeypatch: pytest.MonkeyPatch):
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
        _mock_topology_and_cluster(
            monkeypatch,
            topology=[
                {"node_id": "n1", "num_gpus": 4, "is_head": False},
                {"node_id": "n2", "num_gpus": 4, "is_head": False},
            ],
            total_gpus=8,
        )
        with pytest.raises(ValueError, match="does not support multi-node TP"):
            DynamoBackend._validate_gpu_requirements(server)

    def test_rejects_aggregate_gpu_overcommit(self, monkeypatch: pytest.MonkeyPatch):
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
        _mock_topology_and_cluster(
            monkeypatch,
            topology=[{"node_id": "n1", "num_gpus": 4, "is_head": False}],
            total_gpus=4,
        )
        with pytest.raises(RuntimeError, match="Need 8 GPUs"):
            DynamoBackend._validate_gpu_requirements(server)

    def test_accepts_valid_config(self, monkeypatch: pytest.MonkeyPatch):
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
        _mock_topology_and_cluster(
            monkeypatch,
            topology=[{"node_id": "n1", "num_gpus": 4, "is_head": False}],
            total_gpus=4,
        )
        DynamoBackend._validate_gpu_requirements(server)  # should not raise

    def test_disagg_aggregate_check_counts_prefill_and_decode(self, monkeypatch: pytest.MonkeyPatch):
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
        _mock_topology_and_cluster(
            monkeypatch,
            topology=[{"node_id": "n1", "num_gpus": 4, "is_head": False}],
            total_gpus=4,
        )
        with pytest.raises(RuntimeError, match="Need 8 GPUs"):
            DynamoBackend._validate_gpu_requirements(server)

    def test_disagg_ignores_deployment_config_num_replicas(self, monkeypatch: pytest.MonkeyPatch):
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
        _mock_topology_and_cluster(
            monkeypatch,
            topology=[{"node_id": "n1", "num_gpus": 2, "is_head": False}],
            total_gpus=2,
        )
        DynamoBackend._validate_gpu_requirements(server)  # should not raise

    def test_asymmetric_tp_rejects_role_exceeding_node(self, monkeypatch: pytest.MonkeyPatch):
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
        _mock_topology_and_cluster(
            monkeypatch,
            topology=[
                {"node_id": "n1", "num_gpus": 4, "is_head": False},
                {"node_id": "n2", "num_gpus": 4, "is_head": False},
            ],
            total_gpus=8,
        )
        with pytest.raises(ValueError, match="prefill requests TP=8"):
            DynamoBackend._validate_gpu_requirements(server)


# ---------------------------------------------------------------------------
# Worker/frontend launch args (with PG-aware spawn_actor mocked)
# ---------------------------------------------------------------------------


def _capture_spawn(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture ``spawn_actor()`` calls without actually launching Ray actors."""
    calls: list[dict] = []

    def fake_spawn_actor(label: str, pg, bundle_index: int, **kwargs) -> ManagedSubprocess:  # noqa: ANN001
        calls.append({"label": label, "pg": pg, "bundle_index": bundle_index, **kwargs})
        return ManagedSubprocess(label=label, actor=object())

    monkeypatch.setattr("nemo_curator.core.serve.internal.dynamo.spawn_actor", fake_spawn_actor)
    return calls


def _single_node_spec(num_gpus: int = 1) -> ReplicaBundleSpec:
    return ReplicaBundleSpec(
        bundles=[{"CPU": 1, "GPU": num_gpus}],
        strategy="STRICT_PACK",
        nnodes=1,
        per_node_gpus=num_gpus,
    )


class TestAggregatedWorkerLaunchArgs:
    """Tests for explicit --kv-events-config in aggregated worker launch paths."""

    def test_launch_worker_disables_kv_events_by_default(self, monkeypatch: pytest.MonkeyPatch):
        calls = _capture_spawn(monkeypatch)
        backend = DynamoBackend(InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo"))
        backend._runtime_dir = "/tmp/runtime"  # noqa: S108

        backend._launch_worker(
            replica_index=0,
            model_config=InferenceModelConfig(model_identifier="Qwen/Qwen3-0.6B"),
            base_env={},
            pg=object(),
            bundle_index=0,
            num_gpus=1,
            namespace="curator",
            request_plane="nats",
            event_plane="nats",
            spec=_single_node_spec(1),
            master_addr=None,
        )

        python_args = calls[0]["python_args"]
        idx = python_args.index("--kv-events-config")
        assert json.loads(python_args[idx + 1]) == {"enable_kv_cache_events": False}

    def test_launch_worker_enables_exact_kv_events_for_kv_router(self, monkeypatch: pytest.MonkeyPatch):
        calls = _capture_spawn(monkeypatch)
        monkeypatch.setattr(
            "nemo_curator.core.serve.internal.dynamo.get_free_port_in_bundle",
            lambda _pg, _bundle, _start: 24567,
        )

        backend = DynamoBackend(InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo"))
        backend._runtime_dir = "/tmp/runtime"  # noqa: S108

        model = InferenceModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            dynamo_config={"router_mode": "kv", "router_kv_events": True},
        )
        backend._launch_worker(
            replica_index=3,
            model_config=model,
            base_env={},
            pg=object(),
            bundle_index=0,
            num_gpus=1,
            namespace="curator",
            request_plane="nats",
            event_plane="zmq",
            spec=_single_node_spec(1),
            master_addr=None,
        )

        python_args = calls[0]["python_args"]
        cfg = json.loads(python_args[python_args.index("--kv-events-config") + 1])
        assert cfg["enable_kv_cache_events"] is True
        assert cfg["endpoint"] == "tcp://*:24567"
        assert cfg["publisher"] == "zmq"
        assert cfg["topic"] == "kv-events"

    def test_launch_worker_multi_node_adds_nnodes_and_master(self, monkeypatch: pytest.MonkeyPatch):
        calls = _capture_spawn(monkeypatch)
        backend = DynamoBackend(InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo"))
        backend._runtime_dir = "/tmp/runtime"  # noqa: S108

        spec = ReplicaBundleSpec(
            bundles=[{"CPU": 1, "GPU": 4}, {"CPU": 1, "GPU": 4}],
            strategy="STRICT_SPREAD",
            nnodes=2,
            per_node_gpus=4,
        )
        backend._launch_worker(
            replica_index=0,
            model_config=InferenceModelConfig(
                model_identifier="Qwen/Qwen3-0.6B",
                engine_kwargs={"tensor_parallel_size": 8},
            ),
            base_env={},
            pg=object(),
            bundle_index=0,
            num_gpus=4,
            namespace="curator",
            request_plane="nats",
            event_plane="nats",
            spec=spec,
            master_addr="10.0.0.5",
        )

        python_args = calls[0]["python_args"]
        assert "--nnodes" in python_args
        assert python_args[python_args.index("--nnodes") + 1] == "2"
        assert python_args[python_args.index("--node-rank") + 1] == "0"
        assert python_args[python_args.index("--master-addr") + 1] == "10.0.0.5"

    def test_launch_headless_worker_uses_bundle_index_as_rank(self, monkeypatch: pytest.MonkeyPatch):
        calls = _capture_spawn(monkeypatch)
        backend = DynamoBackend(InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo"))
        backend._runtime_dir = "/tmp/runtime"  # noqa: S108

        spec = ReplicaBundleSpec(
            bundles=[{"CPU": 1, "GPU": 4}, {"CPU": 1, "GPU": 4}],
            strategy="STRICT_SPREAD",
            nnodes=2,
            per_node_gpus=4,
        )
        backend._launch_headless_worker(
            replica_index=0,
            model_config=InferenceModelConfig(
                model_identifier="Qwen/Qwen3-0.6B",
                engine_kwargs={"tensor_parallel_size": 8},
            ),
            base_env={},
            pg=object(),
            bundle_index=1,
            num_gpus=4,
            spec=spec,
            master_addr="10.0.0.5",
        )

        python_args = calls[0]["python_args"]
        assert "--headless" in python_args
        assert python_args[python_args.index("--node-rank") + 1] == "1"
        assert python_args[python_args.index("--master-addr") + 1] == "10.0.0.5"
        # KV events must be explicitly disabled for headless ranks.
        cfg = json.loads(python_args[python_args.index("--kv-events-config") + 1])
        assert cfg["enable_kv_cache_events"] is False


class TestFrontendLaunchArgs:
    """Tests for Dynamo frontend router CLI args."""

    def test_launch_frontend_uses_router_kv_events_by_default(self, monkeypatch: pytest.MonkeyPatch):
        calls = _capture_spawn(monkeypatch)
        backend = DynamoBackend(InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo"))
        backend._infra_pg = object()
        backend._launch_frontend(
            8000,
            {},
            namespace="curator",
            request_plane="nats",
            event_plane="nats",
            frontend_router_cfg={"router_mode": "kv", "router_kv_events": True},
            runtime_env=None,
        )

        args = calls[0]["python_args"]
        assert "--router-mode" in args
        assert args[args.index("--router-mode") + 1] == "kv"
        assert "--no-router-kv-events" not in args

    def test_launch_frontend_uses_no_router_kv_events_for_approx_mode(self, monkeypatch: pytest.MonkeyPatch):
        calls = _capture_spawn(monkeypatch)
        backend = DynamoBackend(InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo"))
        backend._infra_pg = object()
        backend._launch_frontend(
            8000,
            {},
            namespace="curator",
            request_plane="nats",
            event_plane="nats",
            frontend_router_cfg={
                "router_mode": "kv",
                "router_kv_events": False,
                "router_ttl_secs": 90.0,
                "router_max_tree_size": 2**20,
                "router_prune_target_ratio": 0.8,
            },
            runtime_env=None,
        )

        args = calls[0]["python_args"]
        assert "--no-router-kv-events" in args
        assert "--router-ttl-secs" in args

    def test_launch_frontend_no_router_args_when_no_router_mode(self, monkeypatch: pytest.MonkeyPatch):
        calls = _capture_spawn(monkeypatch)
        backend = DynamoBackend(InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo"))
        backend._infra_pg = object()
        backend._launch_frontend(
            8000,
            {},
            namespace="curator",
            request_plane="nats",
            event_plane="nats",
            frontend_router_cfg={"router_mode": None},
            runtime_env=None,
        )

        args = calls[0]["python_args"]
        assert "--router-mode" not in args
        assert "--no-router-kv-events" not in args

    def test_launch_frontend_does_not_hardcode_router_reset_states(self, monkeypatch: pytest.MonkeyPatch):
        calls = _capture_spawn(monkeypatch)
        backend = DynamoBackend(InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo"))
        backend._infra_pg = object()
        backend._launch_frontend(
            8000,
            {},
            namespace="curator",
            request_plane="nats",
            event_plane="nats",
            frontend_router_cfg={"router_mode": "kv", "router_kv_events": True},
            runtime_env=None,
        )

        args = calls[0]["python_args"]
        assert "--router-reset-states" not in args


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

            backend._check_subprocess_health()  # should not raise
        finally:
            with contextlib.suppress(Exception):
                ray.kill(proc.actor, no_restart=True)

    def test_stop_idempotent_after_partial_failure(self):
        """Calling stop() twice after a partial start should not raise."""
        server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")], backend="dynamo")
        backend = DynamoBackend(server)
        backend._worker_actors = []
        backend._frontend_actor = None
        backend.stop()
        backend.stop()  # second call should be safe

    def test_graceful_stop_also_kills_child_processes(self, tmp_path: os.PathLike):
        """graceful_stop_actor must also terminate child processes spawned by the subprocess.

        Subprocesses are launched with start_new_session=True so the entire
        process group is signaled, preventing orphaned grandchildren (e.g.
        vLLM torch.distributed workers).
        """
        import signal
        import time

        import ray

        pid_file = str(tmp_path / "child_pids.txt")
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

        for pid in [parent_pid, *child_pids]:
            os.kill(pid, 0)

        graceful_stop_actor(ray, actor_name, actor)
        time.sleep(0.5)

        alive = [pid for pid in child_pids if _pid_alive(pid)]
        try:
            assert not alive, f"Child processes {alive} survived graceful_stop_actor (orphaned)"
        finally:
            for pid in [parent_pid, *child_pids]:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)


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
