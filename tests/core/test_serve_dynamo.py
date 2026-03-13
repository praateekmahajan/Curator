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
    WorkerPlacement,
    _engine_kwargs_to_cli_flags,
    build_gpu_placement,
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
# GPU placement planner — real Ray cluster
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestGpuPlacement:
    """Tests against the real shared Ray cluster (session fixture provides 2 GPUs)."""

    def test_placement_uses_available_gpus(self):
        """Placement succeeds with the real cluster's GPU inventory."""
        placements = build_gpu_placement(num_replicas=2, gpus_per_replica=1)
        assert len(placements) == 2
        assert placements[0].gpu_ids == [0]
        assert placements[1].gpu_ids == [1]

    def test_placement_tp2(self):
        """TP=2 on 2 GPUs → 1 worker with GPUs [0,1]."""
        placements = build_gpu_placement(num_replicas=1, gpus_per_replica=2)
        assert len(placements) == 1
        assert placements[0].gpu_ids == [0, 1]

    def test_insufficient_gpus_raises(self):
        """Requesting more GPUs than available raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Need"):
            build_gpu_placement(num_replicas=10, gpus_per_replica=4)

    def test_placement_has_valid_node_info(self):
        """Each placement has non-empty node_id and node_ip."""
        placements = build_gpu_placement(num_replicas=1, gpus_per_replica=1)
        assert placements[0].node_id
        assert placements[0].node_ip


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
        """Verify _active_servers tracks dynamo backend correctly."""
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
        """stop() on a never-started backend is a no-op."""
        server = InferenceServer(
            models=[InferenceModelConfig(model_identifier="m")],
            backend="dynamo",
        )
        backend = DynamoBackend(server)
        backend.stop()  # should not raise


# ---------------------------------------------------------------------------
# WorkerPlacement dataclass
# ---------------------------------------------------------------------------


class TestWorkerPlacement:
    def test_fields(self):
        p = WorkerPlacement(node_id="n1", node_ip="10.0.0.1", gpu_ids=[0, 1], worker_index=0)
        assert p.node_id == "n1"
        assert p.gpu_ids == [0, 1]
        assert p.worker_index == 0
