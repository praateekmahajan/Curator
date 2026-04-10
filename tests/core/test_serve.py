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

from unittest.mock import patch

import pytest
from pytest_httpserver import HTTPServer

LLMConfig = pytest.importorskip("ray.serve.llm", reason="ray[serve] not installed").LLMConfig

from nemo_curator.core.serve import (  # noqa: E402
    InferenceModelConfig,
    InferenceServer,
    _active_servers,
    get_active_backend,
    is_ray_serve_active,
)

INTEGRATION_TEST_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # pragma: allowlist secret
INTEGRATION_TEST_MODEL_2 = "HuggingFaceTB/SmolLM-135M-Instruct"  # pragma: allowlist secret


class TestInferenceModelConfig:
    def test_to_llm_config_falls_back_to_identifier(self) -> None:
        config = InferenceModelConfig(model_identifier="meta-llama/Llama-3-8B")
        result = config.to_llm_config()
        assert isinstance(result, LLMConfig)
        assert result.model_loading_config.model_id == "meta-llama/Llama-3-8B"
        assert result.model_loading_config.model_source == "meta-llama/Llama-3-8B"

    def test_to_llm_config_with_model_name(self) -> None:
        custom = InferenceModelConfig(
            model_identifier="google/gemma-3-27b-it",
            model_name="gemma-27b",
            deployment_config={"autoscaling_config": {"min_replicas": 1}},
            engine_kwargs={"tensor_parallel_size": 4},
        )
        result = custom.to_llm_config()
        assert result.model_loading_config.model_id == "gemma-27b"
        assert result.model_loading_config.model_source == "google/gemma-3-27b-it"
        assert result.engine_kwargs == {"tensor_parallel_size": 4}

    def test_to_llm_config_quiet_env_merges_with_user_runtime_env(self) -> None:
        """Quiet env vars override user's logging vars but preserve other runtime_env keys."""
        config = InferenceModelConfig(
            model_identifier="some-model",
            runtime_env={
                "pip": ["my-package"],
                "env_vars": {"MY_VAR": "1", "VLLM_LOGGING_LEVEL": "DEBUG"},
            },
        )
        from nemo_curator.core.serve.internal.ray_serve import RayServeBackend

        quiet_env = RayServeBackend._quiet_runtime_env()
        result = config.to_llm_config(quiet_runtime_env=quiet_env)

        assert result.runtime_env["pip"] == ["my-package"]
        assert result.runtime_env["env_vars"]["MY_VAR"] == "1"
        # quiet overrides the user's DEBUG with WARNING
        assert result.runtime_env["env_vars"]["VLLM_LOGGING_LEVEL"] == "WARNING"
        assert result.runtime_env["env_vars"]["RAY_SERVE_LOG_TO_STDERR"] == "0"

        # Without quiet_env, user's runtime_env is passed through as-is
        result_verbose = config.to_llm_config()
        assert result_verbose.runtime_env["env_vars"]["VLLM_LOGGING_LEVEL"] == "DEBUG"
        assert "RAY_SERVE_LOG_TO_STDERR" not in result_verbose.runtime_env["env_vars"]

    def test_merge_runtime_envs_concatenates_pip_lists(self) -> None:
        """pip/uv lists are concatenated, not replaced, during merge."""
        base = {"pip": ["ai-dynamo[vllm]"], "env_vars": {"A": "1"}}
        override = {"pip": ["extra-package==1.0"], "env_vars": {"B": "2"}}
        result = InferenceModelConfig._merge_runtime_envs(base, override)

        assert sorted(result["pip"]) == sorted(["ai-dynamo[vllm]", "extra-package==1.0"])
        assert result["env_vars"] == {"A": "1", "B": "2"}

    def test_merge_runtime_envs_concatenates_uv_lists(self) -> None:
        """uv lists are concatenated like pip lists."""
        base = {"uv": ["ai-dynamo[vllm]"]}
        override = {"uv": ["other-pkg"]}
        result = InferenceModelConfig._merge_runtime_envs(base, override)

        assert sorted(result["uv"]) == sorted(["ai-dynamo[vllm]", "other-pkg"])

    def test_merge_runtime_envs_mixed_pip_uv(self) -> None:
        """Base has pip, override has uv -- both preserved."""
        base = {"pip": ["pkg-a"]}
        override = {"uv": ["pkg-b"]}
        result = InferenceModelConfig._merge_runtime_envs(base, override)

        assert result["pip"] == ["pkg-a"]
        assert result["uv"] == ["pkg-b"]


class TestBackendDispatch:
    def test_backend_param_creates_dynamo(self):
        from nemo_curator.core.serve.internal.dynamo import DynamoBackend

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


class TestActiveBackendTracking:
    def test_active_backend_tracking(self):
        _active_servers["test-dynamo"] = "dynamo"
        try:
            assert get_active_backend() == "dynamo"
        finally:
            _active_servers.pop("test-dynamo", None)


class TestInferenceServer:
    def test_endpoint_uses_configured_port(self) -> None:
        assert InferenceServer(models=[], port=9999).endpoint == "http://localhost:9999/v1"

    def test_stop_before_start_is_noop(self) -> None:
        server = InferenceServer(models=[InferenceModelConfig(model_identifier="some-model")])
        server.stop()
        assert server._started is False

    def test_wait_for_healthy_succeeds_when_models_present(self, httpserver: HTTPServer) -> None:
        """Health check succeeds once all expected model names appear."""
        model_a = InferenceModelConfig(model_identifier="model-a")
        model_b = InferenceModelConfig(model_identifier="model-b")
        httpserver.expect_request("/v1/models").respond_with_json({"data": [{"id": "model-a"}, {"id": "model-b"}]})
        server = InferenceServer(models=[model_a, model_b], port=httpserver.port, health_check_timeout_s=5)
        server._wait_for_healthy()

    def test_wait_for_healthy_times_out_on_empty_model_list(self, httpserver: HTTPServer) -> None:
        """HTTP 200 with an empty model list is not sufficient — models must appear."""
        httpserver.expect_request("/v1/models").respond_with_json({"data": []})
        server = InferenceServer(
            models=[InferenceModelConfig(model_identifier="model-a")],
            port=httpserver.port,
            health_check_timeout_s=2,
        )
        with pytest.raises(TimeoutError, match="did not become ready within 2s"):
            server._wait_for_healthy()

    def test_wait_for_healthy_times_out_on_unreachable_port(self) -> None:
        """Health check times out when the port is unreachable."""
        server = InferenceServer(
            models=[InferenceModelConfig(model_identifier="m")], port=19876, health_check_timeout_s=2
        )
        with pytest.raises(TimeoutError, match="did not become ready within 2s"):
            server._wait_for_healthy()

    def test_wait_for_healthy_no_models_succeeds_immediately(self, httpserver: HTTPServer) -> None:
        """With no models configured, health check passes on any 200 with empty list."""
        httpserver.expect_request("/v1/models").respond_with_json({"data": []})
        server = InferenceServer(models=[], port=httpserver.port, health_check_timeout_s=5)
        server._wait_for_healthy()

    def test_start_raises_when_another_server_active(self) -> None:
        """start() raises RuntimeError if another InferenceServer is already active."""
        from nemo_curator.core.serve import _active_servers

        server = InferenceServer(models=[InferenceModelConfig(model_identifier="some-model")])

        _active_servers["other-app"] = "ray_serve"
        try:
            with pytest.raises(RuntimeError, match="already active"):
                server.start()
        finally:
            _active_servers.pop("other-app", None)

    def test_stop_calls_shutdown(self) -> None:
        """stop() calls serve.shutdown() when the server was started."""
        from ray import serve

        from nemo_curator.core.serve import _active_servers

        server = InferenceServer(models=[InferenceModelConfig(model_identifier="m")])
        server._started = True
        _active_servers[server.name] = "ray_serve"
        try:
            with patch.object(serve, "shutdown"):
                server.stop()
            assert server._started is False
            assert server.name not in _active_servers
        finally:
            _active_servers.pop(server.name, None)

    def test_stop_skips_shutdown_when_not_started(self) -> None:
        """stop() on a not-started server is a no-op — serve.shutdown() is not called."""
        from ray import serve

        fresh = InferenceServer(models=[InferenceModelConfig(model_identifier="m")])
        fresh._started = False
        with patch.object(serve, "shutdown") as spy:
            fresh.stop()
            spy.assert_not_called()

    def test_stop_is_idempotent(self) -> None:
        """stop() called twice on a not-started server is safe (atexit double-call)."""
        fresh = InferenceServer(models=[InferenceModelConfig(model_identifier="m")])
        assert fresh._started is False
        fresh.stop()
        fresh.stop()
        assert fresh._started is False


# ---------------------------------------------------------------------------
# Integration tests — real Ray Serve + vLLM
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def model_server(shared_ray_cluster: str) -> InferenceServer:  # noqa: ARG001
    """Start InferenceServer once for all integration tests.

    Uses enforce_eager=True to skip torch.compile and CUDA graph capture,
    cutting vLLM startup from ~30s to ~5s.
    """
    config = InferenceModelConfig(
        model_identifier=INTEGRATION_TEST_MODEL,
        deployment_config={
            "autoscaling_config": {"min_replicas": 1, "max_replicas": 1},
        },
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 512,
            "enforce_eager": True,
        },
    )

    server = InferenceServer(models=[config], health_check_timeout_s=600)
    server.start()

    yield server

    server.stop()


@pytest.mark.gpu
@pytest.mark.usefixtures("model_server")
class TestInferenceServerIntegration:
    """Full lifecycle tests against a real InferenceServer started once for the class."""

    def test_is_active_and_queryable(self, model_server: InferenceServer) -> None:
        """Server is active, lists models, and responds to chat completions."""
        from openai import OpenAI

        assert is_ray_serve_active()
        assert model_server._started is True

        client = OpenAI(base_url=model_server.endpoint, api_key="na")

        # /v1/models lists our model
        model_ids = [m.id for m in client.models.list()]
        assert INTEGRATION_TEST_MODEL in model_ids

        # Chat completion returns a non-empty response
        response = client.chat.completions.create(
            model=INTEGRATION_TEST_MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    def test_second_start_rejected(self, model_server: InferenceServer) -> None:
        """Cannot start a second InferenceServer while one is already active."""
        server2 = InferenceServer(
            models=[
                InferenceModelConfig(
                    model_identifier=INTEGRATION_TEST_MODEL_2,
                    deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
                    engine_kwargs={"tensor_parallel_size": 1, "max_model_len": 512, "enforce_eager": True},
                )
            ],
            health_check_timeout_s=600,
        )
        with pytest.raises(RuntimeError, match="already active"):
            server2.start()

        # First server is still healthy and unaffected
        from openai import OpenAI

        client = OpenAI(base_url=model_server.endpoint, api_key="na")
        assert INTEGRATION_TEST_MODEL in {m.id for m in client.models.list()}

    def test_restart_after_stop(self, model_server: InferenceServer) -> None:
        """A new InferenceServer starts cleanly after the previous one is stopped.

        stop() calls serve.shutdown(), so start() must recreate the
        controller and HTTP proxy from scratch.  This test must run last
        in the class — it stops the shared fixture's server and starts a
        replacement.
        """
        from openai import OpenAI

        # Stop the fixture's server
        model_server.stop()
        assert not is_ray_serve_active()

        # Start a fresh server from scratch (new controller + proxy)
        config = InferenceModelConfig(
            model_identifier=INTEGRATION_TEST_MODEL,
            deployment_config={"autoscaling_config": {"min_replicas": 1, "max_replicas": 1}},
            engine_kwargs={"tensor_parallel_size": 1, "max_model_len": 512, "enforce_eager": True},
        )
        server2 = InferenceServer(models=[config], health_check_timeout_s=600)
        server2.start()

        client = OpenAI(base_url=server2.endpoint, api_key="na")
        assert INTEGRATION_TEST_MODEL in {m.id for m in client.models.list()}

        response = client.chat.completions.create(
            model=INTEGRATION_TEST_MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        assert len(response.choices[0].message.content) > 0

        server2.stop()


class TestDynamoRuntimeEnv:
    def test_default_injects_ai_dynamo_vllm(self) -> None:
        """With no user runtime_env, result contains ai-dynamo[vllm] uv requirement."""
        from nemo_curator.core.serve.internal.dynamo import DynamoBackend

        config = InferenceModelConfig(model_identifier="some-model")
        result = DynamoBackend._dynamo_runtime_env(config)

        assert "uv" in result
        assert "ai-dynamo[vllm]" in result["uv"]

    def test_merges_with_user_runtime_env(self) -> None:
        """User runtime_env is merged: uv/pip lists concatenated, env_vars merged."""
        from nemo_curator.core.serve.internal.dynamo import DynamoBackend

        config = InferenceModelConfig(
            model_identifier="some-model",
            runtime_env={
                "uv": ["extra-package"],
                "env_vars": {"MY_VAR": "1"},
            },
        )
        result = DynamoBackend._dynamo_runtime_env(config)

        assert "ai-dynamo[vllm]" in result["uv"]
        assert "extra-package" in result["uv"]
        assert result["env_vars"]["MY_VAR"] == "1"

    def test_user_env_vars_preserved(self) -> None:
        """User env_vars are not overwritten by the default (which has no env_vars)."""
        from nemo_curator.core.serve.internal.dynamo import DynamoBackend

        config = InferenceModelConfig(
            model_identifier="some-model",
            runtime_env={"env_vars": {"VLLM_LOGGING_LEVEL": "DEBUG"}},
        )
        result = DynamoBackend._dynamo_runtime_env(config)

        assert result["env_vars"]["VLLM_LOGGING_LEVEL"] == "DEBUG"
        assert "ai-dynamo[vllm]" in result["uv"]
