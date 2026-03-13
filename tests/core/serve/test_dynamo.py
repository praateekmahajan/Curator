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

"""Integration tests for the Dynamo inference backend.

These tests start the full Dynamo stack (etcd + NATS + vLLM worker + frontend)
with a tiny model and make real HTTP requests. They require:
  - GPU(s) available
  - ``etcd`` and ``nats-server`` binaries on $PATH
  - ``ai-dynamo`` Python package installed

Typically run inside the Dynamo Docker image. Skipped automatically on bare
metal where the binaries are not installed.
"""

import shutil

import pytest
from openai import OpenAI

from nemo_curator.core.serve import InferenceModelConfig, InferenceServer, is_inference_server_active

# Tiny model used by Dynamo's own test suite — fast to load, low VRAM.
DYNAMO_TEST_MODEL = "Qwen/Qwen3-0.6B"

_missing_binaries = not (shutil.which("etcd") and shutil.which("nats-server"))
_skip_reason = "etcd and/or nats-server not on $PATH (run inside Dynamo Docker image)"


@pytest.fixture(scope="class")
def dynamo_server(shared_ray_cluster: str):  # noqa: ARG001
    """Start a Dynamo-backed InferenceServer for the test class."""
    config = InferenceModelConfig(
        model_identifier=DYNAMO_TEST_MODEL,
        deployment_config={"num_replicas": 1},
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 1024,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.4,
        },
    )

    server = InferenceServer(
        models=[config],
        backend="dynamo",
        health_check_timeout_s=300,
    )
    server.start()

    yield server

    server.stop()


@pytest.mark.gpu
@pytest.mark.skipif(_missing_binaries, reason=_skip_reason)
@pytest.mark.usefixtures("dynamo_server")
class TestDynamoIntegration:
    """Full lifecycle tests: start → query → stop against a real Dynamo stack."""

    def test_server_is_active(self, dynamo_server: InferenceServer):
        assert dynamo_server._started
        assert is_inference_server_active()

    def test_lists_models(self, dynamo_server: InferenceServer):
        """The /v1/models endpoint returns the served model."""
        client = OpenAI(base_url=dynamo_server.endpoint, api_key="na")
        model_ids = [m.id for m in client.models.list()]
        assert DYNAMO_TEST_MODEL in model_ids

    def test_chat_completion(self, dynamo_server: InferenceServer):
        """A chat completion request returns a non-empty response."""
        client = OpenAI(base_url=dynamo_server.endpoint, api_key="na")
        response = client.chat.completions.create(
            model=DYNAMO_TEST_MODEL,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    def test_stop_clears_active(self, dynamo_server: InferenceServer):
        """Stopping the server clears the active server tracking.

        Must run last in the class — it stops the shared fixture's server.
        """
        dynamo_server.stop()
        assert not is_inference_server_active()
