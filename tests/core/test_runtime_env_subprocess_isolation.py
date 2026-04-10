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

"""Tests that runtime_env pip/uv isolation propagates to subprocesses.

This validates the core mechanism used by DynamoBackend to inject
ai-dynamo[vllm] into worker actors without conflicting with the base
environment's vLLM version (used by Ray Serve).

Uses ``pydantic`` as the test package because it's heavily constrained in
uv.lock (with transitive deps like pydantic-core), mirroring the real
conflict scenario where the base env pins one vLLM and Dynamo needs another.
"""

import subprocess
import sys
import time

import pydantic
import pytest
import ray

# Pin versions different from base env's 2.12.5
_PIP_TEST_VERSION = "2.10.0"
_UV_TEST_VERSION = "2.11.0"


@ray.remote
class _VersionCheckActor:
    """Actor that reports pydantic version seen in-process and via subprocess."""

    def get_version_info(self) -> dict:
        import pydantic as _pd

        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", "import pydantic; print(pydantic.__version__)"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "actor_version": _pd.__version__,
            "subprocess_version": result.stdout.strip(),
            "sys_executable": sys.executable,
        }


@pytest.mark.usefixtures("shared_ray_client")
class TestRuntimeEnvSubprocessIsolation:
    """Validates that runtime_env pip isolation carries through to subprocesses."""

    def test_pip_isolation_in_actor(self):
        """Actor with runtime_env pip sees the pinned version, not base."""
        actor = _VersionCheckActor.options(runtime_env={"pip": [f"pydantic=={_PIP_TEST_VERSION}"]}).remote()
        info = ray.get(actor.get_version_info.remote())
        ray.kill(actor)

        assert info["actor_version"] == _PIP_TEST_VERSION

    def test_pip_isolation_in_subprocess(self):
        """Subprocess spawned by actor inherits the runtime_env's package version."""
        actor = _VersionCheckActor.options(runtime_env={"pip": [f"pydantic=={_PIP_TEST_VERSION}"]}).remote()
        info = ray.get(actor.get_version_info.remote())
        ray.kill(actor)

        assert info["subprocess_version"] == _PIP_TEST_VERSION

    def test_uv_isolation_in_subprocess(self):
        """Same test but using the uv runtime_env plugin."""
        actor = _VersionCheckActor.options(runtime_env={"uv": [f"pydantic=={_UV_TEST_VERSION}"]}).remote()
        info = ray.get(actor.get_version_info.remote())
        ray.kill(actor)

        assert info["subprocess_version"] == _UV_TEST_VERSION

    def test_base_env_unaffected(self):
        """Base environment pydantic version is not one of our test versions (sanity check)."""
        assert pydantic.__version__ not in (_PIP_TEST_VERSION, _UV_TEST_VERSION)

    def test_sys_executable_points_to_isolated_env(self):
        """sys.executable inside the actor should NOT be the base env Python."""
        base_executable = sys.executable

        actor = _VersionCheckActor.options(runtime_env={"pip": [f"pydantic=={_PIP_TEST_VERSION}"]}).remote()
        info = ray.get(actor.get_version_info.remote())
        ray.kill(actor)

        assert info["sys_executable"] != base_executable

    def test_cache_reuse_is_fast(self):
        """Second actor with same runtime_env should start quickly (cached virtualenv)."""
        runtime_env = {"pip": [f"pydantic=={_PIP_TEST_VERSION}"]}

        # Warm the cache
        actor1 = _VersionCheckActor.options(runtime_env=runtime_env).remote()
        ray.get(actor1.get_version_info.remote())
        ray.kill(actor1)

        # Second actor should be fast
        start = time.monotonic()
        actor2 = _VersionCheckActor.options(runtime_env=runtime_env).remote()
        ray.get(actor2.get_version_info.remote())
        elapsed = time.monotonic() - start
        ray.kill(actor2)

        assert elapsed < 60, f"Cached runtime_env took {elapsed:.1f}s, expected < 60s"
