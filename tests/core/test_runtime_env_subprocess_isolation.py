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

The ``python_args`` tests verify the fix for a bug where subprocess commands
built on the driver with ``sys.executable`` would use the base env's Python
instead of the runtime_env's.  ``python_args`` defers ``sys.executable``
resolution to the actor, which sees the runtime_env's isolated virtualenv.
"""

import os
import subprocess
import sys
import time

import pydantic
import pytest
import ray

from nemo_curator.core.serve.internal.subprocess_mgr import _define_subprocess_actor

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


@pytest.mark.usefixtures("shared_ray_client")
class TestPythonArgsIsolation:
    """Validates that ``python_args`` on ``_SubprocessActor.initialize()`` uses
    the actor's ``sys.executable`` — not the driver's — so subprocesses load
    packages from the runtime_env's isolated virtualenv.

    This is the pattern used by ``spawn_actor(python_args=...)`` for all
    Dynamo Python subprocesses (workers, frontend).
    """

    @staticmethod
    def _run_python_args(
        runtime_env: dict,
        python_args: list[str],
        tmp_path: os.PathLike,
        label: str = "python_args",
    ) -> str:
        """Spawn an actor with *runtime_env*, run *python_args*, return log output."""
        import contextlib

        actor_cls = _define_subprocess_actor(f"{label}_actor")
        actor = actor_cls.options(
            name=f"test_{label}_{os.getpid()}",
            lifetime="detached",
            runtime_env=runtime_env,
        ).remote()

        try:
            log_file = str(tmp_path / f"{label}.log")
            ray.get(
                actor.initialize.remote(None, {}, log_file, python_args=python_args),
                timeout=120,
            )
            ray.get(actor.run.remote(), timeout=30)
            return ray.get(actor.read_log_tail.remote(), timeout=10)
        finally:
            with contextlib.suppress(Exception):
                ray.kill(actor)

    @pytest.mark.parametrize(
        ("installer", "version"),
        [("pip", _PIP_TEST_VERSION), ("uv", _UV_TEST_VERSION)],
    )
    def test_python_args_uses_runtime_env_packages(self, installer: str, version: str, tmp_path: os.PathLike):
        """python_args subprocess sees the runtime_env's pydantic, not the base env's."""
        output = self._run_python_args(
            runtime_env={installer: [f"pydantic=={version}"]},
            python_args=["-c", "import pydantic; print(pydantic.__version__)"],
            tmp_path=tmp_path,
            label=f"python_args_{installer}",
        )
        assert version in output, f"Expected {version} in output, got: {output!r}"

    def test_driver_executable_would_load_wrong_version(self, tmp_path: os.PathLike):
        """Proves the bug: a command built with the driver's sys.executable loads base packages.

        This is the pattern that ``python_args`` fixes.  We explicitly pass
        the driver's Python to a runtime_env actor and verify it loads the
        *base* env's pydantic, not the runtime_env's.
        """
        import contextlib

        driver_executable = sys.executable

        actor_cls = _define_subprocess_actor("driver_exe_actor")
        actor = actor_cls.options(
            name=f"test_driver_exe_{os.getpid()}",
            lifetime="detached",
            runtime_env={"pip": [f"pydantic=={_PIP_TEST_VERSION}"]},
        ).remote()

        try:
            log_file = str(tmp_path / "driver_exe.log")
            ray.get(
                actor.initialize.remote(
                    [driver_executable, "-c", "import pydantic; print(pydantic.__version__)"],
                    {},
                    log_file,
                ),
                timeout=120,
            )
            ray.get(actor.run.remote(), timeout=30)
            output = ray.get(actor.read_log_tail.remote(), timeout=10)
        finally:
            with contextlib.suppress(Exception):
                ray.kill(actor)

        # Driver executable loads BASE env packages, NOT the runtime_env's
        assert pydantic.__version__ in output, (
            f"Driver executable should load base pydantic {pydantic.__version__}, got: {output!r}"
        )
        assert _PIP_TEST_VERSION not in output, (
            f"Driver executable should NOT load runtime_env pydantic {_PIP_TEST_VERSION}, got: {output!r}"
        )

    def test_command_and_python_args_mutually_exclusive(self):
        """Passing both command and python_args raises ValueError."""
        actor_cls = _define_subprocess_actor("mutual_excl_actor")
        actor = actor_cls.options(
            name=f"test_mutual_excl_{os.getpid()}",
            lifetime="detached",
        ).remote()

        try:
            with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
                ray.get(
                    actor.initialize.remote(
                        ["echo", "hello"],
                        {},
                        None,
                        python_args=["-c", "print('hi')"],
                    ),
                    timeout=30,
                )
            assert "Exactly one" in str(exc_info.value)
        finally:
            ray.kill(actor)

    def test_neither_command_nor_python_args_raises(self):
        """Passing neither command nor python_args raises ValueError."""
        actor_cls = _define_subprocess_actor("neither_actor")
        actor = actor_cls.options(
            name=f"test_neither_{os.getpid()}",
            lifetime="detached",
        ).remote()

        try:
            with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
                ray.get(
                    actor.initialize.remote(None, {}, None),
                    timeout=30,
                )
            assert "Exactly one" in str(exc_info.value)
        finally:
            ray.kill(actor)
