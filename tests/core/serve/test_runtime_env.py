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

"""Runtime-env merging behaviour exposed through the Dynamo backend.

These tests live at the serve-package level (not the dynamo subpackage)
because ``runtime_env`` is a user-facing promise of ``BaseModelConfig``:
Curator-owned defaults (e.g. ``ai-dynamo[vllm]``) must not clobber a
user's ``env_vars``, ``pip``, ``uv``, or ``working_dir``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from nemo_curator.core.serve import DynamoVLLMModelConfig
from nemo_curator.core.serve.dynamo import vllm as _vllm_module
from nemo_curator.core.serve.dynamo.vllm import (
    _DEFAULT_DYNAMO_RUNTIME_VENV,
    _DYNAMO_VLLM_RUNTIME_ENV_UV,
    _prebuilt_runtime_env,
    dynamo_runtime_env,
    merge_model_runtime_envs,
)

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture(autouse=True)
def _force_uv_runtime_env(monkeypatch: MonkeyPatch) -> None:
    # Pin ``DYNAMO_VLLM_RUNTIME_ENV`` to the uv form so the merge tests run
    # the same on CI (no prebuilt venv) and dev hosts where
    # ``/opt/dynamo_vllm_venv`` exists and would otherwise flip the module
    # constant to the PYTHONPATH form.
    monkeypatch.setattr(_vllm_module, "DYNAMO_VLLM_RUNTIME_ENV", _DYNAMO_VLLM_RUNTIME_ENV_UV)


class TestDynamoRuntimeEnv:
    def test_default_runtime_env_only(self) -> None:
        mc = DynamoVLLMModelConfig(model_identifier="Qwen/Qwen3-0.6B")
        env = dynamo_runtime_env(mc)
        assert env == _DYNAMO_VLLM_RUNTIME_ENV_UV

    def test_user_runtime_env_merges_uv_packages(self) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            runtime_env={"uv": ["mypkg==1.0"]},
        )
        env = dynamo_runtime_env(mc)
        # Dict-form uv block + list-form override: override packages append;
        # base ``uv_pip_install_options`` survive.
        expected_packages = [*_DYNAMO_VLLM_RUNTIME_ENV_UV["uv"]["packages"], "mypkg==1.0"]
        assert env["uv"]["packages"] == expected_packages
        assert env["uv"]["uv_pip_install_options"] == _DYNAMO_VLLM_RUNTIME_ENV_UV["uv"]["uv_pip_install_options"]

    def test_user_env_vars_are_preserved(self) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            runtime_env={"env_vars": {"HF_TOKEN": "abc", "TRANSFORMERS_OFFLINE": "1"}},
        )
        env = dynamo_runtime_env(mc)
        # No ``uv`` override → base ``uv`` block is preserved verbatim.
        assert env["uv"] == _DYNAMO_VLLM_RUNTIME_ENV_UV["uv"]
        assert env["env_vars"] == {"HF_TOKEN": "abc", "TRANSFORMERS_OFFLINE": "1"}

    def test_working_dir_is_passed_through(self) -> None:
        mc = DynamoVLLMModelConfig(
            model_identifier="Qwen/Qwen3-0.6B",
            runtime_env={"working_dir": "/workspace"},
        )
        env = dynamo_runtime_env(mc)
        assert env["working_dir"] == "/workspace"

    def test_default_bumps_runtime_env_setup_timeout(self) -> None:
        # Regression guard: Ray's default ``setup_timeout_seconds`` is 600 s,
        # but ``--no-build-isolation-package flash-attn`` triggers a
        # from-source rebuild (~15 min) that would otherwise be cancelled with
        # ``RuntimeEnvSetupError`` before the actor comes up.
        env = dynamo_runtime_env(DynamoVLLMModelConfig(model_identifier="m"))
        assert env["config"]["setup_timeout_seconds"] >= 1800

    def test_default_carries_flash_attn_rebuild_flags(self) -> None:
        # Regression guard: without ``--reinstall-package flash-attn`` +
        # ``--no-build-isolation-package flash-attn`` the actor venv loads
        # ai-dynamo[vllm]'s prebuilt flash-attn wheel against the wrong torch
        # ABI and crashes with ``undefined symbol:
        # c10::cuda::c10_cuda_check_implementation``.
        env = dynamo_runtime_env(DynamoVLLMModelConfig(model_identifier="m"))
        assert "flash-attn" in env["uv"]["packages"]
        opts = env["uv"]["uv_pip_install_options"]
        assert "--reinstall-package" in opts
        assert opts[opts.index("--reinstall-package") + 1] == "flash-attn"
        assert "--no-build-isolation-package" in opts
        assert opts[opts.index("--no-build-isolation-package") + 1] == "flash-attn"

    def test_default_carries_torch_backend_auto(self) -> None:
        # Without this, uv defaults to PyPI's torch 2.9.1 build (cu128)
        # regardless of the host's nvidia driver. cu129 hosts then run via
        # cu128→cu129 forward-compat — works but is gratuitous.
        opts = _DYNAMO_VLLM_RUNTIME_ENV_UV["uv"]["uv_pip_install_options"]
        assert "--torch-backend=auto" in opts

    def test_ai_dynamo_pin_matches_inference_server_extra(self) -> None:
        # ``[inference_server]`` in pyproject.toml pins ``ai-dynamo==1.0.2``;
        # the actor venv must match so docker-prebuild and bare-metal
        # runtime_env paths converge on the same release.
        packages = _DYNAMO_VLLM_RUNTIME_ENV_UV["uv"]["packages"]
        assert "ai-dynamo[vllm]==1.0.2" in packages
        assert "ai-dynamo[vllm]" not in packages, "Unpinned ai-dynamo[vllm] would float to whatever PyPI publishes"

    def test_prebuilt_runtime_env_default_path(self) -> None:
        # Documented contract with docker/Dockerfile's prebuild step.
        assert _DEFAULT_DYNAMO_RUNTIME_VENV == "/opt/dynamo_vllm_venv"

    def test_prebuilt_runtime_env_none_when_path_missing(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("DYNAMO_RUNTIME_VENV", str(tmp_path / "does_not_exist"))
        assert _prebuilt_runtime_env() is None

    def test_prebuilt_runtime_env_returns_pythonpath_when_venv_exists(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        # Mimic the layout produced by ``uv venv``: lib/python<X.Y>/site-packages.
        site = tmp_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        site.mkdir(parents=True)
        monkeypatch.setenv("DYNAMO_RUNTIME_VENV", str(tmp_path))
        env = _prebuilt_runtime_env()
        assert env == {"env_vars": {"PYTHONPATH": str(site)}}

    def test_prebuilt_runtime_env_none_when_site_packages_missing(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        # An empty directory without the venv layout shouldn't be treated as
        # a usable prebuilt — actors would crash on import.
        monkeypatch.setenv("DYNAMO_RUNTIME_VENV", str(tmp_path))
        assert _prebuilt_runtime_env() is None


class TestMergeModelRuntimeEnvs:
    def test_no_models_yields_default_env(self) -> None:
        assert merge_model_runtime_envs([]) == _DYNAMO_VLLM_RUNTIME_ENV_UV

    def test_merges_env_vars_across_models(self) -> None:
        models = [
            DynamoVLLMModelConfig(
                model_identifier="m1",
                runtime_env={"env_vars": {"A": "1"}},
            ),
            DynamoVLLMModelConfig(
                model_identifier="m2",
                runtime_env={"env_vars": {"B": "2"}, "uv": ["userpkg"]},
            ),
        ]
        env = merge_model_runtime_envs(models)
        assert env["env_vars"] == {"A": "1", "B": "2"}
        expected_packages = [*_DYNAMO_VLLM_RUNTIME_ENV_UV["uv"]["packages"], "userpkg"]
        assert env["uv"]["packages"] == expected_packages

    def test_later_model_env_var_overrides_earlier(self) -> None:
        models = [
            DynamoVLLMModelConfig(
                model_identifier="m1",
                runtime_env={"env_vars": {"HF_HOME": "/cache/v1"}},
            ),
            DynamoVLLMModelConfig(
                model_identifier="m2",
                runtime_env={"env_vars": {"HF_HOME": "/cache/v2"}},
            ),
        ]
        env = merge_model_runtime_envs(models)
        assert env["env_vars"]["HF_HOME"] == "/cache/v2"

    def test_ignores_models_with_empty_runtime_env(self) -> None:
        models = [
            DynamoVLLMModelConfig(model_identifier="m1"),
            DynamoVLLMModelConfig(
                model_identifier="m2",
                runtime_env={"env_vars": {"A": "1"}},
            ),
        ]
        env = merge_model_runtime_envs(models)
        assert env["env_vars"] == {"A": "1"}

    def test_user_dict_form_uv_concatenates_install_options(self) -> None:
        # User passes a dict-form ``uv`` override carrying its own
        # ``uv_pip_install_options``; merger appends packages and concatenates
        # options without dropping the Curator-owned flash-attn rebuild flags.
        models = [
            DynamoVLLMModelConfig(
                model_identifier="m",
                runtime_env={
                    "uv": {
                        "packages": ["userpkg"],
                        "uv_pip_install_options": ["--prefer-binary"],
                    },
                },
            ),
        ]
        env = merge_model_runtime_envs(models)
        expected_packages = [*_DYNAMO_VLLM_RUNTIME_ENV_UV["uv"]["packages"], "userpkg"]
        expected_options = [*_DYNAMO_VLLM_RUNTIME_ENV_UV["uv"]["uv_pip_install_options"], "--prefer-binary"]
        assert env["uv"]["packages"] == expected_packages
        assert env["uv"]["uv_pip_install_options"] == expected_options
