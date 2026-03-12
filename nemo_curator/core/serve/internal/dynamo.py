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

from typing import TYPE_CHECKING

from nemo_curator.core.serve.internal.base import InferenceBackend

if TYPE_CHECKING:
    from nemo_curator.core.serve.server import InferenceServer


class DynamoBackend(InferenceBackend):
    """NVIDIA Dynamo inference backend (subprocess-based workers).

    This backend launches Dynamo workers as subprocesses with dedicated
    etcd + NATS infrastructure. It does NOT participate in Ray's GPU
    scheduling — pipelines with GPU stages will fail-fast with a RuntimeError.

    TODO: Implement full lifecycle — infra management, GPU planner,
    worker launching, frontend, health check, and teardown.
    """

    def __init__(self, server: InferenceServer) -> None:
        self._server = server

    def start(self) -> None:
        msg = "Dynamo backend is not yet implemented. Use backend='ray_serve' (the default) for now."
        raise NotImplementedError(msg)

    def stop(self) -> None:
        pass
