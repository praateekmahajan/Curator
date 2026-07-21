# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pynvml
import ray


def align_down_to_256(memory_size: int) -> int:
    """
    Aligns a memory size down to the nearest multiple of 256.
    """
    return (memory_size // 256) * 256


def get_device_memory_info() -> tuple[int, int] | None:
    """Return ``(free, total)`` bytes for the first GPU available to the caller."""
    try:
        index = int(ray.get_gpu_ids()[0]) if ray.is_initialized() else 0
    except IndexError:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except pynvml.NVMLError:
        return None
    else:
        return info.free, info.total


def get_device_free_memory() -> int | None:
    """
    Return free memory of the first GPU the caller has access to.
    Returns None if the GPU is not available or information could not be retrieved.
    """
    memory_info = get_device_memory_info()
    return memory_info[0] if memory_info is not None else None
