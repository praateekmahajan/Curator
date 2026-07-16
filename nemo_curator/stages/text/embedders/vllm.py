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

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import torch
from huggingface_hub import snapshot_download

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.models.utils import format_name_with_suffix
from nemo_curator.tasks import DocumentBatch
from nemo_curator.utils.vllm_utils import create_vllm_llm_with_retry

if TYPE_CHECKING:
    from collections.abc import Iterator

    from transformers import AutoTokenizer
    from vllm import LLM

    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

_EMBEDDING_MATRIX_NDIM = 2


class VLLMEmbeddingModelStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    def __init__(  # noqa: PLR0913
        self,
        model_identifier: str,
        vllm_init_kwargs: dict[str, Any] | None = None,
        text_field: str = "text",
        pretokenize: bool = True,
        embedding_field: str = "embeddings",
        retained_fields: list[str] | None = None,
        model_inference_batch_size: int = 1024,
        max_chars: int | None = None,
        cache_dir: str | None = None,
        hf_token: str | None = None,
        verbose: bool = False,
    ):
        self.model_identifier = model_identifier
        self.vllm_init_kwargs = vllm_init_kwargs or {}

        self.text_field = text_field
        self.pretokenize = pretokenize
        self.embedding_field = embedding_field
        self.retained_fields = list(dict.fromkeys(retained_fields)) if retained_fields is not None else None
        if model_inference_batch_size <= 0:
            msg = f"model_inference_batch_size must be positive, got {model_inference_batch_size}"
            raise ValueError(msg)
        self.model_inference_batch_size = model_inference_batch_size
        self.max_chars = max_chars

        self.cache_dir = cache_dir
        self.hf_token = hf_token

        self.verbose = verbose
        # after setup
        self.model: None | LLM = None
        self.tokenizer: None | AutoTokenizer = None
        # stage setup
        self.resources = Resources(
            cpus=1,
            gpus=1,
        )
        self.name = format_name_with_suffix(model_identifier, suffix="_vllm")

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = self.retained_fields if self.retained_fields is not None else [self.text_field]
        return ["data"], [*output_fields, self.embedding_field]

    def _initialize_vllm(self, local_files_only: bool) -> None:
        """Download (or locate) the model and initialize vLLM.

        We pass the resolved snapshot path to ``LLM(model=...)`` instead of the
        HuggingFace repo ID because vLLM does not pass the ``download_dir`` through
        to its config resolution code — passing a repo ID with a custom cache dir
        fails offline.
        """
        model_path = snapshot_download(
            self.model_identifier,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            local_files_only=local_files_only,
        )

        vllm_init_kwargs = self.vllm_init_kwargs.copy()
        if "enforce_eager" not in vllm_init_kwargs:
            vllm_init_kwargs["enforce_eager"] = False
        if "runner" not in vllm_init_kwargs:
            vllm_init_kwargs["runner"] = "pooling"
        if "model_impl" not in vllm_init_kwargs:
            # TODO: Once transformers is bumped to 5.0 then we should also support transformers
            vllm_init_kwargs["model_impl"] = "vllm"
        if self.cache_dir is not None and "download_dir" not in vllm_init_kwargs:
            vllm_init_kwargs["download_dir"] = self.cache_dir

        # Reduce verbosity when not in verbose mode
        if not self.verbose and "disable_log_stats" not in vllm_init_kwargs:
            vllm_init_kwargs["disable_log_stats"] = True

        # Ray may initialize many vLLM actors concurrently on the same node.
        # Use the shared helper so every engine gets a freshly selected
        # MASTER_PORT and transient EngineCore port collisions are retried.
        self.model = create_vllm_llm_with_retry(model=model_path, **vllm_init_kwargs)

    def setup_on_node(self, node_info: NodeInfo | None = None, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if not self.verbose:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()

        # Download model to cache_dir (or default HF cache) and initialize vLLM.
        # local_files_only=False allows downloading when online; if the model is
        # already cached (e.g. in air-gapped environments), snapshot_download falls
        # back to the local cache automatically.
        self._initialize_vllm(local_files_only=False)

    def teardown(self) -> None:
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        if self.model is None:
            # Load from local cache only — model must already be downloaded (by setup_on_node or pre-cached)
            self._initialize_vllm(local_files_only=True)
        if self.pretokenize:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                local_files_only=True,
            )

    def _prepare_input_chunk(
        self, text_column: pa.ChunkedArray, offset: int, chunk_size: int
    ) -> tuple[list[Any], float]:
        input_data = text_column.slice(offset, chunk_size).to_pylist()
        if self.max_chars is not None:
            input_data = [text[: self.max_chars] if text is not None else None for text in input_data]
        if not self.pretokenize:
            return input_data, 0.0

        from vllm.inputs import TokensPrompt

        if self.tokenizer is None:
            msg = "Tokenizer is not initialized. Please call setup() before processing or set pretokenize to False."
            raise ValueError(msg)
        if self.model is None:
            msg = "vLLM model is not initialized. Please call setup() before processing."
            raise ValueError(msg)

        t0 = time.perf_counter()
        tokenized_data = self.tokenizer.batch_encode_plus(
            input_data,
            truncation=True,
            max_length=self.model.model_config.max_model_len,
        )
        prompts = [TokensPrompt(prompt_token_ids=ids) for ids in tokenized_data.input_ids]
        del tokenized_data
        return prompts, time.perf_counter() - t0

    def _embed_chunk(self, input_data: list[Any], expected_size: int) -> tuple[np.ndarray, float]:
        if self.model is None:
            msg = "vLLM model is not initialized. Please call setup() before processing."
            raise ValueError(msg)

        t0 = time.perf_counter()
        vllm_output = self.model.encode(
            input_data,
            pooling_task="embed",
            tokenization_kwargs={"truncate_prompt_tokens": -1},
            use_tqdm=self.verbose,
        )
        elapsed = time.perf_counter() - t0
        if len(vllm_output) != expected_size:
            msg = f"vLLM returned {len(vllm_output)} embeddings for a {expected_size}-row input chunk"
            raise ValueError(msg)

        chunk_embedding_matrix = torch.stack([output.outputs.data for output in vllm_output])
        chunk_embedding_matrix = chunk_embedding_matrix.to(device="cpu", dtype=torch.float32).contiguous()
        if chunk_embedding_matrix.ndim != _EMBEDDING_MATRIX_NDIM:
            msg = f"Expected a two-dimensional embedding matrix, got shape {tuple(chunk_embedding_matrix.shape)}"
            raise ValueError(msg)
        chunk_embedding_numpy = chunk_embedding_matrix.numpy()
        del vllm_output
        return chunk_embedding_numpy, elapsed

    def _iter_prepared_chunks(
        self, text_column: pa.ChunkedArray, num_rows: int
    ) -> Iterator[tuple[int, int, list[Any], float]]:
        """Prepare bounded input chunks for model inference."""
        for offset in range(0, num_rows, self.model_inference_batch_size):
            chunk_size = min(self.model_inference_batch_size, num_rows - offset)
            input_data, tokenization_time = self._prepare_input_chunk(text_column, offset, chunk_size)
            yield offset, chunk_size, input_data, tokenization_time

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        input_table = batch.to_pyarrow()
        if self.text_field not in input_table.column_names:
            msg = f"Input batch is missing required text field {self.text_field!r}"
            raise ValueError(msg)

        retained_fields = self.retained_fields if self.retained_fields is not None else input_table.column_names
        missing_fields = [field for field in retained_fields if field not in input_table.column_names]
        if missing_fields:
            msg = f"Input batch is missing retained fields: {missing_fields}"
            raise ValueError(msg)

        output_table = input_table.select(retained_fields)
        metrics = {}
        embedding_numpy: np.ndarray | None = None
        tokenization_time = 0.0
        vllm_embedding_time = 0.0
        num_rows = input_table.num_rows
        text_column = input_table[self.text_field]
        if num_rows == 0:
            msg = "Cannot generate embeddings for an empty document batch"
            raise ValueError(msg)

        for offset, chunk_size, input_data, chunk_tokenization_time in self._iter_prepared_chunks(
            text_column, num_rows
        ):
            tokenization_time += chunk_tokenization_time
            chunk_embedding_numpy, chunk_embedding_time = self._embed_chunk(input_data, chunk_size)
            vllm_embedding_time += chunk_embedding_time
            del input_data
            if embedding_numpy is None:
                embedding_numpy = np.empty((num_rows, chunk_embedding_numpy.shape[1]), dtype=np.float32)
            elif embedding_numpy.shape[1] != chunk_embedding_numpy.shape[1]:
                msg = (
                    f"vLLM embedding dimension changed from {embedding_numpy.shape[1]} "
                    f"to {chunk_embedding_numpy.shape[1]}"
                )
                raise ValueError(msg)
            embedding_numpy[offset : offset + chunk_size] = chunk_embedding_numpy
            del chunk_embedding_numpy
            gc.collect()

        if embedding_numpy is None:
            msg = "vLLM did not return embeddings for a non-empty document batch"
            raise RuntimeError(msg)

        del input_table, text_column
        metrics["tokenization_time"] = tokenization_time
        metrics["vllm_embedding_time"] = vllm_embedding_time

        embedding_values = pa.array(embedding_numpy.reshape(-1), type=pa.float32(), from_pandas=False)
        embedding_offsets = pa.array(
            np.arange(
                0,
                (embedding_numpy.shape[0] + 1) * embedding_numpy.shape[1],
                embedding_numpy.shape[1],
                dtype=np.int64,
            )
        )
        embedding_array = pa.ListArray.from_arrays(embedding_offsets, embedding_values)

        if self.embedding_field in output_table.column_names:
            embedding_index = output_table.column_names.index(self.embedding_field)
            output_table = output_table.set_column(embedding_index, self.embedding_field, embedding_array)
        else:
            output_table = output_table.append_column(self.embedding_field, embedding_array)

        self._log_metrics(metrics)

        del embedding_numpy, embedding_values, embedding_offsets, embedding_array
        gc.collect()

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=output_table,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
