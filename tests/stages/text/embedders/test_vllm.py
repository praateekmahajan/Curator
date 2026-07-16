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

from contextlib import suppress
from pathlib import Path
from threading import Event
from typing import Any

import pytest

with suppress(ImportError):
    from sentence_transformers import SentenceTransformer

    from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from nemo_curator.tasks import DocumentBatch

# Test model that works with both VLLM and SentenceTransformer
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def sample_data() -> DocumentBatch:
    """Create sample text data for testing."""
    texts = ["Hello world", "This is a test", "Machine learning is great"]
    data = pd.DataFrame({"text": texts})
    return DocumentBatch(dataset_name="test_dataset", data=data)


@pytest.fixture(scope="module")
def reference_model() -> "SentenceTransformer":
    """Load SentenceTransformer model once for the module."""
    return SentenceTransformer(TEST_MODEL).to("cuda")


@pytest.mark.gpu
class TestVLLMEmbeddingModelStage:
    """Test VLLMEmbeddingModelStage initialization and processing."""

    def test_default_initialization(self) -> None:
        """Test initialization with default parameters."""
        stage = VLLMEmbeddingModelStage(model_identifier=TEST_MODEL)

        assert stage.model_identifier == TEST_MODEL
        assert stage.text_field == "text"
        assert stage.embedding_field == "embeddings"
        assert stage.retained_fields is None
        assert stage.model_inference_batch_size == 1024
        assert stage.pretokenize is True
        assert stage.verbose is False
        assert stage.model is None
        assert stage.tokenizer is None

        assert stage.inputs() == (["data"], ["text"])
        assert stage.outputs() == (["data"], ["text", "embeddings"])

    def test_custom_initialization(self) -> None:
        """Test initialization with custom parameters."""
        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            text_field="content",
            embedding_field="emb",
            retained_fields=["id", "content", "id"],
            model_inference_batch_size=17,
            pretokenize=True,
            cache_dir="/tmp/cache",  # noqa: S108
            hf_token="test-token",  # noqa: S106
            verbose=True,
        )

        assert stage.model_identifier == TEST_MODEL
        assert stage.text_field == "content"
        assert stage.embedding_field == "emb"
        assert stage.retained_fields == ["id", "content"]
        assert stage.model_inference_batch_size == 17
        assert stage.pretokenize is True
        assert stage.cache_dir == "/tmp/cache"  # noqa: S108
        assert stage.hf_token == "test-token"  # noqa: S105
        assert stage.verbose is True

        assert stage.inputs() == (["data"], ["content"])
        assert stage.outputs() == (["data"], ["id", "content", "emb"])

        assert stage.resources.gpus == 1
        assert stage.resources.cpus == 1

    def test_llm_uses_cache_dir_for_download(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Ensure vLLM receives download_dir so weights reuse snapshot cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        hf_token = "test-token"  # noqa: S105

        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            cache_dir=str(cache_dir),
            hf_token=hf_token,
            verbose=True,
        )

        captured: dict[str, Any] = {}

        def _fake_snapshot_download(
            model_identifier: str,
            cache_dir: str | None = None,
            token: str | None = None,
            local_files_only: bool | None = None,
        ) -> str:
            captured.setdefault("snapshot_download_calls", []).append(
                {
                    "model_identifier": model_identifier,
                    "cache_dir": cache_dir,
                    "token": token,
                    "local_files_only": local_files_only,
                }
            )
            return f"/resolved/snapshots/{model_identifier}"

        def _fake_create_vllm_llm(*, model: str, **kwargs: Any) -> object:  # noqa: ANN401
            captured["llm"] = {"model": model, "kwargs": kwargs}
            return object()

        import nemo_curator.stages.text.embedders.vllm as _vllm_mod

        monkeypatch.setattr(_vllm_mod, "snapshot_download", _fake_snapshot_download)
        monkeypatch.setattr(_vllm_mod, "create_vllm_llm_with_retry", _fake_create_vllm_llm)

        stage.setup_on_node()

        # setup_on_node calls snapshot_download(local_files_only=False) to download the model
        download_call = captured["snapshot_download_calls"][0]
        assert download_call["cache_dir"] == str(cache_dir)
        assert download_call["token"] == hf_token
        assert download_call["local_files_only"] is False

        # vLLM receives the resolved snapshot path (not the repo ID)
        assert captured["llm"]["model"] == f"/resolved/snapshots/{TEST_MODEL}"
        assert captured["llm"]["kwargs"]["download_dir"] == str(cache_dir)

    def test_process_returns_projected_arrow_float32_embeddings(self) -> None:
        class _PoolingOutput:
            def __init__(self, values: list[float]):
                self.outputs = type("Outputs", (), {"data": torch.tensor(values, dtype=torch.bfloat16)})()

        class _FakeModel:
            def __init__(self):
                self.call: dict[str, Any] | None = None

            def encode(self, prompts: list[str], **kwargs: object) -> list[_PoolingOutput]:
                self.call = {"prompts": prompts, **kwargs}
                return [_PoolingOutput([1.0, 2.0]), _PoolingOutput([3.0, 4.0])]

        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            retained_fields=["_curator_dedup_id", "adlr_id"],
        )
        fake_model = _FakeModel()
        stage.model = fake_model  # type: ignore[assignment]
        batch = DocumentBatch(
            dataset_name="test_dataset",
            data=pd.DataFrame(
                {
                    "text": ["first", "second"],
                    "_curator_dedup_id": [10, 11],
                    "adlr_id": ["a", "b"],
                }
            ).convert_dtypes(dtype_backend="pyarrow"),
        )

        result = stage.process(batch)

        assert isinstance(result.data, pa.Table)
        assert result.data.column_names == ["_curator_dedup_id", "adlr_id", "embeddings"]
        embedding_type = result.data.schema.field("embeddings").type
        assert pa.types.is_list(embedding_type)
        assert embedding_type.value_type == pa.float32()
        assert result.data["embeddings"].to_pylist() == [[1.0, 2.0], [3.0, 4.0]]
        assert fake_model.call is not None
        assert fake_model.call["pooling_task"] == "embed"

    def test_process_chunks_rows_and_preserves_embedding_order(self) -> None:
        class _PoolingOutput:
            def __init__(self, value: float):
                self.outputs = type("Outputs", (), {"data": torch.tensor([value], dtype=torch.bfloat16)})()

        class _FakeModel:
            def __init__(self):
                self.calls: list[list[str]] = []

            def encode(self, prompts: list[str], **_: object) -> list[_PoolingOutput]:
                self.calls.append(prompts)
                return [_PoolingOutput(float(prompt)) for prompt in prompts]

        stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            retained_fields=["id"],
            model_inference_batch_size=2,
        )
        fake_model = _FakeModel()
        stage.model = fake_model  # type: ignore[assignment]
        batch = DocumentBatch(
            dataset_name="test_dataset",
            data=pd.DataFrame({"text": ["0", "1", "2", "3", "4"], "id": [10, 11, 12, 13, 14]}),
        )

        result = stage.process(batch)

        assert fake_model.calls == [["0", "1"], ["2", "3"], ["4"]]
        assert result.data["embeddings"].to_pylist() == [[0.0], [1.0], [2.0], [3.0], [4.0]]

    def test_process_prepares_exactly_one_chunk_ahead(self) -> None:
        second_chunk_prepared = Event()
        third_chunk_prepared = Event()

        class _PoolingOutput:
            def __init__(self, value: float):
                self.outputs = type("Outputs", (), {"data": torch.tensor([value], dtype=torch.bfloat16)})()

        class _DoubleBufferedStage(VLLMEmbeddingModelStage):
            def _prepare_input_chunk(
                self, text_column: pa.ChunkedArray, offset: int, chunk_size: int
            ) -> tuple[list[Any], float]:
                result = super()._prepare_input_chunk(text_column, offset, chunk_size)
                if offset == 2:
                    second_chunk_prepared.set()
                elif offset == 4:
                    third_chunk_prepared.set()
                return result

        class _FakeModel:
            def __init__(self):
                self.num_calls = 0

            def encode(self, prompts: list[str], **_: object) -> list[_PoolingOutput]:
                if self.num_calls == 0:
                    assert second_chunk_prepared.wait(timeout=1)
                    assert not third_chunk_prepared.is_set()
                self.num_calls += 1
                return [_PoolingOutput(float(prompt)) for prompt in prompts]

        stage = _DoubleBufferedStage(
            model_identifier=TEST_MODEL,
            pretokenize=False,
            retained_fields=["id"],
            model_inference_batch_size=2,
        )
        fake_model = _FakeModel()
        stage.model = fake_model  # type: ignore[assignment]
        batch = DocumentBatch(
            dataset_name="test_dataset",
            data=pd.DataFrame({"text": ["0", "1", "2", "3", "4"], "id": [10, 11, 12, 13, 14]}),
        )

        result = stage.process(batch)

        assert fake_model.num_calls == 3
        assert third_chunk_prepared.is_set()
        assert result.data["embeddings"].to_pylist() == [[0.0], [1.0], [2.0], [3.0], [4.0]]

    def test_rejects_nonpositive_model_inference_batch_size(self) -> None:
        with pytest.raises(ValueError, match="model_inference_batch_size must be positive"):
            VLLMEmbeddingModelStage(model_identifier=TEST_MODEL, model_inference_batch_size=0)

    @pytest.mark.parametrize("pretokenize", [True, False])
    def test_vllm_produces_valid_embeddings(
        self, sample_data: DocumentBatch, pretokenize: bool, reference_model: "SentenceTransformer"
    ) -> None:
        """Test that VLLM produces embeddings matching SentenceTransformer reference."""
        vllm_stage = VLLMEmbeddingModelStage(
            model_identifier=TEST_MODEL,
            pretokenize=pretokenize,
            verbose=False,
        )
        try:
            vllm_stage.setup_on_node()
        except Exception:  # noqa: BLE001
            pytest.skip("Skipping test due to model download failure")
        vllm_stage.setup()
        result = vllm_stage.process(sample_data)

        assert isinstance(result, DocumentBatch)
        result_df = result.to_pandas()
        assert "embeddings" in result_df.columns
        assert len(result_df) == 3

        reference_embeddings = reference_model.encode(sample_data.to_pandas()["text"].tolist())
        vllm_embeddings = np.array(result_df["embeddings"].tolist())

        vllm_embeddings_torch = torch.tensor(vllm_embeddings)
        reference_embeddings_torch = torch.tensor(reference_embeddings)

        cosine_sim = torch.nn.functional.cosine_similarity(vllm_embeddings_torch, reference_embeddings_torch, dim=1)
        assert torch.allclose(cosine_sim, torch.ones_like(cosine_sim), atol=1e-5)
