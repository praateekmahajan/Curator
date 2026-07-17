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

# ruff: noqa: C901, PLR0913, PLR0915

"""Embedding generation benchmarking script.

Supports multiple embedding model backends (through the model_variation argument):
- sentence_transformer: EmbeddingCreatorStage with SentenceTransformer
- pytorch_model: EmbeddingCreatorStage with raw PyTorch model + custom pooling
- vllm_text: VLLMEmbeddingModelStage with text input
- vllm_text_pretokenized: VLLMEmbeddingModelStage with pretokenization
"""

import argparse
import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from utils import load_dataset_files, setup_executor, write_benchmark_results

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
    IdGeneratorBase,
    create_id_generator_actor,
    kill_id_generator_actor,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter
from nemo_curator.tasks.utils import TaskPerfUtils

_VLLM_WORKER_ENV_VARS = (
    "CUDA_HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "VLLM_CACHE_ROOT",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
)


def _vllm_worker_runtime_env() -> dict[str, str]:
    """Propagate the CUDA JIT and persistent-cache setup into Ray actor processes."""
    env_vars = {
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_MOE_USE_DEEP_GEMM": "0",
    }
    env_vars.update({name: value for name in _VLLM_WORKER_ENV_VARS if (value := os.environ.get(name))})
    return env_vars


class EmbeddingModelVariation(Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    PYTORCH_MODEL = "pytorch_model"
    VLLM_TEXT = "vllm_text"
    VLLM_TEXT_PRETOKENIZED = "vllm_text_pretokenized"


def _resolve_max_seq_length(model_identifier: str, cache_dir: str | None = None) -> int:
    """Resolve max_seq_length from the sentence-transformers config.

    vLLM reads max_seq_length from the Sentence Transformers config (e.g. 256
    for MiniLM), which is often lower than max_position_embeddings (512).
    SentenceTransformers also silently truncates to this value, but HF
    AutoModel does not — it uses the full max_position_embeddings.

    We use the sentence-transformers config as the single source of truth
    so all backends process the same number of tokens.
    """
    from huggingface_hub import snapshot_download
    from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config

    # Resolve to a local snapshot path so the vLLM helper can find configs
    # even when the model lives in a non-default cache directory.
    model_path = snapshot_download(model_identifier, cache_dir=cache_dir, local_files_only=True)

    st_config = get_sentence_transformer_tokenizer_config(model_path)
    if st_config is not None and st_config.get("max_seq_length") is not None:
        model_limit = int(st_config["max_seq_length"])
        logger.info(f"Resolved max_seq_length={model_limit} from sentence-transformers config")
        return model_limit

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model_limit = getattr(model_config, "max_position_embeddings", None)
    if model_limit is None:
        msg = (
            f"Could not resolve max_seq_length for {model_identifier}: sentence-transformers config "
            "has no max_seq_length and model config has no max_position_embeddings"
        )
        raise ValueError(msg)
    model_limit = int(model_limit)
    logger.warning(
        f"Sentence-transformers config has no max_seq_length; resolved max_seq_length={model_limit} "
        "from model config max_position_embeddings"
    )
    return model_limit


def _create_embedding_stages(
    model_identifier: str,
    model_variation: EmbeddingModelVariation,
    model_inference_batch_size: int,
    model_num_workers: int | None,
    model_worker_gpus: float | None,
    model_max_tasks_in_flight_per_actor: int | None,
    model_max_restarts: int | None,
    model_gpu_memory_utilization: float | None,
    model_kv_cache_memory_bytes: int | None,
    max_seq_length: int,
    embedding_pooling: str,
    max_chars: int | None = None,
    cache_dir: str | None = None,
    retained_fields: list[str] | None = None,
) -> list:
    """Create the embedding stage(s) for the given model variation."""
    if model_variation in {EmbeddingModelVariation.SENTENCE_TRANSFORMER, EmbeddingModelVariation.PYTORCH_MODEL}:
        from nemo_curator.stages.text.embedders import EmbeddingCreatorStage

        use_sentence_transformer = model_variation == EmbeddingModelVariation.SENTENCE_TRANSFORMER
        return [
            EmbeddingCreatorStage(
                model_identifier=model_identifier,
                use_sentence_transformer=use_sentence_transformer,
                text_field="text",
                embedding_field="embeddings",
                model_inference_batch_size=model_inference_batch_size,
                sort_by_length=True,
                max_seq_length=max_seq_length,
                embedding_pooling=embedding_pooling,
                cache_dir=cache_dir,
            ),
        ]

    if model_variation in {EmbeddingModelVariation.VLLM_TEXT, EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED}:
        from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage

        # vLLM strictly enforces max_model_len from the model config, unlike
        # sentence-transformers which silently truncates.  Pass max_seq_length
        # through so vLLM knows the intended limit and won't error on inputs
        # that exceed the model's default max_position_embeddings.
        # vLLM v0.22's V1 pooling runner can deadlock when a single embedding
        # request is split across its default 16,384-token chunk boundary. Keep
        # each request in one prefill; this preserves the full configured context
        # (rather than truncating it) and is required for Qwen3-Embedding at 32k.
        vllm_init_kwargs: dict[str, Any] = {
            "max_model_len": max_seq_length,
            "max_num_batched_tokens": max_seq_length,
        }
        if model_gpu_memory_utilization is not None:
            vllm_init_kwargs["gpu_memory_utilization"] = model_gpu_memory_utilization
        if model_kv_cache_memory_bytes is not None:
            # Ray fractional GPUs do not isolate CUDA memory.  Give every
            # independently scheduled vLLM engine a deterministic KV budget
            # instead of letting each profile the whole shared device.
            vllm_init_kwargs["kv_cache_memory_bytes"] = model_kv_cache_memory_bytes

        stage = VLLMEmbeddingModelStage(
            model_identifier=model_identifier,
            text_field="text",
            embedding_field="embeddings",
            pretokenize=model_variation == EmbeddingModelVariation.VLLM_TEXT_PRETOKENIZED,
            vllm_init_kwargs=vllm_init_kwargs,
            retained_fields=retained_fields,
            model_inference_batch_size=model_inference_batch_size,
            max_chars=max_chars,
            cache_dir=cache_dir,
        )
        stage_overrides: dict[str, Any] = {}
        # The aarch64 vLLM wheel can select DeepGEMM warmup for BF16 embedding
        # models even when the DeepGEMM backend is unavailable. Disable that
        # optional backend for this direct Ray-actor benchmark.
        stage_overrides["runtime_env"] = {"env_vars": _vllm_worker_runtime_env()}
        if model_worker_gpus is not None:
            stage_overrides["resources"] = Resources(cpus=1.0, gpus=model_worker_gpus)
        if model_num_workers is not None:
            stage_overrides["num_workers"] = model_num_workers
        ray_stage_spec: dict[RayStageSpecKeys, Any] = {}
        if model_max_tasks_in_flight_per_actor is not None:
            ray_stage_spec[RayStageSpecKeys.MAX_TASKS_IN_FLIGHT_PER_ACTOR] = model_max_tasks_in_flight_per_actor
        if model_max_restarts is not None:
            ray_stage_spec[RayStageSpecKeys.RAY_REMOTE_ARGS] = {
                "max_restarts": model_max_restarts,
                "max_task_retries": model_max_restarts,
            }
        if ray_stage_spec:
            stage_overrides["ray_stage_spec"] = ray_stage_spec
        if stage_overrides:
            stage = stage.with_(**stage_overrides)
        return [stage]

    msg = f"Unsupported model variation: {model_variation}"
    raise ValueError(msg)


def run_embedding_generation_benchmark(
    input_path: str,
    output_path: str,
    executor: str,
    dataset_size_gb: float | None,
    load_dataset_ratio: float | None,
    max_input_files: int | None,
    id_generator_path: str | None,
    id_path_mapping_json: str | None,
    metadata_fields: list[str] | None,
    reader_max_workers: int | None,
    model_identifier: str,
    model_inference_batch_size: int,
    model_num_workers: int | None,
    model_worker_gpus: float | None,
    model_max_tasks_in_flight_per_actor: int | None,
    model_max_restarts: int | None,
    model_gpu_memory_utilization: float | None,
    model_kv_cache_memory_bytes: int | None,
    model_variation: str,
    embedding_pooling: str,
    input_format: str = "parquet",
    max_chars: int | None = None,
    cache_dir: str | None = None,
    **kwargs: Any,  # noqa: ANN401, ARG001
) -> dict[str, Any]:
    """Run the embedding generation benchmark and collect comprehensive metrics."""
    variation = EmbeddingModelVariation(model_variation)
    max_seq_length = _resolve_max_seq_length(model_identifier, cache_dir=cache_dir)
    input_path = Path(input_path)
    output_path = Path(output_path).absolute()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting embedding generation benchmark")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Dataset size: {dataset_size_gb} GB")
    logger.info(f"Load dataset ratio: {load_dataset_ratio}")
    logger.info(f"Maximum input files: {max_input_files}")
    logger.info(f"ID generator: {id_generator_path}")
    id_path_mapping = json.loads(id_path_mapping_json) if id_path_mapping_json else {}
    if not isinstance(id_path_mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in id_path_mapping.items()
    ):
        msg = "--id-path-mapping-json must decode to a string-to-string dictionary"
        raise ValueError(msg)
    logger.info(f"ID path mapping: {id_path_mapping}")
    metadata_fields = metadata_fields or []
    logger.info(f"Metadata fields: {metadata_fields}")
    logger.info(f"Maximum JSONL reader workers: {reader_max_workers}")
    logger.info(f"Model: {model_identifier}")
    logger.info(f"Model variation: {variation.name}")
    logger.info(f"Model inference batch size: {model_inference_batch_size}")
    logger.info(f"Model workers: {model_num_workers}")
    logger.info(f"Model worker GPUs: {model_worker_gpus}")
    logger.info(f"Model tasks in flight per actor: {model_max_tasks_in_flight_per_actor}")
    logger.info(f"Model maximum restarts: {model_max_restarts}")
    logger.info(f"Model GPU memory utilization: {model_gpu_memory_utilization}")
    logger.info(f"Model KV cache memory bytes: {model_kv_cache_memory_bytes}")
    logger.info(f"Embedding pooling: {embedding_pooling}")
    logger.info(f"Input format: {input_format}")
    logger.info(f"Max chars: {max_chars}")
    logger.info(f"Executor: {executor}")

    run_start_time = time.perf_counter()

    keep_ext = "jsonl" if input_format == "jsonl" else "parquet"
    input_files = load_dataset_files(
        input_path,
        dataset_size_gb=dataset_size_gb,
        dataset_ratio=1.0 if max_input_files is not None else load_dataset_ratio,
        keep_extensions=keep_ext,
    )
    if max_input_files is not None:
        input_files = sorted(input_files)[:max_input_files]
    logger.info(f"Selected {len(input_files)} input files")
    if id_generator_path is not None:
        id_generator = IdGeneratorBase.from_disk(id_generator_path)
        validation_reader = JsonlReader(file_paths=[], id_generator_path_mapping=id_path_mapping)
        id_input_files = validation_reader.decompose()[-1]._map_id_generator_paths(input_files)
        missing_files = [
            path
            for path, id_path in zip(input_files, id_input_files, strict=True)
            if id_generator.hash_files(id_path) not in id_generator.batch_registry
        ]
        if missing_files:
            examples = "\n".join(missing_files[:5])
            msg = (
                f"ID-generator registry does not contain {len(missing_files)}/{len(input_files)} selected file hashes. "
                "Paths and file grouping must exactly match those used to generate the registry. "
                f"First missing paths:\n{examples}"
            )
            raise ValueError(msg)
    executor_obj = setup_executor(executor)

    retained_fields = (
        [CURATOR_DEDUP_ID_STR, *metadata_fields]
        if input_format == "jsonl" and id_generator_path is not None
        else metadata_fields
    )
    embedding_stages = _create_embedding_stages(
        model_identifier=model_identifier,
        model_variation=variation,
        model_inference_batch_size=model_inference_batch_size,
        model_num_workers=model_num_workers,
        model_worker_gpus=model_worker_gpus,
        model_max_tasks_in_flight_per_actor=model_max_tasks_in_flight_per_actor,
        model_max_restarts=model_max_restarts,
        model_gpu_memory_utilization=model_gpu_memory_utilization,
        model_kv_cache_memory_bytes=model_kv_cache_memory_bytes,
        max_seq_length=max_seq_length,
        embedding_pooling=embedding_pooling,
        max_chars=max_chars,
        cache_dir=cache_dir,
        retained_fields=retained_fields,
    )

    if input_format == "jsonl":
        output_fields = (
            [CURATOR_DEDUP_ID_STR, *metadata_fields, "embeddings"]
            if id_generator_path is not None
            else [*metadata_fields, "embeddings"]
        )
        reader = JsonlReader(
            file_paths=input_files,
            files_per_partition=1,
            fields=["text", *metadata_fields],
            _assign_ids=id_generator_path is not None,
            id_generator_path_mapping=id_path_mapping,
            read_max_workers=reader_max_workers,
        )
        writer = ParquetWriter(
            path=str(output_path),
            fields=output_fields,
        )
    else:
        reader = ParquetReader(
            file_paths=input_files,
            files_per_partition=1,
            fields=["text", *metadata_fields],
            _generate_ids=False,
        )
        writer = ParquetWriter(path=str(output_path), fields=[*metadata_fields, "embeddings"])

    pipeline = Pipeline(
        name="embedding_generation_pipeline",
        stages=[reader, *embedding_stages, writer],
    )
    if id_generator_path is not None:
        create_id_generator_actor(id_generator_path)
    try:
        output_tasks = pipeline.run(executor_obj)
    finally:
        if id_generator_path is not None:
            kill_id_generator_actor()

    run_time_taken = time.perf_counter() - run_start_time

    num_documents_processed = sum(task._stage_perf[-1].num_items_processed for task in output_tasks)
    throughput_docs_per_sec = num_documents_processed / run_time_taken if run_time_taken > 0 else 0
    stage_metrics = TaskPerfUtils.collect_stage_metrics(output_tasks)
    gpu_stage_process_time_sum = sum(
        float(values["process_time"].sum())
        for stage_name, values in stage_metrics.items()
        if stage_name.endswith("_vllm") and "process_time" in values
    )
    gpu_stage_num_items_processed = sum(
        int(values["num_items_processed"].sum())
        for stage_name, values in stage_metrics.items()
        if stage_name.endswith("_vllm") and "num_items_processed" in values
    )
    models_per_gpu = 1.0 / model_worker_gpus if model_worker_gpus else 0.0
    throughput_items_per_gpu_second = (
        models_per_gpu * gpu_stage_num_items_processed / gpu_stage_process_time_sum
        if gpu_stage_process_time_sum > 0
        else 0.0
    )

    logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
    logger.success(f"Processed {num_documents_processed} documents")

    return {
        "params": {
            "max_seq_length": max_seq_length,
            "load_dataset_ratio": load_dataset_ratio,
            "max_input_files": max_input_files,
            "id_generator_path": id_generator_path,
            "id_path_mapping": id_path_mapping,
            "metadata_fields": metadata_fields,
            "reader_max_workers": reader_max_workers,
            "model_num_workers": model_num_workers,
            "model_worker_gpus": model_worker_gpus,
            "model_max_tasks_in_flight_per_actor": model_max_tasks_in_flight_per_actor,
            "model_max_restarts": model_max_restarts,
            "model_gpu_memory_utilization": model_gpu_memory_utilization,
            "model_kv_cache_memory_bytes": model_kv_cache_memory_bytes,
            "max_chars": max_chars,
        },
        "metrics": {
            "is_success": True,
            "time_taken_s": run_time_taken,
            "num_documents_processed": num_documents_processed,
            "throughput_docs_per_sec": throughput_docs_per_sec,
            "gpu_stage_process_time_sum_s": gpu_stage_process_time_sum,
            "gpu_stage_num_items_processed": gpu_stage_num_items_processed,
            "models_per_gpu": models_per_gpu,
            "throughput_items_per_gpu_second": throughput_items_per_gpu_second,
        },
        "tasks": output_tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding generation benchmark for nightly benchmarking")
    parser.add_argument("--benchmark-results-path", required=True, help="Path to benchmark results")
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--output-path", default="./embedding_generation_output", help="Output directory for results")
    parser.add_argument("--executor", default="ray_data", choices=["xenna", "ray_data"], help="Executor to use")
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--dataset-size-gb", type=float, default=None, help="Size of dataset to process in GB")
    size_group.add_argument(
        "--load-dataset-ratio", type=float, default=None, help="Fraction of input files to process"
    )
    size_group.add_argument("--max-input-files", type=int, default=None, help="Exact maximum number of input files")
    parser.add_argument(
        "--id-generator-path",
        default=None,
        help="Saved ID-generator state used to assign existing deduplication IDs",
    )
    parser.add_argument(
        "--id-path-mapping-json",
        default=None,
        help="JSON dictionary mapping physical path prefixes to ID-generator path prefixes",
    )
    parser.add_argument(
        "--metadata-field",
        dest="metadata_fields",
        action="append",
        default=None,
        help="Input metadata field to preserve in output Parquet; may be repeated",
    )
    parser.add_argument(
        "--reader-max-workers",
        type=int,
        default=None,
        help="Maximum number of concurrent JSONL reader actors",
    )
    parser.add_argument(
        "--model-identifier",
        required=True,
        help="Model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument("--model-inference-batch-size", type=int, default=1024, help="Batch size for model inference")
    parser.add_argument("--model-num-workers", type=int, default=None, help="Ray actor count for the model stage")
    parser.add_argument("--model-worker-gpus", type=float, default=None, help="GPUs reserved per model actor")
    parser.add_argument(
        "--model-max-tasks-in-flight-per-actor",
        type=int,
        default=1,
        help="Maximum Ray Data file tasks queued on each model actor",
    )
    parser.add_argument(
        "--model-max-restarts",
        type=int,
        default=1,
        help="Maximum Ray actor restarts and task retries for the model stage",
    )
    parser.add_argument(
        "--model-gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU-memory fraction per model actor",
    )
    parser.add_argument(
        "--model-kv-cache-memory-bytes",
        type=int,
        default=None,
        help="Explicit per-actor vLLM KV-cache reservation in bytes",
    )
    parser.add_argument(
        "--model-variation",
        default="vllm_text",
        choices=[v.value for v in EmbeddingModelVariation],
        help="Embedding model backend (default: vllm_text)",
    )
    parser.add_argument(
        "--embedding-pooling",
        default="mean_pooling",
        choices=["mean_pooling", "last_token"],
        help="Pooling strategy for pytorch_model variation (ignored by sentence_transformer)",
    )
    parser.add_argument(
        "--input-format",
        default="parquet",
        choices=["parquet", "jsonl"],
        help="Input file format (default: parquet)",
    )
    parser.add_argument("--max-chars", type=int, default=None, help="Maximum characters per input text")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace cache directory for model weights (uses default HF cache if not set)",
    )

    args = parser.parse_args()

    logger.info("=== Embedding Generation Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    success_code = 1
    result_dict: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        run_result = run_embedding_generation_benchmark(**vars(args))
        result_dict["params"].update(run_result.pop("params", {}))
        result_dict.update(run_result)
        success_code = 0 if result_dict["metrics"]["is_success"] else 1
    finally:
        write_benchmark_results(result_dict, args.benchmark_results_path)
    return success_code


if __name__ == "__main__":
    raise SystemExit(main())
