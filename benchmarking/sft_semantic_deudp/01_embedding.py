"""Generate multi-column SFT semantic-dedup embeddings."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from utils import load_dataset_files, setup_executor, write_benchmark_results  # noqa: E402
from nemo_curator.backends.utils import get_available_cpu_gpu_resources
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.embedders.vllm import VLLMEmbeddingModelStage
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

EMBEDDING_FIELDS = {
    "question": "question_embedding",
    "combined_question_answer": "combined_question_answer_embedding",
    "original_text": "original_text_embedding",
}


def effective_entry_name() -> str:
    return "harrier_oss_270m_{}_{}".format(
        os.environ.get("SLURM_ARRAY_TASK_ID", "0"), os.environ.get("SLURM_JOB_ID", "local")
    )


def run_embedding_benchmark(
    input_path: str,
    output_path: str,
    benchmark_results_path: str,
    executor: str,
    model_identifier: str,
    model_inference_batch_size: int,
    num_vllm_replicas_per_gpu: int,
    dataset_ratio: float,
    cache_dir: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    if not 0 < dataset_ratio <= 1:
        raise ValueError("dataset_ratio must be in (0, 1]")
    if num_vllm_replicas_per_gpu <= 0:
        raise ValueError("num_vllm_replicas_per_gpu must be positive")

    input_files = load_dataset_files(Path(input_path), dataset_ratio=dataset_ratio, keep_extensions="parquet")
    _, num_gpus = get_available_cpu_gpu_resources(init_and_shutdown=True)
    if num_gpus <= 0:
        raise RuntimeError("Ray reported no GPUs")
    num_workers = num_vllm_replicas_per_gpu * num_gpus
    logger.info(f"Using {num_gpus} GPUs and {num_workers} vLLM workers")

    stage = VLLMEmbeddingModelStage(
        model_identifier=model_identifier,
        embedding_fields=EMBEDDING_FIELDS,
        metadata_fields=["int_id"],
        model_inference_batch_size=model_inference_batch_size,
        cache_dir=cache_dir,
    ).with_(num_workers=num_workers)
    reader = ParquetReader(
        file_paths=input_files,
        files_per_partition=1,
        fields=["int_id", *EMBEDDING_FIELDS],
        _generate_ids=False,
    )
    writer = ParquetWriter(path=output_path, fields=["int_id", *EMBEDDING_FIELDS.values()])
    pipeline = Pipeline(name="sft_semantic_embedding", stages=[reader, stage, writer])
    started = time.perf_counter()
    tasks = pipeline.run(setup_executor(executor), checkpoint_path=Path(benchmark_results_path).absolute().parent)
    elapsed = time.perf_counter() - started
    processed = sum(task._stage_perf[-1].num_items_processed for task in tasks or [])
    return {
        "params": {"entry_name": effective_entry_name(), "num_gpus": num_gpus, "num_workers": num_workers},
        "metrics": {
            "is_success": True,
            "time_taken_s": elapsed,
            "num_documents_processed": processed,
            "throughput_docs_per_sec": processed / elapsed if elapsed else 0,
        },
        "tasks": tasks or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-results-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--executor", default="ray_data")
    parser.add_argument("--model-identifier", required=True)
    parser.add_argument("--model-inference-batch-size", type=int, default=1024)
    parser.add_argument("--num-vllm-replicas-per-gpu", type=int, default=4)
    parser.add_argument("--dataset-ratio", type=float, default=0.1)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    result: dict[str, Any] = {"params": vars(args), "metrics": {"is_success": False}, "tasks": []}
    try:
        result = run_embedding_benchmark(**vars(args))
        return 0
    finally:
        write_benchmark_results(result, args.benchmark_results_path)


if __name__ == "__main__":
    raise SystemExit(main())
