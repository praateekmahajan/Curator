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

"""Prepare sharded seed datasets for the SLURM array DataDesigner tutorial."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATASETS_ROOT = WORKSPACE_ROOT / "datasets"

GRETEL_BASE_URL = "https://huggingface.co/datasets/gretelai/symptom_to_diagnosis/resolve/main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and shard a seed dataset into JSONL and Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["ndd_gretel_symptoms", "wikimedia_wikipedia"],
        default="ndd_gretel_symptoms",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--rows", type=int, default=1065, help="Maximum rows to write after filtering.")
    parser.add_argument("--rows-per-file", type=int, default=25)
    parser.add_argument("--formats", default="jsonl,parquet", help="Comma-separated subset of jsonl,parquet.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing shard files.")
    parser.add_argument(
        "--wiki-config",
        default="20231101.simple",
        help="Hugging Face config for wikimedia/wikipedia, used only for wikimedia_wikipedia.",
    )
    parser.add_argument("--wiki-min-chars", type=int, default=1000)
    parser.add_argument("--wiki-max-chars", type=int, default=6000)
    return parser.parse_args()


def _read_gretel_split(split: str) -> pd.DataFrame:
    url = f"{GRETEL_BASE_URL}/{split}.jsonl"
    logger.info(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        records = [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line.strip()]
    df = pd.DataFrame.from_records(records)
    df["source_split"] = split
    df["source_row_id"] = range(len(df))
    return df


def _load_gretel(rows: int) -> pd.DataFrame:
    df = pd.concat([_read_gretel_split("train"), _read_gretel_split("test")], ignore_index=True)
    df = df.rename(columns={"input_text": "patient_summary", "output_text": "diagnosis"})
    df = df[["source_split", "source_row_id", "diagnosis", "patient_summary"]]
    return df.head(rows)


def _load_wikipedia(args: argparse.Namespace) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        msg = (
            "The wikimedia_wikipedia dataset mode requires the optional `datasets` package. "
            "Run with `uv run --with datasets python prepare_seed_dataset.py --dataset wikimedia_wikipedia ...`."
        )
        raise ModuleNotFoundError(msg) from exc

    logger.info(f"Streaming wikimedia/wikipedia config={args.wiki_config}")
    stream = load_dataset("wikimedia/wikipedia", args.wiki_config, split="train", streaming=True)
    records: list[dict[str, Any]] = []
    for row in stream:
        text = str(row.get("text") or "").strip()
        if len(text) < args.wiki_min_chars:
            continue
        records.append(
            {
                "source_id": str(row.get("id") or ""),
                "source_url": str(row.get("url") or ""),
                "wikipedia_article_title": str(row.get("title") or ""),
                "wikipedia_article_content": text[: args.wiki_max_chars],
            }
        )
        if len(records) >= args.rows:
            break
    if not records:
        msg = "No Wikipedia rows matched the requested filters"
        raise RuntimeError(msg)
    return pd.DataFrame.from_records(records)


def _dataset_dir(output_root: Path, dataset_name: str) -> Path:
    return output_root / dataset_name


def _write_shards(df: pd.DataFrame, dataset_dir: Path, rows_per_file: int, formats: set[str], force: bool) -> None:
    if rows_per_file < 1:
        msg = f"rows_per_file must be >= 1, got {rows_per_file}"
        raise ValueError(msg)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for output_format in formats:
        if output_format not in {"jsonl", "parquet"}:
            msg = f"Unsupported output format: {output_format}"
            raise ValueError(msg)
        format_dir = dataset_dir / output_format
        format_dir.mkdir(parents=True, exist_ok=True)
        if force:
            for old_file in format_dir.glob(f"*.{output_format}"):
                old_file.unlink()

        for shard_idx, start in enumerate(range(0, len(df), rows_per_file)):
            shard = df.iloc[start : start + rows_per_file].reset_index(drop=True)
            if output_format == "jsonl":
                output_file = format_dir / f"part_{shard_idx:05d}.jsonl"
                if output_file.exists() and not force:
                    continue
                shard.to_json(output_file, orient="records", lines=True, force_ascii=False)
            else:
                output_file = format_dir / f"part_{shard_idx:05d}.parquet"
                if output_file.exists() and not force:
                    continue
                shard.to_parquet(output_file, index=False)
        logger.info(f"Wrote {output_format} shards under {format_dir}")

    manifest = {
        "dataset_dir": str(dataset_dir),
        "num_rows": len(df),
        "rows_per_file": rows_per_file,
        "formats": sorted(formats),
        "columns": list(df.columns),
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    formats = {value.strip() for value in args.formats.split(",") if value.strip()}
    dataset_dir = _dataset_dir(args.output_root, args.dataset)

    df = _load_gretel(args.rows) if args.dataset == "ndd_gretel_symptoms" else _load_wikipedia(args)

    _write_shards(df, dataset_dir, args.rows_per_file, formats, args.force)
    logger.info(f"Prepared {len(df)} row(s) in {dataset_dir}")


if __name__ == "__main__":
    main()
