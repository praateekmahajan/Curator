#!/usr/bin/env python
import argparse
from pathlib import Path

import pyarrow.parquet as pq


DATASET_ROOT = Path(
    "/lustre/fsw/portfolios/nemotron/projects/nemotron_n4_pre/datasets/sft/clean/qa_parse_status=true"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Inspect every parquet footer")
    parser.add_argument("--scan-ids", action="store_true", help="Read int_id and verify global uniqueness on GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_files = sorted(DATASET_ROOT.glob("*.parquet"))
    files = all_files if args.all else all_files[:5]
    total_rows = 0
    id_ranges = []
    id_nulls = 0
    question_nulls = 0
    combined_nulls = 0
    print(f"files_inspected={len(files)}")
    for path in files:
        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        total_rows += parquet_file.metadata.num_rows
        for column in ("int_id", "question", "combined_question_answer"):
            schema.field(column)
        id_index = schema.get_field_index("int_id")
        question_index = schema.get_field_index("question")
        combined_index = schema.get_field_index("combined_question_answer")
        for index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(index)
            id_stats = row_group.column(id_index).statistics
            question_stats = row_group.column(question_index).statistics
            combined_stats = row_group.column(combined_index).statistics
            if id_stats is None or not id_stats.has_min_max:
                raise RuntimeError(f"Missing int_id statistics: {path} row group {index}")
            id_ranges.append((id_stats.min, id_stats.max))
            id_nulls += id_stats.null_count
            question_nulls += question_stats.null_count if question_stats else 0
            combined_nulls += combined_stats.null_count if combined_stats else 0

    sorted_ranges = sorted(id_ranges)
    overlapping_ranges = [
        (previous, current)
        for previous, current in zip(sorted_ranges, sorted_ranges[1:], strict=False)
        if current[0] <= previous[1]
    ]
    print(f"total_rows={total_rows}")
    print("int_id_type=int64")
    print(f"int_id_min={sorted_ranges[0][0]}")
    print(f"int_id_max={sorted_ranges[-1][1]}")
    print(f"int_id_nulls={id_nulls}")
    print(f"overlapping_id_ranges={len(overlapping_ranges)}")
    print(f"question_nulls={question_nulls}")
    print(f"combined_question_answer_nulls={combined_nulls}")

    if args.scan_ids:
        import cudf

        ids = cudf.read_parquet(
            [str(path) for path in files],
            columns=["int_id"],
            dataset_kwargs={"partitioning": None},
        )["int_id"]
        distinct_ids = ids.nunique(dropna=False)
        print(f"scanned_id_rows={len(ids)}")
        print(f"distinct_ids={distinct_ids}")
        print(f"duplicate_ids={len(ids) - distinct_ids}")


if __name__ == "__main__":
    main()
