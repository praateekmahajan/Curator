# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apply per-cluster boolean removal masks without CPU embedding materialization."""

from dataclasses import dataclass
from pathlib import Path

import cudf

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.semantic.utils import read_parquet_chunks
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import get_all_file_paths_under


@dataclass
class SemanticDuplicateFilterStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """GPU filtering over cluster-partitioned Parquet and matching pairwise masks.

    Embeddings/payload are chunked; one cluster's ID mask must fit GPU memory.
    Paths must be local/shared. Output directory must be new for each run.
    """

    masks_path: str
    output_path: str
    columns: list[str]
    id_field: str = "id"
    read_memory_bytes: int = 256 * 1024**2
    name: str = "semantic_duplicate_filter"

    def __post_init__(self) -> None:
        super().__init__()
        if self.id_field not in self.columns or self.read_memory_bytes <= 0:
            msg = "columns must include id_field and read_memory_bytes must be positive"
            raise ValueError(msg)
        self.resources = Resources(cpus=1.0, gpus=1.0)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        cluster = int(task._metadata["centroid_id"])
        mask = cudf.read_parquet(f"{self.masks_path}/cluster_{cluster}.parquet", columns=["id", "removed"])
        mask = mask.rename(columns={"id": self.id_field})
        if mask.isnull().any().any() or mask[self.id_field].duplicated().any() or mask.removed.dtype != "bool":
            msg = "removal masks require unique non-null IDs and non-null boolean removed flags"
            raise ValueError(msg)
        removed = mask.loc[mask.removed, self.id_field]
        files = sorted(
            file
            for path in task.data
            for file in get_all_file_paths_under(
                path,
                recurse_subdirectories=True,
                keep_extensions=[".parquet"],
            )
        )
        root = Path(self.output_path)
        root.mkdir(parents=True, exist_ok=True)
        outputs, input_rows, kept_rows = [], 0, 0
        for index, frame in enumerate(read_parquet_chunks(files, self.columns, self.read_memory_bytes, None)):
            if not frame[self.id_field].isin(mask[self.id_field]).all():
                msg = "input contains IDs absent from its cluster mask"
                raise ValueError(msg)
            input_rows += len(frame)
            retained = frame[~frame[self.id_field].isin(removed)]
            kept_rows += len(retained)
            if len(retained):
                output = root / f"cluster-{cluster}-{index}.parquet"
                if output.exists():
                    raise FileExistsError(output)
                retained.to_parquet(str(output), index=False, compression="zstd")
                outputs.append(str(output))
        if input_rows != len(mask) or input_rows - kept_rows != len(removed):
            msg = "cluster rows and removal mask cardinalities differ"
            raise ValueError(msg)
        return FileGroupTask(
            data=outputs,
            dataset_name=task.dataset_name,
            _metadata={**task._metadata, "input_rows": input_rows, "kept_rows": kept_rows},
            _stage_perf=task._stage_perf,
        )
