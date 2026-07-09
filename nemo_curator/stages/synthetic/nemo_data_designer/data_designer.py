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

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    import data_designer.config as dd
    from data_designer.interface import DataDesigner


@dataclass
class DataDesignerStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Data Designer stage.

    This class provides a Data Designer stage.
    To request GPUs, use: DataDesignerStage(...).with_(resources=Resources(gpus=X)).

    When ``verbose`` is False (default), NeMo Data Designer (NDD) log output is suppressed
    (e.g. "Preview generation in progress", "Preview complete!") so the stage is less verbose.
    Set ``verbose=True`` to see full NDD logging.

    Optional ``model_providers``: pass a list of :class:`data_designer.config.models.ModelProvider`
    to use custom or test endpoints (e.g. a mock LLM server). If None, the default DataDesigner
    providers are used.
    """

    config_builder: dd.DataDesignerConfigBuilder | None = None
    data_designer_config_file: str | None = None
    model_providers: list | None = None
    verbose: bool = False
    run_mode: str = "preview"
    artifact_path: str | Path | None = None
    resume_mode: str = "if_possible"
    run_config: dict | object | None = None
    data_designer: DataDesigner = field(init=False)

    def __post_init__(self) -> None:
        import data_designer.config as dd

        # Set in __post_init__ so they are not constructor args; use .with_(resources=..., name=...) to customize.
        self.resources = Resources(gpus=0.0)
        self.name = "DataDesignerStage"

        # check config_builder and data_designer_config_file
        if self.config_builder is None and self.data_designer_config_file is None:
            msg = "Either 'config_builder' or 'data_designer_config_file' must be set."
            raise ValueError(msg)
        if self.config_builder is not None and self.data_designer_config_file is not None:
            msg = "Only one of 'config_builder' or 'data_designer_config_file' can be set, not both."
            raise ValueError(msg)
        if self.run_mode not in {"preview", "create"}:
            msg = "run_mode must be either 'preview' or 'create'."
            raise ValueError(msg)
        if self.run_mode == "create" and self.artifact_path is None:
            msg = "artifact_path must be set when run_mode='create'."
            raise ValueError(msg)

        # read config from file if config_builder is not set
        if self.config_builder is None:
            self.config_builder = dd.DataDesignerConfigBuilder.from_config(self.data_designer_config_file)
        self._init_data_designer()

    def __getstate__(self) -> dict:
        """Return deepcopy/pickle state without the live DataDesigner client.

        DataDesigner 0.7.0 owns runtime objects such as locks that are not deepcopyable.
        Executors can safely copy the stage configuration and recreate the client afterward.
        """
        state = self.__dict__.copy()
        state.pop("data_designer", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._init_data_designer()

    def _init_data_designer(self) -> None:
        from data_designer.interface import DataDesigner

        if self.model_providers is not None:
            self.data_designer = DataDesigner(model_providers=self.model_providers)
        else:
            self.data_designer = DataDesigner()

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        import data_designer.config as dd

        num_input_records = batch.num_items
        # set seed dataframe from batch
        self.config_builder.with_seed_dataset(dd.DataFrameSeedSource(df=batch.to_pandas()))

        # When verbose is False, suppress NDD's logging (it logs "Preview generation in progress", etc.)
        ndd_logger = logging.getLogger("data_designer")
        if not self.verbose:
            _old_ndd_level = ndd_logger.level
            ndd_logger.setLevel(logging.WARNING)

        try:
            t1 = time.perf_counter()
            if self.run_mode == "preview":
                results = self.data_designer.preview(self.config_builder, num_records=num_input_records)
                df = results.dataset
            else:
                results = self._create_dataset(num_records=num_input_records, batch=batch)
                df = results.load_dataset()
            ndd_running_time = time.perf_counter() - t1
        finally:
            if not self.verbose:
                ndd_logger.setLevel(_old_ndd_level)

        num_output_records = len(df)

        # Token metrics from NDD stats analysis
        # (these stats are available for LLM columns only)
        output_medians = []
        input_medians = []
        analysis = self._get_analysis(results)
        if analysis:
            # Loop through all columns in the analysis that has LLM token stats
            for col_stat in analysis.column_statistics:
                in_median = getattr(col_stat, "input_tokens_median", None)
                out_median = getattr(col_stat, "output_tokens_median", None)
                if isinstance(in_median, (int, float)):
                    input_medians.append(float(in_median))
                if isinstance(out_median, (int, float)):
                    output_medians.append(float(out_median))
        # Sum across all columns that have LLM token stats
        output_tokens_median_per_record = sum(output_medians) if output_medians else 0.0
        input_tokens_median_per_record = sum(input_medians) if input_medians else 0.0

        self._log_metrics(
            {
                "ndd_running_time": ndd_running_time,
                "num_input_records": float(num_input_records),
                "num_output_records": float(num_output_records),
                "input_tokens_median_per_record": float(input_tokens_median_per_record),
                "output_tokens_median_per_record": float(output_tokens_median_per_record),
            }
        )

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def _create_dataset(self, *, num_records: int, batch: DocumentBatch) -> object:
        import data_designer.config as dd
        from data_designer.interface import ResumeMode

        if self.run_config is not None:
            run_config = self.run_config if not isinstance(self.run_config, dict) else dd.RunConfig(**self.run_config)
            self.data_designer.set_run_config(run_config)

        return self.data_designer.create(
            self.config_builder,
            num_records=num_records,
            dataset_name=self._dataset_name_for_batch(batch),
            artifact_path=Path(self.artifact_path),
            resume=ResumeMode(self.resume_mode),
        )

    def _dataset_name_for_batch(self, batch: DocumentBatch) -> str:
        source_files = batch._metadata.get("source_files")
        if source_files:
            identity = json.dumps(sorted(str(path) for path in source_files), sort_keys=True)
        elif batch.task_id:
            identity = batch.task_id
        else:
            identity = f"{batch.dataset_name}:{batch.num_items}"

        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        safe_dataset_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", batch.dataset_name).strip("-.") or "dataset"
        return f"{safe_dataset_name[:80]}-{digest}"

    @staticmethod
    def _get_analysis(results: object) -> object | None:
        analysis = getattr(results, "analysis", None)
        if analysis is not None:
            return analysis
        load_analysis = getattr(results, "load_analysis", None)
        if load_analysis is None or not callable(load_analysis):
            return None
        return load_analysis()


# Explicitly export the class
__all__ = ["DataDesignerStage"]
