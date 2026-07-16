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

import pandas as pd
import pyarrow as pa

from nemo_curator.tasks import DocumentBatch


def test_arrow_table_to_pandas_preserves_arrow_backed_dtypes() -> None:
    batch = DocumentBatch(
        dataset_name="dataset",
        data=pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "text": pa.array(["a", "b"], type=pa.string()),
            }
        ),
    )

    result = batch.to_pandas()

    assert isinstance(result["id"].dtype, pd.ArrowDtype)
    assert isinstance(result["text"].dtype, pd.ArrowDtype)
