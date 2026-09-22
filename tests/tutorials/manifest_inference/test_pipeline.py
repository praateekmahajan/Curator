# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def test_failed_source_replays_and_completed_source_skips_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from nemo_curator.backends.base import BaseStageAdapter
    from nemo_curator.tasks import EmptyTask
    from nemo_curator.utils.resumability_actor import ResumabilityActor
    from tutorials.text.manifest_inference import inference
    from tutorials.text.manifest_inference.manifest import generate_manifest
    from tutorials.text.manifest_inference.stages import ManifestFilePartitioningStage, SpecificJsonlReader
    from tutorials.text.manifest_inference.writer import NamedJsonlWriter

    class Client:
        fail = True

        def __init__(self, **kwargs):
            self.client = SimpleNamespace(with_options=lambda **_kw: self.client, close=AsyncMock())

        def setup(self) -> None:
            pass

        async def query_model_response(self, *, messages: list[dict[str, str]], **kwargs) -> SimpleNamespace:
            if self.fail and messages[0]["content"] == "retry":
                msg = "temporary failure"
                raise RuntimeError(msg)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="answer", reasoning=None, reasoning_content=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
            )

    monkeypatch.setattr(inference, "AsyncOpenAIClient", Client)
    source = tmp_path / "input"
    source.mkdir()
    (source / "part.jsonl").write_text('{"question":"ok"}\n{"question":"retry"}\n')
    manifest = tmp_path / "manifest.jsonl"
    generate_manifest(source, manifest, 1)
    source_stage = ManifestFilePartitioningStage(str(manifest), str(source))
    source_stage.is_source_stage = True
    sink = NamedJsonlWriter(str(tmp_path / "output"))
    sink.is_sink_stage = True
    client = inference.ResumableInferenceStage(
        model_alias="qwen",
        model={
            "model": "test",
            "endpoint": "http://unused/v1",
            "replicas": 1,
            "gpus_per_replica": 1,
            "chat_template_kwargs": {},
        },
        prompt_field="question",
        generation={"max_tokens": 4, "temperature": 0, "top_p": 1, "seed": 42},
        max_concurrent_requests=1,
        max_retries=0,
        failure_dir=str(tmp_path / "failures"),
    )
    actor_class = ResumabilityActor.__ray_metadata__.modified_class

    def attempt(writer_id: str) -> int:
        actor = actor_class(str(tmp_path / "checkpoint"), writer_id=writer_id)

        def completed(ids: list[str]) -> set[str]:
            return {key for key, done in zip(ids, actor.are_completed(ids), strict=True) if done}

        try:
            with (
                patch("nemo_curator.backends.base.is_resumability_actor_active", return_value=True),
                patch("nemo_curator.backends.base.flush_resumability_deltas", side_effect=actor.apply_deltas),
                patch("nemo_curator.backends.base.completed_resumability_sources", side_effect=completed),
            ):
                pending = BaseStageAdapter(source_stage).process_batch([EmptyTask()])
                for task in pending:
                    rows = BaseStageAdapter(SpecificJsonlReader()).process_batch([task])
                    results = BaseStageAdapter(client).process_batch(rows)
                    if results:
                        BaseStageAdapter(sink).process_batch(results)
                return len(pending)
        finally:
            actor.close()

    assert attempt("first") == 2
    outputs = [path for path in (tmp_path / "output").rglob("*.jsonl") if path.is_file()]
    assert len(outputs) == 1
    first_mtime = outputs[0].stat().st_mtime_ns
    Client.fail = False
    assert attempt("second") == 1
    assert outputs[0].stat().st_mtime_ns == first_mtime
    assert len([path for path in (tmp_path / "output").rglob("*.jsonl") if path.is_file()]) == 2
    assert attempt("third") == 0
