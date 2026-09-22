import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))
from runner.env_capture import publish_session_environment


def test_parallel_array_environment_publication_preserves_attempts(tmp_path: Path):
    attempts = []
    for index in range(8):
        path = tmp_path / "array_environments" / str(index)
        path.mkdir(parents=True)
        (path / "env.json").write_text(json.dumps({"hostname": str(index)}))
        (path / "packages.txt").write_text(f"package-{index}")
        attempts.append(path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda path: publish_session_environment(path, tmp_path), attempts))
    winner = json.loads((tmp_path / "env.json").read_text())["hostname"]
    assert (tmp_path / "packages.txt").read_text() == f"package-{winner}"
    assert all((path / "env.json").exists() for path in attempts)
    publish_session_environment(attempts[0], tmp_path)
    assert json.loads((tmp_path / "env.json").read_text())["hostname"] == winner
