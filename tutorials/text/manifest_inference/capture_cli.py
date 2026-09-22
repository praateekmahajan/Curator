"""Capture the exact vLLM argv before the server is executed."""

import json
import sys
from pathlib import Path


def server_lineage(argv: list[str]) -> dict:
    prefix_length = 3
    if len(argv) < prefix_length or argv[:2] != ["vllm", "serve"]:
        msg = "Expected vllm serve MODEL followed by CLI options"
        raise ValueError(msg)
    kwargs = {}
    index = 3
    while index < len(argv):
        option = argv[index]
        if not option.startswith("--"):
            msg = f"Unexpected positional server argument: {option}"
            raise ValueError(msg)
        key, separator, value = option[2:].partition("=")
        if not separator and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            index += 1
            value = argv[index]
            separator = "="
        if separator:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
        else:
            parsed = True
            if key.startswith("no-"):
                key, parsed = key[3:], False
        kwargs[key.replace("-", "_")] = parsed
        index += 1
    return {"model": argv[2], "server_cli_kwargs": kwargs, "server_argv": argv}


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(server_lineage(sys.argv[2:]), indent=2) + "\n")
