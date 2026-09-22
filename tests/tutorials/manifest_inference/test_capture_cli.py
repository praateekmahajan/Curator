from tutorials.text.manifest_inference.capture_cli import server_lineage


def test_capture_exact_argv_and_typed_options():
    argv = [
        "vllm",
        "serve",
        "model",
        "--no-enable-prefix-caching",
        "--seed",
        "42",
        "--gpu-memory-utilization=0.95",
        "--engram-config",
        '{"cpu_offload":true}',
        "--reasoning-parser",
        "deepseek_v41",
        "--enable-expert-parallel",
    ]
    lineage = server_lineage(argv)
    assert lineage["server_argv"] == argv
    assert lineage["server_cli_kwargs"] == {
        "enable_prefix_caching": False,
        "seed": 42,
        "gpu_memory_utilization": 0.95,
        "engram_config": {"cpu_offload": True},
        "reasoning_parser": "deepseek_v41",
        "enable_expert_parallel": True,
    }
