import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu-regression",
        action="store_true",
        default=False,
        help="Run GPU-only regression tests that execute end-to-end local model/judge evaluations.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "vllm_engine_test: marks tests that target the vLLM eval engine stubs",
    )
    config.addinivalue_line(
        "markers",
        "transformers_engine_test: marks tests that target the transformers eval engine stubs",
    )
    config.addinivalue_line(
        "markers",
        "gpu_regression: marks slow GPU-only end-to-end regression checks (opt-in).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if not items:
        return
    if config.getoption("--run-gpu-regression"):
        return

    skip_gpu_regression = pytest.mark.skip(
        reason=(
            "GPU regression tests are disabled by default. "
            "Re-run pytest with --run-gpu-regression to execute them."
        )
    )
    for item in items:
        if "gpu_regression" in item.keywords:
            item.add_marker(skip_gpu_regression)
