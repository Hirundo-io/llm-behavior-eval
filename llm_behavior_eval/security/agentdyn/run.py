"""Run AgentDyn on two models (a base and a hardened arm) with identical settings.

Both arms get the same episodes (the same suites, mode, attack and stable
sample IDs) and the same generation settings, in one Inspect evaluation; the
summary CLI (``llm-behavior-eval-agentdyn-summary``) then compares them pair by
pair. Generation settings left unset fall through to the serving stack's
defaults; for a single model, use ``inspect eval llm_behavior_eval/agentdyn``.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, Model, get_model

from .constants import DEFAULT_ATTACK, MODES, SUITES, SUPPORTED_ATTACKS
from .task import agentdyn

logger = logging.getLogger(__name__)

ARMS = ("base", "hardened")
DEFAULT_MODELS = {
    "base": ("openai-api/local/base", "http://127.0.0.1:18000/v1"),
    "hardened": ("openai-api/local/hardened", "http://127.0.0.1:18001/v1"),
}
"""Each arm's Inspect model and endpoint: two local OpenAI-compatible servers
(for example vLLM) serving the aliases ``base`` and ``hardened``."""
DEFAULT_MAX_SAMPLES = 32


def parser() -> argparse.ArgumentParser:
    """The command line of ``llm-behavior-eval-agentdyn``."""
    parser = argparse.ArgumentParser(
        prog="llm-behavior-eval-agentdyn", description=__doc__
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="episodes per arm (a pilot run)")
    parser.add_argument("--suites", default=",".join(SUITES))
    parser.add_argument("--mode", choices=MODES, default="both")
    parser.add_argument("--attack", choices=SUPPORTED_ATTACKS, default=DEFAULT_ATTACK)
    for arm in ARMS:
        model, url = DEFAULT_MODELS[arm]
        parser.add_argument(f"--{arm}-model", default=model, help=f"default: {model}")
        parser.add_argument(f"--{arm}-url", default=url, help=f"default: {url}")
    settings = parser.add_argument_group(
        "generation settings for both arms (default: the server's)"
    )
    settings.add_argument("--temperature", type=float)
    settings.add_argument("--top-p", type=float)
    settings.add_argument("--max-tokens", type=int)
    settings.add_argument("--seed", type=int)
    settings.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        help="sent as the request's reasoning_effort (as vLLM reads it)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"concurrent episodes per arm (default: {DEFAULT_MAX_SAMPLES})",
    )
    return parser


def generate_config(arguments: argparse.Namespace) -> GenerateConfig:
    """The settings both arms share; only those given on the command line."""
    return GenerateConfig(
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        max_tokens=arguments.max_tokens,
        seed=arguments.seed,
        extra_body=(
            None
            if arguments.reasoning_effort is None
            else {"reasoning_effort": arguments.reasoning_effort}
        ),
    )


def validate(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    """Reject out-of-range numbers before any model is contacted."""
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")
    if arguments.max_tokens is not None and arguments.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if arguments.temperature is not None and not 0 <= arguments.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")
    if arguments.top_p is not None and not 0 < arguments.top_p <= 1:
        parser.error("--top-p must be greater than 0 and at most 1")
    if arguments.max_samples < 1:
        parser.error("--max-samples must be positive")
    if arguments.base_model == arguments.hardened_model:
        parser.error("--base-model and --hardened-model must differ")


def failures(logs: Sequence[EvalLog]) -> list[str]:
    """What keeps the logs from being reported: failed runs or episodes."""
    problems = []
    for log in logs:
        if log.status != "success":
            problems.append(f"{log.location}: {log.status}")
        elif log.results is None or (
            log.results.completed_samples != log.results.total_samples
        ):
            problems.append(f"{log.location}: incomplete")
        problems.extend(
            f"{log.location}: {sample.id}: {sample.error.message}"
            for sample in log.samples or []
            if sample.error is not None
        )
    return problems


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate both arms on identical episodes and settings."""
    logging.basicConfig(level=logging.INFO)
    command = parser()
    arguments = command.parse_args(argv)
    validate(command, arguments)
    config = generate_config(arguments)
    task = agentdyn(
        suites=arguments.suites, mode=arguments.mode, attack=arguments.attack
    )
    models: list[Model] = [
        get_model(
            getattr(arguments, f"{arm}_model"),
            base_url=getattr(arguments, f"{arm}_url"),
            config=config,
        )
        for arm in ARMS
    ]
    logs = inspect_eval(
        task,
        model=models,
        limit=arguments.limit,
        log_dir=str(arguments.log_dir),
        max_samples=arguments.max_samples,
        max_tasks=len(models),
        fail_on_error=False,
    )
    problems = failures(logs)
    if problems:
        raise SystemExit(
            "The evaluation has errors; inspect the logs before reporting:\n"
            + "\n".join(problems)
        )
    logger.info("Completed %s logs in %s", len(logs), arguments.log_dir)


if __name__ == "__main__":
    main()
