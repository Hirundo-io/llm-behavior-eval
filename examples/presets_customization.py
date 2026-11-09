"""Run legacy presets as native Inspect tasks, with per-role settings.

Presets choose tasks and task arguments only. Every inference setting below is
the user's choice, passed through Inspect for each role; the benchmarks set
none. Drop a setting to fall through to the serving stack's defaults.

    python examples/presets_customization.py --preset bias:gender --preset hallu-med
    python examples/presets_customization.py --preset refusal:all --print-commands

Credentials belong in the provider's environment (for example HF_TOKEN or
<SERVICE>_API_KEY for openai-api/<service>/<model>), never in task arguments.
To reproduce legacy results instead, run each task with its benchmark's
parity.yaml (see INSPECT_MIGRATION.md).
"""

import argparse
import shlex

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig, get_model

from llm_behavior_eval import _registry, get_dataset_preset

ANSWER_MODEL = "hf/meta-llama/Llama-3.1-8B-Instruct"
GRADER_MODEL = "hf/google/gemma-4-12B-it"


def preset_tasks(preset_names: list[str]) -> list[Task]:
    """One task per preset asset, created from its registered ``@task`` function.

    Calling the registered function keeps the log's task name
    ``llm_behavior_eval/<task>``, which inspect-brief's targets are keyed by.
    """
    tasks = []
    for name in preset_names:
        for mapping in get_dataset_preset(name).inspect_tasks:
            task_function = getattr(
                _registry, mapping.task_name.removeprefix("llm_behavior_eval/")
            )
            tasks.append(task_function(**dict(mapping.task_args)))
    return tasks


def preset_commands(
    preset_names: list[str], answer_model: str, grader_model: str
) -> list[str]:
    """The same runs as ``inspect eval`` commands with the same per-role settings."""
    commands = []
    for name in preset_names:
        for mapping in get_dataset_preset(name).inspect_tasks:
            commands.append(
                shlex.join(
                    [
                        "inspect",
                        "eval",
                        mapping.task_name,
                        *mapping.cli_args,
                        "--model",
                        answer_model,
                        "-M",
                        "do_sample=false",
                        "--temperature",
                        "0",
                        "--max-tokens",
                        "512",
                        "--model-role",
                        f"grader={{model: {grader_model}, temperature: 0.0, "
                        "max_tokens: 64, model_args: {do_sample: false}}",
                    ]
                )
            )
    return commands


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", action="append", dest="presets")
    parser.add_argument("--model", default=ANSWER_MODEL)
    parser.add_argument("--grader", default=GRADER_MODEL)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--log-dir", default="results/inspect")
    parser.add_argument("--print-commands", action="store_true")
    args = parser.parse_args(argv)
    presets = args.presets or ["bias:gender", "hallu-med"]

    if args.print_commands:
        for command in preset_commands(presets, args.model, args.grader):
            print(command)
        return

    # Grader role: its own model, GenerateConfig and provider arguments. The
    # eval-level settings below never reach it.
    grader = get_model(
        args.grader,
        config=GenerateConfig(temperature=0.0, max_tokens=64),
        do_sample=False,  # Inspect's HF provider samples unless told otherwise.
        memoize=False,
    )
    inspect_eval(
        preset_tasks(presets),
        # Model under test: eval-level GenerateConfig and provider arguments.
        model=args.model,
        model_args={"do_sample": False},
        temperature=0.0,
        max_tokens=512,
        model_roles={"grader": grader},
        limit=args.limit,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
