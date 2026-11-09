"""Native preset expansion. Credentials belong in the provider environment."""

from importlib import import_module

from inspect_ai import eval as inspect_eval
from inspect_ai import task_with
from inspect_ai.model import get_model

from llm_behavior_eval import list_dataset_presets


def main() -> None:
    catalog = {preset.name: preset for preset in list_dataset_presets()}
    tasks = []
    for name in ("bias:gender", "hallu-med"):
        for mapping in catalog[name].inspect_tasks:
            task_name = mapping.task_name.removeprefix("llm_behavior_eval/")
            constructor = getattr(
                import_module(f"llm_behavior_eval.{task_name}.task"), task_name
            )
            tasks.append(
                task_with(
                    constructor(**dict(mapping.task_args), max_samples=200),
                    name=mapping.task_name,
                )
            )
    # Configure HF thinking independently for answer and grader.
    # For API deployments see the per-family attestation flags in INSPECT_MIGRATION.md.
    inspect_eval(
        tasks,
        model=get_model(
            "hf/meta-llama/Llama-3.1-8B-Instruct",
            enable_thinking=False,
            do_sample=False,
        ),
        model_roles={
            "grader": get_model(
                "hf/google/gemma-4-12B-it", enable_thinking=False, do_sample=False
            )
        },
        log_dir="results/inspect",
    )


if __name__ == "__main__":
    main()
