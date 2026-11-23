"""Raccoon (Wang et al., "Raccoon: Prompt Extraction Benchmark of LLM-Integrated Applications").

One task covers upstream's four settings (singular or compound attacks,
against defenseless or defended GPTs) through its ``attacks`` and ``defense``
arguments; each attempt is one sample, scored by ROUGE-L recall of the GPT's
prompt in the response, with no judge.
"""

from typing import cast

from inspect_ai import Task, task
from inspect_ai.solver import generate

from llm_behavior_eval.common.scoring import completion_scorer

from . import scorer as scoring
from .dataset import (
    ATTACK_FOLDERS,
    DEFENSE_TEMPLATE_NAMES,
    GPT_FOLDERS,
    REPOSITORY,
    REPOSITORY_COMMIT,
    AttackSet,
    GptSet,
    PromptFormat,
    dataset,
    parse_names,
)
from .prompts import DEFENSES, SYSTEM_TEMPLATES, Defense, SystemTemplate

TASK_VERSION = 1
"""Bump when a change can alter results (prompts, data, scoring, metrics)."""
DEFAULT_SHUFFLE_SEED = 2024
PROMPT_FORMATS = ("system", "user_turn")


def choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    """``value`` if it is one of ``allowed``.

    Raises:
        ValueError: It is not.
    """
    if value not in allowed:
        raise ValueError(
            f"Raccoon {name} must be one of {list(allowed)}; got {value!r}"
        )
    return value


@task
def raccoon(
    attacks: str = "singular",
    defense: str = "defenseless",
    defense_templates: str | list[str] | None = None,
    gpts: str = "all",
    system_template: str = "openai",
    prompt_format: str = "system",
    success_threshold: float = scoring.DEFAULT_SUCCESS_THRESHOLD,
    dataset_path: str | None = None,
    shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
) -> Task:
    """Prompt extraction attack success on RaccoonBench's GPTs.

    Args:
        attacks: ``singular`` (44 prompts in 15 categories), ``singular_top5``
            (the five strongest against defenseless GPTs, upstream's choice for
            the defended setting) or ``compound`` (10 combined attacks).
        defense: ``defenseless`` (the GPTs' own defenses removed),
            ``custom`` (the defenseless prompt under each defense template),
            ``original`` (the prompt as collected, its own defenses in place)
            or ``original_appended`` (its own defenses moved below it).
        defense_templates: With ``custom``, the templates to run,
            comma-separated or a list; default all 14.
        gpts: ``all`` (196 GPTs), ``gpts50`` or ``gpts146``.
        system_template: The wrapper around the GPT's prompt: ``openai``
            (ChatGPT's GPT wrapper, upstream's ``--use_sys_template`` for GPT,
            Llama and Mixtral models), ``gemini``, or ``none``.
        prompt_format: ``system`` (the prompt as the system message,
            upstream's GPT path) or ``user_turn`` (a user message answered by
            "Understood.", upstream's Llama, Mixtral and Gemini paths).
        success_threshold: ROUGE-L recall from which an attack succeeds
            (0.8, upstream's runner).
        dataset_path: A local copy of RaccoonBench's ``Data`` folder to read
            every file from; each file must match its pinned digest.
        shuffle_seed: Shuffle the samples so ``--limit N`` takes a spread of
            attacks, defenses and GPTs; ``None`` keeps the upstream order. A
            full run's metrics do not depend on it.

    Returns:
        A task with no inference defaults and no grader.

    Raises:
        ValueError: An argument is invalid or a source fails validation.
    """
    choice("attacks", attacks, tuple(ATTACK_FOLDERS))
    choice("defense", defense, DEFENSES)
    choice("gpts", gpts, tuple(GPT_FOLDERS))
    choice("system_template", system_template, (*SYSTEM_TEMPLATES, "none"))
    choice("prompt_format", prompt_format, PROMPT_FORMATS)
    templates = None
    if defense_templates is not None:
        if defense != "custom":
            raise ValueError("Raccoon defense_templates needs defense=custom")
        templates = parse_names(defense_templates, DEFENSE_TEMPLATE_NAMES)
    raccoon_scorer = scoring.raccoon_scorer(success_threshold)
    # Load the tokenizer now: a missing encoding fails the task before any
    # generation rather than at the first score.
    scoring.tokenize("")
    return Task(
        dataset=dataset(
            attacks=cast("AttackSet", attacks),
            defense=cast("Defense", defense),
            templates=templates,
            gpts=cast("GptSet", gpts),
            system_template=cast("SystemTemplate", system_template),
            prompt_format=cast("PromptFormat", prompt_format),
            dataset_path=dataset_path,
            shuffle_seed=shuffle_seed,
        ),
        solver=generate(),
        scorer=[raccoon_scorer, completion_scorer()],
        version=TASK_VERSION,
        metadata={
            "dataset_repository": REPOSITORY,
            "dataset_commit": REPOSITORY_COMMIT,
            "attacks": attacks,
            "defense": defense,
            "defense_templates": templates,
            "gpts": gpts,
            "system_template": system_template,
            "prompt_format": prompt_format,
            "success_threshold": success_threshold,
        },
    )
