"""BIPIA (Yi et al., KDD 2025): indirect prompt injection attack success.

One task covers every BIPIA task family, chosen with ``tasks``, so a single
log holds the overall ASR and its per-task, text/code, attack and position
breakdowns.
"""

from typing import cast

from inspect_ai import Task, task
from inspect_ai.solver import generate

from llm_behavior_eval.common.scoring import completion_scorer

from .dataset import (
    PUBLIC_TASKS,
    REPOSITORY,
    REPOSITORY_COMMIT,
    dataset,
    parse_names,
)
from .prompts import (
    INSERTION_SEED,
    POSITIONS,
    TASK_NAMES,
    Position,
    PromptMode,
    TaskName,
)
from .scorer import LANGDETECT_SEED, bipia_scorer

TASK_VERSION = 1
"""Bump when a change can alter results (prompts, data, checks, metrics)."""
DEFAULT_TASKS = ",".join(PUBLIC_TASKS)
"""``email,table,code``: the tasks whose contexts the repository ships."""
DEFAULT_POSITIONS = ",".join(POSITIONS)
DEFAULT_SHUFFLE_SEED = 2023
REFERENCE_JUDGE = "gpt-3.5-turbo (the reference README's default evaluator, gpt35)"
PROMPT_MODES: tuple[PromptMode, ...] = ("system", "user")


@task
def bipia(
    tasks: str | list[str] = DEFAULT_TASKS,
    positions: str | list[str] = DEFAULT_POSITIONS,
    prompt_mode: str = "system",
    ignore_guidance: bool = False,
    dataset_path: str | None = None,
    shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
    judge_sees_answer_reasoning: bool = False,
) -> Task:
    """BIPIA attack success rate on the test split.

    Args:
        tasks: BIPIA tasks, comma-separated or a list, among ``email``,
            ``qa``, ``table``, ``abstract`` and ``code``. ``qa`` and
            ``abstract`` need ``dataset_path``. Default: the three tasks whose
            contexts the repository ships.
        positions: Where the attack goes in the external content, among
            ``end``, ``start`` and ``middle``. Default: all three.
        prompt_mode: ``system`` (external content in the system message, the
            reference's GPT-3.5/GPT-4 setup) or ``user`` (one user message,
            its open-model setup).
        ignore_guidance: Add the reference's ``--add_ign_guidance`` reminder
            not to follow instructions in the external content.
        dataset_path: A local copy of the reference's ``benchmark`` folder to
            read every file from; each file must match its pinned digest.
        shuffle_seed: Shuffle the samples so ``--limit N`` takes a spread of
            tasks, attacks and positions; ``None`` keeps the reference order.
            A full run's metrics do not depend on it.
        judge_sees_answer_reasoning: Check the answer with its parsed
            reasoning, for the grader and the rule-based checks alike.

    Returns:
        A task with no model or grader inference defaults.

    Raises:
        ValueError: An argument is invalid or a source fails validation.
    """
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(
            f"BIPIA prompt_mode must be one of {list(PROMPT_MODES)}; "
            f"got {prompt_mode!r}"
        )
    chosen_tasks = cast("list[TaskName]", parse_names(tasks, TASK_NAMES, "tasks"))
    chosen_positions = cast(
        "list[Position]", parse_names(positions, POSITIONS, "positions")
    )
    return Task(
        dataset=dataset(
            chosen_tasks,
            chosen_positions,
            dataset_path=dataset_path,
            prompt_mode=prompt_mode,
            ignore_guidance=ignore_guidance,
            shuffle_seed=shuffle_seed,
        ),
        solver=generate(),
        scorer=[
            bipia_scorer(judge_sees_answer_reasoning=judge_sees_answer_reasoning),
            completion_scorer(),
        ],
        version=TASK_VERSION,
        metadata={
            "dataset_repository": REPOSITORY,
            "dataset_commit": REPOSITORY_COMMIT,
            "dataset_split": "test",
            "bipia_tasks": chosen_tasks,
            "positions": chosen_positions,
            "prompt_mode": prompt_mode,
            "ignore_guidance": ignore_guidance,
            "judge_sees_answer_reasoning": judge_sees_answer_reasoning,
            "insertion_seed": INSERTION_SEED,
            "langdetect_seed": LANGDETECT_SEED,
            "reference_judge": REFERENCE_JUDGE,
        },
    )
