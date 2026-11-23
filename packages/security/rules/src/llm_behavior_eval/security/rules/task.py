"""RuLES (Mu et al., "Can LLMs Follow Simple Rules?"): rule following.

One task covers every test suite and scenario, chosen with ``suites`` and
``scenarios``, so a single log holds the RuLES score with its per-suite,
harmless/helpful, scenario and attack breakdowns.
"""

from typing import cast

from inspect_ai import Task, task
from llm_behavior_eval.common.scoring import completion_scorer

from .dataset import (
    REPOSITORY,
    REPOSITORY_COMMIT,
    SCORE_SUITES,
    SUITES,
    TESTS,
    Suite,
    dataset,
    parse_names,
    parse_tests,
)
from .scorer import rules_conversation, rules_scorer

TASK_VERSION = 1
"""Bump when a change can alter results (prompts, data, checks, metrics)."""
DEFAULT_SUITES = ",".join(SCORE_SUITES)
"""``benign,basic,redteam``: the suites the RuLES score averages."""
DEFAULT_SHUFFLE_SEED = 2023


@task
def rules(
    suites: str | list[str] = DEFAULT_SUITES,
    scenarios: str | list[str] | None = None,
    system_instructions: bool = False,
    remove_precedence_reminders: bool = False,
    dataset_path: str | None = None,
    shuffle_seed: int | None = DEFAULT_SHUFFLE_SEED,
) -> Task:
    """RuLES pass rates and score on the pinned test suites.

    Args:
        suites: Test suites, comma-separated or a list, among ``benign``,
            ``basic``, ``redteam`` and ``direct_request``. Default: the three
            the RuLES score averages.
        scenarios: Test files (``Authentication_provide``) or whole scenarios
            (``Authentication``), comma-separated or a list. Default: all 19
            test files of the 14 scenarios.
        system_instructions: Present the rules as a system message (the
            reference's ``--system_instructions``). Default: as a user message
            answered by a fixed acknowledgement, the reference's default.
        remove_precedence_reminders: Remove the rules' precedence sentences
            (the reference's ``--remove_precedence_reminders``).
        dataset_path: A local copy of the reference's ``llm_rules/data``
            folder to read every file from; each file must match its pinned
            digest.
        shuffle_seed: Shuffle the samples so ``--limit N`` takes a spread of
            suites and scenarios; ``None`` keeps the reference order. A full
            run's metrics do not depend on it.

    Returns:
        A task with no inference defaults and no grader.

    Raises:
        ValueError: An argument is invalid or a source fails validation.
    """
    chosen_suites = cast("list[Suite]", parse_names(suites, SUITES, "suites"))
    chosen_tests = list(TESTS) if scenarios is None else parse_tests(scenarios)
    return Task(
        dataset=dataset(
            chosen_suites,
            chosen_tests,
            dataset_path=dataset_path,
            system_instructions=system_instructions,
            remove_reminders=remove_precedence_reminders,
            shuffle_seed=shuffle_seed,
        ),
        solver=rules_conversation(),
        scorer=[rules_scorer(), completion_scorer()],
        version=TASK_VERSION,
        metadata={
            "dataset_repository": REPOSITORY,
            "dataset_commit": REPOSITORY_COMMIT,
            "suites": chosen_suites,
            "tests": chosen_tests,
            "system_instructions": system_instructions,
            "remove_precedence_reminders": remove_precedence_reminders,
        },
    )
