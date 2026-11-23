"""LLMail-Inject tasks: replayed challenge attacks, or prepared records, against the model alone."""

from inspect_ai import Task, task
from inspect_ai.solver import generate, use_tools
from llm_behavior_eval.common.scoring import completion_scorer
from llm_behavior_eval.security.llmail_inject.dataset import (
    ANALYSIS_COMMIT,
    ANALYSIS_REPOSITORY,
    ATTACK_FILES,
    CHALLENGE_COMMIT,
    CHALLENGE_REPOSITORY,
    DEFAULT_ATTACKS_PER_PHASE,
    EMAIL_FILES,
    EXPECTED_ATTACKS,
    dataset,
)
from llm_behavior_eval.security.llmail_inject.prepared import prepared_dataset
from llm_behavior_eval.security.llmail_inject.prompts import TOOL_NAME
from llm_behavior_eval.security.llmail_inject.scenario import PromptDefense, send_email
from llm_behavior_eval.security.llmail_inject.scorer import (
    llmail_inject_prepared_scorer,
    llmail_inject_scorer,
)

TASK_VERSION = 1
"""Bump when a change can alter results (prompts, data, sampling, scoring)."""
PREPARED_TASK_VERSION = 1
"""``llmail_inject_prepared``'s version, bumped on the same rule."""


@task
def llmail_inject(
    levels: int | list[int] | None = None,
    attacks_per_phase: int | None = DEFAULT_ATTACKS_PER_PHASE,
    prompt_defense: PromptDefense = "none",
    dataset_path: str | None = None,
) -> Task:
    """LLMail-Inject attack success rate of the model under test, no grader.

    Args:
        levels: Challenge levels to replay every attack at: 1 (two emails,
            attacker last), 2 (ten emails, attacker fifth), 3 or 4 (ten
            emails ranked by TF-IDF similarity to the level's query, attacker
            assumed retrieved; needs the ``tfidf`` extra). Default: 1 and 2.
        attacks_per_phase: Attacks sampled from each challenge phase (the
            smallest SHA-256 of the attack text first); ``None`` replays all
            25,323 + 3,688.
        prompt_defense: ``none`` (default, the model alone), ``spotlight``
            (the phase 1 spotlighting sub-level) or ``delimiters`` (the phase 2
            instructions and delimiters).
        dataset_path: Folder holding the pinned files under their published
            names, read instead of the pinned GitHub URLs.

    Returns:
        A task with no inference settings; tool calls are recorded, never run.
    """
    return Task(
        dataset=dataset(levels, attacks_per_phase, prompt_defense, dataset_path),
        solver=[use_tools(send_email()), generate(tool_calls="none")],
        scorer=[llmail_inject_scorer(), completion_scorer()],
        version=TASK_VERSION,
        metadata={
            "attack_repository": ANALYSIS_REPOSITORY,
            "attack_commit": ANALYSIS_COMMIT,
            "attack_sha256": {
                str(phase): source.sha256 for phase, source in ATTACK_FILES.items()
            },
            "attack_pool_sizes": {
                str(phase): count for phase, count in EXPECTED_ATTACKS.items()
            },
            "challenge_repository": CHALLENGE_REPOSITORY,
            "challenge_commit": CHALLENGE_COMMIT,
            "email_sha256": {
                str(level): source.sha256 for level, source in EMAIL_FILES.items()
            },
            "tool_name": TOOL_NAME,
        },
    )


@task
def llmail_inject_prepared(dataset_path: str) -> Task:
    """LLMail-Inject attack success rate on prepared email contexts, no grader.

    Args:
        dataset_path: A ``.json`` array or ``.jsonl`` file of records with
            non-empty string fields ``id``, ``system``, ``query``,
            ``retrieved`` (the whole email search result, attack included),
            ``target_to`` and ``target_body``.

    Returns:
        A task with the conversation, tool and scoring of ``llmail_inject``
        and no inference settings; tool calls are recorded, never run.
    """
    return Task(
        dataset=prepared_dataset(dataset_path),
        solver=[use_tools(send_email()), generate(tool_calls="none")],
        scorer=[llmail_inject_prepared_scorer(), completion_scorer()],
        version=PREPARED_TASK_VERSION,
        metadata={"tool_name": TOOL_NAME},
    )
