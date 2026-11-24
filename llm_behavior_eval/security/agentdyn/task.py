"""AgentDyn (Li et al., "AgentDyn: Are Your Agent Security Defenses Deployable
in Real-World Dynamic Environments?"): prompt injection against tool-using agents.

One task covers AgentDyn's three suites (Shopping, GitHub, Daily Life), with
and without injections, chosen with ``suites`` and ``mode``, so a single log
holds utility, utility under attack and the attack success rate.
"""

from inspect_ai import Task, task

from llm_behavior_eval.common.scoring import completion_scorer

from .constants import (
    AGENTDYN_COMMIT,
    AGENTDYN_REPOSITORY,
    BENCHMARK_VERSION,
    DEFAULT_ATTACK,
    DEFAULT_MODEL_NAME,
    SUITES,
)
from .scorer import agentdyn_scorer

TASK_VERSION = 1
"""Bump when a change can alter results (episodes, agent loop, checks, metrics)."""
DEFAULT_SUITES = ",".join(SUITES)
"""``shopping,github,dailylife``: every AgentDyn suite."""
DEFAULT_MAX_ITERATIONS = 15
"""Upstream's ``ToolsExecutionLoop(max_iters=15)``."""


@task
def agentdyn(
    suites: str | list[str] = DEFAULT_SUITES,
    mode: str = "both",
    attack: str = DEFAULT_ATTACK,
    model_name: str = DEFAULT_MODEL_NAME,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Task:
    """AgentDyn utility and attack success on the pinned suites.

    Args:
        suites: Suites, comma-separated or a list, among ``shopping``,
            ``github`` and ``dailylife``. Default: all three.
        mode: ``clean`` (each user task once, no injection: 60 episodes),
            ``attack`` (each user task under each injection task: 560) or
            ``both`` (620).
        attack: Upstream's attack (``--attack``) that fills the injection
            vectors: ``important_instructions`` (default),
            ``important_instructions_no_user_name``,
            ``important_instructions_no_model_name``,
            ``important_instructions_no_names``,
            ``important_instructions_wrong_user_name``, ``direct``, ``ignore_previous``, ``system_message`` or
            ``injecagent``.
        model_name: How the ``important_instructions`` family addresses the
            model under test; upstream looks it
            up from its own model registry. Default: upstream's generic
            ``the AI language model``.
        max_iterations: Tool-execution rounds after the first answer
            (upstream's ``max_iters``).

    Returns:
        A task with no inference defaults and no grader.

    Raises:
        ValueError: An argument is invalid.
        InstalledRevisionError: The installed ``agentdojo`` is not the pinned
            AgentDyn fork.
    """
    # agentdojo is slow to import: only when the task is built, not when
    # Inspect loads this entry point to resolve another benchmark's task.
    from .agent import agentdyn_agent
    from .dataset import dataset, parse_names

    chosen = parse_names(suites, SUITES, "suites")
    return Task(
        dataset=dataset(chosen, mode=mode, attack=attack, model_name=model_name),
        solver=agentdyn_agent(max_iterations),
        scorer=[agentdyn_scorer(), completion_scorer()],
        version=TASK_VERSION,
        metadata={
            "agentdyn_repository": AGENTDYN_REPOSITORY,
            "agentdyn_commit": AGENTDYN_COMMIT,
            "benchmark_version": BENCHMARK_VERSION,
            "suites": chosen,
            "mode": mode,
            "attack": attack if mode != "clean" else None,
            "model_name": model_name,
            "max_iterations": max_iterations,
        },
    )
