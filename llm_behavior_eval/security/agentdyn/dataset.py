"""AgentDyn episodes: the pinned suites, the attacks, and one sample per episode.

The suites, their tools, user and injection tasks, injection vectors, default
system prompt and attack templates are upstream's own objects, imported from
the ``agentdojo`` distribution that SaFo-Lab/AgentDyn publishes (installed at
``AGENTDYN_COMMIT``). Nothing here copies their data.
"""

import json
from importlib.metadata import PackageNotFoundError, distribution

from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.attacks import BaseAttack, FixedJailbreakAttack
from agentdojo.attacks.attack_registry import ATTACKS
from agentdojo.models import MODEL_NAMES
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.task_suite.task_suite import TaskSuite
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from .constants import (
    AGENTDYN_COMMIT,
    AGENTDYN_REPOSITORY,
    BENCHMARK_VERSION,
    MODEL_NAME_ATTACKS,
    MODES,
    SUITES,
    SUPPORTED_ATTACKS,
)


class InstalledRevisionError(RuntimeError):
    """The installed ``agentdojo`` is not the pinned AgentDyn fork."""


def installed_commit() -> str | None:
    """The commit the installed ``agentdojo`` was built from, when pip recorded one.

    Returns:
        ``vcs_info.commit_id`` of its ``direct_url.json`` (a Git install), or
        ``None`` (an install from an index, a local directory or an archive).
    """
    try:
        text = distribution("agentdojo").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


def check_installed_suites() -> dict[str, TaskSuite]:
    """The AgentDyn suites of the installed ``agentdojo``.

    Raises:
        InstalledRevisionError: ``agentdojo`` was installed from Git at another
            commit, or it lacks AgentDyn's suites (stock AgentDojo).
    """
    commit = installed_commit()
    if commit is not None and commit != AGENTDYN_COMMIT:
        raise InstalledRevisionError(
            f"agentdojo is installed from commit {commit}; this task needs "
            f"{AGENTDYN_REPOSITORY} at {AGENTDYN_COMMIT}. Reinstall "
            "llm-behavior-eval-agentdyn in an environment of its own."
        )
    suites = get_suites(BENCHMARK_VERSION)
    missing = [name for name in SUITES if name not in suites]
    if missing:
        raise InstalledRevisionError(
            f"The installed agentdojo has no {', '.join(missing)} suite: it is "
            f"stock AgentDojo, not {AGENTDYN_REPOSITORY} at {AGENTDYN_COMMIT}."
        )
    return {name: suites[name] for name in SUITES}


def get_suite(name: str) -> TaskSuite:
    """One AgentDyn suite, at ``BENCHMARK_VERSION``."""
    return check_installed_suites()[name]


def parse_names(
    value: str | list[str], allowed: tuple[str, ...], argument: str
) -> list[str]:
    """Names from a comma-separated string or a list, deduplicated, in order.

    Raises:
        ValueError: A name is not in ``allowed``, or none is given.
    """
    names = value.split(",") if isinstance(value, str) else list(value)
    names = list(dict.fromkeys(name.strip() for name in names if name.strip()))
    unknown = [name for name in names if name not in allowed]
    if unknown or not names:
        raise ValueError(
            f"{argument} must name one or more of {', '.join(allowed)}; got {value!r}"
        )
    return names


def attacked_pipeline() -> BasePipelineElement:
    """A stand-in for the attacked pipeline: attacks only read its ``name``.

    Upstream's model-addressing attacks look the model's prose name up from
    the pipeline name in ``agentdojo.models.MODEL_NAMES`` while they are built;
    any registered name lets them build, and ``build_attack`` then sets the
    name the task was given.
    """
    pipeline = GroundTruthPipeline(None)
    pipeline.name = next(iter(MODEL_NAMES))
    return pipeline


def build_attack(name: str, suite: TaskSuite, model_name: str) -> BaseAttack:
    """Upstream's attack ``name`` on ``suite``.

    Args:
        name: One of ``SUPPORTED_ATTACKS``.
        suite: The suite whose injection vectors it fills.
        model_name: How a ``MODEL_NAME_ATTACKS`` template addresses the model
            (upstream derives it from its pipeline name).

    Raises:
        ValueError: ``name`` is not supported.
    """
    if name not in SUPPORTED_ATTACKS:
        raise ValueError(
            f"attack must be one of {', '.join(SUPPORTED_ATTACKS)}; got {name!r}"
        )
    attack = ATTACKS[name](suite, attacked_pipeline())
    if name in MODEL_NAME_ATTACKS and isinstance(attack, FixedJailbreakAttack):
        attack.model_name = model_name
    return attack


def system_prompt() -> str:
    """Upstream's default system message (``data/system_messages.yaml``)."""
    return load_system_message(None)


def sample_id(suite: str, user_task: str, injection_task: str | None) -> str:
    """``<suite>/<user task>/<injection task>``, or ``.../clean`` without an attack."""
    return f"{suite}/{user_task}/{injection_task or 'clean'}"


def suite_samples(
    suite_name: str,
    *,
    clean: bool,
    attack: str | None,
    model_name: str,
    system: str,
) -> list[Sample]:
    """One suite's episodes: each user task clean, then under each injection task.

    The order is upstream's: user tasks in registration order, and for each
    one every injection task, the injections computed by the attack exactly
    as ``benchmark_suite_with_injections`` does.
    """
    suite = get_suite(suite_name)
    attacker = build_attack(attack, suite, model_name) if attack is not None else None
    samples = []
    for user_task in suite.user_tasks.values():
        runs: list[str | None] = [None] if clean else []
        if attacker is not None:
            runs.extend(suite.injection_tasks)
        for injection_id in runs:
            injections: dict[str, str] = {}
            if attacker is not None and injection_id is not None:
                injections = attacker.attack(
                    user_task, suite.get_injection_task_by_id(injection_id)
                )
            samples.append(
                Sample(
                    id=sample_id(suite_name, user_task.ID, injection_id),
                    input=[
                        ChatMessageSystem(content=system),
                        ChatMessageUser(content=user_task.PROMPT),
                    ],
                    metadata={
                        "suite": suite_name,
                        "user_task": user_task.ID,
                        "injection_task": injection_id,
                        "attack": None if injection_id is None else attack,
                        "injections": injections,
                    },
                )
            )
    return samples


def dataset(
    suites: list[str], *, mode: str, attack: str, model_name: str
) -> MemoryDataset:
    """Every episode of ``suites`` for ``mode`` (``clean``, ``attack`` or ``both``)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}; got {mode!r}")
    if attack not in SUPPORTED_ATTACKS:
        raise ValueError(
            f"attack must be one of {', '.join(SUPPORTED_ATTACKS)}; got {attack!r}"
        )
    system = system_prompt()
    samples = [
        sample
        for suite in suites
        for sample in suite_samples(
            suite,
            clean=mode in ("clean", "both"),
            attack=attack if mode in ("attack", "both") else None,
            model_name=model_name,
            system=system,
        )
    ]
    return MemoryDataset(samples, name="agentdyn")
