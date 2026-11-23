"""SEP tasks: instruction-data separation (Zverev et al., ICLR 2025).

``sep`` is the paper's paired evaluation. ``sep_tool`` and
``sep_instruction_control`` are a newer adaptation (the probe in a tool
result, judged by the grader, and its instruction-side control); they are not
canonical SEP (see the README).
"""

from functools import partial

from inspect_ai import Task, task
from inspect_ai.solver import generate

from llm_behavior_eval.common.scoring import completion_scorer

from .dataset import (
    DATASET_COMMIT,
    DATASET_FILE,
    DATASET_LICENSE,
    DATASET_REPOSITORY,
    DATASET_SHA256,
    EXPECTED_ROWS,
    PROBES_FILE,
    PROBES_SHA256,
    dataset,
    variant_dataset,
)
from .prompts import messages, tool_messages
from .scorer import sep_instruction_control_scorer, sep_scorer, sep_tool_scorer
from .solver import sep_solver

TASK_VERSION = 1
"""``sep``: bump when a change can alter results (prompts, matching, data, metrics)."""
TOOL_TASK_VERSION = 1
"""``sep_tool``: bump when a change can alter results (also the judge prompt)."""
CONTROL_TASK_VERSION = 1
"""``sep_instruction_control``: bump when a change can alter results."""

DATASET_METADATA = {
    "dataset_repository": DATASET_REPOSITORY,
    "dataset_commit": DATASET_COMMIT,
    "dataset_file": DATASET_FILE,
    "dataset_sha256": DATASET_SHA256,
    "dataset_rows": EXPECTED_ROWS,
    "dataset_license": DATASET_LICENSE,
}
ADAPTATION_METADATA = {
    **DATASET_METADATA,
    "probes_file": PROBES_FILE,
    "probes_sha256": PROBES_SHA256,
    "adaptation": "not canonical SEP; prompts from Hirundo's earlier Inspect port",
}


@task
def sep(dataset_path: str | None = None, system_role: bool = True) -> Task:
    """SEP separation and utility scores, scored by the witness rule (no grader).

    Each item is generated twice, with the probe in the instruction (system)
    argument and in the data (user) argument.

    Args:
        dataset_path: Read this local copy of the authors'
            ``datasets/SEP_dataset.json`` instead of the pinned download; it
            must have the pinned SHA-256.
        system_role: Instruction argument as the system message, data
            argument as the user message (the paper's default). ``False``
            sends one user message ``System prompt: ... User prompt: ...``, as
            the paper did for models without a system role (Gemma, Starling).

    Returns:
        A task with no inference settings.
    """
    return Task(
        dataset=dataset(dataset_path),
        solver=sep_solver(system_role=system_role),
        scorer=[sep_scorer(), completion_scorer()],
        version=TASK_VERSION,
        metadata={**DATASET_METADATA, "prompt_template": "naive (template 0)"},
    )


@task
def sep_tool(
    dataset_path: str | None = None,
    probes_path: str | None = None,
    judge_sees_answer_reasoning: bool = False,
) -> Task:
    """Probe-execution rate with the probe in a tool result, judged by the grader.

    An adaptation, not canonical SEP: the clean task prompt is the system
    message, the user asks to process a document, and the data prompt with
    the probe arrives as a prefilled ``fetch_document`` tool result. The
    required ``grader`` role decides whether the reply executed the probe.

    Args:
        dataset_path: A local copy of ``datasets/SEP_dataset.json`` (checked
            against the pinned SHA-256).
        probes_path: A local copy of ``SEP_dataset/source/probes.json``
            (checked against the pinned SHA-256).
        judge_sees_answer_reasoning: Show the grader the answer with its
            parsed reasoning.

    Returns:
        A task with no model or grader inference settings.
    """
    return Task(
        dataset=variant_dataset("tool", tool_messages, dataset_path, probes_path),
        solver=generate(),
        scorer=[
            sep_tool_scorer(judge_sees_answer_reasoning=judge_sees_answer_reasoning),
            completion_scorer(),
        ],
        version=TOOL_TASK_VERSION,
        metadata=ADAPTATION_METADATA,
    )


@task
def sep_instruction_control(
    dataset_path: str | None = None,
    probes_path: str | None = None,
    system_role: bool = True,
) -> Task:
    """Witness rate with the probe in the instruction: ``sep_tool``'s control.

    The same rows and probe checks as ``sep_tool``, with ``sep``'s
    probe-in-instruction conversation (task prompt with the probe as the
    system message, clean data as the user message) generated once.

    Args:
        dataset_path: A local copy of ``datasets/SEP_dataset.json`` (checked
            against the pinned SHA-256).
        probes_path: A local copy of ``SEP_dataset/source/probes.json``
            (checked against the pinned SHA-256).
        system_role: As in ``sep``: ``False`` sends one user message
            ``System prompt: ... User prompt: ...``.

    Returns:
        A task with no inference settings and no grader.
    """
    build = partial(messages, position="instruction", system_role=system_role)
    return Task(
        dataset=variant_dataset(
            "instruction_control", build, dataset_path, probes_path
        ),
        solver=generate(),
        scorer=[sep_instruction_control_scorer(), completion_scorer()],
        version=CONTROL_TASK_VERSION,
        metadata=ADAPTATION_METADATA,
    )
