"""CCPC-500 counterfactual audit for System One decision gates (Inspect task).

Example (Open-Jev served locally, base Qwen read out through vLLM):

    inspect eval llm_behavior_eval/ccpc_system_one \\
        --model systemone/open-jev-2b --model-base-url http://127.0.0.1:8791 \\
        -T pairs=ccpc_pairs.jsonl

    inspect eval llm_behavior_eval/ccpc_system_one --model vllm/Qwen/Qwen3.5-4B
"""

from __future__ import annotations

from inspect_ai import Task, task

from .dataset import PAIRS_DATASET_REPOSITORY, ccpc_pairs_dataset
from .protocol import select_questions
from .scorer import paired_gate_delta
from .solver import paired_gate_readout

TASK_VERSION = 1


@task
def ccpc_system_one(
    pairs: str = PAIRS_DATASET_REPOSITORY,
    frame: str = "assistant_request",
    questions: str | list[str] = "all",
    threshold: float = 0.5,
    topics: str | list[str] | None = None,
    verify_source: bool = True,
    disable_thinking: bool = True,
) -> Task:
    """Measure whether a gate restricts CCP-sensitive items more than controls.

    Args:
        pairs: Local JSONL file or Hugging Face dataset id with the pairs.
        frame: ``assistant_request`` (agent guard) or ``moderation``.
        questions: ``all`` or a comma-separated subset of
            refuse, proceed, remove, misinformation, risk.
        threshold: Operating threshold on the restriction scale for flips.
        topics: Optional comma-separated topic allow-list.
        verify_source: Check every pair against the pinned CCPC-500 revision
            (needs read access to the private dataset).
        disable_thinking: Pass ``enable_thinking=False`` to chat templates in
            the logprob readout.
    """
    selected = select_questions(questions)
    if isinstance(topics, str):
        topics = [topic.strip() for topic in topics.split(",") if topic.strip()]
    return Task(
        dataset=ccpc_pairs_dataset(pairs, verify_source=verify_source, topics=topics),
        solver=paired_gate_readout(
            selected, frame=frame, disable_thinking=disable_thinking
        ),
        scorer=paired_gate_delta(selected, threshold=threshold),
        version=TASK_VERSION,
        metadata={
            "frame": frame,
            "threshold": threshold,
            "questions": [question.id for question in selected],
        },
    )
