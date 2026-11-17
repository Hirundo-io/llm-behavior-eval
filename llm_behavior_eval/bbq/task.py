"""BBQ task composed from Inspect dataset, solver and scorer APIs."""

from inspect_ai import Task, task
from inspect_ai.solver import generate, prompt_template, system_message

from llm_behavior_eval.bbq.dataset import dataset
from llm_behavior_eval.bbq.prompts import SYSTEM_MESSAGE
from llm_behavior_eval.bbq.scorer import bbq_scorer
from llm_behavior_eval.common.scoring import completion_scorer

TASK_VERSION = 1
"""Bump when a change can alter BBQ results (prompts, parsing, data, metrics).

Version 1 is the native Inspect BBQ; its results are not comparable with the
legacy engine's (see the README).
"""


@task
def bbq(
    bias_type: str = "gender",
    kind: str = "bias",
    dataset_path: str | None = None,
    revision: str | None = None,
    split: str = "train",
    seed: int = 42,
    judge_sees_answer_reasoning: bool = False,
) -> Task:
    """Evaluate one BBQ variant with a required independently configured grader.

    Args:
        bias_type: Gender, race, nationality, physical, age or religion.
        kind: Bias (ambiguous) or unbias (disambiguated).
        dataset_path: Override the Hub source with any ``datasets.load_dataset``
            source, such as a local folder of JSONL or Parquet files or a Hub
            snapshot, or with a ``save_to_disk`` folder.
        revision: Dataset revision. Defaults to the pinned commit in
            ``DATASET_REVISIONS`` for the Hub dataset, and to none (the
            source as it is) with ``dataset_path``.
        split: Dataset split.
        seed: Dataset shuffle seed, not a model generation setting.
        judge_sees_answer_reasoning: Give the judge the answer's parsed
            reasoning too, as the legacy engine did by default.

    Returns:
        A task with no model or grader inference defaults.

    Raises:
        ValueError: Unknown ``bias_type`` or ``kind``.
    """
    return Task(
        dataset=dataset(
            bias_type,
            kind,
            dataset_path=dataset_path,
            revision=revision,
            split=split,
            seed=seed,
        ),
        solver=[
            system_message(SYSTEM_MESSAGE),
            prompt_template("{prompt}\n"),
            generate(),
        ],
        scorer=[
            bbq_scorer(
                kind=kind, judge_sees_answer_reasoning=judge_sees_answer_reasoning
            ),
            completion_scorer(),
        ],
        version=TASK_VERSION,
    )
