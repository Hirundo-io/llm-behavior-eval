"""Credentials outside task arguments and logged model arguments.

Inspect records every task argument in the ``.eval`` log, so no benchmark takes
a credential:

* Dataset access reads ``DATASET_CREDENTIAL_VAR``, then ``HF_TOKEN``
  (``resolve_dataset_token``), so datasets can use a credential separate from
  the models' (the legacy ``--dataset-token``).
* The model under test and the grader each take their own credential as an
  explicit model object: ``get_model(..., api_key=...)``. Inspect never logs
  ``api_key``, and every model object keeps its own key, so two models whose
  providers read the same variable (two ``hf/`` models, two accounts) stay
  separate. Without ``api_key``, a provider reads its own environment
  variables (``HF_TOKEN``, ``OPENAI_API_KEY``, ``<SERVICE>_API_KEY``, ...).

``credentials`` sets the dataset variable for programmatic callers, from an
ordinary Python argument, for the duration of a ``with`` block. Environment
variables are process-global; a provider or subprocess that reads a credential
only from the environment needs process-level setup instead (an exported
variable, or one process per credential set).
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

DATASET_CREDENTIAL_VAR = "LLM_BEHAVIOR_EVAL_DATASET_TOKEN"
"""Hugging Face token for dataset access only; takes precedence over ``HF_TOKEN``."""

HF_CREDENTIAL_VAR = "HF_TOKEN"
"""Hugging Face's own token variable, the dataset fallback."""


def _clean(value: str | None) -> str | None:
    """``value`` without line breaks or surrounding space; ``None`` if empty.

    Matches how ``huggingface_hub`` reads ``HF_TOKEN``.
    """
    if value is None:
        return None
    return value.replace("\r", "").replace("\n", "").strip() or None


def resolve_dataset_token() -> str | None:
    """The token for loading Hugging Face Hub datasets, from the environment.

    Returns:
        ``DATASET_CREDENTIAL_VAR`` when set and non-empty, otherwise ``HF_TOKEN``
        when set and non-empty, otherwise ``None``, which lets
        ``huggingface_hub`` use its remaining sources (such as a cached
        ``hf auth login``). Credentials given only to a model (an ``api_key``
        model argument, ``<SERVICE>_API_KEY``) are never used.
    """
    return _clean(os.environ.get(DATASET_CREDENTIAL_VAR)) or _clean(
        os.environ.get(HF_CREDENTIAL_VAR)
    )


def _validated_dataset_token(dataset_token: str) -> str:
    """Reject a value no environment variable can hold; messages omit it.

    Raises:
        TypeError: ``dataset_token`` is not a string.
        ValueError: It is empty, only whitespace, or contains a NUL character.
    """
    if not isinstance(dataset_token, str):
        raise TypeError(
            f"the dataset credential {DATASET_CREDENTIAL_VAR} must be a string"
        )
    if not dataset_token.strip() or "\0" in dataset_token:
        raise ValueError(
            f"the dataset credential {DATASET_CREDENTIAL_VAR} is empty or contains"
            " a NUL character; omit it to leave the variable unchanged"
        )
    return dataset_token


@contextmanager
def credentials(*, dataset_token: str | None = None) -> Generator[None]:
    """Set the dataset credential for the duration of a ``with`` block.

    The token stays an ordinary Python argument: it is never a task argument,
    so it never reaches the ``.eval`` log. Keep the block open while the task
    is built (``inspect_ai.eval`` builds it), because that is when datasets
    load. Models take their credentials as ``get_model(..., api_key=...)``,
    not from this context.

    On exit, normal or by exception, ``DATASET_CREDENTIAL_VAR`` is restored to
    its previous value, or removed if it was unset before; a change made inside
    the block is discarded. Other variables are untouched. Contexts nest; the
    innermost value wins while it is open.

    The variable is process-global: every thread and async task in the
    process sees it. Do not overlap contexts that set different tokens from
    several threads or concurrent evaluations in one process; run those in
    separate processes.

    Args:
        dataset_token: Hugging Face token for dataset access only, set as
            ``DATASET_CREDENTIAL_VAR``. ``None`` leaves the variable unchanged.

    Yields:
        Nothing; the variable is set while the block runs.

    Raises:
        TypeError: ``dataset_token`` is not a string.
        ValueError: ``dataset_token`` is empty, only whitespace, or contains a
            NUL character. Nothing is changed in that case.

    Example::

        with credentials(dataset_token=dataset_secret):
            eval(
                "llm_behavior_eval/<benchmark>",
                model=get_model("openai-api/answer/<model>", api_key=model_secret),
                model_roles={
                    "grader": get_model(
                        "openai-api/airgap/<judge>", api_key=judge_secret
                    )
                },
            )
    """
    if dataset_token is None:
        yield
        return
    value = _validated_dataset_token(dataset_token)
    previous = os.environ.get(DATASET_CREDENTIAL_VAR)
    try:
        os.environ[DATASET_CREDENTIAL_VAR] = value
        yield
    finally:
        if previous is None:
            os.environ.pop(DATASET_CREDENTIAL_VAR, None)
        else:
            os.environ[DATASET_CREDENTIAL_VAR] = previous
