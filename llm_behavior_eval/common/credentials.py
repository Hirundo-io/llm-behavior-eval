"""Credentials from environment variables, never from task or logged arguments.

Inspect records every task argument in the ``.eval`` log, so no benchmark takes
a credential. Each role reads its own environment variables instead:

* Dataset access reads ``DATASET_TOKEN_ENV``, then ``HF_TOKEN``
  (``resolve_dataset_token``), so datasets can use a credential separate from
  the models' (the legacy ``--dataset-token``).
* The model under test and the grader use their Inspect provider's own
  variables (``HF_TOKEN``, ``OPENAI_API_KEY``, ``<SERVICE>_API_KEY``, ...).

``credentials`` sets these variables for programmatic callers, from ordinary
Python arguments, for the duration of a ``with`` block. Environment variables
are process-global: do not overlap contexts that set different credentials in
one process (threads, concurrent ``eval`` calls); run such evaluations in
separate processes.
"""

import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager

# S105: these constants are environment variable names, not secrets.
DATASET_TOKEN_ENV = "LLM_BEHAVIOR_EVAL_DATASET_TOKEN"  # noqa: S105
"""Hugging Face token for dataset access only; takes precedence over ``HF_TOKEN``."""

HF_TOKEN_ENV = "HF_TOKEN"  # noqa: S105
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
        ``DATASET_TOKEN_ENV`` when set and non-empty, otherwise ``HF_TOKEN``
        when set and non-empty, otherwise ``None``, which lets
        ``huggingface_hub`` use its remaining sources (such as a cached
        ``hf auth login``). Credentials given only to a model (an ``api_key``
        model argument, ``<SERVICE>_API_KEY``) are never used.
    """
    return _clean(os.environ.get(DATASET_TOKEN_ENV)) or _clean(
        os.environ.get(HF_TOKEN_ENV)
    )


def _credential_variables(
    dataset_token: str | None,
    model_env: Mapping[str, str] | None,
    judge_env: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate the requested variables and merge them into one mapping.

    Error messages name variables and roles, never values.

    Raises:
        TypeError: A name or value is not a string.
        ValueError: A name or value is empty or contains ``=`` (names) or a
            NUL character, or two roles set one variable to different values.
    """
    requested: list[tuple[str, Mapping[str, str]]] = []
    if dataset_token is not None:
        requested.append(("dataset", {DATASET_TOKEN_ENV: dataset_token}))
    if model_env is not None:
        requested.append(("model", model_env))
    if judge_env is not None:
        requested.append(("judge", judge_env))

    variables: dict[str, str] = {}
    owners: dict[str, str] = {}
    for role, env in requested:
        for name, value in env.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError(
                    f"{role} credential names and values must be strings"
                    f" (variable {name!r})"
                )
            if not name or "=" in name or "\0" in name:
                raise ValueError(f"invalid environment variable name {name!r}")
            if not value.strip() or "\0" in value:
                raise ValueError(
                    f"{role} credential {name} is empty or contains a NUL"
                    " character; omit it to leave the variable unchanged"
                )
            if name in variables and variables[name] != value:
                raise ValueError(
                    f"the {owners[name]} and {role} credentials both set {name}"
                    " to different values. One process environment cannot hold"
                    " both; see INSPECT_MIGRATION.md, Credentials, for per-model"
                    " api_key arguments or separate processes."
                )
            variables[name] = value
            owners.setdefault(name, role)
    return variables


@contextmanager
def credentials(
    *,
    dataset_token: str | None = None,
    model_env: Mapping[str, str] | None = None,
    judge_env: Mapping[str, str] | None = None,
) -> Generator[None]:
    """Set credential environment variables for the duration of a ``with`` block.

    Credentials stay ordinary Python arguments: they are never task arguments
    or logged model arguments, so they never reach the ``.eval`` log. Keep the
    block open around the whole ``inspect_ai.eval`` call, because datasets
    load and providers read their variables while it runs.

    On exit, normal or by exception, every variable the context set is
    restored to its previous value, or removed if it was unset before. Changes
    made inside the block to those variables are discarded; other variables
    are untouched. Contexts nest; the innermost value wins while it is open.

    Environment variables are process-global. Do not overlap contexts that set
    different credentials from several threads or concurrent evaluations in
    one process; run those in separate processes instead.

    Args:
        dataset_token: Hugging Face token for dataset access only, set as
            ``DATASET_TOKEN_ENV``. ``None`` leaves the variable unchanged.
        model_env: Variables for the model under test's provider, for example
            ``{"HF_TOKEN": ...}`` for ``hf/`` or ``{"ANSWER_API_KEY": ...}`` for
            ``openai-api/answer/<model>``.
        judge_env: Variables for the grader's provider, for example
            ``{"AIRGAP_API_KEY": ...}`` for ``openai-api/airgap/<model>``.

    Yields:
        Nothing; the variables are set while the block runs.

    Raises:
        TypeError: A name or value is not a string.
        ValueError: A name or value is invalid or empty, or two roles set one
            variable to different values (for example two ``hf/`` models with
            different ``HF_TOKEN`` values). Nothing is changed in that case.

    Example::

        with credentials(
            dataset_token=dataset_secret,
            model_env={"HF_TOKEN": model_secret},
            judge_env={"AIRGAP_API_KEY": judge_secret},
        ):
            eval(
                "llm_behavior_eval/<benchmark>",
                model="hf/org/answer",
                model_roles={"grader": "openai-api/airgap/judge"},
            )
    """
    variables = _credential_variables(dataset_token, model_env, judge_env)
    previous = {name: os.environ.get(name) for name in variables}
    try:
        for name, value in variables.items():
            os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
