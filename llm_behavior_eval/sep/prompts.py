"""SEP prompt construction, as in the authors' evaluation code.

Source: ``model_eval/get_model_outputs.py`` and ``model_eval/prompt_templates.json``
in https://github.com/egozverev/Should-It-Be-Executed-Or-Processed at commit
``7606c0696f20f5aa433169fd2221f76852d1d4f5``.

Each item is run twice (``inference``):

* probe in data (``mode='data_with_probe'``): instruction argument
  ``system_prompt_clean``, data argument ``prompt_instructed``;
* probe in instruction (``mode='probe_with_task'``): instruction argument
  ``system_prompt_instructed``, data argument ``prompt_clean``.

The paper's "Naive" results use template 0 (``{"system": "{}", "main": "{}"}``),
so both arguments are the dataset strings unchanged. The instruction argument
is the system message and the data argument the user message
(``_format_model_input``). For models without a system role (the paper's Gemma
and Starling runs) both go into one user message as
``f"System prompt: {system_instruction} User prompt: {user_instruction}"``.
"""

from typing import Any, Literal

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser

ProbePosition = Literal["instruction", "data"]
PROBE_POSITIONS: tuple[ProbePosition, ...] = ("data", "instruction")
"""Generation order, as in the authors' ``inference`` loop: data first."""

NO_SYSTEM_ROLE_TEMPLATE = "System prompt: {system} User prompt: {user}"
"""The single user message for models without a system role (verbatim)."""


def arguments(metadata: dict[str, Any], position: ProbePosition) -> tuple[str, str]:
    """The (instruction, data) arguments with the probe at ``position``."""
    if position == "instruction":
        return metadata["system_prompt_instructed"], metadata["prompt_clean"]
    return metadata["system_prompt_clean"], metadata["prompt_instructed"]


def messages(
    metadata: dict[str, Any], position: ProbePosition, *, system_role: bool = True
) -> list[ChatMessage]:
    """The chat input for one SEP item with the probe at ``position``.

    Args:
        metadata: The sample's metadata (the four dataset prompts).
        position: Where the probe is: the instruction or the data argument.
        system_role: Send the instruction argument as a system message and the
            data argument as a user message. ``False`` puts both in one user
            message, ``System prompt: ... User prompt: ...``.

    Returns:
        The messages to generate from.
    """
    system, user = arguments(metadata, position)
    if system_role:
        return [ChatMessageSystem(content=system), ChatMessageUser(content=user)]
    # Braces inside the substituted prompts are not re-parsed by str.format.
    content = NO_SYSTEM_ROLE_TEMPLATE.format(system=system, user=user)
    return [ChatMessageUser(content=content)]
