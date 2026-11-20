"""The challenge's email-assistant context, rebuilt for one attack email.

Mirrors the GPT path of the challenge agent at the commit in ``prompts``:
``email_retriever.filter_emails`` and ``BasicEmailRetriever`` (levels 1 and 2),
``prompt_utils.format_emails``, the spotlighting branch of
``LLMWithDefenses.prompt`` and the message layout of ``GPTLLM.call_model``.
"""

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Literal

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import Tool, ToolCall, ToolDef, ToolParam, ToolParams

from llm_behavior_eval.llmail_inject.prompts import (
    MAX_LEN,
    READ_EMAILS_CALL_ID,
    READ_EMAILS_FUNCTION,
    SPECIAL_TOKENS,
    SPOTLIGHT_DATA_MARK,
    SPOTLIGHT_DATA_MARK_SUFFIX,
    SPOTLIGHT_EMAILS_FORMAT,
    SPOTLIGHT_QUERY_FORMAT,
    SPOTLIGHT_SYS_SUFFIX,
    SYSTEM_PROMPT,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    TOOL_PARAMETERS,
    TRUNCATION_MODEL,
    USER_QUERY,
)

if TYPE_CHECKING:
    from tiktoken import Encoding

PromptDefense = Literal["none", "spotlight", "delimiters"]
PROMPT_DEFENSES: tuple[PromptDefense, ...] = ("none", "spotlight", "delimiters")


@dataclass(frozen=True)
class Level:
    """A retrieval scenario without retrieval (``rag: False``)."""

    number: int
    top_k: int
    """Emails given to the model, the attacker's included."""
    emails_file: str


LEVELS: dict[int, Level] = {
    1: Level(number=1, top_k=2, emails_file="level1.json"),
    2: Level(number=2, top_k=10, emails_file="level2.json"),
}
"""``scenario1`` and ``scenario2`` of ``config.yaml``. Levels 3 and 4 rank the
emails with a retriever and are not supported (see the README)."""


@cache
def _encoding() -> "Encoding":
    import tiktoken

    return tiktoken.encoding_for_model(TRUNCATION_MODEL)


def filter_email(attacker_email: str) -> str:
    """``filter_emails``: the challenge's sanitization of the attacker's email.

    Removes the tool name, turns newlines into spaces, replaces special tokens
    by a space and keeps the first ``MAX_LEN`` ``gpt-4o`` tokens. Benign emails
    are not filtered.
    """
    attacker_email = attacker_email.replace(TOOL_NAME, "")
    attacker_email = attacker_email.replace("\n", " ")
    for token in SPECIAL_TOKENS:
        attacker_email = attacker_email.replace(token, " ")
    # A byte-level BPE token covers at least one byte, so text of at most
    # MAX_LEN UTF-8 bytes is never truncated; skip loading the encoding then.
    if len(attacker_email.encode("utf-8")) <= MAX_LEN:
        return attacker_email
    encoding = _encoding()
    return encoding.decode(encoding.encode(attacker_email)[0:MAX_LEN])


def retrieve(
    level: Level, emails: list[str], attacker_email: str
) -> tuple[list[str], int]:
    """``BasicEmailRetriever``: the level's emails with the attacker's inserted.

    Keeps the first ``top_k - 1`` benign emails and puts the filtered attacker
    email after the first half of them (level 1: last of two; level 2: fifth
    of ten).

    Returns:
        The emails and the attacker email's index among them.
    """
    kept = emails[: level.top_k - 1]
    new_email = filter_email(attacker_email)
    if len(kept) > 1:
        index = int(len(kept) / 2)
        return [*kept[:index], new_email, *kept[index:]], index
    return [*kept, new_email], len(kept)


def format_emails(emails: list[str]) -> str:
    """``format_emails``: each email prefixed by ``" email: "``, concatenated."""
    return "".join(f" email: {email}" for email in emails)


def build_messages(
    emails: list[str], prompt_defense: PromptDefense = "none"
) -> list[ChatMessage]:
    """The GPT-path conversation for the retrieved emails.

    System prompt, user query, a fabricated ``read_emails`` call and its tool
    result holding the emails. ``spotlight`` is the phase 1 spotlighting
    sub-level (instructions, delimiters and data marking); ``delimiters`` is
    what phase 2 added to every level (instructions and delimiters only).
    """
    system_prompt = SYSTEM_PROMPT
    query = USER_QUERY
    formatted_emails = format_emails(emails)
    if prompt_defense not in PROMPT_DEFENSES:
        raise ValueError(
            f"prompt_defense must be one of {list(PROMPT_DEFENSES)}; "
            f"got {prompt_defense!r}"
        )
    if prompt_defense == "spotlight":
        system_prompt += SPOTLIGHT_SYS_SUFFIX
        query = SPOTLIGHT_QUERY_FORMAT.format(query)
        system_prompt += SPOTLIGHT_DATA_MARK_SUFFIX
        formatted_emails = SPOTLIGHT_DATA_MARK.join(formatted_emails.split(" "))
        formatted_emails = SPOTLIGHT_EMAILS_FORMAT.format(formatted_emails)
    elif prompt_defense == "delimiters":
        system_prompt += SPOTLIGHT_SYS_SUFFIX
        query = SPOTLIGHT_QUERY_FORMAT.format(query)
        formatted_emails = SPOTLIGHT_EMAILS_FORMAT.format(formatted_emails)
    return [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=query),
        ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(
                    id=READ_EMAILS_CALL_ID,
                    function=READ_EMAILS_FUNCTION,
                    arguments={"query": query},
                )
            ],
        ),
        ChatMessageTool(
            content=formatted_emails,
            tool_call_id=READ_EMAILS_CALL_ID,
            function=READ_EMAILS_FUNCTION,
        ),
    ]


def send_email() -> Tool:
    """The challenge's ``send_email`` tool, as a stub that sends nothing.

    The task never executes tool calls (``generate(tool_calls="none")``); the
    scorer reads them from the model's reply. The stub only exists to give the
    model the schema, and returns a fixed string if run by another solver.
    """

    async def execute(to: str, body: str) -> str:
        return "Email not sent: this tool is a stub."

    return ToolDef(
        execute,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        parameters=ToolParams(
            properties={
                name: ToolParam(type="string", description=description)
                for name, description in TOOL_PARAMETERS.items()
            },
            required=list(TOOL_PARAMETERS),
        ),
    ).as_tool()
