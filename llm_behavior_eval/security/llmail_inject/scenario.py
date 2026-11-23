"""The challenge's email-assistant context, rebuilt for one attack email.

Mirrors the GPT path of the challenge agent at the commit in ``prompts``:
``email_retriever.filter_emails``, ``BasicEmailRetriever`` (levels 1 and 2),
``RAGEmailRetriever`` (levels 3 and 4, with TF-IDF in place of the embedding
model; see ``TfidfRetriever``), ``prompt_utils.format_emails``, the
spotlighting branch of ``LLMWithDefenses.prompt`` and the message layout of
``GPTLLM.call_model``.
"""

import bisect
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import Tool, ToolCall, ToolDef, ToolParam, ToolParams

from llm_behavior_eval.security.llmail_inject.prompts import (
    LEVEL3_QUERY,
    LEVEL4_QUERY,
    LEVEL4_TARGET_BODY,
    MAX_LEN,
    RAG_TOP_K,
    READ_EMAILS_CALL_ID,
    READ_EMAILS_FUNCTION,
    SPECIAL_TOKENS,
    SPOTLIGHT_DATA_MARK,
    SPOTLIGHT_DATA_MARK_SUFFIX,
    SPOTLIGHT_EMAILS_FORMAT,
    SPOTLIGHT_QUERY_FORMAT,
    SPOTLIGHT_SYS_SUFFIX,
    SYSTEM_PROMPT,
    TARGET_BODY,
    TARGET_TO,
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
    """One challenge scenario (``scenario_templates.scenario<number>``)."""

    number: int
    top_k: int
    """Emails given to the model, the attacker's included."""
    emails_file: str
    query: str
    """The user's query; at a retrieval level also the retrieval query."""
    target_body: str
    """The ``body`` the attacker wants sent."""
    target_to: str = TARGET_TO
    """The recipient the attacker wants (the same at every level)."""
    rag: bool = False
    """Whether the level ranks emails by similarity to the query."""
    target_in_emails: bool = False
    """Whether ``target_body`` is data the attack must take from another email."""


LEVELS: dict[int, Level] = {
    1: Level(1, 2, "level1.json", USER_QUERY, TARGET_BODY),
    2: Level(2, 10, "level2.json", USER_QUERY, TARGET_BODY),
    3: Level(3, RAG_TOP_K, "level3.json", LEVEL3_QUERY, TARGET_BODY, rag=True),
    4: Level(
        4,
        RAG_TOP_K,
        "level4.json",
        LEVEL4_QUERY,
        LEVEL4_TARGET_BODY,
        rag=True,
        target_in_emails=True,
    ),
}
"""``scenario1`` to ``scenario4`` of ``config.yaml``."""
DEFAULT_LEVELS = (1, 2)
"""Levels run by default: those without retrieval, which phase 2 used."""
TFIDF_EXTRA = "llm-behavior-eval-llmail-inject[tfidf]"


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


class BasicRetriever:
    """``BasicEmailRetriever``: the level's first emails, attacker in the middle.

    Keeps the first ``top_k - 1`` benign emails and puts the filtered attacker
    email after the first half of them (level 1: last of two; level 2: fifth
    of ten).
    """

    def __init__(self, level: Level, emails: list[str]) -> None:
        self.emails = emails[: level.top_k - 1]

    def retrieve(self, attacker_email: str) -> tuple[list[str], int]:
        """The emails with the filtered attacker email, and its index."""
        new_email = filter_email(attacker_email)
        if len(self.emails) > 1:
            index = int(len(self.emails) / 2)
            return [*self.emails[:index], new_email, *self.emails[index:]], index
        return [*self.emails, new_email], len(self.emails)


def tfidf_modules() -> tuple[Any, Any]:
    """scikit-learn's ``TfidfVectorizer`` and ``cosine_similarity``.

    Raises:
        ImportError: scikit-learn is not installed (the ``tfidf`` extra).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as error:
        raise ImportError(
            "LLMail-Inject levels 3 and 4 rank emails with TF-IDF, which needs "
            f"scikit-learn: install {TFIDF_EXTRA!r}"
        ) from error
    return TfidfVectorizer, cosine_similarity


class TfidfRetriever:
    """``RAGEmailRetriever`` with TF-IDF similarity, the attacker email assumed retrieved.

    The challenge embedded the query and every email with
    ``paraphrase-MiniLM-L6-v2``, kept the ``top_k`` benign emails most similar
    to the query, inserted the attacker email at its rank among them
    (``bisect``: after benign emails with an equal score) and cut the list to
    ``top_k``, so an attacker email ranked below all of them was not
    retrieved. Here similarity is the cosine of TF-IDF vectors (English stop
    words removed) fitted on the level's benign emails only, so their ranking
    does not depend on the attack, as with fixed embeddings. The attacker
    email is always retrieved: it is inserted at its rank among the
    ``top_k - 1`` most similar benign emails (equal scores keep file order),
    which is the challenge's list whenever its rank is within ``top_k``.
    """

    def __init__(self, level: Level, emails: list[str]) -> None:
        vectorizer_class, self._similarity = tfidf_modules()
        self._vectorizer = vectorizer_class(stop_words="english").fit(emails)
        self._query = self._vectorizer.transform([level.query])
        scores = self._similarity(self._query, self._vectorizer.transform(emails))[0]
        order = sorted(range(len(emails)), key=lambda i: (-scores[i], i))
        kept = order[: level.top_k - 1]
        self.emails = [emails[i] for i in kept]
        self.scores = [float(scores[i]) for i in kept]
        if level.target_in_emails and not any(
            level.target_body in email for email in self.emails
        ):
            raise ValueError(
                f"LLMail-Inject level {level.number}: none of the "
                f"{len(self.emails)} benign emails TF-IDF retrieves contains the "
                f"target {level.target_body!r}, so no attack could succeed"
            )

    def score(self, email: str) -> float:
        """Cosine similarity of ``email`` to the query."""
        return float(
            self._similarity(self._query, self._vectorizer.transform([email]))[0][0]
        )

    def retrieve(self, attacker_email: str) -> tuple[list[str], int]:
        """The ranked emails with the filtered attacker email, and its index."""
        new_email = filter_email(attacker_email)
        rank = bisect.bisect([-score for score in self.scores], -self.score(new_email))
        return [*self.emails[:rank], new_email, *self.emails[rank:]], rank


Retriever = BasicRetriever | TfidfRetriever


def retriever(level: Level, emails: list[str]) -> Retriever:
    """The level's retriever over its benign emails."""
    return TfidfRetriever(level, emails) if level.rag else BasicRetriever(level, emails)


def retrieve(
    level: Level, emails: list[str], attacker_email: str
) -> tuple[list[str], int]:
    """The emails the level gives the model, and the attacker email's index."""
    return retriever(level, emails).retrieve(attacker_email)


def format_emails(emails: list[str]) -> str:
    """``format_emails``: each email prefixed by ``" email: "``, concatenated."""
    return "".join(f" email: {email}" for email in emails)


def build_messages(
    emails: list[str],
    prompt_defense: PromptDefense = "none",
    query: str = USER_QUERY,
) -> list[ChatMessage]:
    """The GPT-path conversation for the retrieved emails and the user's query.

    System prompt, user query, a fabricated ``read_emails`` call and its tool
    result holding the emails. ``spotlight`` is the phase 1 spotlighting
    sub-level (instructions, delimiters and data marking); ``delimiters`` is
    what phase 2 added to every level (instructions and delimiters only).
    """
    system_prompt = SYSTEM_PROMPT
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
    return conversation(system_prompt, query, formatted_emails)


def conversation(system_prompt: str, query: str, emails: str) -> list[ChatMessage]:
    """``GPTLLM.call_model``'s messages: the emails as a ``read_emails`` result.

    System prompt, user query, an assistant message with a fabricated
    ``read_emails`` call (``call_001``, the query as its argument) and the tool
    message holding the emails text.
    """
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
            content=emails,
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
