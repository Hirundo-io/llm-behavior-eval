"""Opt-in legacy-engine text preprocessing, for parity runs only.

The legacy engine did not judge the raw sample text. It rendered the chat
template, tokenized the result with truncation to 1024 tokens and
``padding="max_length"``, and gave the judge the *decoded* prompt as the
question. Targets, stereotyped answers and per-row judge questions took a
256-token round trip without special tokens. This module reproduces those text
views from the answer model's own tokenizer, so a benchmark can show the judge
exactly what the legacy judge saw.

It is a prompt transform, not an inference setting. It never changes the
messages sent to the model under test, never sets a ``GenerateConfig`` for
either role, and never inspects or restricts the provider, so every Inspect
provider still works. Truncating the answer model's own input to 1024 tokens is
a provider setting; parity runs pass it with the benchmark's ``parity.yaml``
(vLLM/OpenAI-compatible servers) or ``-M tokenizer_call_args`` (HF).

Nothing here runs unless a benchmark opts in: each benchmark adds
:func:`legacy_text_preprocessing` to its solver only when its
``legacy_tokenizer`` task argument is set, and its scorer reads the recorded
views with :func:`legacy_prompt` and :func:`legacy_text`. Which views a
benchmark uses, and what it shows its judge, is that benchmark's decision.
"""

import hashlib
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from inspect_ai.solver import Generate, Solver, TaskState, solver

# Legacy engine tokenization (evaluation_utils/custom_dataset.py,
# free_text_preprocess_function): prompts use max_length=1024 with special
# tokens; targets, stereotypes and judge questions use gt_max_length=256 without.
LEGACY_PROMPT_MAX_LENGTH = 1024
LEGACY_TEXT_MAX_LENGTH = 256
METADATA_KEY = "legacy_text"
SCHEMA = "legacy-text-v1"

# Templates whose legacy rendering differed from a plain apply_chat_template
# (evaluation_utils/util_functions.py SafeApplyChatTemplate): Nemotron
# "/no_think" injection, Muse Glimmer reasoning surgery, and the Gemma v1
# system-role merge. They fail loudly instead of producing a wrong prompt.
_UNSUPPORTED_TEMPLATE_MARKERS = {
    "/no_think": "Nemotron /no_think injection",
    "to=self": "Muse Glimmer reasoning surgery",
    "reasoning_strength": "Muse Glimmer reasoning surgery",
    "System role not supported": "Gemma v1 system-role merge",
}
_REVISION = re.compile(r"[0-9a-f]{40}")


@lru_cache(maxsize=8)
def load_tokenizer(path: str, revision: str | None = None) -> Any:
    """Load the answer model's tokenizer from local files only.

    ``path`` is a local snapshot directory, or a repository already in the HF
    cache together with an immutable 40-character commit ``revision``. Nothing
    is downloaded during an evaluation.
    """
    from transformers import AutoTokenizer

    if revision is not None and not _REVISION.fullmatch(revision):
        raise ValueError(
            "legacy_tokenizer_revision must be an immutable 40-character commit"
        )
    if revision is None and not Path(path).is_dir():
        raise ValueError(
            "legacy_tokenizer must be a local snapshot directory, or a cached "
            "repository with legacy_tokenizer_revision"
        )
    return AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=True)


def render_chat(
    tokenizer: Any, messages: list[dict[str, str]], *, legacy_thinking: bool
) -> str:
    """Render ``messages`` with the tokenizer's chat template, as the legacy engine did.

    ``legacy_thinking`` is the ``enable_thinking`` chat-template variable the
    legacy engine rendered with; templates that do not read it ignore it. It
    only shapes this text view, never a model request. Templates the legacy
    engine rewrote raise ``ValueError``.
    """
    template = tokenizer.get_chat_template()
    for marker, reason in _UNSUPPORTED_TEMPLATE_MARKERS.items():
        if marker in template:
            raise ValueError(f"Unsupported legacy chat template ({reason})")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=legacy_thinking,
    )
    if not legacy_thinking and rendered.rstrip().endswith("<think>"):
        raise ValueError(
            "Unsupported legacy chat template (open <think> with thinking off)"
        )
    return rendered


def roundtrip(
    tokenizer: Any, text: str, *, max_length: int, add_special_tokens: bool
) -> dict[str, Any]:
    """Tokenize as the legacy engine did, then decode without special tokens.

    Keeps the tokenizer's own truncation side. Returns the unpadded
    ``input_ids`` (the tokens the model would see) and the ``decoded`` text.
    """
    encoded = tokenizer(
        [text],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    ids = list(encoded["input_ids"][0])
    mask = list(encoded["attention_mask"][0])
    return {
        "input_ids": [token for token, keep in zip(ids, mask, strict=True) if keep],
        "decoded": tokenizer.decode(ids, skip_special_tokens=True),
    }


@solver
def legacy_text_preprocessing(
    tokenizer: str,
    revision: str | None = None,
    *,
    legacy_thinking: bool,
    target: bool = False,
    metadata_fields: Sequence[str] = (),
) -> Solver:
    """Record the legacy engine's text views of a sample in its metadata.

    Place it after the steps that build the prompt and before ``generate()``.
    It renders the current messages with the answer tokenizer and stores, under
    ``metadata["legacy_text"]``: the decoded 1024-token prompt with its token
    IDs (rendered with the chat-template variable ``enable_thinking`` set to
    ``legacy_thinking``, the benchmark's legacy value), and 256-token round trips of the target (``target=True``) and of each
    string ``metadata_fields`` entry. Messages, target and config are unchanged.
    The tokenizer is loaded here, so a bad path fails before any generation.
    """
    loaded = load_tokenizer(tokenizer, revision)
    template = loaded.get_chat_template()
    identity = {
        "path": tokenizer,
        "revision": revision,
        "truncation_side": loaded.truncation_side,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    }

    def text_view(text: str) -> dict[str, Any]:
        return roundtrip(
            loaded, text, max_length=LEGACY_TEXT_MAX_LENGTH, add_special_tokens=False
        )

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        messages = [{"role": m.role, "content": m.text} for m in state.messages]
        rendered = render_chat(loaded, messages, legacy_thinking=legacy_thinking)
        texts = {}
        if target:
            texts["target"] = text_view(state.target.text)
        for name in metadata_fields:
            value = state.metadata.get(name)
            if isinstance(value, str):
                texts[name] = text_view(value)
        state.metadata[METADATA_KEY] = {
            "schema": SCHEMA,
            "legacy_thinking": legacy_thinking,
            "prompt": roundtrip(
                loaded,
                rendered,
                max_length=LEGACY_PROMPT_MAX_LENGTH,
                add_special_tokens=True,
            ),
            "texts": texts,
            "tokenizer": identity,
        }
        return state

    return solve


def legacy_prompt(state: TaskState, default: str) -> str:
    """The decoded legacy prompt if preprocessing ran, else ``default``."""
    views = state.metadata.get(METADATA_KEY)
    return views["prompt"]["decoded"] if views else default


def legacy_text(state: TaskState, name: str, default: str) -> str:
    """The decoded round trip of ``name`` if preprocessing recorded it, else ``default``."""
    views = state.metadata.get(METADATA_KEY)
    if views and name in views["texts"]:
        return views["texts"][name]["decoded"]
    return default
