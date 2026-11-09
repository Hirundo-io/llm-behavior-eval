"""Legacy text preprocessing at the native provider boundary (no inference engine)."""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from inspect_ai import Task
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import Generate, TaskState, solver


@lru_cache(maxsize=8)
def load_tokenizer(path: str, revision: str | None) -> Any:
    from transformers import AutoTokenizer

    if not Path(path).is_dir() and not revision:
        raise ValueError(
            "compatibility_tokenizer requires a local snapshot or immutable revision"
        )
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "compatibility_tokenizer_revision must be an immutable 40-character commit"
        )
    return AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=True)


def preprocess(
    tokenizer: Any,
    messages: list[dict[str, str]],
    target: str,
    *,
    thinking: bool = False,
    question: str | None = None,
    stereotype: str | None = None,
) -> dict[str, Any]:
    """Retain padding, special tokens, whitespace and tokenizer truncation direction."""
    template = tokenizer.get_chat_template()
    if any(x in template for x in ("/no_think", "to=self", "reasoning_strength")):
        raise ValueError(
            "This template requires unsupported legacy thinking transformations"
        )
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )

    if not thinking and rendered.rstrip().endswith("<think>"):
        raise ValueError(
            "Native template requires unsupported legacy thinking-token closure"
        )

    def roundtrip(text: str, limit: int, special: bool) -> dict[str, Any]:
        encoded = tokenizer(
            [text],
            truncation=True,
            padding="max_length",
            max_length=limit,
            add_special_tokens=special,
        )
        ids = list(encoded["input_ids"][0])
        mask = list(encoded["attention_mask"][0])
        return dict(
            input_ids=ids,
            attention_mask=mask,
            decoded=tokenizer.decode(ids, skip_special_tokens=True),
        )

    prompt = roundtrip(rendered, 1024, True)
    result = dict(
        schema="legacy-text-v1",
        prompt=prompt,
        target=roundtrip(target, 256, False),
        judge_question=prompt["decoded"],
    )
    if question is not None:
        result["judge_question"] = roundtrip(question, 256, False)["decoded"]
    if stereotype is not None:
        result["stereotype"] = roundtrip(stereotype, 256, False)
    return result


def judge_question(state: TaskState) -> str:
    return state.metadata.get("compatibility", {}).get(
        "judge_question", state.metadata.get("judge_question", state.input_text)
    )


def judge_target(state: TaskState, target: str) -> str:
    return (
        state.metadata.get("compatibility", {}).get("target", {}).get("decoded", target)
    )


def validate_hf(api: Any, tokenizer: Any | None = None) -> None:
    """Validate native HF configuration without mutating shared provider state."""
    args = api.tokenizer_call_args
    if not isinstance(args, dict) or set(args) - {
        "truncation",
        "max_length",
        "add_special_tokens",
    }:
        raise ValueError(
            "Unsupported HF tokenizer_call_args; omit padding/return_tensors"
        )
    if args.get("add_special_tokens", True) is not True:
        raise ValueError("HF requires add_special_tokens=True")
    if tokenizer is None:
        if args.get("truncation", False) is not False or "max_length" in args:
            raise ValueError(
                "Legacy HF judge must not have answer truncation/max_length configured"
            )
        return
    if args != dict(truncation=True, max_length=1024, add_special_tokens=True):
        raise ValueError(
            "HF answer requires tokenizer_call_args={truncation: true, max_length: 1024, add_special_tokens: true}"
        )
    if api.use_chat_template is False:
        raise ValueError("HF answer requires use_chat_template=True")
    if (
        api.chat_template is not None
        and api.chat_template != tokenizer.get_chat_template()
    ):
        raise ValueError(
            "HF chat_template override differs from compatibility tokenizer"
        )
    if not isinstance(api.enable_thinking, bool):
        raise ValueError("HF requires explicit enable_thinking")

    def identity(value: Any) -> dict[str, Any]:
        if not getattr(value, "is_fast", False):
            raise ValueError(
                "HF compatibility requires a fast tokenizer with inspectable identity"
            )
        backend = json.loads(value.backend_tokenizer.to_str())
        # Runtime encode settings differ: native HF owns dynamic left EOS pads.
        backend.pop("padding", None)
        backend.pop("truncation", None)
        return backend

    if (
        identity(api.tokenizer) != identity(tokenizer)
        or api.tokenizer.get_chat_template() != tokenizer.get_chat_template()
        or api.tokenizer.truncation_side != tokenizer.truncation_side
    ):
        raise ValueError(
            "HF provider tokenizer differs from compatibility tokenizer content/template/truncation"
        )


def compatibility_task(
    task: Task, tokenizer_path: str | None, revision: str | None, *, family: str
) -> Task:
    """Wrap generation, preserving original input and writing explicit judge metadata.

    Real providers require an explicit matching tokenizer. Mock smoke runs are
    deliberately unqualified. HF/vLLM own templating and generation; encoded text
    is never sent back as a user message.
    """
    original = task.solver

    @solver
    def compatibility():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            async def boundary(
                state: TaskState,
                tool_calls: Literal["loop", "single", "none"] = "loop",
                **kwargs: Any,
            ) -> TaskState:
                model = get_model()
                provider = str(model).split("/")[0]
                if provider == "mockllm" and tokenizer_path is None:
                    return await generate(state, tool_calls=tool_calls, **kwargs)
                if tokenizer_path is None:
                    raise ValueError(
                        "Supply compatibility_tokenizer (matching local snapshot) for production preprocessing"
                    )
                if provider not in {"vllm", "hf", "mockllm"}:
                    raise ValueError(
                        "Production preprocessing requires native hf or vllm; API answer token boundaries are unsupported"
                    )
                tokenizer = load_tokenizer(tokenizer_path, revision)
                messages = [dict(role=m.role, content=m.text) for m in state.messages]
                # Gemma 3's pinned template folds system content into the user turn.
                # Let the actual template do this; do not demote roles generically.
                template = tokenizer.get_chat_template()
                if "System role not supported" in template:
                    raise ValueError(
                        "Old Gemma system-role merge is unsupported by the native transport"
                    )
                extra = dict(kwargs.get("extra_body") or {})
                template_args = extra.get("chat_template_kwargs", {})
                if set(template_args) - {"enable_thinking"}:
                    raise ValueError(
                        "Additional chat template arguments require separate legacy qualification"
                    )
                thinking = template_args.get("enable_thinking", False)
                if provider == "hf":
                    api: Any = model.api
                    validate_hf(api, tokenizer)
                    thinking = api.enable_thinking
                    rendered = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=thinking,
                    )
                    if api.hf_chat(state.messages, []) != rendered:
                        raise ValueError(
                            "HF provider message/template transformation differs from legacy rendering"
                        )
                probe = preprocess(
                    tokenizer,
                    messages,
                    state.target.text,
                    thinking=thinking,
                    question=state.metadata.get("judge_question")
                    if family == "prompt_injection"
                    else None,
                    stereotype=state.metadata.get("stereotyped_answer"),
                )
                probe["thinking"] = thinking
                if family == "ccpc":
                    probe["judge_question"] = state.input_text
                state.metadata["compatibility"] = probe
                state.metadata["compatibility_tokenizer"] = dict(
                    path=tokenizer_path,
                    revision=revision,
                    truncation_side=tokenizer.truncation_side,
                    template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                )
                state.metadata["prompt_parity"] = (
                    "production legacy-text-v1; live token equality requires qualification"
                )
                if provider == "hf":
                    probe["provider_padding"] = (
                        "dynamic left EOS; unpadded IDs equivalent, generation unqualified"
                    )
                    return await generate(state, tool_calls=tool_calls, **kwargs)
                extra.update(
                    return_token_ids=True,
                    add_special_tokens=True,
                    truncate_prompt_tokens=1024,
                    truncation_side=tokenizer.truncation_side,
                )
                kwargs["extra_body"] = extra
                return await generate(state, tool_calls=tool_calls, **kwargs)

            return await original(state, boundary)

        return solve

    task.solver = compatibility()
    return task


def judge_config(state: TaskState, config: GenerateConfig) -> GenerateConfig:
    """Validate HF and configure vLLM graders without applying answer caps."""
    if "compatibility" not in state.metadata:
        return config
    model = get_model(role="grader", required=True)
    if str(model).startswith("hf/"):
        validate_hf(model.api)
        return config
    if not str(model).startswith("vllm/"):
        return config
    extra = dict(config.extra_body or {})
    if extra.get("truncate_prompt_tokens") is not None:
        raise ValueError("Legacy judge must not have truncate_prompt_tokens configured")
    extra.update(add_special_tokens=True, return_token_ids=True)
    return config.model_copy(update={"extra_body": extra})
