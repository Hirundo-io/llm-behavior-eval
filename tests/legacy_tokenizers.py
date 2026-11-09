"""Tokenizer fixtures for the opt-in legacy text preprocessing tests.

``tiny_tokenizer`` builds a byte-level fast tokenizer with a chat template
offline, so the tests run in CI. It adds ``<bos>`` both from the template and as
a special token, like Gemma, so the legacy double-BOS tokenization is exercised.
The pinned Gemma 3 and Qwen3 snapshots are optional: tests using them skip when
the files are absent (see scripts/parity/README.md).
"""

import hashlib
import os
from pathlib import Path

import pytest

TINY_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}"
    "<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>"
    "{% if enable_thinking is defined and enable_thinking %}<think>{% endif %}"
    "{% endif %}"
)
PINNED = {
    "gemma3": (
        "LLM169_TOKENIZER",
        "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
    ),
    "qwen3": (
        "LLM169_QWEN3_TOKENIZER",
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
}


def tiny_tokenizer(directory: Path) -> Path:
    """Save the tiny tokenizer to ``directory`` and return the snapshot path."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    vocab.update({char: index + 3 for index, char in enumerate(alphabet)})
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.chat_template = TINY_TEMPLATE
    path = directory / "tiny-tokenizer"
    tokenizer.save_pretrained(str(path))
    return path


def pinned_tokenizer(name: str) -> Path:
    """A pinned local tokenizer snapshot, or skip when it is absent."""
    variable, digest = PINNED[name]
    path = Path(os.environ.get(variable, f"results/tokenizers/{name}")).resolve()
    if not path.is_dir():
        pytest.skip(f"Pinned {name} tokenizer absent; see scripts/parity/README.md")
    assert hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest() == digest
    return path
