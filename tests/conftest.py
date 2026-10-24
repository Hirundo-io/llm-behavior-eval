import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

HARMONY_SPECIAL_TOKENS = [
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|constrain|>",
]
HARMONY_TEST_WORDS = [
    "[UNK]",
    "assistant",
    "analysis",
    "final",
    "commentary",
    "to=refusal",
    "refusal",
    "We",
    "must",
    "refuse.",
    "I",
    "cannot",
    "share",
    "that.",
    "Sure.",
]


def _word_tokenizer(*, harmony: bool) -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            vocab={word: index for index, word in enumerate(HARMONY_TEST_WORDS)},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    backend.decoder = WordPiece(prefix="##", cleanup=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    if harmony:
        tokenizer.add_tokens(HARMONY_SPECIAL_TOKENS, special_tokens=True)
    return tokenizer


@pytest.fixture
def harmony_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer that registers the GPT-OSS Harmony control tokens."""
    return _word_tokenizer(harmony=True)


@pytest.fixture
def plain_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer without Harmony control tokens."""
    return _word_tokenizer(harmony=False)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "vllm_engine_test: marks tests that target the vLLM eval engine stubs",
    )
    config.addinivalue_line(
        "markers",
        "transformers_engine_test: marks tests that target the transformers eval engine stubs",
    )
