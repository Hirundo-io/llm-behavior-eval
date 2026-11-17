"""Credentials: dataset token resolution, ``credentials`` and per-model api_keys."""

import inspect
import os
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import hf_dataset
from inspect_ai.model import get_model
from inspect_ai.solver import generate

from llm_behavior_eval.common.completion import completion_scorer
from llm_behavior_eval.common.credentials import (
    DATASET_CREDENTIAL_VAR,
    HF_CREDENTIAL_VAR,
    credentials,
    resolve_dataset_token,
)
from llm_behavior_eval.common.dataset import (
    LEGACY_SHUFFLE_SEED,
    load_hf_dataset,
    to_inspect_dataset,
)
from llm_behavior_eval.inspect_template.dataset import record_to_sample
from llm_behavior_eval.inspect_template.scorer import scorer

from inspect_helpers import (
    assert_no_credentials_logged,
    local_openai_judge,
    sample_scores,
)

DATASET_SENTINEL = "hf_DATASET_SENTINEL_NOT_A_REAL_SECRET"
MODEL_SENTINEL = "sk-MODEL_SENTINEL_NOT_A_REAL_SECRET"
JUDGE_SENTINEL = "sk-JUDGE_SENTINEL_NOT_A_REAL_SECRET"
HF_SENTINEL = "hf_ENVIRONMENT_SENTINEL_NOT_A_REAL_SECRET"
SHARED_SENTINEL = "sk-SHARED_ENVIRONMENT_SENTINEL_NOT_A_REAL_SECRET"

# Variables these tests set; each test starts with all of them unset.
MANAGED = (
    DATASET_CREDENTIAL_VAR,
    HF_CREDENTIAL_VAR,
    "ANSWER_API_KEY",
    "ANSWER_BASE_URL",
    "AIRGAP_API_KEY",
    "AIRGAP_BASE_URL",
    "SHARED_API_KEY",
    "SHARED_BASE_URL",
    "CACHED_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MANAGED:
        monkeypatch.delenv(name, raising=False)


def environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in MANAGED}


@pytest.mark.parametrize(
    ("dataset", "hf", "expected"),
    [
        (None, None, None),
        (None, "hf-env", "hf-env"),
        ("dataset-env", None, "dataset-env"),
        ("dataset-env", "hf-env", "dataset-env"),
        ("", "hf-env", "hf-env"),
        (" \n", "hf-env", "hf-env"),
        (" dataset-env\n", "hf-env", "dataset-env"),
        (None, "", None),
    ],
)
def test_dataset_token_precedence(
    monkeypatch: pytest.MonkeyPatch,
    dataset: str | None,
    hf: str | None,
    expected: str | None,
) -> None:
    if dataset is not None:
        monkeypatch.setenv(DATASET_CREDENTIAL_VAR, dataset)
    if hf is not None:
        monkeypatch.setenv(HF_CREDENTIAL_VAR, hf)
    assert resolve_dataset_token() == expected


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({}, {}),
        ({HF_CREDENTIAL_VAR: "hf-env"}, {"token": "hf-env"}),
        (
            {DATASET_CREDENTIAL_VAR: "dataset-env", HF_CREDENTIAL_VAR: "hf-env"},
            {"token": "dataset-env"},
        ),
    ],
)
def test_load_hf_dataset_passes_the_resolved_token(
    monkeypatch: pytest.MonkeyPatch,
    variables: dict[str, str],
    expected: dict[str, str],
) -> None:
    import datasets

    calls: list[dict[str, Any]] = []
    rows = Dataset.from_list([{"id": "a"}])

    def load(source: str, **kwargs: Any) -> Dataset:
        calls.append(kwargs)
        return rows

    monkeypatch.setattr(datasets, "load_dataset", load)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    load_hf_dataset("org/gated", split="train", revision="abc")
    assert calls == [{"split": "train", "revision": "abc"} | expected]


def test_snapshot_loads_ignore_the_dataset_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Dataset.from_list([{"id": "a"}]).save_to_disk(tmp_path / "snapshot")
    monkeypatch.setenv(DATASET_CREDENTIAL_VAR, DATASET_SENTINEL)
    assert (
        len(
            load_hf_dataset(
                "org/id", split="train", dataset_path=str(tmp_path / "snapshot")
            )
        )
        == 1
    )


def test_credentials_set_and_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HF_CREDENTIAL_VAR, "ambient")
    monkeypatch.setenv("UNRELATED", "kept")
    with credentials(dataset_token="dataset"):
        assert os.environ[DATASET_CREDENTIAL_VAR] == "dataset"
        # Only the dataset variable: model credentials are per-model api_keys.
        assert os.environ[HF_CREDENTIAL_VAR] == "ambient"
        # Changes to the managed variable inside the block are discarded.
        os.environ[DATASET_CREDENTIAL_VAR] = "changed"
    assert environment() == {name: None for name in MANAGED} | {
        HF_CREDENTIAL_VAR: "ambient"
    }
    assert os.environ["UNRELATED"] == "kept"
    monkeypatch.setenv(DATASET_CREDENTIAL_VAR, "previous")
    with credentials(dataset_token="dataset"):
        del os.environ[DATASET_CREDENTIAL_VAR]
    assert os.environ[DATASET_CREDENTIAL_VAR] == "previous"


def test_credentials_none_leaves_variables_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DATASET_CREDENTIAL_VAR, "ambient")
    with credentials():
        assert os.environ[DATASET_CREDENTIAL_VAR] == "ambient"
    with credentials(dataset_token=None):
        assert os.environ[DATASET_CREDENTIAL_VAR] == "ambient"
    assert os.environ[DATASET_CREDENTIAL_VAR] == "ambient"


def test_nested_credentials() -> None:
    with credentials(dataset_token="outer"):
        with credentials(dataset_token="inner"):
            assert os.environ[DATASET_CREDENTIAL_VAR] == "inner"
            assert resolve_dataset_token() == "inner"
        assert os.environ[DATASET_CREDENTIAL_VAR] == "outer"
    assert environment() == {name: None for name in MANAGED}


def test_credentials_restore_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HF_CREDENTIAL_VAR, "ambient")
    with pytest.raises(RuntimeError, match="boom"):
        with credentials(dataset_token="outer"):
            with credentials(dataset_token="inner"):
                raise RuntimeError("boom")
    # A previously unset variable is removed again.
    assert environment() == {name: None for name in MANAGED} | {
        HF_CREDENTIAL_VAR: "ambient"
    }


@pytest.mark.parametrize("token", ["", " \n", "data\0set"])
def test_invalid_credentials_change_nothing(
    monkeypatch: pytest.MonkeyPatch, token: str
) -> None:
    monkeypatch.setenv(HF_CREDENTIAL_VAR, "ambient")
    before = dict(os.environ)
    with pytest.raises(ValueError) as error:
        with credentials(dataset_token=token):
            pytest.fail("the block must not run")
    assert dict(os.environ) == before
    # Messages name the variable, never the value.
    assert DATASET_CREDENTIAL_VAR in str(error.value)
    if token.strip():
        assert token not in str(error.value)


def test_non_string_credentials_are_rejected() -> None:
    not_a_string: Any = 1
    with pytest.raises(TypeError):
        with credentials(dataset_token=not_a_string):
            pass
    assert DATASET_CREDENTIAL_VAR not in os.environ


def test_credentials_are_dataset_only() -> None:
    # Model and grader credentials are per-model api_key arguments; the
    # context no longer sets provider variables.
    parameters = inspect.signature(credentials).parameters
    assert list(parameters) == ["dataset_token"]
    assert parameters["dataset_token"].kind is inspect.Parameter.KEYWORD_ONLY
    removed: dict[str, Any] = {"model_env": {HF_CREDENTIAL_VAR: "model"}}
    with pytest.raises(TypeError):
        with credentials(**removed):
            pass
    assert HF_CREDENTIAL_VAR not in os.environ


def test_dataset_credential_precedence_inside_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(HF_CREDENTIAL_VAR, "hf-env")
    with credentials(dataset_token="dataset"):
        # The dataset uses its own credential; HF_TOKEN is untouched.
        assert resolve_dataset_token() == "dataset"
        assert os.environ[HF_CREDENTIAL_VAR] == "hf-env"
    # Without a dataset credential, datasets fall back to HF_TOKEN.
    assert resolve_dataset_token() == "hf-env"


def credentialed_task() -> Task:
    """A benchmark-shaped task loading a gated Hub dataset (patched loader).

    It follows the recommended dataset call: native ``hf_dataset`` with the
    token computed in the task body, never a task argument.
    """
    return Task(
        dataset=hf_dataset(
            "org/gated",
            split="train",
            sample_fields=record_to_sample,
            shuffle=LEGACY_SHUFFLE_SEED,
            token=resolve_dataset_token(),
        ),
        solver=generate(),
        scorer=[
            scorer(
                exclude_thinking_trace_for_judge=True,
                thinking_start_token="<think>",
                thinking_end_token="</think>",
            ),
            completion_scorer(),
        ],
    )


def test_eval_separates_credentials_and_logs_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hf_dataset_cache: Path
) -> None:
    # The primary Python pattern: explicit model objects with per-model
    # api_keys, and the dataset token from credentials(). Both models use
    # one service name, so their provider reads the same SHARED_API_KEY
    # variable; api_key keeps them apart anyway.
    import datasets

    tokens: list[str | None] = []

    def load(**kwargs: Any) -> Dataset:
        tokens.append(kwargs.get("token"))
        return Dataset.from_list([{"id": "a", "question": "1 + 1?", "answer": "2"}])

    monkeypatch.setattr(datasets, "load_dataset", load)
    # Set in the environment directly, outside the context manager.
    monkeypatch.setenv(HF_CREDENTIAL_VAR, HF_SENTINEL)
    monkeypatch.setenv("SHARED_API_KEY", SHARED_SENTINEL)
    with (
        local_openai_judge("2") as answer,
        local_openai_judge("SCORE: 1") as judge,
        credentials(dataset_token=DATASET_SENTINEL),
    ):
        [log] = inspect_eval(
            credentialed_task(),
            model=get_model(
                "openai-api/shared/answer",
                base_url=answer.base_url,
                api_key=MODEL_SENTINEL,
            ),
            model_roles={
                "grader": get_model(
                    "openai-api/shared/judge",
                    base_url=judge.base_url,
                    api_key=JUDGE_SENTINEL,
                )
            },
            log_dir=str(tmp_path),
            display="none",
        )
    assert log.status == "success", log.error
    [score] = sample_scores(log, "judge_scorer")
    assert score.value == 1
    # Each role got only its own credential.
    assert tokens == [DATASET_SENTINEL]
    assert answer.authorizations == [f"Bearer {MODEL_SENTINEL}"]
    assert judge.authorizations == [f"Bearer {JUDGE_SENTINEL}"]
    # Inspect logs neither key: api_key is not a model argument it records.
    assert "api_key" not in log.eval.model_args
    assert log.eval.model_roles
    grader = log.eval.model_roles["grader"]
    assert not isinstance(grader, list)
    assert "api_key" not in grader.args
    for sentinel in (
        DATASET_SENTINEL,
        MODEL_SENTINEL,
        JUDGE_SENTINEL,
        HF_SENTINEL,
        SHARED_SENTINEL,
    ):
        assert_no_credentials_logged(log, sentinel)
    assert DATASET_CREDENTIAL_VAR not in os.environ
    assert os.environ[HF_CREDENTIAL_VAR] == HF_SENTINEL


def test_per_model_api_keys_are_not_logged(tmp_path: Path) -> None:
    # The CLI-shaped form of the same credential: an api_key model argument
    # (-M api_key=... / a grader role's model_args), which Inspect redacts
    # from logged model arguments.
    with (
        local_openai_judge("2") as answer,
        local_openai_judge("SCORE: 1") as judge,
    ):
        [log] = inspect_eval(
            credentialed_task_from_rows(),
            model="openai-api/airgap/answer",
            model_base_url=answer.base_url,
            model_args={"api_key": MODEL_SENTINEL},
            model_roles={
                "grader": get_model(
                    "openai-api/airgap/judge",
                    base_url=judge.base_url,
                    memoize=False,
                    api_key=JUDGE_SENTINEL,
                )
            },
            log_dir=str(tmp_path),
            display="none",
        )
    assert log.status == "success", log.error
    assert answer.authorizations == [f"Bearer {MODEL_SENTINEL}"]
    assert judge.authorizations == [f"Bearer {JUDGE_SENTINEL}"]
    assert_no_credentials_logged(log, MODEL_SENTINEL)
    assert_no_credentials_logged(log, JUDGE_SENTINEL)


def credentialed_task_from_rows() -> Task:
    records = [{"id": "a", "question": "1 + 1?", "answer": "2"}]
    return Task(
        dataset=to_inspect_dataset(records, record_to_sample, dataset_id="rows"),
        solver=generate(),
        scorer=scorer(
            exclude_thinking_trace_for_judge=True,
            thinking_start_token="<think>",
            thinking_end_token="</think>",
        ),
    )


def test_hf_models_take_separate_tokens_through_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two hf/ models both read HF_TOKEN, so the environment cannot separate
    # them; Inspect's api_key argument passes each its own Hub token.
    hf = pytest.importorskip("inspect_ai.model._providers.hf")
    loads: list[tuple[str, str, str | None]] = []

    def loader(kind: str) -> type:
        def from_pretrained(source: str, **kwargs: Any) -> Any:
            loads.append((kind, source, kwargs.get("token")))
            return type("Loaded", (), {"eos_token": "</s>"})()

        return type(kind, (), {"from_pretrained": staticmethod(from_pretrained)})

    monkeypatch.setattr(hf, "AutoModelForCausalLM", loader("model"))
    monkeypatch.setattr(hf, "AutoTokenizer", loader("tokenizer"))
    monkeypatch.setenv(HF_CREDENTIAL_VAR, HF_SENTINEL)
    get_model("hf/org/answer", api_key="answer-token", memoize=False, device="cpu")
    get_model("hf/org/judge", api_key="judge-token", memoize=False, device="cpu")
    # Without api_key, no token is passed and huggingface_hub reads HF_TOKEN.
    get_model("hf/org/default", memoize=False, device="cpu")
    assert loads == [
        ("model", "org/answer", "answer-token"),
        ("tokenizer", "org/answer", "answer-token"),
        ("model", "org/judge", "judge-token"),
        ("tokenizer", "org/judge", "judge-token"),
        ("model", "org/default", None),
        ("tokenizer", "org/default", None),
    ]


def test_memoized_models_keep_their_first_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pins the caveat in INSPECT_MIGRATION.md, "Credentials": Inspect caches a
    # model given by name, and the provider read its environment key when
    # first built. A model built with memoize=False reads the new key.
    with (
        local_openai_judge("2") as answer,
        local_openai_judge("SCORE: 1") as judge,
    ):
        grader = get_model(
            "openai-api/airgap/judge",
            base_url=judge.base_url,
            api_key="judge-placeholder",
            memoize=False,
        )
        for key in ("first", "second", "third"):
            monkeypatch.setenv("CACHED_API_KEY", key)
            model: Any = "openai-api/cached/model"
            if key == "third":
                model = get_model(model, base_url=answer.base_url, memoize=False)
            [log] = inspect_eval(
                credentialed_task_from_rows(),
                model=model,
                model_base_url=answer.base_url,
                model_roles={"grader": grader},
                log_dir=str(tmp_path / key),
                display="none",
            )
            assert log.status == "success", log.error
    assert answer.authorizations == ["Bearer first", "Bearer first", "Bearer third"]


def test_api_key_models_are_memoized_per_key() -> None:
    # Inspect's memoization key includes api_key, so a model given its own
    # key never reuses a cached model built with another one.
    first = get_model("openai-api/keyed/model", base_url="http://x/v1", api_key="a")
    again = get_model("openai-api/keyed/model", base_url="http://x/v1", api_key="a")
    other = get_model("openai-api/keyed/model", base_url="http://x/v1", api_key="b")
    assert first is again
    assert other is not first
    assert (first.api.api_key, other.api.api_key) == ("a", "b")
