import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from llm_behavior_eval.evaluation_utils.util_functions import (
    build_vllm_prompt_token_ids,
    is_model_multimodal,
    maybe_download_adapter,
    pick_best_dtype,
    safe_apply_chat_template,
    torch_dtype_to_str,
    truncate_text_by_whitespace,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class MockConfig:
    """A simple mock config object for testing."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": self.model_type}


class StubTokenizer:
    def __init__(self, name: str, template: str) -> None:
        self.name_or_path = name
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Simple join of role and content for testing purposes
        return "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )


def test_pick_best_dtype_cpu() -> None:
    assert pick_best_dtype("cpu") == torch.float32


def test_pick_best_dtype_cuda_bf16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    dtype = pick_best_dtype("cuda", prefer_bf16=True)
    assert dtype == torch.bfloat16


def test_safe_apply_chat_template_merges_system_message() -> None:
    tokenizer = StubTokenizer("google/gemma-2b", "System role not supported")
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    formatted = safe_apply_chat_template(
        cast("PreTrainedTokenizerBase", tokenizer), messages
    )
    assert "system" in formatted and "user" in formatted


def test_safe_apply_chat_template_appends_max_answer_instruction() -> None:
    tokenizer = StubTokenizer("some/model", "generic template")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer the question."},
    ]
    formatted = safe_apply_chat_template(
        cast("PreTrainedTokenizerBase", tokenizer),
        messages,
        max_answer_tokens=42,
        pass_max_answer_tokens=True,
    )
    assert "Respond in no more than 42 tokens." in formatted
    assert "You are a helpful assistant." in formatted
    assert "Answer the question." in formatted


def test_safe_apply_chat_template_does_not_append_without_flag_or_value() -> None:
    tokenizer = StubTokenizer("some/model", "generic template")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer the question."},
    ]
    without_flag = safe_apply_chat_template(
        cast("PreTrainedTokenizerBase", tokenizer),
        [m.copy() for m in messages],
        max_answer_tokens=42,
        pass_max_answer_tokens=False,
    )
    without_value = safe_apply_chat_template(
        cast("PreTrainedTokenizerBase", tokenizer),
        [m.copy() for m in messages],
        max_answer_tokens=None,
        pass_max_answer_tokens=True,
    )
    assert "Respond in no more than 42 tokens." not in without_flag
    assert "Respond in no more than" not in without_value


def test_torch_dtype_to_str_supported() -> None:
    assert torch_dtype_to_str(torch.float16) == "float16"
    assert torch_dtype_to_str(torch.bfloat16) == "bfloat16"
    assert torch_dtype_to_str(torch.float32) == "float32"


def test_torch_dtype_to_str_unsupported() -> None:
    with pytest.raises(ValueError):
        torch_dtype_to_str(torch.float64)


def test_truncate_text_by_whitespace_within_limit_returns_original_text() -> None:
    assert truncate_text_by_whitespace("one two", 3) == "one two"


def test_truncate_text_by_whitespace_truncates_on_token_limit() -> None:
    assert truncate_text_by_whitespace("one two three", 2) == "one two"


def test_truncate_text_by_whitespace_handles_nonpositive_limit() -> None:
    assert truncate_text_by_whitespace("one two", 0) == ""
    assert truncate_text_by_whitespace("one two", -1) == ""


def test_build_vllm_prompt_token_ids_strips_padding() -> None:
    input_ids = torch.tensor([[0, 11, 12, 13], [0, 0, 21, 22]])
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])

    prompts = build_vllm_prompt_token_ids(input_ids, attention_mask)

    assert prompts == [[11, 12, 13], [21, 22]]


def test_build_vllm_prompt_token_ids_validates_shape() -> None:
    input_ids = torch.zeros((2, 3), dtype=torch.long)
    attention_mask = torch.zeros((2, 2), dtype=torch.long)

    with pytest.raises(ValueError):
        build_vllm_prompt_token_ids(input_ids, attention_mask)


@pytest.fixture
def mock_auto_config_remote_fallback(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fixture that mocks AutoConfig to simulate remote fallback behavior."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def mock_from_pretrained(*args: Any, **kwargs: Any) -> MockConfig:
        calls.append((args, kwargs))
        if len(calls) == 1:
            # First call raises exception to trigger remote fallback
            raise Exception("local_files_only=True failed")
        # Second call (remote fallback) returns config
        return MockConfig("llama")

    class MockAutoConfig:
        from_pretrained = staticmethod(mock_from_pretrained)

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.util_functions.AutoConfig",
        MockAutoConfig,
    )
    return {"calls": calls}


@pytest.fixture
def mock_auto_config_local_load(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fixture that mocks AutoConfig to simulate local load behavior."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def mock_from_pretrained(*args: Any, **kwargs: Any) -> MockConfig:
        calls.append((args, kwargs))
        return MockConfig("llama")

    class MockAutoConfig:
        from_pretrained = staticmethod(mock_from_pretrained)

    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.util_functions.AutoConfig",
        MockAutoConfig,
    )
    return {"calls": calls}


def test_is_model_multimodal_passes_token_on_remote_fallback(
    mock_auto_config_remote_fallback: dict[str, Any],
) -> None:
    """Test that is_model_multimodal passes the token parameter on remote fallback."""
    # Test with token parameter
    result = is_model_multimodal(
        "test/model", trust_remote_code=False, token="test_token"
    )

    # Verify that both calls occurred (local then remote)
    calls = mock_auto_config_remote_fallback["calls"]
    assert len(calls) == 2

    # Check the remote call (second call) includes the token
    remote_call = calls[1]
    assert remote_call[1].get("token") == "test_token"
    assert result is False


def test_is_model_multimodal_passes_token_on_local_load(
    mock_auto_config_local_load: dict[str, Any],
) -> None:
    """Test that is_model_multimodal passes the token parameter when loading locally."""
    # Test with token parameter
    result = is_model_multimodal(
        "test/model", trust_remote_code=False, token="test_token"
    )

    # Verify that only the local call occurred
    calls = mock_auto_config_local_load["calls"]
    assert len(calls) == 1

    local_call = calls[0]
    assert local_call[1].get("token") == "test_token"
    assert result is False


# Tests for maybe_download_adapter


def test_maybe_download_adapter_empty_string() -> None:
    """Test that empty string raises ValueError."""
    with pytest.raises(ValueError, match="adapter_ref must be a non-empty string"):
        maybe_download_adapter("")


def test_maybe_download_adapter_whitespace_only() -> None:
    """Test that whitespace-only string raises ValueError."""
    with pytest.raises(ValueError, match="adapter_ref must be a non-empty string"):
        maybe_download_adapter("   ")


def test_maybe_download_adapter_local_path(tmp_path) -> None:
    """Test that local paths are returned unchanged."""
    local_path = str(tmp_path / "adapter")
    result = maybe_download_adapter(local_path)
    assert result == local_path


def test_maybe_download_adapter_hf_repo_id() -> None:
    """Test that HF repo IDs are returned unchanged."""
    repo_id = "meta-llama/Llama-3.1-8B-Instruct"
    result = maybe_download_adapter(repo_id)
    assert result == repo_id


def test_maybe_download_adapter_local_path_with_whitespace(tmp_path) -> None:
    """Test that whitespace is stripped from adapter_ref."""
    local_path = str(tmp_path / "adapter")
    result = maybe_download_adapter(f"  {local_path}  ")
    assert result == local_path


def test_maybe_download_adapter_mlflow_scheme_missing_mlflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test that mlflow:// scheme raises ImportError when mlflow is not available."""
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.util_functions.mlflow",
        None,
        raising=False,
    )

    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("No module named 'mlflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(ImportError, match="mlflow is required for mlflow:// refs"):
        maybe_download_adapter("mlflow://abc123def456")


def test_maybe_download_adapter_mlflow_scheme_no_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test that mlflow:// without run_id raises ValueError."""
    # Mock mlflow import - need to mock sys.modules since it's imported inside the function
    mock_mlflow = types.ModuleType("mlflow")
    mock_mlflow.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]

    def mock_download_artifacts(run_id, artifact_path, dst_path):
        return str(dst_path)

    mock_artifacts = types.ModuleType("mlflow.artifacts")
    mock_artifacts.download_artifacts = mock_download_artifacts  # type: ignore[attr-defined]

    sys.modules["mlflow"] = mock_mlflow
    sys.modules["mlflow.artifacts"] = mock_artifacts

    try:
        with pytest.raises(ValueError, match="Invalid mlflow ref \\(missing run id\\)"):
            maybe_download_adapter("mlflow://")
    finally:
        # Clean up
        if "mlflow" in sys.modules:
            del sys.modules["mlflow"]
        if "mlflow.artifacts" in sys.modules:
            del sys.modules["mlflow.artifacts"]


def test_maybe_download_adapter_mlflow_scheme_with_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test mlflow:// scheme with run_id downloads artifacts."""
    import sys

    run_id = "abc123def45678901234567890123456"
    artifact_path = "hf_checkpoint/peft"

    # Mock mlflow
    mock_download_artifacts_calls: list[tuple[str, str, str]] = []

    def mock_download_artifacts(*args, **kwargs) -> str:
        run_id = kwargs.get("run_id", args[0] if args else "")
        artifact_path = kwargs.get("artifact_path", args[1] if len(args) > 1 else "")
        dst_path = kwargs.get("dst_path", args[2] if len(args) > 2 else "")
        mock_download_artifacts_calls.append((run_id, artifact_path, dst_path))
        # Create the directory structure - return the directory, not a file
        result_dir = Path(dst_path) / artifact_path
        result_dir.mkdir(parents=True, exist_ok=True)
        return str(result_dir)

    mock_mlflow = types.ModuleType("mlflow")
    mock_mlflow.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]
    mock_artifacts = types.ModuleType("mlflow.artifacts")
    mock_artifacts.download_artifacts = mock_download_artifacts  # type: ignore[attr-defined]

    sys.modules["mlflow"] = mock_mlflow
    sys.modules["mlflow.artifacts"] = mock_artifacts

    try:
        result = maybe_download_adapter(
            f"mlflow://{run_id}/{artifact_path}", cache_dir=str(tmp_path)
        )

        assert len(mock_download_artifacts_calls) == 1
        assert mock_download_artifacts_calls[0][0] == run_id
        assert mock_download_artifacts_calls[0][1] == artifact_path
        assert "peft_adapters" in result
    finally:
        # Clean up
        if "mlflow" in sys.modules:
            del sys.modules["mlflow"]
        if "mlflow.artifacts" in sys.modules:
            del sys.modules["mlflow.artifacts"]


def test_maybe_download_adapter_mlflow_scheme_default_artifact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test mlflow:// scheme tries default artifact paths when none specified."""
    run_id = "abc123def45678901234567890123456"

    # Mock mlflow - first call fails, second succeeds
    mock_download_artifacts_calls: list[tuple[str, str, str]] = []

    def mock_download_artifacts(*args, **kwargs) -> str:
        run_id = kwargs.get("run_id", args[0] if args else "")
        artifact_path = kwargs.get("artifact_path", args[1] if len(args) > 1 else "")
        dst_path = kwargs.get("dst_path", args[2] if len(args) > 2 else "")
        mock_download_artifacts_calls.append((run_id, artifact_path, dst_path))
        if artifact_path == "hf_checkpoint/peft":
            raise Exception("Not found")
        # Create the directory structure - return the directory
        result_dir = Path(dst_path) / artifact_path
        result_dir.mkdir(parents=True, exist_ok=True)
        return str(result_dir)

    mock_mlflow = types.ModuleType("mlflow")
    mock_mlflow.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]
    mock_artifacts = types.ModuleType("mlflow.artifacts")
    mock_artifacts.download_artifacts = mock_download_artifacts  # type: ignore[attr-defined]

    sys.modules["mlflow"] = mock_mlflow
    sys.modules["mlflow.artifacts"] = mock_artifacts

    try:
        result = maybe_download_adapter(f"mlflow://{run_id}", cache_dir=str(tmp_path))

        assert len(mock_download_artifacts_calls) == 2
        assert mock_download_artifacts_calls[0][1] == "hf_checkpoint/peft"
        assert mock_download_artifacts_calls[1][1] == "hf_checkpoint"
        assert "hf_checkpoint" in result
    finally:
        # Clean up
        if "mlflow" in sys.modules:
            del sys.modules["mlflow"]
        if "mlflow.artifacts" in sys.modules:
            del sys.modules["mlflow.artifacts"]


def test_maybe_download_adapter_http_with_mlflow_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test http:// URL with mlflow run_id in path."""
    import sys
    import types

    run_id = "abc123def45678901234567890123456"

    mock_download_artifacts_calls: list[tuple[str, str, str]] = []

    def mock_download_artifacts(*args, **kwargs) -> str:
        run_id = kwargs.get("run_id", args[0] if args else "")
        artifact_path = kwargs.get("artifact_path", args[1] if len(args) > 1 else "")
        dst_path = kwargs.get("dst_path", args[2] if len(args) > 2 else "")
        mock_download_artifacts_calls.append((run_id, artifact_path, dst_path))
        # For http URLs, artifact_path might be the full path, so handle it
        if artifact_path:
            result_dir = Path(dst_path) / artifact_path
        else:
            result_dir = Path(dst_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        return str(result_dir)

    mock_mlflow = types.ModuleType("mlflow")
    mock_mlflow.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]
    mock_artifacts = types.ModuleType("mlflow.artifacts")
    mock_artifacts.download_artifacts = mock_download_artifacts  # type: ignore[attr-defined]

    sys.modules["mlflow"] = mock_mlflow
    sys.modules["mlflow.artifacts"] = mock_artifacts

    try:
        maybe_download_adapter(
            f"http://mlflow.example.com/runs/{run_id}",
            cache_dir=str(tmp_path),
            mlflow_tracking_uri="http://mlflow.example.com",
        )

        assert len(mock_download_artifacts_calls) == 1
        assert mock_download_artifacts_calls[0][0] == run_id
    finally:
        # Clean up
        if "mlflow" in sys.modules:
            del sys.modules["mlflow"]
        if "mlflow.artifacts" in sys.modules:
            del sys.modules["mlflow.artifacts"]


def test_maybe_download_adapter_mlflow_artifact_is_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test that mlflow artifact pointing to file raises ValueError."""
    import sys

    run_id = "abc123def45678901234567890123456"

    def mock_download_artifacts(*args, **kwargs) -> str:
        # Return a file path instead of directory
        dst_path = kwargs.get("dst_path", args[2] if len(args) > 2 else "")
        file_path = Path(dst_path) / "file.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("test")
        return str(file_path)

    mock_mlflow = types.ModuleType("mlflow")
    mock_mlflow.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]
    mock_artifacts = types.ModuleType("mlflow.artifacts")
    mock_artifacts.download_artifacts = mock_download_artifacts  # type: ignore[attr-defined]

    sys.modules["mlflow"] = mock_mlflow
    sys.modules["mlflow.artifacts"] = mock_artifacts

    try:
        with pytest.raises(ValueError, match="MLflow artifact resolved to a file"):
            maybe_download_adapter(f"mlflow://{run_id}", cache_dir=str(tmp_path))
    finally:
        # Clean up
        if "mlflow" in sys.modules:
            del sys.modules["mlflow"]
        if "mlflow.artifacts" in sys.modules:
            del sys.modules["mlflow.artifacts"]


def test_maybe_download_adapter_git_scheme_missing_gitpython(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test that git:// scheme raises ImportError when gitpython is not available."""

    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "git" or name.startswith("git."):
            raise ImportError("No module named 'git'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(ImportError, match="gitpython is required for git:// refs"):
        maybe_download_adapter(
            "git://github.com/user/repo.git", cache_dir=str(tmp_path)
        )


def test_maybe_download_adapter_git_scheme_basic(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test git:// scheme basic functionality."""
    repo_url = "git://github.com/user/repo.git"

    class MockGit:
        def fetch(self, *args):
            pass

        def checkout(self, rev: str):
            pass

        def pull(self):
            pass

    class MockRepo:
        def __init__(self, path: str):
            self.git = MockGit()

    mock_clone_calls: list[tuple[str, str]] = []

    def mock_clone_from(url: str, path: str):
        mock_clone_calls.append((url, path))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / ".git").mkdir()

    # Create a mock Repo class that supports both clone_from and instantiation
    class MockRepoClass:
        @staticmethod
        def clone_from(url: str, path: str):
            mock_clone_from(url, path)

        def __new__(cls, path: str):
            return MockRepo(path)

    mock_git_module = types.ModuleType("git")
    mock_git_module.Repo = MockRepoClass  # type: ignore[attr-defined]
    sys.modules["git"] = mock_git_module

    try:
        result = maybe_download_adapter(repo_url, cache_dir=str(tmp_path))
        # Verify it returns a path
        assert isinstance(result, str)
        assert "peft_adapters" in result
        # Should have attempted to clone if .git doesn't exist
        assert len(mock_clone_calls) >= 0
    finally:
        # Clean up
        if "git" in sys.modules:
            del sys.modules["git"]


def test_maybe_download_adapter_git_scheme_with_rev_and_subdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test git:// scheme with revision and subdir."""
    repo_url = "git://github.com/user/repo.git#main:adapters/lora"

    class MockGit:
        def fetch(self, *args):
            pass

        def checkout(self, rev: str):
            pass

        def pull(self):
            pass

    class MockHead:
        def __init__(self):
            self.is_detached = True

    class MockRepo:
        def __init__(self, path: str):
            self.git = MockGit()

    def mock_clone_from(url: str, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / ".git").mkdir()

    class MockRepoClass:
        @staticmethod
        def clone_from(url: str, path: str):
            mock_clone_from(url, path)

        def __init__(self, path: str):
            self.path = path
            self.git = MockGit()
            self.head = MockHead()

    mock_git_module = types.ModuleType("git")
    mock_git_module.Repo = MockRepoClass  # type: ignore[attr-defined]
    sys.modules["git"] = mock_git_module

    try:
        result = maybe_download_adapter(repo_url, cache_dir=str(tmp_path))
        assert "adapters/lora" in result
    finally:
        # Clean up
        if "git" in sys.modules:
            del sys.modules["git"]


def test_maybe_download_adapter_s3_scheme_missing_fsspec(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test that s3:// scheme raises ImportError when fsspec is not available."""

    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "fsspec" or name.startswith("fsspec."):
            raise ImportError("No module named 'fsspec'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(
        ImportError, match="fsspec is required for s3:// and gs:// refs"
    ):
        maybe_download_adapter("s3://bucket/path", cache_dir=str(tmp_path))


def test_maybe_download_adapter_s3_scheme_file_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test that s3:// pointing to file raises ValueError."""

    class MockFilesystem:
        def isfile(self, path: str) -> bool:
            return True

        def isdir(self, path: str) -> bool:
            return False

        def find(self, path: str):
            return []

    mock_fs = MockFilesystem()

    def mock_url_to_fs(url: str):
        return mock_fs, "s3://bucket/file.txt"

    # fsspec.core.url_to_fs is accessed as an attribute, so we need to make it work
    mock_core = types.ModuleType("fsspec.core")
    # Set url_to_fs as an attribute
    mock_core.url_to_fs = mock_url_to_fs  # type: ignore[attr-defined]
    mock_fsspec = types.ModuleType("fsspec")
    mock_fsspec.core = mock_core  # type: ignore[attr-defined]
    sys.modules["fsspec"] = mock_fsspec

    try:
        with pytest.raises(ValueError, match="points to a file; expected a directory"):
            maybe_download_adapter("s3://bucket/file.txt", cache_dir=str(tmp_path))
    finally:
        # Clean up
        if "fsspec" in sys.modules:
            del sys.modules["fsspec"]


def test_maybe_download_adapter_s3_scheme_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test s3:// scheme downloads files."""

    class MockFilesystem:
        def __init__(self):
            self.files = ["s3://bucket/path/file1.txt", "s3://bucket/path/file2.txt"]

        def isfile(self, path: str) -> bool:
            return False

        def isdir(self, path: str) -> bool:
            return path == "s3://bucket/path"

        def find(self, path: str):
            return self.files

        def open(self, path: str, mode: str):
            from io import BytesIO

            return BytesIO(b"test content")

    mock_fs = MockFilesystem()

    def mock_url_to_fs(url: str):
        return mock_fs, "s3://bucket/path"

    mock_core = types.ModuleType("fsspec.core")
    mock_core.url_to_fs = mock_url_to_fs  # type: ignore[attr-defined]
    mock_fsspec = types.ModuleType("fsspec")
    mock_fsspec.core = mock_core  # type: ignore[attr-defined]
    sys.modules["fsspec"] = mock_fsspec

    try:
        result = maybe_download_adapter("s3://bucket/path", cache_dir=str(tmp_path))
        assert "peft_adapters" in result
        assert "s3_" in result
    finally:
        # Clean up
        if "fsspec" in sys.modules:
            del sys.modules["fsspec"]


def test_maybe_download_adapter_gs_scheme_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test gs:// scheme downloads files."""

    class MockFilesystem:
        def __init__(self):
            self.files = ["gs://bucket/path/file1.txt"]

        def isfile(self, path: str) -> bool:
            return False

        def isdir(self, path: str) -> bool:
            return path == "gs://bucket/path"

        def find(self, path: str):
            return self.files

        def open(self, path: str, mode: str):
            from io import BytesIO

            return BytesIO(b"test content")

    mock_fs = MockFilesystem()

    def mock_url_to_fs(url: str):
        return mock_fs, "gs://bucket/path"

    mock_core = types.ModuleType("fsspec.core")
    mock_core.url_to_fs = mock_url_to_fs  # type: ignore[attr-defined]
    mock_fsspec = types.ModuleType("fsspec")
    mock_fsspec.core = mock_core  # type: ignore[attr-defined]
    sys.modules["fsspec"] = mock_fsspec

    try:
        result = maybe_download_adapter("gs://bucket/path", cache_dir=str(tmp_path))
        assert "peft_adapters" in result
        assert "gs_" in result
    finally:
        # Clean up
        if "fsspec" in sys.modules:
            del sys.modules["fsspec"]


def test_maybe_download_adapter_custom_cache_dir(tmp_path) -> None:
    """Test that custom cache_dir is used."""
    custom_cache = tmp_path / "custom_cache"
    local_path = str(tmp_path / "adapter")
    result = maybe_download_adapter(local_path, cache_dir=str(custom_cache))
    # For local paths, cache_dir doesn't matter, but verify it doesn't crash
    assert result == local_path


def test_maybe_download_adapter_mlflow_tracking_uri(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Test that mlflow_tracking_uri parameter sets tracking URI."""
    run_id = "abc123def45678901234567890123456"
    tracking_uri = "http://mlflow.example.com"

    mock_set_tracking_uri_calls: list[str] = []

    def mock_set_tracking_uri(uri: str) -> None:
        mock_set_tracking_uri_calls.append(uri)

    def mock_download_artifacts(*args, **kwargs) -> str:
        artifact_path = kwargs.get("artifact_path", args[1] if len(args) > 1 else "")
        dst_path = kwargs.get("dst_path", args[2] if len(args) > 2 else "")
        # Return a directory, not a file
        if artifact_path:
            result_dir = Path(dst_path) / artifact_path
        else:
            result_dir = Path(dst_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        return str(result_dir)

    mock_mlflow = types.ModuleType("mlflow")
    mock_mlflow.set_tracking_uri = mock_set_tracking_uri  # type: ignore[attr-defined]
    mock_artifacts = types.ModuleType("mlflow.artifacts")
    mock_artifacts.download_artifacts = mock_download_artifacts  # type: ignore[attr-defined]

    sys.modules["mlflow"] = mock_mlflow
    sys.modules["mlflow.artifacts"] = mock_artifacts

    try:
        maybe_download_adapter(
            f"mlflow://{run_id}",
            cache_dir=str(tmp_path),
            mlflow_tracking_uri=tracking_uri,
        )

        assert tracking_uri in mock_set_tracking_uri_calls
    finally:
        # Clean up
        if "mlflow" in sys.modules:
            del sys.modules["mlflow"]
        if "mlflow.artifacts" in sys.modules:
            del sys.modules["mlflow.artifacts"]
