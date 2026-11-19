"""SEP dataset: the authors' published file at a pinned commit, SHA-256 checked.

SEP has no faithful Hugging Face copy, so it is read from the authors'
repository (``egozverev/Should-It-Be-Executed-Or-Processed``) at a fixed commit
and its bytes are checked against a pinned SHA-256 before anything is parsed.
The verified file is cached; a cached copy is checked again on every load and
downloaded afresh when it does not match. Every row is validated before Inspect
applies any limit. Rows keep their published order (no shuffle), so ``--limit
N`` takes the first N items.
"""

import hashlib
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

DATASET_REPOSITORY = "egozverev/Should-It-Be-Executed-Or-Processed"
DATASET_COMMIT = "7606c0696f20f5aa433169fd2221f76852d1d4f5"
"""Repository head on 2026-04-20; the file is unchanged since ``252cfa1`` (2024-06-03)."""
DATASET_FILE = "datasets/SEP_dataset.json"
DATASET_URL = (
    f"https://raw.githubusercontent.com/{DATASET_REPOSITORY}/{DATASET_COMMIT}/"
    f"{DATASET_FILE}"
)
DATASET_SHA256 = "9f81a52ba089073251793f8499fbb79bb4112e94cf48d31b7f8805e8b44fa3ce"
"""SHA-256 of the published file's bytes (12,089,182 bytes)."""
DATASET_LICENSE = "CC BY 4.0"
EXPECTED_ROWS = 9160

CACHE_DIR = INSPECT_EVALS_CACHE_PATH / "llm_behavior_eval" / "sep"
"""Where the verified file is cached (``INSPECT_EVALS_CACHE_DIR`` moves it)."""
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_TIMEOUT_SECONDS = 120

TEXT_FIELDS = (
    "system_prompt_clean",
    "prompt_instructed",
    "system_prompt_instructed",
    "prompt_clean",
    "witness",
)
"""Each row's string fields: the task prompt and data prompt with and without
the probe, and the probe's witness."""
INFO_FIELDS = frozenset(
    {
        "type",
        "subtask",
        "subtask_descr",
        "appended_task_id",
        "appended_type",
        "is_insistent",
    }
)
"""The row's ``info`` object: task domain, subtask, probe ID, probe placement
(``ll``, ``lr``, ``rl`` or ``rr``) and whether the probe is insistent."""
APPENDED_TYPES = frozenset({"ll", "lr", "rl", "rr"})


def sha256(content: bytes) -> str:
    """Hex SHA-256 of ``content``."""
    return hashlib.sha256(content).hexdigest()


def verify(content: bytes, source: str) -> bytes:
    """Return ``content`` if it matches ``DATASET_SHA256``.

    Raises:
        ValueError: The checksum differs.
    """
    found = sha256(content)
    if found != DATASET_SHA256:
        raise ValueError(
            f"SEP dataset at {source} does not match the pinned release; "
            f"expected SHA-256 {DATASET_SHA256}, found {found}"
        )
    return content


def download(url: str) -> bytes:
    """Fetch ``url``, retrying network errors with exponential backoff.

    Raises:
        OSError: Every attempt failed.
    """
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            # The URL is the pinned https constant above.
            with urllib.request.urlopen(  # noqa: S310
                url, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                return response.read()
        except OSError:
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def cached_path() -> Path:
    """The cache file for the pinned release (named by its checksum)."""
    return CACHE_DIR / f"SEP_dataset-{DATASET_SHA256[:16]}.json"


def read_pinned() -> bytes:
    """The pinned file's bytes, from the cache or downloaded and verified.

    A cached copy that fails the checksum is replaced. A download that fails
    it is not cached.

    Raises:
        ValueError: The downloaded file does not match the pinned checksum.
    """
    path = cached_path()
    if path.is_file():
        content = path.read_bytes()
        if sha256(content) == DATASET_SHA256:
            return content
    content = verify(download(DATASET_URL), DATASET_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(content)
    os.replace(handle.name, path)
    return content


def _require_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"SEP row field {field!r} must be a non-empty string")
    return value


def record_to_sample(record: dict[str, Any], index: int) -> Sample:
    """One SEP row as a sample with ID ``index`` (its position in the file).

    The input is the probe-in-instruction conversation (task prompt with the
    probe as the system message, clean data as the user message); the solver
    builds both conversations from metadata. The target is the witness.
    Metadata holds the four prompts, the witness and the ``info`` fields.

    Raises:
        ValueError: The row does not have exactly the published fields, or a
            field has the wrong type.
    """
    if set(record) != {*TEXT_FIELDS, "info"}:
        raise ValueError(
            f"SEP row fields must be exactly {sorted({*TEXT_FIELDS, 'info'})}; "
            f"found {sorted(record)}"
        )
    texts = {field: _require_text(record, field) for field in TEXT_FIELDS}
    info = record["info"]
    if not isinstance(info, dict) or set(info) != INFO_FIELDS:
        raise ValueError(f"SEP row info must have exactly {sorted(INFO_FIELDS)}")
    for field in ("type", "subtask", "subtask_descr"):
        _require_text(info, field)
    if info["appended_type"] not in APPENDED_TYPES:
        raise ValueError(f"SEP appended_type must be one of {sorted(APPENDED_TYPES)}")
    if not isinstance(info["is_insistent"], bool):
        raise ValueError("SEP is_insistent must be a boolean")
    if type(info["appended_task_id"]) is not int:
        raise ValueError("SEP appended_task_id must be an integer")
    return Sample(
        id=index,
        input=[
            ChatMessageSystem(content=texts["system_prompt_instructed"]),
            ChatMessageUser(content=texts["prompt_clean"]),
        ],
        target=texts["witness"],
        metadata={**texts, **info},
    )


def parse(content: bytes, source: str) -> list[Sample]:
    """Every row of the published JSON list, validated, in published order.

    Raises:
        ValueError: The file is not a JSON list of ``EXPECTED_ROWS`` valid rows.
    """
    rows = json.loads(content.decode("utf-8"))
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        found = len(rows) if isinstance(rows, list) else type(rows).__name__
        raise ValueError(
            f"SEP dataset at {source} must be a list of exactly {EXPECTED_ROWS} "
            f"rows; found {found}"
        )
    samples = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"SEP row {index} is not an object")
        samples.append(record_to_sample(row, index))
    return samples


def dataset(dataset_path: str | None = None) -> MemoryDataset:
    """The 9160 SEP samples in published order, checksum-verified and validated.

    Args:
        dataset_path: Read this local copy of ``datasets/SEP_dataset.json``
            instead of downloading it. It must be byte-identical to the pinned
            release (the same SHA-256).

    Returns:
        Samples with integer IDs (file positions) and the witness as target.

    Raises:
        ValueError: The checksum, the row count or a row is invalid.
    """
    if dataset_path is None:
        source = DATASET_URL
        content = read_pinned()
    else:
        source = dataset_path
        content = verify(Path(dataset_path).read_bytes(), dataset_path)
    return MemoryDataset(parse(content, source), name="sep", location=source)
