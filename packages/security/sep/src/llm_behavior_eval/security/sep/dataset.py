"""SEP dataset: the authors' published files at a pinned commit, SHA-256 checked.

SEP has no faithful Hugging Face copy, so it is read from the authors'
repository (``egozverev/Should-It-Be-Executed-Or-Processed``) at a fixed commit
and each file's bytes are checked against a pinned SHA-256 before anything is
parsed. A verified file is cached; a cached copy is checked again on every load
and downloaded afresh when it does not match. Every row is validated before
Inspect applies any limit. Rows keep their published order (no shuffle), so
``--limit N`` takes the first N items.

``sep`` reads only the dataset. The adaptations (``sep_tool`` and
``sep_instruction_control``) also read the probe list,
``SEP_dataset/source/probes.json``, from the same commit: each row's
``appended_task_id`` indexes it, and the probe's answer must be the row's
witness.
"""

import hashlib
import json
import os
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
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

PROBES_FILE = "SEP_dataset/source/probes.json"
PROBES_URL = (
    f"https://raw.githubusercontent.com/{DATASET_REPOSITORY}/{DATASET_COMMIT}/"
    f"{PROBES_FILE}"
)
PROBES_SHA256 = "adc7ab362c894ca868ad49137172a879d504714b093c30643e5ed0ae8d4228ed"
"""SHA-256 of the published probe list's bytes (11,630 bytes)."""
EXPECTED_PROBES = 100
PROBE_FIELDS = frozenset({"instruction", "answer"})
"""Each probe: the instruction inserted into the prompts and its witness."""

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

MessageBuilder = Callable[[dict[str, Any]], list[ChatMessage]]
"""Builds a sample's chat input from its metadata."""


def sha256(content: bytes) -> str:
    """Hex SHA-256 of ``content``."""
    return hashlib.sha256(content).hexdigest()


def verify(content: bytes, source: str, expected: str) -> bytes:
    """Return ``content`` if its SHA-256 is ``expected``.

    Raises:
        ValueError: The checksum differs.
    """
    found = sha256(content)
    if found != expected:
        raise ValueError(
            f"SEP file at {source} does not match the pinned release; "
            f"expected SHA-256 {expected}, found {found}"
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


def cached_path(url: str, expected: str) -> Path:
    """The cache file for the pinned file at ``url`` (named by its checksum)."""
    stem = Path(url).stem
    return CACHE_DIR / f"{stem}-{expected[:16]}.json"


def read_pinned(url: str, expected: str) -> bytes:
    """The bytes of the file at ``url``, from the cache or downloaded and verified.

    A cached copy that fails the checksum is replaced. A download that fails
    it is not cached.

    Raises:
        ValueError: The downloaded file does not match ``expected``.
    """
    path = cached_path(url, expected)
    if path.is_file():
        content = path.read_bytes()
        if sha256(content) == expected:
            return content
    content = verify(download(url), url, expected)
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


def validate_row(record: dict[str, Any]) -> dict[str, Any]:
    """The row's four prompts, witness and ``info`` fields, flattened.

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
    return {**texts, **info}


def record_to_sample(record: dict[str, Any], index: int) -> Sample:
    """One SEP row as a ``sep`` sample with ID ``index`` (its position in the file).

    The input is the probe-in-instruction conversation (task prompt with the
    probe as the system message, clean data as the user message); the solver
    builds both conversations from metadata. The target is the witness.
    Metadata holds the four prompts, the witness and the ``info`` fields.

    Raises:
        ValueError: The row does not have exactly the published fields, or a
            field has the wrong type.
    """
    return _paired_sample(validate_row(record), index)


def _paired_sample(metadata: dict[str, Any], index: int) -> Sample:
    return Sample(
        id=index,
        input=[
            ChatMessageSystem(content=metadata["system_prompt_instructed"]),
            ChatMessageUser(content=metadata["prompt_clean"]),
        ],
        target=metadata["witness"],
        metadata=metadata,
    )


def _load_json(content: bytes) -> Any:
    return json.loads(content.decode("utf-8"))


def parse_rows(content: bytes, source: str) -> list[dict[str, Any]]:
    """Every row of the published JSON list, validated, in published order.

    Returns:
        Each row's ``validate_row`` metadata.

    Raises:
        ValueError: The file is not a JSON list of ``EXPECTED_ROWS`` valid rows.
    """
    rows = _load_json(content)
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        found = len(rows) if isinstance(rows, list) else type(rows).__name__
        raise ValueError(
            f"SEP dataset at {source} must be a list of exactly {EXPECTED_ROWS} "
            f"rows; found {found}"
        )
    validated = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"SEP row {index} is not an object")
        validated.append(validate_row(row))
    return validated


def parse(content: bytes, source: str) -> list[Sample]:
    """Every row of the published JSON list as a ``sep`` sample, in published order.

    Raises:
        ValueError: The file is not a JSON list of ``EXPECTED_ROWS`` valid rows.
    """
    rows = parse_rows(content, source)
    return [_paired_sample(row, index) for index, row in enumerate(rows)]


def parse_probes(content: bytes, source: str) -> list[dict[str, str]]:
    """The published probe list, validated, in published order.

    Raises:
        ValueError: The file is not a JSON list of ``EXPECTED_PROBES`` objects
            with exactly a non-empty ``instruction`` and ``answer``.
    """
    probes = _load_json(content)
    if not isinstance(probes, list) or len(probes) != EXPECTED_PROBES:
        found = len(probes) if isinstance(probes, list) else type(probes).__name__
        raise ValueError(
            f"SEP probes at {source} must be a list of exactly {EXPECTED_PROBES} "
            f"probes; found {found}"
        )
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict) or set(probe) != PROBE_FIELDS:
            raise ValueError(
                f"SEP probe {index} must be an object with exactly "
                f"{sorted(PROBE_FIELDS)}"
            )
        for field in PROBE_FIELDS:
            _require_text(probe, field)
    return probes


def probe_for(row: dict[str, Any], probes: list[dict[str, str]], index: int) -> str:
    """The probe instruction of ``row`` (validated metadata), checked against it.

    ``appended_task_id`` must index ``probes``, the probe's answer must be the
    row's witness, and its instruction must appear in both instructed prompts.

    Raises:
        ValueError: The row and the probe list disagree.
    """
    probe_id = row["appended_task_id"]
    if not 0 <= probe_id < len(probes):
        raise ValueError(
            f"SEP row {index}: appended_task_id {probe_id} is not a probe index"
        )
    probe = probes[probe_id]
    if probe["answer"] != row["witness"]:
        raise ValueError(
            f"SEP row {index}: probe {probe_id}'s answer {probe['answer']!r} is "
            f"not the row's witness {row['witness']!r}"
        )
    instruction = probe["instruction"]
    for field in ("prompt_instructed", "system_prompt_instructed"):
        if instruction not in row[field]:
            raise ValueError(
                f"SEP row {index}: probe {probe_id}'s instruction is not in {field}"
            )
    return instruction


def _read(path: str | None, url: str, expected: str) -> tuple[bytes, str]:
    """A pinned file's verified bytes and where they came from."""
    if path is None:
        return read_pinned(url, expected), url
    return verify(Path(path).read_bytes(), path, expected), path


def dataset(dataset_path: str | None = None) -> MemoryDataset:
    """The 9160 ``sep`` samples in published order, checksum-verified and validated.

    Args:
        dataset_path: Read this local copy of ``datasets/SEP_dataset.json``
            instead of downloading it. It must be byte-identical to the pinned
            release (the same SHA-256).

    Returns:
        Samples with integer IDs (file positions) and the witness as target.

    Raises:
        ValueError: The checksum, the row count or a row is invalid.
    """
    content, source = _read(dataset_path, DATASET_URL, DATASET_SHA256)
    return MemoryDataset(parse(content, source), name="sep", location=source)


Variant = Literal["tool", "instruction_control"]


def variant_dataset(
    variant: Variant,
    messages: MessageBuilder,
    dataset_path: str | None = None,
    probes_path: str | None = None,
) -> MemoryDataset:
    """SEP rows with their probe, for the single-conversation adaptations.

    Args:
        variant: The task (``tool`` or ``instruction_control``), for the
            dataset name.
        messages: Builds a sample's input from its metadata.
        dataset_path: A local copy of ``datasets/SEP_dataset.json`` (checked
            against the pinned SHA-256).
        probes_path: A local copy of ``SEP_dataset/source/probes.json``
            (checked against the pinned SHA-256).

    Returns:
        Samples with integer IDs (file positions), the witness as target and
        the row's fields plus ``probe`` (the probe instruction) as metadata.

    Raises:
        ValueError: A checksum, a count, a row, a probe, or the agreement
            between a row and its probe is invalid.
    """
    content, source = _read(dataset_path, DATASET_URL, DATASET_SHA256)
    probe_content, probe_source = _read(probes_path, PROBES_URL, PROBES_SHA256)
    rows = parse_rows(content, source)
    probes = parse_probes(probe_content, probe_source)
    samples = []
    for index, row in enumerate(rows):
        metadata = {**row, "probe": probe_for(row, probes, index)}
        samples.append(
            Sample(
                id=index,
                input=messages(metadata),
                target=row["witness"],
                metadata=metadata,
            )
        )
    return MemoryDataset(samples, name=f"sep_{variant}", location=source)
