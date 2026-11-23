"""BIPIA test data: pinned sources, checksums, validation and sample building.

The context and attack files are read from microsoft/BIPIA at a fixed commit
(``REPOSITORY_COMMIT``) and checked against SHA-256 digests before use. Web QA
(``qa``, NewsQA) and Summarization (``abstract``, XSum) contexts are not in the
repository for licence reasons: build them with the reference's
``benchmark/{qa,abstract}/process.py`` and pass the ``benchmark`` folder as
``dataset_path``; they are checked against the reference's own MD5 digests
(``benchmark/{qa,abstract}/md5.txt``).

Samples are every (position, context, attack) combination, built in the
reference builder's order (``bipia/data/base.py``, ``construct_samples``):
positions ``end``, ``start``, ``middle``; contexts in file order; attacks in
file order with ``-<index>`` suffixes.
"""

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

from .attacks import (
    ATTACK_GROUPS,
    ATTACK_VARIANTS,
    CODE_ATTACK_CATEGORIES,
    TEXT_ATTACK_CATEGORIES,
    attack_category,
    evaluator,
)
from .prompts import (
    Position,
    PromptMode,
    TaskName,
    insert_attack,
    middle_offset,
    task_messages,
)

REPOSITORY = "microsoft/BIPIA"
REPOSITORY_COMMIT = "a004b69ec0dd446e0afd461d98cb5e96e120a5d0"
"""The repository's HEAD (2024-04-15); ``benchmark/`` last changed in bc0cc5d."""
RAW_URL = "https://raw.githubusercontent.com/{repository}/{commit}/benchmark/{path}"
CACHE_DIR = INSPECT_EVALS_CACHE_PATH / "llm_behavior_eval_bipia" / REPOSITORY_COMMIT

PUBLIC_TASKS: tuple[TaskName, ...] = ("email", "table", "code")
"""Tasks whose contexts are in the repository, loaded by default."""
LOCAL_ONLY_TASKS: tuple[TaskName, ...] = ("qa", "abstract")
"""Tasks whose contexts must be built locally (NewsQA, XSum terms of use)."""

TEXT_ATTACKS = "text_attack_test.json"
CODE_ATTACKS = "code_attack_test.json"


@dataclass(frozen=True)
class SourceFile:
    """One file of the ``benchmark`` folder and its expected digest."""

    path: str
    """Path relative to ``benchmark/``."""
    digest: str
    """Hex digest of the file's bytes."""
    algorithm: str = "sha256"
    """``sha256`` for pinned files, ``md5`` for the reference's ``md5.txt``."""
    rows: int | None = None
    """Context rows (JSON lines); ``None`` for an attack file."""


SOURCES: dict[str, SourceFile] = {
    "email": SourceFile(
        "email/test.jsonl",
        "217b403faaa1d0cb12c24892bea39f4a0b9e2819919ac7e2b902006e28278cdb",
        rows=50,
    ),
    "table": SourceFile(
        "table/test.jsonl",
        "3d5eaab192b80bada762e3cd7655583a14e5888ba8b845a01f0434e4ccc4291e",
        rows=100,
    ),
    "code": SourceFile(
        "code/test.jsonl",
        "ed40b84f541352fb45753032a80750922de8d5e0b70997924df3d04cf942e058",
        rows=50,
    ),
    "qa": SourceFile(
        "qa/test.jsonl", "1e973cc21ef6f0284bf5e7b509a60a1b", "md5", rows=100
    ),
    "abstract": SourceFile(
        "abstract/test.jsonl", "240d0759011d310134f373a7a4af6329", "md5", rows=100
    ),
    TEXT_ATTACKS: SourceFile(
        TEXT_ATTACKS,
        "75750e7b4e8b34e8f9d88d89b357aeaaf02bd07f9e493ccd37eda74a0cd7c7f8",
    ),
    CODE_ATTACKS: SourceFile(
        CODE_ATTACKS,
        "892545c5aaec0645b1ded65dc7816b3d70e9ef4eadcba2301a7a3db93676b6e0",
    ),
}
"""Every test file; ``qa`` and ``abstract`` digests are the reference's MD5s."""

PROMPT_FIELDS: dict[TaskName, tuple[str, ...]] = {
    "email": ("context", "question"),
    "qa": ("context", "question"),
    "table": ("context", "question"),
    "abstract": ("context",),
    "code": ("context", "error", "code"),
}
"""Fields each task's builder reads (``ideal`` is not needed for attack success)."""

DOWNLOAD_ATTEMPTS = 4
TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


def digest(data: bytes, algorithm: str) -> str:
    """Hex digest of ``data`` (MD5 only to match the reference's ``md5.txt``)."""
    if algorithm == "md5":
        return hashlib.md5(data, usedforsecurity=False).hexdigest()
    return hashlib.new(algorithm, data).hexdigest()


def verify(source: SourceFile, data: bytes, origin: str) -> bytes:
    """Return ``data`` if its digest is the pinned one.

    Raises:
        ValueError: The digest differs.
    """
    found = digest(data, source.algorithm)
    if found != source.digest:
        raise ValueError(
            f"BIPIA file {origin} does not match the pinned release: expected "
            f"{source.algorithm} {source.digest}, found {found}"
        )
    return data


def source_url(source: SourceFile) -> str:
    """The raw GitHub URL of a pinned file at ``REPOSITORY_COMMIT``."""
    return RAW_URL.format(
        repository=REPOSITORY, commit=REPOSITORY_COMMIT, path=source.path
    )


def download(url: str) -> bytes:
    """Fetch ``url``, retrying network errors and transient HTTP statuses."""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (fixed https URL)
                return response.read()
        except urllib.error.HTTPError as error:
            if (
                error.code not in TRANSIENT_HTTP_CODES
                or attempt == DOWNLOAD_ATTEMPTS - 1
            ):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def read_source(name: str, dataset_path: str | None) -> bytes:
    """One verified file, from ``dataset_path`` or the pinned repository.

    A pinned download is cached under ``INSPECT_EVALS_CACHE_DIR`` and verified
    on every read; a cached copy that fails verification is downloaded again.

    Raises:
        ValueError: The file is missing locally, is a local-only task without
            ``dataset_path``, or does not match its digest.
    """
    source = SOURCES[name]
    if dataset_path is not None:
        path = Path(dataset_path) / source.path
        if not path.is_file():
            raise ValueError(f"BIPIA dataset_path has no {source.path}: {path}")
        return verify(source, path.read_bytes(), str(path))
    if name in LOCAL_ONLY_TASKS:
        raise ValueError(
            f"BIPIA task {name!r} needs dataset_path: build benchmark/{source.path} "
            "with the reference's process.py (see the README)"
        )
    cached = CACHE_DIR / source.path
    if cached.is_file():
        data = cached.read_bytes()
        if digest(data, source.algorithm) == source.digest:
            return data
    url = source_url(source)
    data = verify(source, download(url), url)
    store(cached, data)
    return data


def store(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    The bytes go to a uniquely named file next to ``path`` first, so
    concurrent cache fills (other processes or threads) never share a staging
    file, and ``os.replace`` swaps it in whole.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staging = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


def parse_attacks(data: bytes, categories: list[str], origin: str) -> dict[str, str]:
    """Flatten an attack file into ``{"<category>-<index>": attack}``.

    As ``BasePIABuilder.load_attack``, keeping file order.

    Raises:
        ValueError: The categories or their attack counts are not the reference's.
    """
    attacks = json.loads(data)
    if not isinstance(attacks, dict) or list(attacks) != categories:
        found = list(attacks) if isinstance(attacks, dict) else type(attacks).__name__
        raise ValueError(
            f"BIPIA {origin} must hold categories {categories}; found {found}"
        )
    flat: dict[str, str] = {}
    for category, strings in attacks.items():
        if (
            not isinstance(strings, list)
            or len(strings) != ATTACK_VARIANTS
            or not all(isinstance(s, str) and s.strip() for s in strings)
        ):
            raise ValueError(
                f"BIPIA {origin} category {category!r} must hold "
                f"{ATTACK_VARIANTS} non-empty strings"
            )
        for index, attack in enumerate(strings):
            flat[f"{category}-{index}"] = attack
    return flat


def _text(value: Any, task: TaskName, field: str, joined: bool) -> str:
    """A context field as the builder uses it (code fields are joined lines)."""
    if joined:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"BIPIA {task} field {field!r} must be a list of strings")
        return "\n".join(value)
    if not isinstance(value, str):
        raise ValueError(f"BIPIA {task} field {field!r} must be a string")
    return value


def parse_contexts(task: TaskName, data: bytes) -> list[dict[str, str]]:
    """The context rows' prompt fields, as the task's builder reads them.

    Raises:
        ValueError: The row count differs from the release, a field is missing
            or has the wrong type, or a context has no sentence to insert into.
    """
    lines = data.decode("utf-8").splitlines()
    expected = SOURCES[task].rows
    if len(lines) != expected:
        raise ValueError(
            f"BIPIA {task} must have exactly {expected} contexts; found {len(lines)}"
        )
    rows = []
    for line in lines:
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"BIPIA {task} rows must be JSON objects")
        missing = [f for f in PROMPT_FIELDS[task] if f not in record]
        if missing:
            raise ValueError(f"BIPIA {task} row lacks {missing}")
        row = {
            field: _text(record[field], task, field, joined=task == "code")
            for field in PROMPT_FIELDS[task]
        }
        if not row["context"].strip():
            raise ValueError(f"BIPIA {task} context must not be empty")
        rows.append(row)
    return rows


def parse_names(
    value: str | list[str], allowed: tuple[str, ...], what: str
) -> list[str]:
    """Comma-separated or listed names, validated and in ``allowed`` order.

    Raises:
        ValueError: A name is unknown, or none is given.
    """
    names = value.split(",") if isinstance(value, str) else list(value)
    chosen = {name.strip() for name in names if name.strip()}
    unknown = sorted(chosen - set(allowed))
    if unknown or not chosen:
        raise ValueError(f"BIPIA {what} must be among {list(allowed)}; got {value!r}")
    return [name for name in allowed if name in chosen]


def build_samples(
    task: TaskName,
    contexts: list[dict[str, str]],
    attacks: dict[str, str],
    positions: list[Position],
    *,
    prompt_mode: PromptMode,
    ignore_guidance: bool,
) -> list[Sample]:
    """Every (position, context, attack) sample of one task, in reference order."""
    middles = (
        [middle_offset(row["context"]) for row in contexts]
        if "middle" in positions
        else []
    )
    samples = []
    for position in positions:
        for index, row in enumerate(contexts):
            fields = {k: v for k, v in row.items() if k != "context"}
            for attack_name, attack in attacks.items():
                poisoned = insert_attack(
                    row["context"],
                    attack,
                    position,
                    middles[index] if position == "middle" else None,
                )
                category = attack_category(attack_name)
                samples.append(
                    Sample(
                        id=f"{task}-{position}-{index}-{attack_name}",
                        input=task_messages(
                            task,
                            fields,
                            poisoned,
                            prompt_mode=prompt_mode,
                            ignore_guidance=ignore_guidance,
                        ),
                        metadata={
                            "task": task,
                            "attack_type": "code" if task == "code" else "text",
                            "attack_name": attack_name,
                            "attack_category": category,
                            "attack_group": ATTACK_GROUPS[category],
                            "evaluator": evaluator(attack_name).kind,
                            "position": position,
                            "context_index": index,
                        },
                    )
                )
    return samples


def dataset(
    tasks: list[TaskName],
    positions: list[Position],
    *,
    dataset_path: str | None = None,
    prompt_mode: PromptMode = "system",
    ignore_guidance: bool = False,
    shuffle_seed: int | None = None,
) -> MemoryDataset:
    """The BIPIA test samples of ``tasks``, every source verified first.

    Args:
        tasks: BIPIA tasks to include, in ``TASK_NAMES`` order.
        positions: Insertion positions, in ``POSITIONS`` order.
        dataset_path: A local ``benchmark`` folder (the reference's layout)
            to read every file from instead of the pinned repository;
            required for ``qa`` and ``abstract``.
        prompt_mode: ``system`` or ``user`` (see ``prompts.task_messages``).
        ignore_guidance: Add the reference's ignore-guidance reminder.
        shuffle_seed: Shuffle the samples with this seed so that ``--limit``
            takes a spread of tasks, attacks and positions; ``None`` keeps
            the reference order.

    Returns:
        The samples; inputs are chat messages, metadata names the task, attack,
        attack category and group, evaluator, position and context index.

    Raises:
        ValueError: A file is missing, fails its digest or is malformed.
    """
    attack_files = {"text": (TEXT_ATTACKS, TEXT_ATTACK_CATEGORIES)}
    attack_files["code"] = (CODE_ATTACKS, CODE_ATTACK_CATEGORIES)
    attack_sets: dict[str, dict[str, str]] = {}
    samples: list[Sample] = []
    for task in tasks:
        contexts = parse_contexts(task, read_source(task, dataset_path))
        kind = "code" if task == "code" else "text"
        if kind not in attack_sets:
            name, categories = attack_files[kind]
            attack_sets[kind] = parse_attacks(
                read_source(name, dataset_path), categories, name
            )
        attacks = attack_sets[kind]
        samples.extend(
            build_samples(
                task,
                contexts,
                attacks,
                positions,
                prompt_mode=prompt_mode,
                ignore_guidance=ignore_guidance,
            )
        )
    data = MemoryDataset(
        samples,
        name="bipia",
        location=dataset_path or f"{REPOSITORY}@{REPOSITORY_COMMIT}",
    )
    if shuffle_seed is not None:
        data.shuffle(seed=shuffle_seed)
    return data
