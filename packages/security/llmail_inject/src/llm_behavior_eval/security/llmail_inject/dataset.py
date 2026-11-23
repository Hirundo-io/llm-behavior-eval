"""LLMail-Inject data: pinned attack and email files, validation and sampling.

Attacks are the paper's own evaluation subset (Section 4.5, Table 3): every
unique challenge submission that made the challenge's model call ``send_email``
(25,323 from phase 1, 3,688 from phase 2), as the authors published them in
``microsoft/llmail-inject-challenge-analysis``. Each key is the attacker's
email exactly as the challenge formatted it (``Subject of the email: ...
Body: ...``); each value lists the submissions' ``job_ids`` in the Hugging Face
dataset ``microsoft/llmail-inject-challenge``. The benign emails of each level
come from the challenge code. Both repositories are MIT-licensed.

Every file is fetched from a pinned commit (or read from ``dataset_path``) and
checked against its SHA-256 before it is parsed; nothing is committed here.
"""

import gzip
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_evals_utils import INSPECT_EVALS_CACHE_PATH
from llm_behavior_eval.security.llmail_inject.scenario import (
    DEFAULT_LEVELS,
    LEVELS,
    PromptDefense,
    Retriever,
    build_messages,
    retriever,
    tfidf_modules,
)

ANALYSIS_REPOSITORY = "microsoft/llmail-inject-challenge-analysis"
ANALYSIS_COMMIT = "3c6e0069e7609d7ef100694c9c29e12aa488452e"
CHALLENGE_REPOSITORY = "microsoft/llmail-inject-challenge"
CHALLENGE_COMMIT = "3da2399839b285e129888161fcc8cb03169d1b43"
RAW_URL = "https://raw.githubusercontent.com/{repository}/{commit}/{path}"


@dataclass(frozen=True)
class SourceFile:
    """One pinned file: its name in ``dataset_path``, URL and SHA-256."""

    name: str
    url: str
    sha256: str


def _github(repository: str, commit: str, path: str, sha256: str) -> SourceFile:
    return SourceFile(
        name=Path(path).name,
        url=RAW_URL.format(repository=repository, commit=commit, path=path),
        sha256=sha256,
    )


ATTACK_FILES: dict[int, SourceFile] = {
    1: _github(
        ANALYSIS_REPOSITORY,
        ANALYSIS_COMMIT,
        "defense_eval/defenses_results/defenses_results_phase1.json.gz",
        "529ee766601be1b694992cf804247e6d30fc0d034ef9d813abc1fa65d3829bdb",
    ),
    2: _github(
        ANALYSIS_REPOSITORY,
        ANALYSIS_COMMIT,
        "defense_eval/defenses_results/defenses_results_phase2.json",
        "095ff33da18e6fdf7519d98b3a8889417fa0dc1f1483a29419d20688ece4584b",
    ),
}
"""Attack files per challenge phase."""

EMAIL_FILES: dict[int, SourceFile] = {
    1: _github(
        CHALLENGE_REPOSITORY,
        CHALLENGE_COMMIT,
        "src/agent/data/level1.json",
        "e5f301ffe805fdca2061cce52e37d2b07e220a4ab85f679daccdfd7df1b343d7",
    ),
    2: _github(
        CHALLENGE_REPOSITORY,
        CHALLENGE_COMMIT,
        "src/agent/data/level2.json",
        "8fa84308dbbbaec70a3ae1dd3096e738a2419922eacd0264f874a21cb82931e6",
    ),
    3: _github(
        CHALLENGE_REPOSITORY,
        CHALLENGE_COMMIT,
        "src/agent/data/level3.json",
        "52da9798debe330a08417338b6478cec8e47f2f38f2fa2361c57ee78dfb5e3ce",
    ),
    4: _github(
        CHALLENGE_REPOSITORY,
        CHALLENGE_COMMIT,
        "src/agent/data/level4.json",
        "0e010305f5fd1c48bed931ee44f190e01da6e5081bcae5a2289c73d25cf7d485",
    ),
}
"""Benign emails per level (``{"emails": [...]}``)."""

EXPECTED_ATTACKS = {1: 25_323, 2: 3_688}
"""Unique attacks per phase (paper, Section 3 and Table 3)."""
EXPECTED_EMAILS = {1: 1, 2: 20, 3: 40, 4: 18}
"""Benign emails per level file; level 2 uses the first nine, levels 3 and 4
the nine TF-IDF ranks most similar to the query."""
ATTACK_PREFIX = "Subject of the email: "
PHASES = (1, 2)
DEFAULT_ATTACKS_PER_PHASE = 250
CACHE_DIR = INSPECT_EVALS_CACHE_PATH / "llm_behavior_eval" / "llmail_inject"
DOWNLOAD_ATTEMPTS = 4


@dataclass(frozen=True)
class Attack:
    """One unique attack email of one phase."""

    phase: int
    text: str
    job_ids: list[str]

    @property
    def sha256(self) -> str:
        """SHA-256 of the attack text (UTF-8): its ID and sampling key."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def download(url: str) -> bytes:
    """Fetch ``url``, retrying network errors and 429/5xx with backoff."""
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https URL)
                return response.read()
        except HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def check_sha256(source: SourceFile, data: bytes) -> None:
    """Raise unless ``data`` has the pinned checksum.

    Raises:
        ValueError: The checksum differs.
    """
    found = hashlib.sha256(data).hexdigest()
    if found != source.sha256:
        raise ValueError(
            f"LLMail-Inject file {source.name} does not match its pinned "
            f"checksum; expected SHA-256 {source.sha256}, found {found}"
        )


def fetch(source: SourceFile, dataset_path: str | None = None) -> bytes:
    """A pinned file's bytes, checksum-verified.

    Reads ``dataset_path/<name>`` when ``dataset_path`` is set; otherwise the
    Inspect evals cache, downloading the pinned URL on a miss or a stale copy.
    """
    if dataset_path is not None:
        data = (Path(dataset_path) / source.name).read_bytes()
        check_sha256(source, data)
        return data
    cached = CACHE_DIR / f"{source.sha256[:16]}-{source.name}"
    if cached.is_file():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == source.sha256:
            return data
    data = download(source.url)
    check_sha256(source, data)
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


def parse_attacks(phase: int, data: bytes) -> list[Attack]:
    """Validate one phase's attack file and return its attacks in file order.

    Raises:
        ValueError: The file is not a JSON object of attack text to
            ``{"job_ids": [...], ...}``, an attack is not a formatted email,
            or the attack count differs from the paper's.
    """
    source = ATTACK_FILES[phase]
    if source.name.endswith(".gz"):
        data = gzip.decompress(data)
    pool: Any = json.loads(data.decode("utf-8"))
    if not isinstance(pool, dict):
        raise ValueError(f"{source.name} must hold a JSON object")
    attacks = []
    for text, entry in pool.items():
        if not text.startswith(ATTACK_PREFIX):
            raise ValueError(
                f"{source.name}: every attack must start with {ATTACK_PREFIX!r}"
            )
        job_ids = entry.get("job_ids") if isinstance(entry, dict) else None
        if (
            not isinstance(job_ids, list)
            or not job_ids
            or not all(isinstance(job, str) and job for job in job_ids)
        ):
            raise ValueError(
                f"{source.name}: every attack needs a non-empty job_ids list"
            )
        attacks.append(Attack(phase=phase, text=text, job_ids=job_ids))
    if len(attacks) != EXPECTED_ATTACKS[phase]:
        raise ValueError(
            f"{source.name} must hold exactly {EXPECTED_ATTACKS[phase]} attacks; "
            f"found {len(attacks)}"
        )
    return attacks


def parse_emails(level: int, data: bytes) -> list[str]:
    """Validate one level's benign-email file and return its emails.

    Raises:
        ValueError: The file is not ``{"emails": [non-empty strings]}`` with
            the expected count.
    """
    source = EMAIL_FILES[level]
    content: Any = json.loads(data.decode("utf-8"))
    emails = content.get("emails") if isinstance(content, dict) else None
    if not isinstance(emails, list) or not all(
        isinstance(email, str) and email.strip() for email in emails
    ):
        raise ValueError(f"{source.name} must hold a list of non-empty emails")
    if len(emails) != EXPECTED_EMAILS[level]:
        raise ValueError(
            f"{source.name} must hold exactly {EXPECTED_EMAILS[level]} emails; "
            f"found {len(emails)}"
        )
    return emails


def sample_attacks(attacks: list[Attack], count: int | None) -> list[Attack]:
    """The ``count`` attacks with the smallest SHA-256, in that order.

    A deterministic uniform sample that ignores file order; a larger
    ``count`` extends a smaller one. ``None`` keeps every attack.

    Raises:
        ValueError: ``count`` is not positive.
    """
    if count is not None and count < 1:
        raise ValueError(f"attacks_per_phase must be positive; got {count}")
    ranked = sorted(attacks, key=lambda attack: attack.sha256)
    return ranked if count is None else ranked[:count]


def normalize_levels(levels: int | list[int] | None) -> list[int]:
    """Requested levels as a list, defaulting to levels 1 and 2.

    Raises:
        ValueError: A level is unsupported or repeated, or none is given.
        ImportError: Level 3 or 4 is requested without scikit-learn.
    """
    requested = (
        list(DEFAULT_LEVELS)
        if levels is None
        else [levels]
        if isinstance(levels, int)
        else levels
    )
    if not requested or len(set(requested)) != len(requested):
        raise ValueError(f"levels must be distinct and non-empty; got {levels!r}")
    unsupported = [level for level in requested if level not in LEVELS]
    if unsupported:
        raise ValueError(
            f"Unsupported LLMail-Inject levels {unsupported}: the challenge has "
            f"levels {list(LEVELS)}"
        )
    if any(LEVELS[level].rag for level in requested):
        tfidf_modules()
    return requested


def attack_sample(
    attack: Attack,
    level: int,
    level_retriever: Retriever,
    prompt_defense: PromptDefense,
) -> Sample:
    """One attack replayed at one level, as the GPT-path conversation."""
    scenario = LEVELS[level]
    emails, position = level_retriever.retrieve(attack.text)
    return Sample(
        id=f"phase{attack.phase}-{attack.sha256[:16]}-level{level}",
        input=build_messages(emails, prompt_defense, scenario.query),
        metadata={
            "phase": attack.phase,
            "level": level,
            "attack_sha256": attack.sha256,
            "attack_position": position,
            "job_ids": attack.job_ids,
            "target_to": scenario.target_to,
            "target_body": scenario.target_body,
        },
    )


def dataset(
    levels: int | list[int] | None = None,
    attacks_per_phase: int | None = DEFAULT_ATTACKS_PER_PHASE,
    prompt_defense: PromptDefense = "none",
    dataset_path: str | None = None,
) -> MemoryDataset:
    """The sampled attacks of both phases, each replayed at every level.

    Samples interleave by sampling rank (phase 1 then phase 2, each at every
    level), so ``--limit N`` keeps the phases and levels balanced.

    Args:
        levels: Levels to replay each attack at (1 to 4; default 1 and 2).
        attacks_per_phase: Attacks sampled per phase; ``None`` keeps all.
        prompt_defense: ``none``, ``spotlight`` or ``delimiters``.
        dataset_path: Folder holding the pinned files (both attack files
            and the selected levels' email files) under their published
            names, read instead of the pinned URLs.

    Returns:
        The samples, with phase, level, attack hash, attack position and job
        IDs as metadata.

    Raises:
        ValueError: A file fails its checksum or validation, an argument
            is invalid, or level 4's retrieved emails lack its target.
        ImportError: Level 3 or 4 is requested without scikit-learn.
    """
    selected_levels = normalize_levels(levels)
    retrievers = {
        level: retriever(
            LEVELS[level],
            parse_emails(level, fetch(EMAIL_FILES[level], dataset_path)),
        )
        for level in selected_levels
    }
    sampled = {
        phase: sample_attacks(
            parse_attacks(phase, fetch(ATTACK_FILES[phase], dataset_path)),
            attacks_per_phase,
        )
        for phase in PHASES
    }
    samples = [
        attack_sample(by_phase[rank], level, retrievers[level], prompt_defense)
        for rank in range(max(len(attacks) for attacks in sampled.values()))
        for by_phase in sampled.values()
        if rank < len(by_phase)
        for level in selected_levels
    ]
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("LLMail-Inject sample IDs must be unique")
    return MemoryDataset(samples, name="llmail_inject", location=ANALYSIS_REPOSITORY)
