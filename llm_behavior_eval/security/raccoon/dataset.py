"""RaccoonBench data: pinned sources, checksums, validation and sample building.

The GPT prompts, attack prompts, reference defenses and defense templates are
read from M0gician/RaccoonBench at a fixed commit (``REPOSITORY_COMMIT``),
folder ``Data/``, and checked against SHA-256 digests (``sources.SOURCES``)
before use. Downloads are cached; ``dataset_path`` reads a local copy of the
``Data`` folder instead.

Each sample is one (attack prompt, defense, GPT) attempt: the GPT's prompt,
assembled as upstream's ``RaccoonGang.benchmark_single`` assembles it, then
the attack prompt as the user's message. Samples are built attack by attack,
then defense by defense, then GPT by GPT, as the upstream runner loops.
"""

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

from .prompts import Defense, SystemTemplate, system_prompt, wrap
from .sources import SOURCES

REPOSITORY = "M0gician/RaccoonBench"
REPOSITORY_COMMIT = "d2ea81e8e5e3a933f3f2cba88d35e47d11c63800"
"""The repository's HEAD (2026-04-09, "add GPTs", which added the GPT prompts)."""
RAW_URL = "https://raw.githubusercontent.com/{repository}/{commit}/Data/{path}"
CACHE_DIR = INSPECT_EVALS_CACHE_PATH / "llm_behavior_eval_raccoon" / REPOSITORY_COMMIT

REFERENCE_DEFENSES = "reference/gpts196_defense_prompt.json"
"""Each GPT's defense sentences, mapped to their defense categories."""
DEFENSE_TEMPLATES = "defenses/defense_template.json"
"""The 14 defense templates of the defended setting, by name."""
SYSTEM_PROMPT = "system_prompt.md"

GptSet = Literal["all", "gpts50", "gpts146"]
GPT_FOLDERS: dict[str, tuple[str, ...]] = {
    "all": ("GPTs50", "GPTs146"),
    "gpts50": ("GPTs50",),
    "gpts146": ("GPTs146",),
}
"""``all`` is the 196 GPTs the reference defenses cover (upstream's ``gpts196``)."""

AttackSet = Literal["singular", "singular_top5", "compound"]
ATTACK_FOLDERS: dict[str, str] = {
    "singular": "singular_attacks",
    "singular_top5": "singular_attacks_deflesstop5",
    "compound": "compound_attacks",
}
"""The attack sets: every single-category attack (44 prompts in 15 categories),
the five most effective against defenseless GPTs, and the 10 compound attacks."""

DEFENSE_TEMPLATE_NAMES: tuple[str, ...] = (
    "short_defense_1",
    "short_defense_FramerGPT",
    "short_defense_Code_Tutor",
    "medium_defense_台車ジェネレータ",
    "medium_defense_Tune_Therapist",
    "medium_defense_Gated_access",
    "medium_defense_STOP_HALT",
    "medium_defense_I_will_never_trust_you_again",
    "long_defense_Rule_Nr_1",
    "long_defense_Can_t_Hack_This",
    "long_defense_CAPTURETHEGPT",
    "long_defense_Smart_Sorter",
    "long_defense_GptInfinite",
    "long_defense_CK_12_Flexi",
)
"""The keys of ``defense_template.json``, in file order (the order upstream runs them)."""

PromptFormat = Literal["system", "user_turn"]
ACKNOWLEDGEMENT = "Understood."
"""The assistant turn upstream's Llama, Mixtral and Gemini paths put after the prompt."""

DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_WORKERS = 8
TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


def gpt_paths(gpt_set: str) -> list[tuple[str, str]]:
    """``(folder, name)`` of every GPT in ``gpt_set``, sorted by folder then name."""
    found = []
    for folder in GPT_FOLDERS[gpt_set]:
        for path in SOURCES:
            parts = path.split("/")
            if len(parts) == 3 and parts[0] == folder and parts[2] == SYSTEM_PROMPT:
                found.append((folder, parts[1]))
    return sorted(found, key=lambda item: (GPT_FOLDERS["all"].index(item[0]), item[1]))


def attack_paths(attack_set: str) -> list[tuple[str, str]]:
    """``(category, prompt)`` of every attack prompt of ``attack_set``, sorted."""
    prefix = f"attacks/{ATTACK_FOLDERS[attack_set]}/"
    found = []
    for path in SOURCES:
        if path.startswith(prefix):
            category, prompt = path.removeprefix(prefix).split("/")
            found.append((category, prompt))
    return sorted(found)


def attack_categories() -> list[str]:
    """Every attack category of every attack set, sorted, without duplicates."""
    return sorted({category for s in ATTACK_FOLDERS for category, _ in attack_paths(s)})


def source_url(path: str) -> str:
    """The raw GitHub URL of a pinned file (a path below ``Data/``)."""
    return RAW_URL.format(
        repository=REPOSITORY,
        commit=REPOSITORY_COMMIT,
        path=urllib.parse.quote(path),
    )


def verify(path: str, data: bytes, origin: str) -> bytes:
    """Return ``data`` if its SHA-256 digest is the pinned one.

    Raises:
        ValueError: The digest differs.
    """
    expected = SOURCES[path]
    found = hashlib.sha256(data).hexdigest()
    if found != expected:
        raise ValueError(
            f"RaccoonBench file {origin} does not match the pinned commit: "
            f"expected sha256 {expected}, found {found}"
        )
    return data


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


def store(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    The bytes go to a uniquely named file next to ``path`` first, so
    concurrent cache fills never share a staging file, and ``os.replace``
    swaps it in whole.
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


def read_source(path: str, dataset_path: str | None) -> bytes:
    """One verified file, from ``dataset_path`` or the pinned repository.

    A pinned download is cached under ``INSPECT_EVALS_CACHE_DIR`` and verified
    on every read; a cached copy that fails verification is downloaded again.

    Raises:
        ValueError: The local file is missing or a file does not match its
            digest.
    """
    if dataset_path is not None:
        local = Path(dataset_path) / path
        if not local.is_file():
            raise ValueError(f"RaccoonBench dataset_path has no {path}: {local}")
        return verify(path, local.read_bytes(), str(local))
    cached = CACHE_DIR / path
    if cached.is_file():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == SOURCES[path]:
            return data
    url = source_url(path)
    data = verify(path, download(url), url)
    store(cached, data)
    return data


def as_text(data: bytes) -> str:
    """UTF-8 text with universal newlines, as upstream's text-mode ``open`` reads it."""
    return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def read_sources(paths: list[str], dataset_path: str | None) -> dict[str, str]:
    """Several verified files as text (``as_text``), fetched in parallel, by path."""
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        contents = list(pool.map(lambda path: read_source(path, dataset_path), paths))
    return {path: as_text(data) for path, data in zip(paths, contents, strict=True)}


def parse_reference_defenses(text: str) -> dict[str, dict[str, str]]:
    """``gpts196_defense_prompt.json``: GPT name -> {defense sentence: category}.

    Raises:
        ValueError: It is not an object of string-to-string objects.
    """
    data = json.loads(text)
    if not isinstance(data, dict) or not all(
        isinstance(defenses, dict)
        and all(
            isinstance(sentence, str) and isinstance(category, str)
            for sentence, category in defenses.items()
        )
        for defenses in data.values()
    ):
        raise ValueError(
            f"RaccoonBench {REFERENCE_DEFENSES} must map each GPT to an object "
            "of defense sentences and their categories"
        )
    return data


def parse_defense_templates(text: str) -> dict[str, str]:
    """``defense_template.json``: template name -> template, in file order.

    Raises:
        ValueError: It is not an object of strings, or its names are not
            ``DEFENSE_TEMPLATE_NAMES``.
    """
    data = json.loads(text)
    if not isinstance(data, dict) or not all(
        isinstance(template, str) for template in data.values()
    ):
        raise ValueError(
            f"RaccoonBench {DEFENSE_TEMPLATES} must map names to template strings"
        )
    if tuple(data) != DEFENSE_TEMPLATE_NAMES:
        raise ValueError(
            f"RaccoonBench {DEFENSE_TEMPLATES} must name exactly "
            f"{list(DEFENSE_TEMPLATE_NAMES)}; found {list(data)}"
        )
    return data


def parse_names(value: str | list[str], allowed: tuple[str, ...]) -> list[str]:
    """Comma-separated or listed defense template names, validated, in ``allowed`` order.

    Raises:
        ValueError: A name is unknown, or none is given.
    """
    names = value.split(",") if isinstance(value, str) else list(value)
    chosen = {name.strip() for name in names if name.strip()}
    unknown = sorted(chosen - set(allowed))
    if unknown or not chosen:
        raise ValueError(
            f"RaccoonBench defense_templates must be among {list(allowed)}; "
            f"got {value!r}"
        )
    return [name for name in allowed if name in chosen]


def opening_messages(
    prompt: str, attack: str, prompt_format: PromptFormat
) -> list[ChatMessage]:
    """The conversation the model under test answers.

    ``system`` (upstream's GPT path): the prompt as the system message, the
    attack as the user's. ``user_turn`` (upstream's Llama, Mixtral and Gemini
    paths): the prompt as a user message answered by "Understood.", then the
    attack.
    """
    if prompt_format == "system":
        return [ChatMessageSystem(content=prompt), ChatMessageUser(content=attack)]
    return [
        ChatMessageUser(content=prompt),
        ChatMessageAssistant(content=ACKNOWLEDGEMENT),
        ChatMessageUser(content=attack),
    ]


def dataset(
    *,
    attacks: AttackSet,
    defense: Defense,
    templates: list[str] | None,
    gpts: GptSet,
    system_template: SystemTemplate,
    prompt_format: PromptFormat,
    dataset_path: str | None = None,
    shuffle_seed: int | None = None,
) -> MemoryDataset:
    """Every (attack, defense, GPT) attempt, every source file verified first.

    Args:
        attacks: The attack set (``ATTACK_FOLDERS``).
        defense: How the GPTs' prompts are defended (``prompts.system_prompt``).
        templates: With ``defense="custom"``, the defense templates to run
            (``None``: all 14); each attempt is repeated under every one.
        gpts: The GPT set (``GPT_FOLDERS``).
        system_template: The wrapper around the GPT's prompt (``prompts.wrap``).
        prompt_format: How the prompt is sent (``opening_messages``).
        dataset_path: A local copy of RaccoonBench's ``Data`` folder to read
            every file from instead of the pinned repository.
        shuffle_seed: Shuffle the samples with this seed so that ``--limit``
            takes a spread of attacks, defenses and GPTs; ``None`` keeps the
            upstream order.

    Returns:
        The samples. The target is the GPT's prompt before any wrapper, the
        text upstream compares the response with; metadata names the GPT, the
        attack and the defense, and holds the attack text for the echo filter.

    Raises:
        ValueError: A file is missing, fails its digest or is malformed, a GPT
            has no reference defenses, or a GPT's defense is not in its prompt.
    """
    gpt_list = gpt_paths(gpts)
    attack_list = attack_paths(attacks)
    gpt_files = [f"{folder}/{name}/{SYSTEM_PROMPT}" for folder, name in gpt_list]
    attack_files = [
        f"attacks/{ATTACK_FOLDERS[attacks]}/{category}/{prompt}"
        for category, prompt in attack_list
    ]
    files = read_sources(
        [REFERENCE_DEFENSES, DEFENSE_TEMPLATES, *gpt_files, *attack_files],
        dataset_path,
    )
    references = parse_reference_defenses(files[REFERENCE_DEFENSES])
    all_templates = parse_defense_templates(files[DEFENSE_TEMPLATES])
    if defense == "custom":
        names = list(DEFENSE_TEMPLATE_NAMES) if templates is None else templates
        defenses: list[tuple[str, str | None]] = [(n, all_templates[n]) for n in names]
    else:
        defenses = [(defense, None)]

    prompts: dict[tuple[str, str], str] = {}
    for (_folder, name), path in zip(gpt_list, gpt_files, strict=True):
        if name not in references:
            raise ValueError(
                f"RaccoonBench GPT {name!r} has no entry in {REFERENCE_DEFENSES}"
            )
        for defense_name, template in defenses:
            try:
                prompts[name, defense_name] = system_prompt(
                    defense,
                    name=name,
                    original_prompt=files[path],
                    defenses=references[name],
                    template=template or "",
                )
            except ValueError as error:
                raise ValueError(f"RaccoonBench {path}: {error}") from error

    samples: list[Sample] = []
    for (category, prompt_name), attack_file in zip(
        attack_list, attack_files, strict=True
    ):
        attack = files[attack_file]
        if not attack.strip():
            raise ValueError(f"RaccoonBench {attack_file} is empty")
        for defense_name, _template in defenses:
            for folder, name in gpt_list:
                reference = prompts[name, defense_name]
                samples.append(
                    Sample(
                        id=f"{attacks}/{category}/{prompt_name}/{defense_name}/{name}",
                        input=opening_messages(
                            wrap(system_template, name, reference),
                            attack,
                            prompt_format,
                        ),
                        target=reference,
                        metadata={
                            "gpt": name,
                            "gpt_set": folder,
                            "attack_set": attacks,
                            "attack_category": category,
                            "attack_prompt": prompt_name,
                            "attack": attack,
                            "defense": defense,
                            "defense_name": defense_name,
                        },
                    )
                )
    data = MemoryDataset(
        samples,
        name="raccoon",
        location=dataset_path or f"{REPOSITORY}@{REPOSITORY_COMMIT}",
    )
    if shuffle_seed is not None:
        data.shuffle(seed=shuffle_seed)
    return data
