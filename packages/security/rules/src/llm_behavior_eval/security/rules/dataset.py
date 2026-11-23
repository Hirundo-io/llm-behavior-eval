"""RuLES test suites: pinned sources, checksums, validation and sample building.

The JSONL test cases are read from normster/llm_rules at a fixed commit
(``REPOSITORY_COMMIT``), folder ``llm_rules/data/<suite>/<test>.jsonl``, and
checked against SHA-256 digests and row counts before use. Downloads are
cached; ``dataset_path`` reads a local copy of the ``data`` folder instead.

Each test case is one sample. As the reference's ``evaluate_single``
(``llm_rules/scripts/evaluate.py``) builds it, the input is the scenario's
rules (a system message, or a user message answered by the fixed
acknowledgement), then the test case's messages up to its last prefilled
assistant message; the remaining user messages are sent one at a time by the
solver.
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
from typing import Any, Literal

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_evals_utils.cache import INSPECT_EVALS_CACHE_PATH

from .scenarios import (
    SCENARIOS,
    BaseScenario,
    Message,
    Role,
    remove_precedence_reminders,
)

REPOSITORY = "normster/llm_rules"
REPOSITORY_COMMIT = "f627e569146015d7fd6f200bb758b2591f9eb6c6"
"""The repository's HEAD (2025-02-12), RuLES v3.0.0."""
RAW_URL = (
    "https://raw.githubusercontent.com/{repository}/{commit}/llm_rules/data/{path}"
)
CACHE_DIR = INSPECT_EVALS_CACHE_PATH / "llm_behavior_eval_rules" / REPOSITORY_COMMIT

Suite = Literal["benign", "basic", "redteam", "direct_request"]
SUITES: tuple[Suite, ...] = ("benign", "basic", "redteam", "direct_request")
SCORE_SUITES: tuple[Suite, ...] = ("benign", "basic", "redteam")
"""The suites the RuLES score averages (``llm_rules/scripts/read_scores.py``)."""

SCENARIO_CATEGORIES: dict[str, str] = {
    "AccessControl": "harmless",
    "AnimalSounds": "helpful",
    "Authentication_provide": "helpful",
    "Authentication_withholdpassword": "harmless",
    "Authentication_withholdsecret": "harmless",
    "BinarySearch_confirm": "helpful",
    "BinarySearch_withhold": "harmless",
    "Commitment": "helpful",
    "Confidentiality": "harmless",
    "DiningCryptographers": "helpful",
    "Encryption": "harmless",
    "ForbiddenWord": "harmless",
    "Integrity": "helpful",
    "Millionaires_identify": "helpful",
    "Millionaires_withhold": "harmless",
    "Questions_confirm": "helpful",
    "Questions_withhold": "harmless",
    "RockPaperScissors": "helpful",
    "SimonSays": "helpful",
}
"""Each test file's category, from ``llm_rules/scripts/metadata.yaml``."""
TESTS: tuple[str, ...] = tuple(SCENARIO_CATEGORIES)
"""The test files of every suite, by name (``<Scenario>[_<behavior>]``), sorted."""
CATEGORIES = ("harmless", "helpful")


@dataclass(frozen=True)
class SourceFile:
    """One pinned test file's SHA-256 digest and row count."""

    digest: str
    rows: int


SOURCES: dict[str, SourceFile] = {
    "benign/AccessControl.jsonl": SourceFile(
        "628db0c607315bc7df8002ba028ab571260cf80ac7cca4288ec36d20802cbb25", 25
    ),
    "benign/AnimalSounds.jsonl": SourceFile(
        "1406edfa556e09aa9eac8c45d085eb671be00f26ed0a71aaf59bb7bcc39c3e6c", 25
    ),
    "benign/Authentication_provide.jsonl": SourceFile(
        "7bbcde5d01c391e593e34269bfcb039cd152bd515d08afb7e4319205cf5e1374", 25
    ),
    "benign/Authentication_withholdpassword.jsonl": SourceFile(
        "33d4da510b08a4abf067f337cf17d675f41101bc7cac8a47c4c279bde5677960", 25
    ),
    "benign/Authentication_withholdsecret.jsonl": SourceFile(
        "300428bf2c9a993c3fd9b3b3de86888f3665a46c62df122c43ea9318c76a9690", 25
    ),
    "benign/BinarySearch_confirm.jsonl": SourceFile(
        "441f37f86f4f3303ff975ddaccab2f6b3a7b0ca48af5114dcc54ca8b97c3c8c4", 25
    ),
    "benign/BinarySearch_withhold.jsonl": SourceFile(
        "9ad748e366e85a03effe540b007087cafb981c5053c033159768c866d9f11183", 25
    ),
    "benign/Commitment.jsonl": SourceFile(
        "4a3c9d023d7239478421ea84194721202a8e0b53832127ba23b3a9fffe7a6212", 25
    ),
    "benign/Confidentiality.jsonl": SourceFile(
        "15a366846f7bfcd5775ec9bb0190fc6e534e499fd601ddd806de541102b2adbc", 25
    ),
    "benign/DiningCryptographers.jsonl": SourceFile(
        "0661349c9ab7d1316d87b370889c0699875293a5c7829f19be94dd9bbc549e36", 25
    ),
    "benign/Encryption.jsonl": SourceFile(
        "d63bf7798f4db260cdc8d7dee9e3a6f86581f524782bcdd30b6760bebde4d081", 25
    ),
    "benign/ForbiddenWord.jsonl": SourceFile(
        "062d8b5d07b216697e4242edf4a23fa339f9b06fc97cf042dfd1bc72fe3a0fe4", 25
    ),
    "benign/Integrity.jsonl": SourceFile(
        "f4776935696306c9054a61c46a9f6f79fecf25ebddf6279343963e47d3b6e7c9", 25
    ),
    "benign/Millionaires_identify.jsonl": SourceFile(
        "c5c0c76b0cf41137ee98c94d7cedbbb43d6132301cca84a4a85729068f1398f7", 25
    ),
    "benign/Millionaires_withhold.jsonl": SourceFile(
        "a67f3d89c2e2ddf2f8ad50138216759d18de8080235a81b56bae023748be900f", 25
    ),
    "benign/Questions_confirm.jsonl": SourceFile(
        "c73ba10a2f3cb0336a48f45577967d3d028f8519fcb4c28f53bb76898c13559d", 25
    ),
    "benign/Questions_withhold.jsonl": SourceFile(
        "426771da5581e35b2f18b40fd4dd616c3cdb24e3efbe9e372bb0d53563e6be65", 25
    ),
    "benign/RockPaperScissors.jsonl": SourceFile(
        "2f2b9a4c60e226796faf043d8e0a8fa9ee5c4ad7d7d899778cb23afcae3c31bd", 25
    ),
    "benign/SimonSays.jsonl": SourceFile(
        "cc9a0b479b7e27b1e4896712bceddf12eaab65044b56b1506178111aefaabe74", 25
    ),
    "basic/AccessControl.jsonl": SourceFile(
        "b30e9f990318ac38289ca7ff31d0469d2fc8a913f945d51f4ee653b45616f4a0", 25
    ),
    "basic/AnimalSounds.jsonl": SourceFile(
        "3619d38a9779badf88f9b44f5ea20aa9a9a6fbae242912c3629bfbf841330599", 25
    ),
    "basic/Authentication_provide.jsonl": SourceFile(
        "7300913453ecd123d7c6e323903d434faf865ea5d961debdedb67fdbf7246cd5", 25
    ),
    "basic/Authentication_withholdpassword.jsonl": SourceFile(
        "9bc923e36023e33d9d788398abd1600df96ed390222f57e71a3f46d4809c7e06", 25
    ),
    "basic/Authentication_withholdsecret.jsonl": SourceFile(
        "330cc21a87a8b9d6f5aaf73d094522176fbaccb4121d1e031cdf239a0037e7f2", 25
    ),
    "basic/BinarySearch_confirm.jsonl": SourceFile(
        "0e60248c0cab803247433daa091d9fa7bd5ff42bd866b67bf9e4c801ed8f908d", 25
    ),
    "basic/BinarySearch_withhold.jsonl": SourceFile(
        "0bc8a3065f0e6a59c523dd6ae024b67c7497ad2c1f9f5e1b37c85e0244d908d4", 25
    ),
    "basic/Commitment.jsonl": SourceFile(
        "545e5fc4fdb1a30d851245f4cfbef45df0add53908bcc17be6ea9ade85b998fb", 25
    ),
    "basic/Confidentiality.jsonl": SourceFile(
        "c14088a2e6ba4c6dbdaeffea6635fe007662f3201ac62c8a8b65658aa8fd6781", 25
    ),
    "basic/DiningCryptographers.jsonl": SourceFile(
        "cced10938eb0ddc2459e6b9524812ac7c1ae1206c717270ef3f3ddb54e63e432", 25
    ),
    "basic/Encryption.jsonl": SourceFile(
        "962ab24a8e900e6cc704df1ef84a37ce81273d6b296ff68ab8bd56c35fc2916c", 25
    ),
    "basic/ForbiddenWord.jsonl": SourceFile(
        "286820d64b7bb2392db80bf8b942033a906d7427da9b7a05e035a4282e058910", 25
    ),
    "basic/Integrity.jsonl": SourceFile(
        "f4c8af9cb068eecced5cdc6dd2d3efdec9c0dcec45960fd0f1b897aea6d2211e", 25
    ),
    "basic/Millionaires_identify.jsonl": SourceFile(
        "6953ae4a9e99e9036bc685d53e3f2c91791f72187e4da69b1c4e5d13328f32a7", 25
    ),
    "basic/Millionaires_withhold.jsonl": SourceFile(
        "0416e7a7fda3f11ce00cbad3aa2cf3cfedce649b33291412fb36628356610ef6", 25
    ),
    "basic/Questions_confirm.jsonl": SourceFile(
        "bd63f112e7f945b760d0cc1fecaad91b991db8f296975e28f4621272feb11400", 25
    ),
    "basic/Questions_withhold.jsonl": SourceFile(
        "77520cd3097161c13a5a1406d3535accf1acd7f09173b1db1372b4cbb989d882", 25
    ),
    "basic/RockPaperScissors.jsonl": SourceFile(
        "964d0c169ca1c3f6fc5d758ddf60731ee28d44102bdb50e71bd1719d6e10fd1b", 25
    ),
    "basic/SimonSays.jsonl": SourceFile(
        "8f26f690bab69df34baa74d46e9403a58929e81016ad04d409bd839c4b3845e5", 25
    ),
    "redteam/AccessControl.jsonl": SourceFile(
        "8a309747c78a1eb60eb9578d9aa106b2309689b5782332448c1b08fbffd66909", 41
    ),
    "redteam/AnimalSounds.jsonl": SourceFile(
        "7568f1790fe7f8e000510ed6d5f46613318b6dfc777da2b2ca1549957deae562", 39
    ),
    "redteam/Authentication_provide.jsonl": SourceFile(
        "8cd355e4e85002d61207d5f2ff0ac7b6ca20efc1187f6d80e6c2c76f9ec03377", 39
    ),
    "redteam/Authentication_withholdpassword.jsonl": SourceFile(
        "dbeb865b74783c8c18964b401744f7ff478532e40d41bd6cb1ed616fbce4cb09", 39
    ),
    "redteam/Authentication_withholdsecret.jsonl": SourceFile(
        "5e2f2118a5e759e3fcc7332c068af9d8d8a2f1163636ed13dd408f74c6883888", 39
    ),
    "redteam/BinarySearch_confirm.jsonl": SourceFile(
        "f438b9aaf4997ace7920da1ec57d1cc1fd485fca0101d16c1bdc788d18002f50", 39
    ),
    "redteam/BinarySearch_withhold.jsonl": SourceFile(
        "6dbd227df82fc3edb531a2c3eeae6510bb87f82be32726802ed3180d088f7b81", 39
    ),
    "redteam/Commitment.jsonl": SourceFile(
        "eae07b2aa0e54e8dba2b78abf1c98018d120288a0e2af159f7c59ae810ae6171", 39
    ),
    "redteam/Confidentiality.jsonl": SourceFile(
        "fd9b470a9a6e100f67d0c193c945dd3c89de72466064afada4230dfd4b06d80f", 41
    ),
    "redteam/DiningCryptographers.jsonl": SourceFile(
        "64984d8168fffc274fe55493915b73cd4faba02620fa9cf7ce43ddf534b8b3ee", 39
    ),
    "redteam/Encryption.jsonl": SourceFile(
        "2728acd5fb4e2efe15e016e40f095179394a19bcfb4a817797d8bacd61ad72da", 39
    ),
    "redteam/ForbiddenWord.jsonl": SourceFile(
        "ec4cf56db6452054fda24b072b6379036cc5919cf1d46f41cdf1595f13eb28fd", 39
    ),
    "redteam/Integrity.jsonl": SourceFile(
        "567a78c11340a82c09f8f86bc83cac5c3fb7c21f45067f0e9737065c36e6a7f4", 39
    ),
    "redteam/Millionaires_identify.jsonl": SourceFile(
        "2bf107f3576b245ee20259b757b94cfdcaf412fb851de39a774fb78e9d9bb77f", 39
    ),
    "redteam/Millionaires_withhold.jsonl": SourceFile(
        "f5e9e8f77c99801098a51f7d06c99c27218cd3ff6b93c3f96fc7c9fa61eb3534", 39
    ),
    "redteam/Questions_confirm.jsonl": SourceFile(
        "78b24b3a9455d554f5025d4ff0149d594b7940747362beca20c36a18a0f16d39", 39
    ),
    "redteam/Questions_withhold.jsonl": SourceFile(
        "99d145d90072453df40fe4bca9d90276899129e0dafb398cdd86aa9e92d03ef6", 39
    ),
    "redteam/RockPaperScissors.jsonl": SourceFile(
        "a6e0d68ee7aea59a57a01f1b775e947b43bc13c66fd41be60df42d1f11816596", 39
    ),
    "redteam/SimonSays.jsonl": SourceFile(
        "0d3c084909bad8199d8ac8d4acd66f41aa680077ae7ffc385c59c96d5890c3f7", 39
    ),
    "direct_request/AccessControl.jsonl": SourceFile(
        "e7c3c79b6cecdc286450cbf4bcc820788d80a581f8b43b1400b70a9bb6ecf223", 15
    ),
    "direct_request/AnimalSounds.jsonl": SourceFile(
        "d60d6613c533af780aabe0329ca6e9065e1f918bfa72b7ade4c06ad4a0ed64d4", 15
    ),
    "direct_request/Authentication_provide.jsonl": SourceFile(
        "512141d8ef32f7c9ab3c768bf98bbdb15f58219b5ee5b61d2a1484258cc52efb", 15
    ),
    "direct_request/Authentication_withholdpassword.jsonl": SourceFile(
        "1b4d972e3c2f3e08aee72a689a494bfdf2919215238a7e6b7b5243091569bcd7", 15
    ),
    "direct_request/Authentication_withholdsecret.jsonl": SourceFile(
        "2844b0ad3f9f53b0517dc06af45b09bb7c962d0d95474d79c1dabfafcb13bb0d", 15
    ),
    "direct_request/BinarySearch_confirm.jsonl": SourceFile(
        "99f41cea631836efec08d25dfb974604ee75c190b8dc143fb666a9721445d349", 15
    ),
    "direct_request/BinarySearch_withhold.jsonl": SourceFile(
        "4369456d86185d3a9c56e71a5a4b83e3085ccf86ef9af014ff2c0375dcb23839", 15
    ),
    "direct_request/Commitment.jsonl": SourceFile(
        "c8ba9a4a9251540f959201aa39b5edf3eadc1c2cd4406583f72a2f8b43f2683c", 15
    ),
    "direct_request/Confidentiality.jsonl": SourceFile(
        "3b7e5871e90bd8813c807a6d7c5d9ed14935705dc7d11444861d82571562a859", 15
    ),
    "direct_request/DiningCryptographers.jsonl": SourceFile(
        "aa3f1b6c27209afc61941704bf5993a06478a2e1983f0265a9377a41b40fdcc3", 15
    ),
    "direct_request/Encryption.jsonl": SourceFile(
        "b264d4427c4a85bb2d7393afe7a45f1734502945134e5b475d90e95df85d5a1d", 15
    ),
    "direct_request/ForbiddenWord.jsonl": SourceFile(
        "036c3e04cfd69dfd95333d98951cf0e6b97abc27bb83eda1d39121ec5a6663ce", 15
    ),
    "direct_request/Integrity.jsonl": SourceFile(
        "daf3d8d96fa149615ac74af3751229bd4c1c311788c6da5c146a72900ef3e547", 15
    ),
    "direct_request/Millionaires_identify.jsonl": SourceFile(
        "910b0cbdecffbbebcbed8d4e6a07405be5359e2ba86c2d71054bd7cd2ddfd35a", 15
    ),
    "direct_request/Millionaires_withhold.jsonl": SourceFile(
        "84a7759d9d990b0766968dfd925bf18b357d173e5f87c6228b343900a2035a38", 15
    ),
    "direct_request/Questions_confirm.jsonl": SourceFile(
        "00a2b2fd5b2caa64602daba8194533336aea86817d9bcceeb5efdcc24dcb0709", 15
    ),
    "direct_request/Questions_withhold.jsonl": SourceFile(
        "46df9e0ec486c8c314ed80e5ffa2fa52c928303515649b9055a214164d33c362", 15
    ),
    "direct_request/RockPaperScissors.jsonl": SourceFile(
        "33f1530de48b7bbf2892dc198271ba3b723678c6c66a3eb9df67ebf0e9b18bf4", 15
    ),
    "direct_request/SimonSays.jsonl": SourceFile(
        "8da63830089fa7b7513f36c854cbbb077e7decd8567725acf4fd18377e1d5208", 15
    ),
}
"""Every test file at ``REPOSITORY_COMMIT``, by path below ``llm_rules/data/``."""

DOWNLOAD_ATTEMPTS = 4
TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


def source_path(suite: str, test: str) -> str:
    """A test file's path below ``llm_rules/data/``."""
    return f"{suite}/{test}.jsonl"


def source_url(path: str) -> str:
    """The raw GitHub URL of a pinned test file at ``REPOSITORY_COMMIT``."""
    return RAW_URL.format(repository=REPOSITORY, commit=REPOSITORY_COMMIT, path=path)


def verify(path: str, data: bytes, origin: str) -> bytes:
    """Return ``data`` if its SHA-256 digest is the pinned one.

    Raises:
        ValueError: The digest differs.
    """
    expected = SOURCES[path].digest
    found = hashlib.sha256(data).hexdigest()
    if found != expected:
        raise ValueError(
            f"RuLES file {origin} does not match the pinned release: expected "
            f"sha256 {expected}, found {found}"
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
    """One verified test file, from ``dataset_path`` or the pinned repository.

    A pinned download is cached under ``INSPECT_EVALS_CACHE_DIR`` and verified
    on every read; a cached copy that fails verification is downloaded again.

    Raises:
        ValueError: The local file is missing or a file does not match its
            digest.
    """
    if dataset_path is not None:
        local = Path(dataset_path) / path
        if not local.is_file():
            raise ValueError(f"RuLES dataset_path has no {path}: {local}")
        return verify(path, local.read_bytes(), str(local))
    cached = CACHE_DIR / path
    if cached.is_file():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == SOURCES[path].digest:
            return data
    url = source_url(path)
    data = verify(path, download(url), url)
    store(cached, data)
    return data


def scenario_name(test: str) -> str:
    """The scenario class of a test file: ``Authentication_provide`` -> ``Authentication``."""
    return test.split("_")[0]


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
        raise ValueError(f"RuLES {what} must be among {list(allowed)}; got {value!r}")
    return [name for name in allowed if name in chosen]


def parse_tests(value: str | list[str]) -> list[str]:
    """Test files named directly or by scenario (all of its behaviors).

    Raises:
        ValueError: A name is neither a test file nor a scenario.
    """
    allowed = (*TESTS, *(s for s in SCENARIOS if s not in TESTS))
    names = parse_names(value, allowed, "scenarios")
    return [t for t in TESTS if t in names or scenario_name(t) in names]


def opening_messages(
    scenario: BaseScenario,
    *,
    system_instructions: bool,
    remove_reminders: bool,
) -> list[Message]:
    """The rules, as ``evaluate_single`` presents them before any test message.

    With ``system_instructions`` (``--system_instructions``), one system
    message holding the rules. Otherwise (the reference's default) an empty
    system message, the rules as a user message and the scenario's fixed
    acknowledgement ``initial_response`` as the assistant's reply.
    """
    prompt = scenario.prompt
    if remove_reminders:
        prompt = remove_precedence_reminders(prompt)
    if system_instructions:
        return [Message(Role.SYSTEM, prompt)]
    return [
        Message(Role.SYSTEM, ""),
        Message(Role.USER, prompt),
        Message(Role.ASSISTANT, scenario.initial_response),
    ]


def chat_message(message: Message) -> ChatMessage:
    """An Inspect chat message with the same role and content."""
    if message.role == Role.SYSTEM:
        return ChatMessageSystem(content=message.content)
    if message.role == Role.USER:
        return ChatMessageUser(content=message.content)
    return ChatMessageAssistant(content=message.content)


def _test_messages(value: Any, origin: str) -> list[Message]:
    """A test case's messages (``Message.unserialize``: roles in any case)."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"RuLES {origin}: messages must be a non-empty list")
    messages = []
    for item in value:
        role = item.get("role") if isinstance(item, dict) else None
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(role, str) or role.lower() not in ("user", "assistant"):
            raise ValueError(
                f"RuLES {origin}: message roles must be user or assistant; got {role!r}"
            )
        if not isinstance(content, str):
            raise ValueError(f"RuLES {origin}: message content must be a string")
        messages.append(Message(Role[role.upper()], content))
    return messages


def build_sample(
    suite: str,
    test: str,
    record: Any,
    *,
    system_instructions: bool,
    remove_reminders: bool,
) -> Sample:
    """One test case as a sample, checked as the reference would use it.

    Raises:
        ValueError: The record lacks an ``id`` or ``params``, its parameters
            do not fit the scenario, it has no user message after its last
            prefilled assistant message, or the scenario rejects one of its
            user messages (the reference raises then).
    """
    origin = f"{source_path(suite, test)}"
    if not isinstance(record, dict):
        raise ValueError(f"RuLES {origin}: rows must be JSON objects")
    case_id = record.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"RuLES {origin}: every row needs a non-empty string id")
    origin = f"{origin} id {case_id}"
    params = record.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"RuLES {origin}: params must be a JSON object")
    category = record.get("category", "default")
    if not isinstance(category, str):
        raise ValueError(f"RuLES {origin}: category must be a string")
    try:
        scenario = SCENARIOS[scenario_name(test)](params)
    except (TypeError, KeyError, IndexError) as error:
        raise ValueError(f"RuLES {origin}: params do not fit the scenario") from error
    messages = _test_messages(record.get("messages"), origin)
    last = max(
        (i for i, m in enumerate(messages) if m.role == Role.ASSISTANT), default=-1
    )
    prefilled, remaining = messages[: last + 1], messages[last + 1 :]
    if not remaining or any(m.role != Role.USER for m in remaining):
        raise ValueError(
            f"RuLES {origin}: needs user messages after the last assistant message"
        )
    opening = opening_messages(
        scenario,
        system_instructions=system_instructions,
        remove_reminders=remove_reminders,
    )
    try:
        scenario.evaluate([*opening, *messages], system_instructions)
    except ValueError as error:
        raise ValueError(f"RuLES {origin}: {error}") from error
    return Sample(
        id=f"{suite}/{test}/{case_id}",
        input=[chat_message(m) for m in [*opening, *prefilled]],
        metadata={
            "suite": suite,
            "test": test,
            "scenario": scenario_name(test),
            "scenario_category": SCENARIO_CATEGORIES[test],
            "test_category": category,
            "params": params,
            "turns": [m.content for m in remaining],
            "system_instructions": system_instructions,
        },
    )


def parse_test_file(
    suite: str,
    test: str,
    data: bytes,
    *,
    system_instructions: bool,
    remove_reminders: bool,
) -> list[Sample]:
    """Every test case of one file, as samples, in file order.

    Raises:
        ValueError: The row count differs from the pinned file's, or a row is
            invalid (see ``build_sample``).
    """
    path = source_path(suite, test)
    lines = data.decode("utf-8").splitlines()
    if len(lines) != SOURCES[path].rows:
        raise ValueError(
            f"RuLES {path} must have exactly {SOURCES[path].rows} test cases; "
            f"found {len(lines)}"
        )
    return [
        build_sample(
            suite,
            test,
            json.loads(line),
            system_instructions=system_instructions,
            remove_reminders=remove_reminders,
        )
        for line in lines
    ]


def dataset(
    suites: list[Suite],
    tests: list[str],
    *,
    dataset_path: str | None = None,
    system_instructions: bool = False,
    remove_reminders: bool = False,
    shuffle_seed: int | None = None,
) -> MemoryDataset:
    """The RuLES test cases of ``suites`` and ``tests``, every file verified first.

    Args:
        suites: Test suites, in ``SUITES`` order.
        tests: Test files (``<Scenario>[_<behavior>]``), in ``TESTS`` order.
        dataset_path: A local copy of the reference's ``llm_rules/data``
            folder to read every file from instead of the pinned repository.
        system_instructions: Rules as a system message (see
            ``opening_messages``).
        remove_reminders: Remove the rule-precedence sentences from the rules.
        shuffle_seed: Shuffle the samples with this seed so that ``--limit``
            takes a spread of suites and scenarios; ``None`` keeps the
            reference order (suites, then test files, then file order).

    Returns:
        The samples; metadata names the suite, test file, scenario, its
        harmless/helpful category, the test case's category, its parameters,
        the user turns still to send and how the rules were presented.

    Raises:
        ValueError: A file is missing, fails its digest or is malformed.
    """
    samples: list[Sample] = []
    for suite in suites:
        for test in tests:
            path = source_path(suite, test)
            samples.extend(
                parse_test_file(
                    suite,
                    test,
                    read_source(path, dataset_path),
                    system_instructions=system_instructions,
                    remove_reminders=remove_reminders,
                )
            )
    data = MemoryDataset(
        samples,
        name="rules",
        location=dataset_path or f"{REPOSITORY}@{REPOSITORY_COMMIT}",
    )
    if shuffle_seed is not None:
        data.shuffle(seed=shuffle_seed)
    return data
