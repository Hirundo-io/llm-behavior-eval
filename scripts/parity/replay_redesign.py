"""Replay recorded qualification evidence through the current tasks, on CPU.

The representative GPU qualification (REPORT.md) ran the pre-redesign tasks,
which set inference settings in code. Its raw archive recorded every native
answer and judge request with its settings, the token IDs the vLLM server
actually received, and every response. This script re-runs each recorded case
with the *current* production task, configured the redesigned way:

* the benchmark's ``parity.yaml``: its ``generate_config`` for the model under
  test and its grader ``config`` for the judge;
* the benchmark's opt-in legacy text preprocessing (``legacy_tokenizer``);
* mock answer and grader models that return the recorded response for a
  byte-identical recorded request, and record a mismatch otherwise.

For every sample it checks that the current code sends exactly the recorded
requests (messages, order and settings, ignoring only the capture fields the
qualification runner added), that the preprocessing records the token IDs the
server actually received and the judge texts the legacy-compatible run used,
and that per-sample scores and every recorded metric are identical.

It performs no inference, so it cannot show that real models answer the same
under the new configuration path; only a GPU re-run can (see README.md).

    python -m scripts.parity.replay_redesign --evidence EXTRACTED_ARCHIVE \\
      --tokenizer ANSWER_TOKENIZER_SNAPSHOT --output replay.json
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

# Request fields the qualification runner added for evidence capture
# (prompt token IDs in responses, per-case prefix-cache isolation); they are not
# legacy settings and are not in parity.yaml.
CAPTURE_ONLY = {"return_token_ids", "cache_salt"}
# Where the archive's absolute paths pointed on the GPU host.
RECORDED_ROOT = "/mnt/filesystem-x0/llm169_f8a66042/"
OLD_SCORERS = {
    "bbq": "bias_scorer",
    "bloom": "bias_scorer",
    "unqover": "bias_scorer",
    "halueval": "hallucination_scorer",
    "medhallu": "hallucination_scorer",
    "xstest": "refusal_scorer",
    "orbench": "refusal_scorer",
    "prompt_injection": "injection_scorer",
    "ccpc": "censorship_judge",
}
NEW_SCORERS = {family: f"{family}_scorer" for family in OLD_SCORERS}
# Per-sample verdict fields compared in score metadata.
VERDICT_FIELDS = {
    "bias": ("correct", "empty", "unknown", "stereotyped", "gt_agree", "uncertain"),
    "hallucination": ("grade", "skipped"),
    "refusal": ("judge_label", "refusal", "grading_status", "heuristic_refusal"),
    "prompt_injection": ("skipped", "response_tokens", "formatted_answer"),
    "ccpc": ("verdict",),
}
GROUP = {
    "bbq": "bias",
    "bloom": "bias",
    "unqover": "bias",
    "halueval": "hallucination",
    "medhallu": "hallucination",
    "xstest": "refusal",
    "orbench": "refusal",
    "prompt_injection": "prompt_injection",
    "ccpc": "ccpc",
}


def request_key(messages: Any) -> str:
    """A request's identity: its exact roles and texts, in order."""
    return json.dumps([[m.role, m.text] for m in messages], ensure_ascii=False)


def settings(config: Any) -> dict[str, Any]:
    """Recorded request settings, without concurrency and capture-only fields."""
    data = config.model_dump(exclude_none=True)
    data.pop("max_connections", None)
    body = {
        key: value
        for key, value in (data.pop("extra_body", None) or {}).items()
        if key not in CAPTURE_ONLY
    }
    if body:
        data["extra_body"] = body
    return data


def same(a: Any, b: Any) -> bool:
    """Exact equality, with NaN equal to NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def parity_settings(family: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The answer and grader GenerateConfig fields from a benchmark's parity.yaml."""
    import yaml

    from llm_behavior_eval.common.parity import parity_config_path

    raw = yaml.safe_load(parity_config_path(f"llm_behavior_eval.{family}").read_text())
    return raw["generate_config"], raw["model_roles"]["grader"]["config"]


def case_roots(evidence: Path) -> list[Path]:
    """The 31 final sampled case roots, with the recorded follow-up replacement."""
    from scripts.parity.qualify_matrix import case_identity

    replacements = json.loads(
        (evidence / "sample100-final-case-roots.json").read_text()
    )
    roots = {}
    for root in sorted((evidence / "sample100-qualified").iterdir()):
        identity = root / "identity-config.json"
        if identity.exists() and (root / "scored.eval").exists():
            roots[json.loads(identity.read_text())["case"]] = root
    for case, recorded in replacements.items():
        roots[case] = evidence / recorded.removeprefix(RECORDED_ROOT)
    expected = json.loads(Path(__file__).with_name("pending.json").read_text())[
        "expected_cases"
    ]
    missing = sorted(set(expected) - set(roots))
    if missing:
        raise ValueError(f"Missing recorded cases: {missing}")
    for case in expected:
        case_identity(case)
    return [roots[case] for case in expected]


def replay_case(
    root: Path, tokenizer: str, log_dir: Path, evidence: Path | None = None
) -> dict[str, Any]:
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import GenerateConfig, ModelOutput, get_model

    from llm_behavior_eval import _registry
    from llm_behavior_eval.common.preprocessing import METADATA_KEY
    from scripts.parity.qualify_matrix import case_identity

    identity = json.loads((root / "identity-config.json").read_text())
    source = json.loads((root / "source.json").read_text())
    family, _, task_args = case_identity(identity["case"])
    group = GROUP[family]
    old = read_eval_log(str(root / "scored.eval"), resolve_attachments=True)
    mismatches: list[str] = []

    answers: dict[str, deque] = defaultdict(deque)
    judges: dict[str, deque] = defaultdict(deque)
    old_samples: dict[Any, Any] = {}
    old_judge_keys: dict[Any, list[str]] = {}
    for sample in old.samples or []:
        events = [event for event in sample.events if event.event == "model"]
        first = events[0]
        if first.call is None or first.call.response is None:
            raise ValueError(f"{sample.id}: no recorded answer call")
        answers[request_key(first.input)].append(
            dict(
                id=sample.id,
                output=first.output,
                settings=settings(first.config),
                server_ids=first.call.response["prompt_token_ids"],
            )
        )
        for event in events[1:]:
            judges[request_key(event.input)].append(
                dict(id=sample.id, output=event.output, settings=settings(event.config))
            )
        old_samples[sample.id] = sample
        old_judge_keys[sample.id] = [request_key(event.input) for event in events[1:]]

    seen_answers: dict[Any, dict[str, Any]] = {}
    seen_judges: list[tuple[str, dict[str, Any]]] = []
    # Samples run one at a time (max_samples=1), so judge requests belong to
    # the sample whose answer was requested last.
    current: list[Any] = [None]

    def take(queue: deque, sample_id: Any) -> dict[str, Any]:
        """The recorded entry of this sample; identical requests can recur."""
        for index, entry in enumerate(queue):
            if entry["id"] == sample_id:
                del queue[index]
                return entry
        return queue.popleft()

    def answer(messages, tools, tool_choice, config):
        queue = answers.get(request_key(messages))
        if not queue:
            mismatches.append("answer request not in the recorded evidence")
            return ModelOutput.from_content("replay/answer", "")
        recorded = queue.popleft()
        current[0] = recorded["id"]
        seen_answers[recorded["id"]] = dict(recorded, new_settings=settings(config))
        return recorded["output"]

    def judge(messages, tools, tool_choice, config):
        key = request_key(messages)
        queue = judges.get(key)
        if not queue:
            mismatches.append(f"{current[0]}: judge request not in the evidence")
            return ModelOutput.from_content("replay/grader", "")
        recorded = take(queue, current[0])
        seen_judges.append(
            (key, dict(recorded, id=current[0], new_settings=settings(config)))
        )
        return recorded["output"]

    answer_config, grader_config = parity_settings(family)
    task = getattr(_registry, family)(
        dataset_path=str(root / "dataset"), legacy_tokenizer=tokenizer, **task_args
    )
    ids = source["ids"]
    if [sample.id for sample in task.dataset][: len(ids)] != ids:
        mismatches.append("current task selects different samples")
    log = inspect_eval(
        task,
        model=get_model("mockllm/answer", custom_outputs=answer, memoize=False),
        model_roles={
            "grader": get_model(
                "mockllm/grader",
                custom_outputs=judge,
                config=GenerateConfig(**grader_config),
                memoize=False,
            )
        },
        limit=len(ids),
        max_samples=1,
        log_dir=str(log_dir),
        display="none",
        **answer_config,
    )[0]
    if log.status != "success":
        raise RuntimeError(f"{identity['case']}: {log.error}")

    judge_requests = 0
    for _, row in seen_judges:
        judge_requests += 1
        if row["settings"] != row["new_settings"]:
            mismatches.append(f"{row['id']}: judge settings differ")
    leftover = sum(len(queue) for queue in judges.values())
    if leftover:
        mismatches.append(f"{leftover} recorded judge requests were not sent")
    new_order = defaultdict(list)
    for key, row in seen_judges:
        new_order[row["id"]].append(key)
    server_boundary = views = verdicts = 0
    for sample in log.samples or []:
        old_sample = old_samples[sample.id]
        sent = seen_answers.get(sample.id)
        if sent is None:
            mismatches.append(f"{sample.id}: no answer request")
            continue
        if sent["settings"] != sent["new_settings"]:
            mismatches.append(
                f"{sample.id}: answer settings differ: {sent['settings']} != "
                f"{sent['new_settings']}"
            )
        if new_order[sample.id] != old_judge_keys[sample.id]:
            mismatches.append(f"{sample.id}: judge requests or their order differ")
        legacy = (sample.metadata or {}).get(METADATA_KEY)
        if legacy is None:
            mismatches.append(f"{sample.id}: no legacy text views")
            continue
        if legacy["prompt"]["input_ids"] == sent["server_ids"]:
            server_boundary += 1
        else:
            mismatches.append(f"{sample.id}: prompt IDs differ from server IDs")
        compat = old_sample.metadata["compatibility"]
        expected = {"prompt": compat["prompt"]["decoded"]}
        actual = {"prompt": legacy["prompt"]["decoded"]}
        for old_name, new_name in (
            ("target", "target"),
            ("stereotype", "stereotyped_answer"),
        ):
            if old_name in compat and new_name in legacy["texts"]:
                expected[old_name] = compat[old_name]["decoded"]
                actual[old_name] = legacy["texts"][new_name]["decoded"]
        if family == "prompt_injection":
            expected["judge_question"] = compat["judge_question"]
            actual["judge_question"] = legacy["texts"]["judge_question"]["decoded"]
        if expected == actual:
            views += 1
        else:
            mismatches.append(f"{sample.id}: legacy text views differ")
        old_score = (old_sample.scores or {})[OLD_SCORERS[family]]
        new_score = (sample.scores or {})[NEW_SCORERS[family]]
        old_meta, new_meta = old_score.metadata or {}, new_score.metadata or {}
        fields = VERDICT_FIELDS[group]
        if same(old_score.value, new_score.value) and all(
            same(old_meta.get(name), new_meta.get(name)) for name in fields
        ):
            verdicts += 1
        else:
            mismatches.append(f"{sample.id}: per-sample score differs")

    old_metrics = {
        name: metric.value
        for score in (old.results.scores if old.results else [])
        for name, metric in score.metrics.items()
    }
    new_metrics = {
        name: metric.value
        for score in (log.results.scores if log.results else [])
        for name, metric in score.metrics.items()
    }
    shared = sorted(set(old_metrics) & set(new_metrics))
    for name in shared:
        if not same(old_metrics[name], new_metrics[name]):
            mismatches.append(
                f"metric {name}: {old_metrics[name]!r} != {new_metrics[name]!r}"
            )
    samples = len(log.samples or [])
    return dict(
        case=identity["case"],
        evidence_root=str(root.relative_to(evidence) if evidence else root),
        scored_log_sha256=digest(root / "scored.eval"),
        samples=samples,
        answer_requests=len(seen_answers),
        judge_requests=judge_requests,
        recorded_judge_requests=sum(len(keys) for keys in old_judge_keys.values()),
        prompt_ids_equal_server_ids=server_boundary,
        legacy_text_views_equal=views,
        per_sample_scores_equal=verdicts,
        metrics_compared=shared,
        metrics_only_recorded=sorted(set(old_metrics) - set(new_metrics)),
        metrics_only_current=sorted(set(new_metrics) - set(old_metrics)),
        metrics=new_metrics,
        mismatches=mismatches,
        exact=not mismatches
        and server_boundary == views == verdicts == samples == len(ids),
    )


def source_state() -> tuple[str | None, bool | None]:
    """The checked-out commit and whether tracked files differ from it."""
    git = shutil.which("git")
    if git is None:
        return None, None
    head = subprocess.run(  # noqa: S603 - resolved git, fixed argv
        [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(  # noqa: S603 - resolved git, fixed argv
        [git, "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return head.stdout.strip() or None, bool(status.stdout.strip())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    tokenizer = str(args.tokenizer.resolve())
    roots = case_roots(evidence)
    rows = []
    with tempfile.TemporaryDirectory() as logs:
        for root in roots:
            case = json.loads((root / "identity-config.json").read_text())["case"]
            if args.cases and case not in args.cases:
                continue
            rows.append(replay_case(root, tokenizer, Path(logs) / root.name, evidence))
            print(case, "exact" if rows[-1]["exact"] else rows[-1]["mismatches"][:3])
    commit, dirty = source_state()
    report = dict(
        scope=(
            "CPU replay of the recorded pre-redesign sampled qualification "
            "through the current tasks with parity.yaml settings and the opt-in "
            "legacy preprocessing; mock models return recorded responses for "
            "identical requests. No inference: not a GPU re-run."
        ),
        code_commit=commit,
        code_dirty=dirty,
        archive_sha256=(
            "c9791b0d5d943de3a5cb640847ecf4273eff94b2ef02e795892bffc657614ec7"
            " (production-evidence-archive.json; paths are relative to it)"
        ),
        tokenizer_json_sha256=digest(args.tokenizer / "tokenizer.json"),
        cases=rows,
        totals=dict(
            cases=len(rows),
            exact_cases=sum(row["exact"] for row in rows),
            samples=sum(row["samples"] for row in rows),
            judge_requests=sum(row["judge_requests"] for row in rows),
        ),
        exact=bool(rows) and all(row["exact"] for row in rows),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
