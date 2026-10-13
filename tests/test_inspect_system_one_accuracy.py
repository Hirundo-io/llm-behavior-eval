import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.inspect_tasks import (
    _registry,  # noqa: F401  (registers systemone)
)
from llm_behavior_eval.inspect_tasks.system_one_censorship import accuracy_task
from llm_behavior_eval.inspect_tasks.system_one_censorship.accuracy import (
    distribution_from_answer,
    distribution_from_logits,
    expected_calibration_error,
    normalize_gold,
    option_keys,
    score_distribution,
)

NOUL = {"type": "noul", "instructions": "Is a refund requested?"}
SCORE = {
    "type": "score",
    "instructions": "How urgent?",
    "criteria": ["low", "mid", "high"],
}
CHOICE = {
    "type": "choice",
    "instructions": "Which team?",
    "criteria": {"billing": "Charges", "shipping": None},
}


def test_option_keys_and_gold_normalization():
    assert option_keys(NOUL) == ["false", "true"]
    assert option_keys(SCORE) == ["0", "1", "2"]
    assert option_keys(CHOICE) == ["billing", "shipping"]
    assert normalize_gold(NOUL, "yes") == "true"
    assert normalize_gold(NOUL, "0") == "false"
    assert normalize_gold(SCORE, 2) == "2"
    with pytest.raises(ValueError, match="not an option"):
        normalize_gold(CHOICE, "returns")


def test_distribution_from_answer_per_kind():
    assert distribution_from_answer(
        NOUL, {"type": "noul", "noul": 0.8}
    ) == pytest.approx({"false": 0.2, "true": 0.8})
    rounded = {"type": "choice", "probabilities": {"billing": 0.62, "shipping": 0.40}}
    assert distribution_from_answer(CHOICE, rounded)["billing"] == pytest.approx(
        0.62 / 1.02
    )
    with_levels = {
        "type": "score",
        "score": 1.2,
        "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
    }
    assert distribution_from_answer(SCORE, with_levels)["1"] == pytest.approx(0.6)
    expected_only = {"type": "score", "score": 1.7}
    assert distribution_from_answer(SCORE, expected_only) == {
        "0": 0.0,
        "1": 0.0,
        "2": 1.0,
    }
    with pytest.raises(ValueError, match="wrong type"):
        distribution_from_answer(NOUL, {"type": "score", "score": 1})


def test_distribution_from_logits_matches_readout():
    assert distribution_from_logits(NOUL, [0.0])["true"] == pytest.approx(0.5)
    dist = distribution_from_logits(CHOICE, [2.0, 0.0])
    assert max(dist, key=dist.__getitem__) == "billing"
    assert sum(dist.values()) == pytest.approx(1.0)


def test_score_distribution_and_ece():
    values = score_distribution({"false": 0.3, "true": 0.7}, "true")
    assert values["correct"] == 1.0
    assert values["brier"] == pytest.approx(0.3**2 + 0.3**2)
    assert values["confidence"] == pytest.approx(0.7)
    # One bin at 0.9 confidence, half right: |0.9 - 0.5| = 0.4.
    assert expected_calibration_error([(0.9, 1.0), (0.9, 0.0)]) == pytest.approx(0.4)


class _FakeGate(BaseHTTPRequestHandler):
    """Answers 'true' / 'billing' / level 2 with probability 0.9."""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        answers = {}
        for question_id, definition in body["questions"].items():
            kind = definition["type"]
            if kind == "noul":
                answers[question_id] = {"type": "noul", "noul": 0.9}
            elif kind == "score":
                answers[question_id] = {
                    "type": "score",
                    "score": 1.85,
                    "probabilities": {"0": 0.05, "1": 0.05, "2": 0.9},
                }
            else:
                answers[question_id] = {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.9, "shipping": 0.1},
                }
        payload = json.dumps({"answers": answers}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_gate():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGate)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _sample(sample_id, source, question, gold):
    return Sample(
        id=sample_id,
        input="x",
        target=gold,
        metadata={"source": source, "task": "t", "state": "s", "question": question},
    )


def test_eval_accuracy_against_system_one_server(fake_gate, tmp_path, monkeypatch):
    samples = [
        _sample("a", "jevbench", NOUL, "true"),  # right, conf 0.9
        _sample("b", "jevbench", CHOICE, "shipping"),  # wrong, conf 0.9
        _sample("c", "human", SCORE, "2"),  # right, conf 0.9
    ]
    monkeypatch.setattr(
        accuracy_task,
        "accuracy_dataset",
        lambda *a, **k: MemoryDataset(samples=samples),
    )
    [log] = inspect_eval(
        accuracy_task.system_one_accuracy(suites="jevbench,human"),
        model="systemone/fake-gate",
        model_base_url=fake_gate,
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.status == "success", log.error
    by_name = {score.name: score for score in log.results.scores}
    correct = by_name["correct"].metrics
    assert correct["mean"].value == pytest.approx(2 / 3)
    assert correct["grouped_jevbench"].value == pytest.approx(0.5)
    assert correct["grouped_human"].value == pytest.approx(1.0)
    # All three answers at 0.9 confidence, 2 of 3 right.
    assert by_name["confidence"].metrics["ece"].value == pytest.approx(abs(0.9 - 2 / 3))


def test_intent_items_always_include_gold_and_fixed_options():
    import random

    from llm_behavior_eval.inspect_tasks.system_one_censorship import usecases

    names = [f"intent_{i}" for i in range(30)] + ["oos"]
    pairs = [(f"text {i}", names[i % 31]) for i in range(60)]
    items = usecases._intent_items(
        "t", random.Random(0), 20, pairs, names, "q?", always={"oos": "none"}
    )
    assert len(items) == 20
    for item in items:
        options = item["question"]["criteria"]
        assert len(options) == usecases.INTENT_OPTIONS
        assert item["gold"] in options
        assert options["oos"] == "none"


def test_balanced_sampling_halves():
    import random

    from llm_behavior_eval.inspect_tasks.system_one_censorship.usecases import _balanced

    picked = _balanced(random.Random(0), list(range(50)), list(range(100, 150)), 20)
    assert sum(gold for _, gold in picked) == 10
    assert len(picked) == 20


def test_group_items_are_clear_cut_and_balanced(monkeypatch):
    from llm_behavior_eval.inspect_tasks.system_one_censorship import grouptoxicity

    monkeypatch.setattr(grouptoxicity, "GROUPS", {"chinese": 2, "muslim": 1})
    rows = []
    for group in ("chinese", "muslim"):
        for i in range(4):
            rows.append({"target_group": group, "text": f"{group} toxic {i}", "toxicity_human": 5.0})
            rows.append({"target_group": group, "text": f"{group} benign {i}", "toxicity_human": 1.0})
        rows.append({"target_group": group, "text": f"{group} unclear", "toxicity_human": 3.0})
    # Two annotators of one text are averaged before thresholding.
    rows.append({"target_group": "chinese", "text": "chinese toxic 0", "toxicity_human": 1.0})
    items = grouptoxicity.group_items(rows)
    tasks = [item["task"] for item in items]
    assert tasks.count("toxigen/chinese/toxic") == 2
    assert tasks.count("toxigen/chinese/benign") == 2
    assert tasks.count("toxigen/muslim/toxic") == 1
    texts = {item["state"]["text"] for item in items}
    assert not any("unclear" in t or t == "chinese toxic 0" for t in texts)
    assert all(item["gold"] == ("true" if "/toxic" in item["task"] else "false") for item in items)
