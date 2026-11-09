# HaluEval (native Inspect, LLM-160)

Free-text hallucination benchmark on `hirundo-io/halueval` (legacy preset `hallu`
/ `hallucination`). The model answers each question; a judge model in the
`grader` role grades the answer against the gold answer as CORRECT, INCORRECT or
NOT_ATTEMPTED.

```sh
inspect eval llm_behavior_eval/halueval --model hf/<org>/<model> \
  --model-role grader=hf/google/gemma-4-12B-it
```

The task binds no grader, so runs without `--model-role grader=...` fail. It sets
no inference settings for either role (temperature, top_p, top_k, max_tokens,
thinking): unset settings fall through to the serving stack's defaults. Pass
them per role as described in `INSPECT_MIGRATION.md`, or use `parity.yaml`.

## Task arguments

| Argument | Meaning |
| --- | --- |
| `dataset_path` | Read the data from a local path or `save_to_disk` snapshot instead of the Hub. The logical dataset id stays `hirundo-io/halueval`. |
| `dataset_token` | HF token for Hub loading. Prefer `HF_TOKEN` to keep it out of logs. |

All rows are validated (nonempty string `question` and `answer`), then every row
is converted in the legacy order: the full split shuffled with seed 42. `--limit
N` therefore selects the same rows as the legacy `max_samples=N`. Sample IDs are
`hirundo-io/halueval:<zero-based source row>` and do not depend on the limit.

## Prompts and scoring

* System prompt: the legacy free-text system message, verbatim. User message:
  `f"{question}\n"`, as in the legacy engine.
* Only answers that finished with `stop` are judged. Answers that hit
  `max_tokens`, or have any other or no stop reason, are graded NOT_ATTEMPTED
  without a judge call.
* Inline `<think>...</think>` traces are stripped from the answer before judging.
* The judge gets the legacy SimpleQA-style `GRADER_TEMPLATE` as a single user
  message. The first standalone, case-sensitive `A`, `B` or `C` in its reply
  (`\b([ABC])\b`) maps to CORRECT, INCORRECT or NOT_ATTEMPTED. No match is
  NOT_ATTEMPTED.
* Each sample's score is 1 unless the grade is INCORRECT. Score metadata holds
  the grade, the stop reason, the raw judge reply, the rendered judge prompt
  (when judged), whether the reply failed to parse, and the known parity gaps.

## Metrics

| Name | Unit | Definition | Direction |
| --- | --- | --- | --- |
| `accuracy` | percentage | `100 * (1 - INCORRECT / all samples)` | higher |
| `empty_responses` | count | NOT_ATTEMPTED grades, including unjudged and unparsable answers | lower |
| `incomplete_response_rate` | fraction | answers that stopped at `max_tokens` / all samples | lower |

Every sample stays in the denominator, so NOT_ATTEMPTED answers count as not
incorrect. This is the legacy definition and is kept for parity. The legacy
engine stored accuracy as a fraction and displayed a percentage; this task
reports the percentage.

`accuracy` and `empty_responses` are registered with Inspect as
`llm_behavior_eval/halueval_accuracy` and `llm_behavior_eval/halueval_empty_responses`
and report under the names above. The benchmark-specific registry names keep
another benchmark's metric of the same name from replacing them when Inspect
rebuilds metrics from a log. `target_metrics.json` lists the report metrics.

## Parity

`parity.yaml` holds the legacy `hallu` settings: answers greedy with 128 tokens
and thinking off; judge `hf/google/gemma-3-12b-it`, greedy, 32 tokens, thinking
off; limit 500.

The full parity command is under [Parity run](#parity-run-llm-169) below.

The model-under-test provider args stay on the command line because the file
does not name that model. For a vLLM model under test, the file's `extra_body`
already turns thinking off. To keep the parity settings with another judge, copy
the file and change the grader's `model`, `args` or `base_url`.

Known differences from the legacy engine, recorded in every sample's
`parity_gaps` metadata:

* The legacy judge received the **decoded chat-templated input** as the
  question, not the raw question.
* Legacy inputs were truncated to 1024 tokenizer tokens.
* Legacy targets were truncated to 256 tokens and decoded.

Reproducing these needs the evaluated model's exact tokenizer and chat template,
which arbitrary API endpoints cannot expose. CPU tests compare grade parsing and
aggregate metrics directly with the legacy `FreeTextHaluEvaluator`; they do not
establish live-model parity.

## Parity run (LLM-169)
A parity run passes three things: `parity.yaml` (`--run-config`), the answer
model's provider args (`-M`), and the opt-in legacy text preprocessing (`-T
legacy_tokenizer`, the answer model's local tokenizer snapshot; add `-T
legacy_tokenizer_revision=<commit>` for a cached repository instead):

```sh
inspect eval llm_behavior_eval/halueval \
  -T legacy_tokenizer="$ANSWER" \
  --model hf/local -M model_path="$ANSWER" \
  -M do_sample=false -M enable_thinking=false \
  -M tokenizer_call_args='{truncation: true, max_length: 1024}' \
  --run-config llm_behavior_eval/halueval/parity.yaml
```

On vLLM/SGLang/OpenAI-compatible servers, `parity.yaml`'s `extra_body` already
carries the legacy thinking switch, special-token handling and 1024-token prompt
truncation, so only the HF `-M` flags change; see `scripts/parity/README.md` for
the vLLM variant the qualification used. Do not add `--model-role grader=`: it
replaces the file's whole grader entry, `config` included.

`legacy_tokenizer` is off by default, so ordinary runs are native Inspect. When
set, the judge sees the decoded, 1024-token chat-template prompt as the judge
question and the target after the legacy 256-token round trip. The model's own
input never changes. The prompt-parity gaps described above apply only to runs
without it.

Judge: Gemma 3 12B (`google/gemma-3-12b-it`); `google/gemma-4-12B-it` is not
parity-qualified.

Evidence status: the representative GPU qualification (1 configuration, 100
paired rows, 100 judge requests per side; an earlier eager 500-row run was also
exact (89.8% on both sides)) was exact, but it ran the pre-redesign
implementation, which set these settings in code. A CPU replay of its recorded
requests through this package, configured only by `parity.yaml` and `-T
legacy_tokenizer`, reproduces every request, setting, per-sample score and
metric exactly. A GPU re-run with `parity.yaml` is still needed to confirm the
result (`scripts/parity/README.md`).
