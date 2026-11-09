# Med-Hallu (LLM-161)

Medical hallucination benchmark: the model answers a medical question, and a
grader model compares the answer with the gold answer using the legacy
SimpleQA-style A/B/C prompt. Native port of the legacy `hallu-med` /
`hallucination-med` preset (`FreeTextHaluEvaluator`). The package is
self-contained: it imports only generic mechanics from `llm_behavior_eval.common`.

```sh
inspect eval llm_behavior_eval/medhallu --model hf/org/model \
  --model-role 'grader={model: openai-api/airgap/judge}'
```

The task sets no inference settings for either role and binds no grader: a run
without `--model-role grader=...` fails. Pass settings per role yourself (see
`INSPECT_MIGRATION.md`), or use `parity.yaml` for the legacy engine's settings.

## Files

| File | Owns |
| --- | --- |
| `task.py` | The `medhallu` task. No `config=`, no default grader. |
| `dataset.py` | Loading, validation, legacy seed-42 order, `record_to_sample`. |
| `prompts.py` | Legacy system prompt, `GRADER_TEMPLATE` and A/B/C label map, verbatim. |
| `solver.py` | System prompt, legacy trailing user newline, `generate()`. |
| `scorer.py` | `medhallu_scorer`: stop-only judging and A/B/C parsing. |
| `metrics.py` | `accuracy` and `empty_responses` (registered as `medhallu_accuracy` and `medhallu_empty_responses`). |
| `parity.yaml` | Legacy engine settings for both roles and the 500-sample cap. |
| `target_metrics.json` | Report metrics for `llm_behavior_eval/medhallu`. |

## Dataset

`hirundo-io/medhallu`, columns `question` and `answer`. Task arguments
`-T dataset_path=...` (HF path or `save_to_disk` snapshot, for offline runs) and
`-T dataset_token=...`. Every row must have a nonempty `question` and `answer`.
Sample IDs are `hirundo-io/medhallu:<zero-based source row>`.

The task returns the full split, ordered as the legacy engine shuffled it
(`Dataset.shuffle(seed=42)`). There is no task-level sample cap: `--limit 500`
(or `eval_config.limit` in `parity.yaml`) selects exactly the legacy 500 samples.

## Scoring

* The model sees the legacy system prompt and the question followed by `\n`.
* Only answers that finished with `stop` are sent to the grader. Truncated
  (`max_tokens`) or other answers are graded NOT_ATTEMPTED without a grader call.
* Inline `<think>...</think>` traces are stripped before judging.
* The first standalone, case-sensitive `A`/`B`/`C` in the grader reply maps to
  CORRECT/INCORRECT/NOT_ATTEMPTED. No match is NOT_ATTEMPTED with `parse_error`.
* Score metadata keeps the question, gold answer, formatted answer, judge prompt,
  raw judge text, stop reason, grade, skip/parse status and `parity_gaps`.

| Metric | Unit | Definition | Direction |
| --- | --- | --- | --- |
| `accuracy` (primary) | percentage | `100 * (1 - INCORRECT / total)` | higher |
| `empty_responses` | count | NOT_ATTEMPTED grades, including skips and parse failures | lower |
| `incomplete_response_rate` | fraction | answers ending at `max_tokens` / total | lower |

Every sample stays in the denominator. Legacy stored accuracy as a fraction and
displayed a percentage; this task reports the percentage, so its
`target_metrics.json` entry is `is_normalized: false`. The old brief label is
`medhallu`.

`accuracy` and `empty_responses` are registered with Inspect as
`llm_behavior_eval/medhallu_accuracy` and `llm_behavior_eval/medhallu_empty_responses`
and report under the names above. The benchmark-specific registry names keep
another benchmark's metric of the same name from replacing them when Inspect
rebuilds metrics from a log.

## Parity

`parity.yaml` holds the legacy settings (from the removed engine's
`evaluation_utils/eval_config.py` at d8c8015
`FAMILY_TOKEN_DEFAULTS["hallucination"]`, `sampling_config.py` and
`dataset_config.py`):

| Setting | Model under test | Grader |
| --- | --- | --- |
| Decoding | greedy (`temperature: 0.0`) | greedy (`temperature: 0.0`, `do_sample: false`) |
| `max_tokens` | 128 | 32 |
| `seed` | 42 | 42 |
| Thinking | off (`extra_body.chat_template_kwargs.enable_thinking: false`) | off (`enable_thinking: false`) |
| Model | from the command line | `hf/google/gemma-3-12b-it` |
| Sample cap | `eval_config.limit: 500` | |

The full parity command is under [Parity run](#parity-run-llm-169) below.

The file does not name the model under test, so its HF provider args
(`-M do_sample=false -M enable_thinking=false`) go on the command line. The
`extra_body` thinking switch applies to vLLM/SGLang and is ignored by Inspect's
HF provider. To use another judge endpoint, copy the file and change the
grader's `model` and `args`.

Known gaps, recorded in every sample's `parity_gaps` metadata: the legacy judge
received the **decoded chat-templated input** as the question, after truncating
the input to 1024 tokenizer tokens, and the target truncated to 256 tokens and
decoded. This task sends the raw question and target. Deterministic fixture
tests compare label parsing, aggregates and sample selection with the legacy
evaluator. Live-model parity is not claimed here.

## Parity run (LLM-169)
A parity run passes three things: `parity.yaml` (`--run-config`), the answer
model's provider args (`-M`), and the opt-in legacy text preprocessing (`-T
legacy_tokenizer`, the answer model's local tokenizer snapshot; add `-T
legacy_tokenizer_revision=<commit>` for a cached repository instead):

```sh
inspect eval llm_behavior_eval/medhallu \
  -T legacy_tokenizer="$ANSWER" \
  --model hf/local -M model_path="$ANSWER" \
  -M do_sample=false -M enable_thinking=false \
  -M tokenizer_call_args='{truncation: true, max_length: 1024}' \
  --run-config llm_behavior_eval/medhallu/parity.yaml
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
paired rows, 54 judge requests per side) was exact, but it ran the pre-redesign
implementation, which set these settings in code. A CPU replay of its recorded
requests through this package, configured only by `parity.yaml` and `-T
legacy_tokenizer`, reproduces every request, setting, per-sample score and
metric exactly. A GPU re-run with `parity.yaml` is still needed to confirm the
result (`scripts/parity/README.md`).
