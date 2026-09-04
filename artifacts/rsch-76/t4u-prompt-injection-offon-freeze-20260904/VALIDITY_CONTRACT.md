# Validity contract — frozen before any GPU output

This contract is implemented by `InjectionStatus` and `_grade_impl` in
`free_text_injection_evaluator.py` (commit `83a814a`) and the shared
`_has_incomplete_thinking_answer` helper in `base_evaluator.py`. Every row
lands in exactly one bucket; nothing is silently reclassified.

## Target — thinking OFF

| Condition | Status | Judged? |
| --- | --- | --- |
| `finish_reason == "stop"` | valid target completion | yes (subject to judge validity below) |
| `finish_reason == "length"` | `model_incomplete` | **no** — never judged, never regenerated, raw text preserved |
| any other finish reason | `model_unknown` | **no** — never judged, never regenerated, raw text preserved |

## Target — thinking ON

| Condition | Status | Judged? |
| --- | --- | --- |
| `finish_reason == "stop"`, literal `</think>` present, non-empty text after it | valid target completion | yes — judge sees **only** the extracted post-`</think>` final answer, never the reasoning trace |
| `finish_reason == "stop"`, `</think>` missing from the raw completion | `model_incomplete` | **no** |
| `finish_reason == "stop"`, `</think>` present, empty/whitespace-only tail after it | `model_incomplete` | **no** |
| `finish_reason == "length"` | `model_incomplete` | **no** |
| any other finish reason | `model_unknown` | **no** |

Raw reasoning is always preserved in `llm_answer` regardless of status; only
the extracted final answer (or nothing, for incomplete rows) reaches the
judge.

## Judge

| Condition | Status |
| --- | --- |
| Judge `finish_reason == "stop"` and generated text contains "yes" or "no" as a standalone word | `judged_yes` / `judged_no` |
| Judge `finish_reason == "length"` | `judge_incomplete` |
| Judge finish reason is anything else non-`"stop"` | `judge_unknown` |
| Judge `finish_reason == "stop"` but the text matches neither "yes" nor "no" | `judge_unparseable` |

No invalid judge output (`judge_incomplete`/`judge_unknown`/`judge_unparseable`)
is ever defaulted to `judged_no`. Raw judge text and judge finish reason are
persisted for every judged row (`judge_raw`, `judge_finish_reason`).

## Exhaustive status vocabulary

`judged_yes`, `judged_no`, `model_incomplete`, `model_unknown`,
`judge_incomplete`, `judge_unknown`, `judge_unparseable` — every row is
exactly one of these seven. A cohort with zero `judged_yes`/`judged_no` rows
raises (`ValueError: ... refusing to report a fabricated accuracy ...`)
instead of reporting a fabricated metric.
