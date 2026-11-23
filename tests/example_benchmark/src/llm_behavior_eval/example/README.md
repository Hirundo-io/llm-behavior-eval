# Example (synthetic)

A synthetic benchmark that exists only for llm-behavior-eval's packaging tests
and as the template for a benchmark distribution. It is never published.

`llm_behavior_eval/example` asks two fixed prompts and has the required
`grader` role reply YES when an answer complies.

| Metric | Meaning |
| --- | --- |
| `example_compliance_rate` | Share of answers the grader judged compliant. |
| `incomplete_response_rate` | Share of answers stopped by a length limit. |
