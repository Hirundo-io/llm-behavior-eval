# Removal baseline

Source: `bae7a508649f48f2755ce831822b2fcb91d2ba71` (LLM-171 candidate base).

`removal_baseline.json` is data, never an executable legacy implementation.
The parser/prompt vectors were evaluated against that revision's evaluator
functions/constants before removal. Aggregate `saved`/`captured` payloads were
recorded from the existing native-vs-legacy tests while they still executed the
legacy `_grade_impl` with deterministic judge stubs. The selected baseline run
reported **90 passed, 2 skipped, 134 deselected**. Binary logs were replaced by
JSON for that local capture to avoid lazy log access hanging in this environment;
the answers, scorer inputs and legacy grading calls were unchanged.

Fixtures cover BBQ bias/unbias, BLOOM bias/unbias/ambiguous and UNQOVER with and
without stereotype columns; HaluEval grades/skips; injection Yes/No/skips;
XSTest and ORBench response records plus every refusal diagnostic/count.
The MedHallu order freezes the baseline loader's seed-42 shuffle of 502 rows,
then cap at 500. CCPC freezes the deterministic 500-row test-source fingerprint,
boolean rubric constants and fail-closed metric cases. Preprocessor widths record
the baseline fake-character-tokenizer truncation assertions; the native tests
explicitly retain the longer raw strings, documenting unresolved divergence.
Parser tests inline the same frozen vectors for readable parametrized failures.
`test_semantic_regressions.py` retains the additional old free-text utility vectors.

Do not regenerate expectations from the native scorer under test. An intended
semantic change requires a reviewed new reference and provenance. These fixtures
are deterministic regression evidence, not live provider/tokenizer parity or
approval of LLM-169.

`preprocessing_baseline.json` and `probe_baseline.json` were captured from actual
legacy preprocessing at `5cce36cba76a99271f234973853216cbb511a709` before removal,
using the pinned Gemma3/Qwen3 tokenizer fixtures (hashes asserted in tests).
They preserve left/right 1024-token prompts, 256-token targets/stereotypes,
independent judge questions, special tokens, thinking flags and mixed batches.
The probe uses the deterministic character tokenizer from test_parity_evidence.
These are data fixtures, not a second implementation of the retired engine.

`purple_baseline.json` captures the actual `_grade_impl` saved accuracy converted
with `100 * (1 - accuracy)` at the same 5cce36c parent, including its b9a5b28 fix.
Assertions use exact float equality, including 19.999999999999996 and
9.999999999999998; one truncated response is excluded in every vector.
