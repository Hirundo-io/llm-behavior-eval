# Dataset identity manifest — prompt-injection Purple Llama

## Primary population (executable now)

| Field | Value |
| --- | --- |
| Hugging Face dataset id | `hirundo-io/prompt-injection-purple-llama` |
| Pinned revision (immutable commit) | `403abe13df3913940c065e5af6ca471c4fb7daf6` |
| Revision recovery method | Resolved from the local HF hub cache ref (`refs/main` under `~/.cache/huggingface/hub/datasets--hirundo-io--prompt-injection-purple-llama/`), not invented. Frozen in `llm_behavior_eval/presets.py::DATASET_REVISIONS` as of commit `361842c`. |
| Split | `train` (only split) |
| Row count | **251** |
| `prompt_id` range | `0..250`, all 251 values unique and contiguous |
| Columns | `prompt_id`, `system_prompt`, `question`, `answer`, `judge_question` |
| Non-empty `system_prompt` | 251 / 251 |
| Non-empty `judge_question` | 251 / 251 |
| Parquet file (data/train-00000-of-00001.parquet) content hash | `ebaa62655c81f74bca803927f923af47a94a3da595bf8dda0479c4dba6c67a56` (SHA-256; this is the HF hub blob hash, i.e. the file's own content-addressed identity) |
| README.md blob hash | `2b5eaa94b2f19c6209156053bc8e9cf23d928d6539c366a1f67241ae326b5c11` (SHA-256) |
| Row-content JSON hash (all 251 rows, sorted keys, order-preserving) | `5d615a9f126f38e5c2bee11f2c09dcb467623905c412543d6c35e49d944058f0` (SHA-256; secondary cross-check, not a substitute for the parquet blob hash above) |
| `prompt_id` sequence hash | `b52b057481f55d1542888c9e2993884b84dad76ed832d87bb089d97a52469a5e` (SHA-256 of the JSON-encoded ordered `prompt_id` list) |

All hashes computed locally on 2026-09-04 by loading the pinned revision with
`datasets.load_dataset("hirundo-io/prompt-injection-purple-llama", revision="403abe13df3913940c065e5af6ca471c4fb7daf6")`
and hashing both the resolved parquet blob (via the HF hub cache's content-addressed
blob store) and the row content directly. Recompute both if this artifact is later
verified from a different machine.

## Secondary / expanded population — NOT executable from this freeze

| Field | Value |
| --- | --- |
| Datasets | `hirundo-io/bloom-prompt-injection-malicious-free-text`, `hirundo-io/bloom-prompt-injection-benign-free-text`, `hirundo-io/bloom-prompt-injection-conflicting-signals-free-text` |
| Row counts (locally cached, unverified against a pinned revision) | malicious: 1000, benign: 1135, conflicting-signals: 1000 |
| Wiring owner | PR #159 / LLM-117 (open, `mergeable=CONFLICTING` against current `origin/main`; not touched by this freeze) |
| Revision pin status | **NOT PINNED.** No immutable revision has been recovered or invented for any of the three Bloom datasets in this freeze. |
| Status | `NOT EXECUTABLE FROM THIS FREEZE UNTIL #149/#159 INTEGRATION IS RESOLVED AND IMMUTABLE BLOOM REVISIONS ARE PINNED.` |

Do not confuse the `-free-text` Bloom datasets above (the ones PR #159 actually wires)
with similarly-named but distinct cached datasets `hirundo-io/bloom-prompt-injection-malicious`
/ `-benign` (no `-free-text` suffix, 500 rows each) found in the local HF cache — those
are not referenced by PR #159 and are not part of any current or planned population here.

### Planned Bloom integration order (not started by this freeze)

1. PR #149 / LLM-119 (shared generation-alignment layer) merges to `main`.
2. PR #159 / LLM-117 (Bloom wiring) is rebased onto the resulting `main` and merged.
3. The residual validity behavior from commits `361842c` + `83a814a` is reconciled against
   whatever fail-open/incomplete-handling logic PR #159 lands with (a real, expected
   conflict — see `EVALUATOR_IDENTITY.md`).
4. Immutable Bloom dataset revisions are recovered (same method as above: resolved local
   hub cache ref, not invented) and pinned in `DATASET_REVISIONS`.
5. A separate Bloom smoke, then a separate Bloom full study, is planned and frozen on its
   own — it does not block or extend the Purple Llama OFF/ON study in this freeze.
