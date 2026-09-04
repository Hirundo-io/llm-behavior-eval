# Uncensored Model adapter identity — BLOCKING CONFLICT, UNRESOLVED

## Verdict

**PROMPT-INJECTION GPU SMOKE BLOCKED — UNCENSORED MODEL ADAPTER IDENTITY
AMBIGUOUS**, for the Uncensored Model arms specifically. The Base Model arms
(no LoRA) have no dependency on this and remain fully preflight-verified.

## What was checked

An exhaustive filesystem search (both this repo and `hirundo-research`,
including all `.claude/worktrees/*` subdirectories, `mlruns/`, and every
config/manifest file) found:

- **Zero `adapter_model.safetensors` files anywhere on this machine.** No
  weights exist locally to hash. `hirundo-research`'s training configs
  (`scripts/rsch76_v2c_config.json`) point to
  `/home/ubuntu/hirundo-research/models/rsch-76-v2c/` — a remote GPU-box path
  not present here. The local MLflow store (`hirundo-research/mlruns/0/meta.yaml`)
  has no logged runs.
- **Two different, well-formed SHA-256 values, both claiming to identify the
  same named snapshot `654d5acdd2eb_0`** (same MLflow run id
  `654d5acdd2eb4b1ba613cb980abdada8`, same base model
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, same rank/alpha/
  dropout metadata):

| Value | Where it's recorded | Provenance claimed |
| --- | --- | --- |
| `9b3158331c001e94046469059a8e8c59d4f2f2095f882cb528f87fb3e8c3e9a2` | Notion "Appendix A — Reference Checkpoint" / "Appendix B" / "CCPC / RSCH-76 Data and Artifact Dictionary" / "CCPC-Bench — Internal Submission Tracker" (five independent pages, all consistent with each other) | Recovered from `run_config_resolved.json` inside the **CCPC-500 v2c continuity run**'s artifacts, dated 2026-08-29, described as fully valid (500/500 rows, "zero incomplete, unknown, unparseable, or judge-overflow rows") |
| `ab5c9beb854884db6c9c44675a2ec1c5a15c8a6e1cd2c173faac2647b6e6c74c` | `scripts/validate_ccpc216_thinking_on.py:20-22` (`EXPECTED_ADAPTER_SHA`) in **this very worktree** — an **untracked** file (`git status` shows `?? scripts/`), not part of any commit | A validator for a **different, separate execution**: the **CCPC-216 thinking-ON** run (`ARM_PATHS["v2c"] = "v2c/v2c-654d5acdd2eb-thinking-on/chinese_censorship"`), cross-referenced against a `run_metadata.txt` marker the validator expects that run to carry |

Both are syntactically valid 64-character hex SHA-256 strings. Neither could
be independently measured against a real file, because no real file exists
on this machine for either claim.

## Which of A/B/C/D from the investigation request applies

Cannot be determined mechanically without the actual weights:

- **(A) different model states** — plausible: the CCPC-500 continuity run
  (2026-08-29) and the CCPC-216 thinking-ON run could have loaded the adapter
  from two different export/materialization events that both happened to
  reuse the snapshot label `654d5acdd2eb_0` without the byte content being
  re-verified identical between them.
- **(B) identical tensors, different serialization** — plausible (e.g. a
  re-export via `hirundo-research/scripts/rsch76_export_qwen_adapter.py`,
  which the filesystem search found but could not run against any actual
  snapshot directory here) — cannot rule in or out without the files.
- **(C) one is a converted/repacked copy of the other** — same as (B); not
  distinguishable from documentation alone.
- **(D) one hash/report is stale or incorrect** — also plausible; the
  `ab5c9beb...` value lives only in an untracked, never-committed script that
  may itself be an abandoned draft, and its own correctness was never
  independently confirmed against a hashed file either.

**No inference from the snapshot name `654d5acdd2eb_0` being shared is drawn
here** — per instruction, snapshot-name equality is not treated as proof of
byte identity.

## Resolution required before the Uncensored Model arms can run

Whoever has access to the actual GPU-box artifact store (or the MLflow
artifact backing run `654d5acdd2eb4b1ba613cb980abdada8`) must:

1. Locate the real `adapter_model.safetensors` for the checkpoint actually
   intended for this study.
2. Compute its SHA-256 directly (`shasum -a 256 adapter_model.safetensors`).
3. Compare against both `9b3158331c...` and `ab5c9beb...`.
4. If it matches neither, treat both existing records as stale and record the
   newly-measured value as canonical, with fresh provenance.
5. Update `run_common.sh`'s `UNCENSORED_LORA_SHA256` (and
   `preflight.py`'s adapter check) to the confirmed value, and re-run
   `preflight.py verify-adapter <path>` against the real materialized
   directory before any smoke.

This freeze **does not pick one hash speculatively**. `preflight.py`'s
adapter check (see `preflight.py`) now requires the expected SHA-256 to be
passed explicitly at invocation time rather than defaulting silently to
either candidate, specifically so this ambiguity cannot be papered over by
a hardcoded default.
