# Uncensored Model adapter identity — RESOLVED

## Verdict

**UNCENSORED MODEL ADAPTER IDENTITY RESOLVED — canonical SHA-256
`ab5c9beb854884db6c9c44675a2ec1c5a15c8a6e1cd2c173faac2647b6e6c74c`;
the previously-cited `9b3158331c...` is a stale/incorrect documentation
value that never corresponded to a real file.**

## What resolved it

The initial search (previous round) covered only `hirundo-research` and the
`llm-behavior-eval` worktree and found no real `adapter_model.safetensors`
anywhere, so the conflict was reported as unresolvable. The corrected search
scope added `/Users/ilana/repos/artifacts/rsch-76/`, a separate local
artifact store (distinct from both `hirundo-research` and this worktree),
which contains a real, materialized copy:

```
/Users/ilana/repos/artifacts/rsch-76/v2c-adapter-654d5acdd2eb_0/
  adapter_model.safetensors   (84,972,248 bytes)
  adapter_config.json
  tokenizer_config.json
  chat_template.jinja
  README.md
```

**Directly measured** (`shasum -a 256`, and independently re-confirmed via
`preflight.py verify-adapter`):

```
sha256:  ab5c9beb854884db6c9c44675a2ec1c5a15c8a6e1cd2c173faac2647b6e6c74c
```

`adapter_config.json` (read directly, not from any doc):

```json
"base_model_name_or_path": "Qwen/Qwen3.5-4B"
"r": 16
"lora_alpha": 16
"lora_dropout": 0.1
"peft_type": "LORA"
"target_modules": ["down_proj","q_proj","up_proj","gate_proj","v_proj","k_proj","o_proj"]
```

**Tensor-level check** (via `safetensors.safe_open`, zero-GPU): 256 tensors,
`lora_A`/`lora_B` pairs per target module per layer, rank dimension 16
confirmed on both sides (e.g. `mlp.down_proj.lora_A.weight` shape
`[16, 9216]`, `lora_B.weight` shape `[2560, 16]`), dtype `F32`.

**This is the only real `adapter_model.safetensors` file found anywhere on
this machine** across both the corrected and original search scopes. There
is nothing to compare it against byte-for-byte, so "same tensors, different
serialization" (B) cannot be distinguished from "this is simply the one real
copy" -- but the latter is now the operative fact.

## Independent corroboration (two more sources, neither is this repo)

1. `/Users/ilana/repos/artifacts/rsch-76/ccpc500-v2c-generalization-v1/T3E_REPORT.md`
   -- the report for the **CCPC-500 v2c generalization run itself** (the
   exact run Notion's "Appendix A" cited as the *source* of `9b3158331c...`)
   records: `"...adapter SHA `ab5c9beb…`, benchmark SHA `77af7195…`, expected
   rows 500..."`. **The run's own report contradicts the Notion transcription
   of its own hash.**
2. `/Users/ilana/repos/artifacts/rsch-76/t4o-t3j-outcome-blind-audit-20260904/T4O_REPORT.md`
   -- an independent, outcome-blind audit of a separate study ("T3J",
   thinking safety/utility preservation), dated 2026-09-04, records:
   `"v2c adapter | SHA-256 `ab5c9beb854884db6c9c44675a2ec1c5a15c8a6e1cd2c173faac2647b6e6c74c`, base `Qwen/Qwen3.5-4B`, rank 16"`.

Both are independent of each other, independent of this freeze, and
independent of the `scripts/validate_ccpc216_thinking_on.py` file that first
surfaced `ab5c9beb...` in the previous round -- and all three, plus the
directly-measured file, agree.

## Relationship between the two hashes (answering A/B/C/D)

**(D) `9b3158331c001e94046469059a8e8c59d4f2f2095f882cb528f87fb3e8c3e9a2` is
stale/incorrect.** It appears in five Notion pages but was never checked
against real bytes -- and the one place it claims to be *sourced from*
(the CCPC-500 continuity run's own artifacts, per `T3E_REPORT.md`) actually
records `ab5c9beb...` instead. This looks like a transcription error that
propagated across Notion pages without independent verification, exactly
the failure mode this whole exercise exists to catch. It is not evidence of
a genuinely different model state, a re-export, or a conversion -- there is
no real file anywhere corresponding to it.

**A second, independent discrepancy was also caught by this correction:**
Notion/this freeze's earlier documents stated the LoRA `alpha` as `32`; the
real `adapter_config.json` records `lora_alpha: 16`. This has been corrected
throughout (`STUDY_CONTRACT.md`, `PARAMETER_PLUMBING.md`,
`preflight.py`, `run_common.sh`).

## What changed as a result

- `run_common.sh`: `UNCENSORED_LORA_SHA256` now defaults to
  `ab5c9beb854884db6c9c44675a2ec1c5a15c8a6e1cd2c173faac2647b6e6c74c`
  (still overridable, still verified by `preflight.py verify-adapter
  --expected-sha256` before every Uncensored Model arm -- this is a default,
  not a bypass of the check).
- `STUDY_CONTRACT.md`: STATUS updated from BLOCKED to resolved/ready for the
  Uncensored Model arms.
- Rank/alpha/dropout corrected to `16/16/0.1` everywhere (was incorrectly
  `16/32/0.1`).

## Residual caveat

This resolves *which SHA-256 is real and canonical on this machine*. It does
**not** independently re-verify that `/Users/ilana/repos/artifacts/rsch-76/v2c-adapter-654d5acdd2eb_0/`
is itself the byte-identical copy that will be materialized on whatever GPU
box actually runs the smoke/full study (Appendix A's documented GPU-box path
is `/home/ubuntu/hirundo-research/models/rsch-76-v2c/adapter_snapshots/654d5acdd2eb_0/`,
a different machine). `preflight.py verify-adapter` must still be run
against whatever `UNCENSORED_LORA_PATH` resolves to on that machine before
any real launch -- this freeze fixes *which hash is correct*, not *that
every future copy will automatically match it*.
