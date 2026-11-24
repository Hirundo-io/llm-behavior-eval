# llm-behavior-eval-security

The security benchmarks of
[llm-behavior-eval](https://github.com/Hirundo-io/llm-behavior-eval) as native
[Inspect](https://inspect.aisi.org.uk/) tasks. This distribution ships no code:
it depends on every published security benchmark distribution, so

```sh
pip install llm-behavior-eval-security
inspect eval llm_behavior_eval/<task> --model <provider/model> --model-role grader=<provider/judge>
```

installs them all, with Inspect and `llm-behavior-eval-common` but without the
legacy engine (no torch or transformers). Each benchmark is also installable on
its own (`pip install llm-behavior-eval-<benchmark>`), and `llm-behavior-eval`
depends on this distribution. Task names do not mention the group:
`llm_behavior_eval/<task>`.

In the repository, the group is `llm_behavior_eval/security/`: this
distribution's `pyproject.toml`, `LICENSE` and this README, and one directory
per benchmark, `llm_behavior_eval/security/<benchmark>/`, which is both the
benchmark's package and its distribution's project directory (a workspace
member nested in this one). In import space it is `llm_behavior_eval.security`,
a namespace package that every security benchmark contributes a portion to;
nothing ships `llm_behavior_eval/security/__init__.py`. This distribution's
wheel and sdist hold no module and none of its benchmarks' files.

## Benchmarks

| Benchmark | Directory and package (`llm_behavior_eval/security/`, `llm_behavior_eval.security.`) | Distribution | License | In this distribution | Status |
| --- | --- | --- | --- | --- | --- |
| Tensor Trust | `tensor_trust` | `llm-behavior-eval-tensor-trust` | MIT | yes | in review |
| SEP | `sep` | `llm-behavior-eval-sep` | MIT | yes | in review |
| LLMail-Inject | `llmail_inject` | `llm-behavior-eval-llmail-inject` | MIT | yes | in review |
| BIPIA | `bipia` | `llm-behavior-eval-bipia` | MIT | yes | in review |
| RuLES | `rules` | `llm-behavior-eval-rules` | Apache-2.0 (`LICENSE` and `NOTICE` of its own) | yes | in review |
| Raccoon | `raccoon` | `llm-behavior-eval-raccoon` | GPL-3.0-only (`LICENSE` and `NOTICE` of its own) | yes, unless it needs a Git dependency | planned |
| AgentDyn | `agentdyn` | `llm-behavior-eval-agentdyn` | MIT | **no**: isolated | planned |

Update the row when a benchmark merges (its "Status" becomes "merged").

## Add a security benchmark

Follow "Add or migrate a benchmark" in `INSPECT_MIGRATION.md` with
`<group>` = `security`:

1. The directory is `llm_behavior_eval/security/<benchmark>/`: the package
   `llm_behavior_eval.security.<benchmark>` and, beside its modules, the
   distribution's `pyproject.toml`, `LICENSE` and any `NOTICE` (never an
   `__init__.py` in `llm_behavior_eval/security/`). Its `pyproject.toml` has
   `packages = ["llm_behavior_eval.security.<benchmark>"]`,
   `package-dir = { "llm_behavior_eval.security.<benchmark>" = "." }`,
   `include-package-data = false`, `readme = "README.md"` and the entry point
   `llm_behavior_eval = "llm_behavior_eval.security.<benchmark>.task"`.
   Code and tests import `llm_behavior_eval.security.<benchmark>`.
2. Register it in the root `pyproject.toml` (workspace member
   `"llm_behavior_eval/security/<benchmark>"`, source, dev group), but **not**
   in the root's `[project] dependencies`: add
   `"llm-behavior-eval-<benchmark>>=<the root version>,<0.2",` to this
   distribution's `dependencies` instead (sorted by name). The root reaches it
   through `llm-behavior-eval-security`.
3. Update its row in the table above (or add one).
4. Run `uv lock`, then the checks in step 7 of the recipe, including
   `python scripts/run_package_tests.py llm-behavior-eval-<benchmark>`.

A benchmark whose code derives from copyleft upstream code keeps its own
license (Raccoon: GPL-3.0-only with a `NOTICE`); this distribution stays MIT and
only depends on it. Installing does not combine the code, but anyone
redistributing an environment that includes it takes on the GPL's obligations
for that distribution. A benchmark that carries permissively licensed upstream
code keeps that license too (RuLES: Apache-2.0, with a `NOTICE` naming the
upstream files and revision).

## Unpublished and isolated benchmarks

PyPI rejects a distribution whose requirements include a direct URL (`name @
git+https://...`), and a user's environment must never get a forked
distribution in place of the one it shadows. A benchmark that needs either:

* stays a workspace member with its own tests (CI runs them in its light
  environment), and declares why in its `pyproject.toml`:

  ```toml
  [tool.llm-behavior-eval]
  unpublished = "depends on the agentdojo fork from git, which replaces stock AgentDojo"
  ```

* is **not** in this distribution's `dependencies`, not in an extra (an extra
  naming an unpublished distribution cannot resolve from PyPI, and opting in
  would still swap the user's AgentDojo), and not in the root's dependencies;
  `tests/test_packaging.py` checks this, and the release workflow skips its
  artifacts;
* is installed explicitly from the repository, in an environment of its own:

  ```sh
  pip install "llm-behavior-eval-agentdyn @ git+https://github.com/Hirundo-io/llm-behavior-eval@<tag>#subdirectory=llm_behavior_eval/security/agentdyn"
  ```

AgentDyn is such a benchmark: it depends on a Git fork published under the
distribution name `agentdojo`, which conflicts with stock AgentDojo. If another
workspace member ever requires stock `agentdojo`, also declare the two members
as conflicting in the root `pyproject.toml`
(`[tool.uv] conflicts = [[{ package = "llm-behavior-eval-agentdyn" }, { package = "<other>" }]]`)
and leave AgentDyn out of the dev group. If Raccoon keeps a direct Git
dependency on its upstream code (as the hirundo-safety adapter does), it is
unpublished the same way until that code is on PyPI. RuLES does not: it
carries the upstream scenario checks it needs, under upstream's Apache-2.0.
