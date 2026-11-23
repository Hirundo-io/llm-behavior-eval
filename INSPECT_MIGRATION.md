# Native Inspect benchmarks

Native benchmarks run on [Inspect AI](https://inspect.aisi.org.uk/) alongside
the legacy engine, which stays unchanged. Inspect is required at
`>=0.3.289,<0.4` (0.3.289 is the version hirundo-core pins; `<0.4` because a
0.x minor release may change its APIs); it needs Python 3.11+, so its
dependencies carry a Python-version marker and the Inspect tests are not
collected on Python 3.10.

Each native benchmark is its own distribution (see [Packaging](#packaging)).
In a checkout, `uv sync` installs the whole workspace editable; a user installs
`llm-behavior-eval` (every merged benchmark, with the legacy engine) or just
`llm-behavior-eval-<benchmark>` (that benchmark, Inspect and
`llm-behavior-eval-common`; no torch or transformers):

```sh
uv sync   # or: pip install llm-behavior-eval-<benchmark>
uv run inspect eval llm_behavior_eval/<benchmark> \
  --model <provider/model> --model-role grader=<provider/judge>
```

## Design rules

1. **Benchmarks are self-contained.** Each benchmark package owns its prompts,
   dataset contract, scorer, metrics, README, `parity.yaml` and
   `target_metrics.json`, and is its own distribution (see
   [Packaging](#packaging)). `llm_behavior_eval.common`
   (`llm-behavior-eval-common`) holds only code that several benchmarks would
   otherwise copy (see [Shared code](#shared-code));
   add to it only what at least two benchmarks need.
2. **No inference settings in code** for the model under test or the grader:
   no temperature, token budgets, reasoning or thinking switches. Unset
   settings fall through to the serving stack's defaults. Users set them per
   role: eval-level flags (`--temperature`, `--max-tokens`, `-M ...`) for the
   model under test, and `--model-role grader=...` with its own config for the
   judge.
3. **Parity is an opt-in run config.** `parity.yaml` holds the legacy engine's
   exact settings for both roles and applies only when passed with
   `--run-config`.
4. **The grader is the required `grader` role** and never falls back to the
   model under test.
5. **No credentials in task arguments**: Inspect writes every task argument to
   the `.eval` log. Datasets read `LLM_BEHAVIOR_EVAL_DATASET_TOKEN`, then
   Hugging Face's own resolution (`HF_TOKEN`, a cached login); models take
   theirs from their provider (`get_model(..., api_key=...)` or the provider's
   environment variable).
6. **Pinned data.** A benchmark pins each Hub dataset's commit as the default
   `revision` and validates what it loads; `dataset_path` reads a local copy
   (a `load_dataset` folder or a `save_to_disk` folder). Every pinned upstream
   commit (data or reference code) is also listed in
   `.github/upstream-revisions.env` (see [Upstream revisions](#upstream-revisions)).
7. **Thinking.** Benchmarks rely on Inspect's native reasoning parsing: the
   judge sees `ModelOutput.completion`, which leaves out reasoning a provider
   parsed out. A `judge_sees_answer_reasoning` task argument gives the judge
   that reasoning too. Each benchmark README documents the limitations
   (unparsed traces, Inspect's `hf/` provider).
8. **Metrics are unreduced** (`@metric(scores="unreduced")`): every (sample,
   epoch) is one observation, so `--epochs N` keeps every epoch's metadata.
   Every task lists `completion_scorer()` beside its grading scorer for
   `incomplete_response_rate`.
9. **Registry names are package-prefixed** (`<benchmark>_scorer`,
   `<benchmark>_metrics`): Inspect keeps one registry per object type and a
   later registration silently replaces an earlier one.

## Packaging

The repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
of several distributions that share one import package, `llm_behavior_eval`:

| Distribution | Directory | Ships | Depends on |
| --- | --- | --- | --- |
| `llm-behavior-eval` | `.` (workspace root) | the legacy engine: `llm_behavior_eval/__init__.py`, `evaluate.py`, `evaluation_utils/` … (unchanged) | the legacy stack (torch, transformers…), `llm-behavior-eval-common` and every merged benchmark (or group) distribution |
| `llm-behavior-eval-common` | `llm_behavior_eval/common/` | `llm_behavior_eval.common` | Inspect, `inspect-evals-utils`, `openai`, `datasets` |
| `llm-behavior-eval-<benchmark>` | `llm_behavior_eval/[<group>/]<benchmark>/` | `llm_behavior_eval.[<group>.]<benchmark>` | `llm-behavior-eval-common` and the benchmark's own libraries |

Each native distribution's `pyproject.toml` lives inside its own package
directory, in place under the legacy package, so the import paths and task
names are what they would be in one package:

```text
pyproject.toml                       llm-behavior-eval (flat layout) + [tool.uv.workspace]
llm_behavior_eval/
  __init__.py  evaluate.py  evaluation_utils/ …     the legacy engine, untouched
  common/                            llm-behavior-eval-common
    pyproject.toml  LICENSE  README.md
    __init__.py  dataset.py  scoring.py  target_metrics.py
  <group>/                           a group of benchmarks: no __init__.py (its README lists them)
    <benchmark>/                     llm-behavior-eval-<benchmark>
      pyproject.toml  LICENSE  [NOTICE]
      __init__.py  task.py …  README.md  target_metrics.json
tests/                               every test module, the shared helpers, and
  example_benchmark/                 the synthetic benchmark distribution (template; never published)
scripts/                             workspace tools (workspace.py reads the layout from pyproject.toml)
```

A native distribution maps its package to its own directory, and ships only
its modules and listed data files:

```toml
[tool.setuptools]
packages = ["llm_behavior_eval.<group>.<benchmark>"]   # plus any subpackage, listed explicitly
package-dir = { "llm_behavior_eval.<group>.<benchmark>" = "." }
include-package-data = false   # otherwise pyproject.toml and LICENSE ship inside the package
```

The root's `[tool.setuptools.packages.find]` includes only `llm_behavior_eval`
and `llm_behavior_eval.evaluation_utils*`, so its wheel never ships `common/`
or a benchmark.

### One import package, several distributions

* Only `llm-behavior-eval` ships `llm_behavior_eval/__init__.py` (the legacy
  file, unchanged). Every other distribution ships a regular package below
  it, and nothing at the namespace levels (`llm_behavior_eval/` and a group
  directory have no `__init__.py` of theirs).
* Wheels installed together land in one `site-packages/llm_behavior_eval/`
  directory, each distribution owning its own files, under the legacy
  `__init__.py`.
* A benchmark installed without `llm-behavior-eval` (Inspect,
  `llm-behavior-eval-common` and the benchmark) makes `llm_behavior_eval` a
  PEP 420 namespace package: the legacy engine is absent, not just unloaded.
* In the editable workspace (`uv sync`) the root is installed in setuptools'
  `compat` editable mode (`[tool.uv.config-settings-package]`): a `.pth` file
  puts the repository root on `sys.path`, `llm_behavior_eval` is the
  repository's regular package, and `common` and every benchmark are found in
  place below it. setuptools' default mode would not work here: it gives the
  root's flat layout an import hook that runs after the regular path search,
  while each member's editable install (its package directory is not
  `<dir>/llm_behavior_eval/…`) registers `llm_behavior_eval` as a namespace
  on `sys.path`. Outside the repository root that namespace would win, and
  `import llm_behavior_eval` would skip the legacy `__init__.py`
  (`__version__`, the lazy exports). In compat mode the repository root is a
  `sys.path` entry, and Python prefers a regular package on any `sys.path`
  entry to namespace portions. To install the root editable outside `uv sync`,
  pass `-C editable_mode=compat` (`uv pip install` run in the repository root
  reads the setting from `pyproject.toml` itself).
* Inspect names a task `llm_behavior_eval/<task>` in all three cases: from
  the module's top-level package when the module is in `site-packages`, or
  when the `llm-behavior-eval` distribution is installed editable (it checks
  the distribution named like the top-level package). A benchmark installed
  editable *without* the root distribution would lose the prefix, so light
  environments install workspace members as wheels (`--no-editable`).
* `scripts/check_installs.py` (CI) proves this: it copies the repository,
  stages the synthetic benchmark where a benchmark lives, builds the wheels,
  installs them light, together and editable in fresh environments, and
  checks `eval.task == "llm_behavior_eval/example"` and which
  `llm_behavior_eval/__init__.py` Python imports, from outside the copy.
* A benchmark's directory is also its project directory, so building it in
  place (`uv build --wheel`, or a non-editable `uv pip install`) leaves a
  git-ignored `build/` copy and `*.egg-info` there. Nothing imports them;
  `load_target_metrics` searches only packages below a benchmark package,
  and `uv build` (sdist, then the wheel from it) leaves no `build/`.

### Entry points

Each benchmark distribution declares one `inspect_ai` entry point, **named
`llm_behavior_eval`**, whose value is its own task module:

```toml
[project.entry-points.inspect_ai]
llm_behavior_eval = "llm_behavior_eval.<group>.<benchmark>.task"
```

Entry-point names are therefore shared and the values unique. Inspect resolves
`llm_behavior_eval/<task>` lazily by loading only the `inspect_ai` entry
points *named* `llm_behavior_eval` (`registry_lookup` →
`ensure_entry_points("llm_behavior_eval")`); with a distinct name such as
`llm_behavior_eval_<benchmark>` that lookup returns nothing, so a scorer or
task name from a log would not resolve. Importing a task module twice is a
no-op, so nothing registers twice. `llm-behavior-eval-common` and
`llm-behavior-eval` declare no entry point (there is no `_registry.py` to scan
for benchmarks any more).

### Versions and dependencies

* Every distribution has the root's version; `bumpver` updates them all
  (`[tool.bumpver.file_patterns]`), and one release publishes them all.
* A workspace dependency follows the dependency cap policy: a lower bound and
  the next minor, `llm-behavior-eval-common>=0.1.12,<0.2`. Raise the lower
  bound when a benchmark needs something new in `common`.
* Benchmark and common distributions never depend on `llm-behavior-eval`,
  torch, transformers, vllm or accelerate (`tests/test_packaging.py` checks).
* `requires-python = ">=3.11,<3.14"` for every native distribution (Inspect
  needs 3.11). uv locks the workspace for the intersection of its members, so
  `uv.lock` covers Python 3.11–3.13 only, although `llm-behavior-eval` itself
  still declares 3.10.12.
* `llm-behavior-eval` lists `llm-behavior-eval-common` and every merged
  benchmark (or its group distribution) in `[project] dependencies`, so
  installing it installs every merged benchmark.

## Benchmark layout

A benchmark is the regular package `llm_behavior_eval.<group>.<benchmark>`
(or `llm_behavior_eval.<benchmark>` without a group) in its own distribution:

| File | Contents |
| --- | --- |
| `task.py` | The `@task`: no `config=`, no default grader; `Task(version=...)` bumped whenever results can change. The entry point imports this module. |
| `dataset.py` | Pinned coordinates, `record_to_sample`, validation; loads with `common.dataset.load_dataset`. |
| `prompts.py` | Prompts, copied verbatim from the legacy engine where they exist. |
| `scorer.py` | The benchmark's scorer, using `common.scoring` for the judged answer and the grader call. |
| `metrics.py` | Unreduced benchmark metrics. |
| `README.md` | What it measures, task arguments, metrics, reasoning handling, parity status. Also the distribution's PyPI description. |
| `parity.yaml` | Legacy settings as an Inspect run config. |
| `target_metrics.json` | Report metrics per task, in inspect-brief's format. |

The same directory is the distribution's: it adds `pyproject.toml`,
`LICENSE` and, when upstream terms ask for one, `NOTICE` (none of them
shipped inside the package). Tests stay in `tests/test_inspect_<benchmark>.py`.

**Groups.** Security benchmarks go in the `security` group. Any other
benchmark is ungrouped (`llm_behavior_eval/<benchmark>/`,
`llm_behavior_eval.<benchmark>`) unless its ticket names a group. A group
directory has no `__init__.py` (several distributions share it); its
`README.md` lists the group's benchmarks and any group-specific steps.

## Add or migrate a benchmark

Follow these steps literally; `tests/test_packaging.py` checks every one that
can be checked. Below, `<benchmark>` is the import name (`tensor_trust`),
`<dist>` the distribution name (`llm-behavior-eval-tensor-trust`: underscores
become hyphens) and `<pkg>` the dotted package
(`llm_behavior_eval.security.tensor_trust`, or `llm_behavior_eval.<benchmark>`
ungrouped); `<dir>` is `llm_behavior_eval/<group>/<benchmark>` (or
`llm_behavior_eval/<benchmark>`): the package's own directory, which is also
the distribution's.

1. **Start from the template.** Copy `tests/example_benchmark/` to `<dir>/`
   (create the group directory with no `__init__.py`) and replace its
   modules, `README.md` and `target_metrics.json` with the benchmark's.
   Migrating a benchmark from a branch that used `llm_behavior_eval/<benchmark>/`:
   `git mv llm_behavior_eval/<benchmark> <dir>` when it moves into a group
   (nothing to move otherwise), copy `pyproject.toml` and `LICENSE` from the
   template into `<dir>/`, then replace `llm_behavior_eval.<benchmark>` with
   `<pkg>` in its code and tests. Task, scorer and metric names do not change.
   Do not carry over the branch's edits to `common/`, `_registry.py`,
   `pyproject.toml` or the legacy engine; rebase onto this layout instead.
2. **`<dir>/pyproject.toml`**, from the template:

   ```toml
   [build-system]
   requires = ["setuptools>=77"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "<dist>"
   version = "<the root version>"
   description = "<one line>"
   readme = "README.md"
   license = "MIT"                 # or the SPDX expression upstream terms require
   license-files = ["LICENSE"]     # plus "NOTICE" when there is one
   authors = [{ name = "Hirundo", email = "dev@hirundo.io" }]
   requires-python = ">=3.11,<3.14"
   # Upper bounds stop at the next release line that may break the APIs used here: the next
   # major version, or the next minor for a 0.x package. Lines with a narrower bound say why.
   dependencies = [
     "llm-behavior-eval-common>=<the root version>,<0.2",  # <0.2 while the version is 0.1.x
     # the benchmark's own libraries, each with a comment if it needs one
   ]

   [project.urls]
   Homepage = "https://github.com/Hirundo-io/llm-behavior-eval"

   [project.entry-points.inspect_ai]
   llm_behavior_eval = "<pkg>.task"

   [tool.setuptools]
   packages = ["<pkg>"]
   package-dir = { "<pkg>" = "." }
   include-package-data = false

   [tool.setuptools.package-data]
   "<pkg>" = ["README.md", "parity.yaml", "target_metrics.json"]
   ```

   `package-dir` maps the package to `<dir>` itself. A subpackage (a
   directory with `__init__.py` inside `<dir>`) must be listed in `packages`
   too (`"<pkg>.<sub>"`; `package-dir` covers it); setuptools ships nothing
   that is not listed. `package-data` must list `README.md` and
   `target_metrics.json`, and `parity.yaml` when the benchmark has one (the
   template lists it either way; a missing file is skipped). Add any other
   data file the package reads. Inspect itself comes from `llm-behavior-eval-common`;
   never depend on `llm-behavior-eval`.
3. **License.** `<dir>/LICENSE` is a byte-for-byte copy of the root `LICENSE`
   for MIT. When upstream terms force another license (for example
   GPL-3.0-only), put that license's text in `LICENSE`, set `license`
   accordingly and add a `NOTICE` describing what derives from upstream, at
   which revision; list it in `license-files`.
4. **Register it in the root `pyproject.toml`** (one line each):
   * `[tool.uv.workspace] members`: `"<dir>"`;
   * `[tool.uv.sources]`: `<dist> = { workspace = true }`;
   * `[dependency-groups] dev`: `"<dist>"`;
   * `[project] dependencies`: `"<dist>>=<the root version>,<0.2; python_version >= '3.11'"`
     unless its group has a group distribution (`llm_behavior_eval/<group>/pyproject.toml`):
     then add it to that distribution's dependencies instead (see the group's
     README).
   Then run `uv lock`.
5. **Upstream pins.** For every upstream commit the code pins (a dataset or
   reference code), add an entry to `.github/upstream-revisions.env` (see
   [Upstream revisions](#upstream-revisions)).
6. **Tests.** `tests/test_inspect_<benchmark>.py` imports from `<pkg>`, runs
   the registered task on mock models and asserts
   `missing_target_metrics(log.location) == []` and
   `reduced_metrics(task) == []` (see [Tests](#tests)). It must pass in the
   benchmark's light environment:
   `python scripts/run_package_tests.py <dist>`.
7. **Check** before committing: `uv sync`, `uv run pytest`,
   `uv run ruff format --check . && uv run ruff check .`,
   `uv run pyrefly check`, `python scripts/run_package_tests.py <dist>`, and,
   when packaging changed, `uv build --all-packages --out-dir dist &&
   python scripts/check_artifact_licenses.py dist`. `*.json` is git-ignored
   except `target_metrics.json` under `llm_behavior_eval/`: add a `.gitignore`
   exception for any other JSON file the benchmark ships.

## Shared code

`llm_behavior_eval.common`, the `llm-behavior-eval-common` distribution
(`llm_behavior_eval/common/`):

| Module | API |
| --- | --- |
| `dataset` | `dataset_token()`; `load_dataset(path, *, split, sample_fields, revision=None, name=None, seed=None)`: a pinned Hub load through `inspect_evals_utils`' retrying `hf_dataset`, a local source through Inspect's `hf_dataset` with its split cache off (a file replaced at the same path is read again; a folder with a stray `state.json` is rejected with an explanation), a `save_to_disk` folder through `load_from_disk` (read-only). With `seed`, HF's `Dataset.shuffle(seed=seed)` runs before conversion on every path; without it, rows keep their published order. |
| `scoring` | `stop_reason(state)`; `judged_answer(state, include_reasoning=...)`; `grader()` and `ask_grader(system, user)` on the required role with no settings; `completion_scorer()` and its unreduced `incomplete_response_rate` metric (1 for `max_tokens`/`model_length`, 0 for another known stop reason, unscored when the provider reports none). |
| `target_metrics` | `load_target_metrics()` validates (a JSON object of non-empty metric lists, string names, boolean flags) and merges every `target_metrics.json` in a package at any depth below `llm_behavior_eval`, in every directory it spans (`package_directories()`; below a benchmark package only its subpackages, so an in-place `build/` copy is skipped); `missing_target_metrics(log)` lists target names a log lacks, as inspect-brief looks them up. |

`inspect-evals-utils` (inspect_evals' utilities, without its benchmarks) is a
dependency: benchmarks may use it directly, for example `create_stable_id` and
`filter_duplicate_ids`.

## Tests

`tests/conftest.py` provides `hf_dataset_cache` (Inspect's split cache in
`tmp_path`) and `hub` (serves rows as the Hub dataset, recording the
`load_dataset` calls). `tests/inspect_helpers.py` holds the helpers both
benchmark suites use: `eval_metrics`, `sample_scores`, `reduced_metrics`,
`mock_grader`, `assert_grader_required`, `assert_no_credentials_logged`,
`local_openai_judge` and `message_text` (a loopback OpenAI-compatible endpoint
that exercises Inspect's real provider parsing).

Every benchmark test suite runs its registered task on mock models and asserts
`missing_target_metrics(log.location) == []` and `reduced_metrics(task) == []`.
Use synthetic rows for private datasets.

Tests stay in the root `tests/` directory and import only pytest, Inspect, the
shared code and their own benchmark, so each one also runs in its
distribution's **light environment** (no legacy engine, no other benchmark):

```sh
python scripts/run_package_tests.py <dist>    # every native distribution without arguments
# which runs, for each distribution with tests/test_inspect_<leaf>.py files:
uv run --isolated --no-editable --locked --package <dist> --with pytest \
  pytest tests/test_inspect_<benchmark>.py
```

`--isolated` uses a throwaway environment (the workspace `.venv` stays as it
is); `--no-editable` installs the workspace members as wheels, so Inspect
names tasks as users see them (see [Packaging](#packaging)). The full suite
runs in the workspace environment with `uv run pytest`.
`tests/test_packaging.py` checks the packaging conventions for every
workspace member and for the synthetic benchmark.

## Upstream revisions

`.github/upstream-revisions.env` lists every upstream commit a native benchmark
pins, three lines per entry:

```sh
# renovate: datasource=github-digest depName=<owner>/<repo> currentValue=<branch>
# source: <file from the repository root>:<CONSTANT>
<BENCHMARK>_<WHAT>_REV=<40-character commit>
```

`<CONSTANT>` is a module-level string constant in the benchmark's code (or
`<CONSTANT>["<key>"]` for one value of a module-level dict literal). A Hugging
Face dataset is a Git repository: use `datasource=git-refs depName=<org>/<dataset>
packageName=https://huggingface.co/datasets/<org>/<dataset> currentValue=main`.
Keep entries sorted by key, separated by a blank line.

* `tests/test_upstream_revisions.py` fails when the code and the manifest
  disagree; `tests/test_renovate_configuration.py` checks that Renovate's
  regex manager extracts every entry.
* Renovate (`.github/renovate.json`, run weekly by
  `.github/workflows/renovate.yaml`) opens a pull request labelled
  `upstream-revision` that moves only the manifest when the tracked branch
  moves. In that pull request, run `python scripts/sync_upstream_revisions.py`
  (it replaces the old commit in every text file of the owning distribution),
  review the upstream changes, and update checksums, row counts and anything
  else derived from the data. `--check` lists mismatches without changing
  anything.

## Licensing

Every distribution carries its own `LICENSE` (an exact copy of the root MIT
`LICENSE` unless upstream terms require another license) and declares it with
an SPDX `license` and `license-files`. A distribution whose code derives from
copyleft upstream code (for example Raccoon, GPL-3.0-only) keeps that license
and a `NOTICE` recording what derives from which upstream revision; it stays
a separate distribution so the others remain MIT. These licenses cover code,
not benchmark datasets: each benchmark README states its data's terms.
`scripts/check_artifact_licenses.py dist` (CI, after `uv build --all-packages
--out-dir dist`) checks that every wheel and sdist has exactly the declared
license files and `License-Expression`; `tests/test_packaging.py` checks the
sources and builds two distributions to check them too.

## CI

* `tests.yaml`: the full suite in the workspace (`uv sync --locked`); each
  native distribution's tests in its light environment
  (`scripts/run_package_tests.py`); `uv build --all-packages`, `twine check
  --strict`, the license check, and the install matrix
  (`scripts/check_installs.py`).
* `pyrefly.yaml` runs the locked pyrefly; `ruff.yaml` is unchanged.
* `vulnerability-scan.yaml`: pip-audit of the installed workspace, `uv audit`
  of `uv.lock` (reporting only, until the lock's known advisories are
  triaged), and zizmor on the workflows (`.github/zizmor.yml`). Every action
  is pinned to a commit with its version in a comment, which Renovate keeps
  current.
* `deploy_to_pypi.yaml` builds and publishes every distribution; each needs
  its own PyPI project with this workflow as a trusted publisher before its
  first release.

## Caveats

* Inspect's `hf/`, `nnterp/` and `transformer_lens/` providers report no stop
  reason (`unknown`); benchmarks that grade only finished answers reject them
  for the model under test, and `incomplete_response_rate` is NaN there.
  `hf/` applies a `revision` model arg to the weights only, not the tokenizer.
* Inspect namespaces registry names from install metadata (see
  [Packaging](#packaging)). setuptools leaves `*.egg-info` folders in each
  project directory (the repository root, and each native package's own
  directory); they must not come before `site-packages` on `sys.path`, which
  is why pytest uses `--import-mode=importlib`.
* In the editable workspace Inspect records `llm-behavior-eval` rather than the
  benchmark's distribution in a log's `eval.packages` (its editable root is
  the repository root, which contains every member); installed wheels record
  the benchmark's own distribution.
