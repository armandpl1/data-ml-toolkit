# data-ml-toolkit

Shared SDK and tool versions for my data engineering and ML repositories:

| Repo                         | Uses                                         |
| ---------------------------- | -------------------------------------------- |
| `ml-claim-triage-engine`     | SDK (logging, config) and the `dev` pins     |
| `movie_recommendation_app`   | SDK (logging, config) and the `dev` pins     |
| `sql-performance-lab`        | The `dev` pins                               |
| `ecom-data-modeling-project` | The `dev` pins                               |

It does two jobs:

1. **A small SDK**, so code isn't copied between repos:
   - `data_ml_toolkit.logging`: structured JSON logs (`get_logger`, `log_event`)
   - `data_ml_toolkit.config`: strict YAML loading and repo-relative paths
   - `data_ml_toolkit.s3`: upload, download and list objects in AWS S3
     (install the `aws` extra)
   - `data_ml_toolkit.llm`: one interface for LLM calls, with record/replay
     so tests never call a model, and a cost ledger with a spending cap
     (install the `llm` extra for live calls)
   - `data_ml_toolkit.evals`: golden datasets with frozen splits, bootstrap
     intervals, regression checks against run-to-run noise, and an LLM judge
     with calibration against human labels (install the `evals` extra)
2. **One set of dev tool versions.** The `dev` extra pins ruff, mypy,
   pytest and pre-commit. Every repo depends on `data-ml-toolkit[dev]`
   instead of pinning them itself, so all repos lint and test with the same
   versions. `dmt check-sync` proves it.

The roadmap is in [docs/STEPS.md](docs/STEPS.md).

## Use it in a repo

The toolkit is installed from GitHub at a version tag; no package index needed.

```bash
# Runtime SDK (logging, config); add [aws], [llm] or [evals] as needed
uv add "data-ml-toolkit[llm,evals] @ git+https://github.com/armandpl1/data-ml-toolkit@v0.2.0"

# Shared dev tool versions, in place of ruff, mypy, pytest and pre-commit
uv add --dev "data-ml-toolkit[dev] @ git+https://github.com/armandpl1/data-ml-toolkit@v0.2.0"
```

```python
import logging
from data_ml_toolkit.logging import get_logger, log_event

logger = get_logger(__name__)
log_event(logger, logging.INFO, "rows_loaded", table="claims", rows=26444)
# {"ts": "...", "level": "INFO", "logger": "...", "event": "rows_loaded", "table": "claims", "rows": 26444}
```

## LLM calls and evals

```python
from data_ml_toolkit.llm import LLM, AnthropicProvider, Ledger, Recorded, Request

llm = LLM(
    Recorded(AnthropicProvider(), "tests/recordings", mode="auto"),
    Ledger(cap_usd=1.00, path="reports/llm_ledger.jsonl"),
)
reply = llm.complete(Request(model="claude-haiku-5-5", prompt="...", effort="low"))
```

- **Record once, replay forever.** Each response is saved under a hash of
  its request. With `mode="replay"` (the default) nothing calls a model: a
  missing recording is an error, so CI is free and deterministic. Record
  with `mode="record"` or `"auto"` and commit the files.
- **Ledger.** Every call's tokens, cost and latency are logged. Live calls
  stop with `BudgetExceeded` once `cap_usd` is spent; replays cost nothing.
  Prices for the Claude models in use are in `llm.PRICES` (list prices, US
  dollars per million tokens).
- **Providers.** `AnthropicProvider` uses the official `anthropic` SDK, with
  credentials from the environment. Anything with
  `complete(request) -> Response` is a provider; others (OpenAI, Google,
  Ollama) are added when a repo needs them.

```python
from data_ml_toolkit.evals import GoldenSet, Judge, calibrate, compare

golden = GoldenSet.load("evals/golden.jsonl")   # hash-based dev/test split
assert golden.version == "3f1c..."               # pin it: edits are noticed
judge = Judge(llm, "claude-haiku-5-5", rubric="...", labels=["faithful", "unfaithful"])
calibrate(judge_labels, human_labels).trusted()  # Cohen's kappa >= 0.6
compare(baseline_runs, candidate_runs).verdict   # improved / regressed / no_change
```

`compare` counts a change only if its paired bootstrap interval excludes 0
and it is bigger than the noise floor (how much the average score moves
between repeated runs of the same system).

A Docker image that installs the toolkit needs `git` in its build stage,
because uv fetches it from GitHub.

## Keep repos in sync

```bash
uv run dmt pins                                   # the versions the toolkit pins
uv run dmt check-sync ../ml-claim-triage-engine ../movie_recommendation_app \
    ../sql-performance-lab ../ecom-data-modeling-project
```

`check-sync` prints a table of each repo's locked versions and exits 1 if a
repo doesn't use the toolkit or locks a different tool version.

**To bump a tool for every repo:**

1. Change the pin in this repo's `pyproject.toml` (`dev` extra), bump
   `version`, update `CHANGELOG.md`, and merge.
2. Tag the release: `git tag v0.2.0 && git push origin v0.2.0`. The release
   workflow builds the wheel and attaches it to a GitHub Release.
3. In each repo: change `@v0.1.0` to `@v0.2.0` in `pyproject.toml`, run
   `uv lock`, and open a pull request. CI checks the repo with the new versions.

## Develop

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
make install   # dependencies and git hooks
make ci        # ruff, mypy (strict), tests
make build     # wheel and sdist into dist/
make help      # every command
```

Versioning follows [semantic versioning](https://semver.org/): a breaking
change to the SDK bumps the major version (minor while below 1.0).
