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
2. **One set of dev tool versions.** The `dev` extra pins ruff, mypy,
   pytest and pre-commit. Every repo depends on `data-ml-toolkit[dev]`
   instead of pinning them itself, so all repos lint and test with the same
   versions. `dmt check-sync` proves it.

The roadmap is in [docs/STEPS.md](docs/STEPS.md).

## Use it in a repo

The toolkit is installed from GitHub at a version tag; no package index needed.

```bash
# Runtime SDK (logging, config); add [aws] for the S3 helpers
uv add "data-ml-toolkit @ git+https://github.com/armandpl1/data-ml-toolkit@v0.1.0"

# Shared dev tool versions, in place of ruff, mypy, pytest and pre-commit
uv add --dev "data-ml-toolkit[dev] @ git+https://github.com/armandpl1/data-ml-toolkit@v0.1.0"
```

```python
import logging
from data_ml_toolkit.logging import get_logger, log_event

logger = get_logger(__name__)
log_event(logger, logging.INFO, "rows_loaded", table="claims", rows=26444)
# {"ts": "...", "level": "INFO", "logger": "...", "event": "rows_loaded", "table": "claims", "rows": 26444}
```

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
