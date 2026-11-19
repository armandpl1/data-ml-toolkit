# Single-command execution. `make help` lists every target.

RUN := uv run
REPOS ?= ../ml-claim-triage-engine ../movie_recommendation_app ../sql-performance-lab ../ecom-data-modeling-project

.PHONY: help install hooks lint typecheck test ci build check-sync clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install dependencies from uv.lock and the git pre-commit hooks
	uv sync --locked
	$(RUN) pre-commit install

hooks:  ## Install the git pre-commit hooks (ruff, mypy, file hygiene)
	$(RUN) pre-commit install

lint:  ## All pre-commit checks: ruff lint and format, mypy, file hygiene
	$(RUN) pre-commit run --all-files

typecheck:  ## mypy (strict) only
	$(RUN) mypy src

test:  ## All tests
	$(RUN) pytest

ci: lint test  ## Everything CI runs

build:  ## Build the wheel and sdist into dist/
	rm -rf dist
	uv build

check-sync:  ## Compare the other repos' uv.lock with the pinned tools (REPOS=...)
	$(RUN) dmt check-sync $(REPOS)

clean:  ## Remove caches and build outputs
	rm -rf dist reports .pytest_cache .mypy_cache .ruff_cache
