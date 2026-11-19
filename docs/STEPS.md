# Roadmap

What this repo is for: learn to build, version and release an internal SDK,
share it across repositories, and deploy to AWS.

| Step | Name                                          | Status  |
| ---- | --------------------------------------------- | ------- |
| 1    | SDK foundation and shared tool pins           | Done    |
| 2    | Adopt the toolkit in the four repos           | Review  |
| 3    | Release automation (GitHub Releases on tags)  | Done    |
| 4    | AWS S3 helpers                                | Done    |
| 5    | AWS account access for CI (OIDC, Terraform)   | Planned |
| 6    | Deploy a container to AWS (ECR + App Runner)  | Planned |
| 7    | Reusable CI workflow shared by every repo     | Planned |
| 8    | Grow the SDK from real duplication            | Planned |

## 1. SDK foundation and shared tool pins

- Package `data_ml_toolkit`, built with `uv_build`, typed (`py.typed`, strict mypy)
- `logging` and `config` moved here from the copies in two repos
- `dev` extra pins ruff, mypy, pytest and pre-commit for every repo
- `dmt check-sync` reports any repo that drifts from those pins

## 2. Adopt the toolkit in the four repos

One branch per repo (`chore/data-ml-toolkit`), pushed for review, not merged:

- Dev tools come from `data-ml-toolkit[dev]` instead of each repo's own pins
- `ml-claim-triage-engine` and `movie_recommendation_app`: `src/common/logging.py`
  re-exports the toolkit's logging, and config loading uses `load_yaml`
- `sql-performance-lab` and `ecom-data-modeling-project` gain ruff 0.17.5
- Status: pushed on all four repos, waiting for review and merge. CI runs
  when a pull request is opened (the repos run CI on pull requests and `main`)

## 3. Release automation

- Pushing a tag `vX.Y.Z` runs `release.yml`: checks the tag matches the
  version in `pyproject.toml`, runs the tests, builds the wheel and sdist,
  and attaches them to a GitHub Release
- Repos install by tag, so a release is immutable once others depend on it
- Verified with v0.1.0: the release workflow passed and the Release has the
  wheel and sdist attached

## 4. AWS S3 helpers

- `s3.upload_file`, `s3.download_file`, `s3.list_keys` with `s3://` URIs
- Tested with botocore's Stubber, so tests never touch AWS
- Still to do: one manual run against a real bucket once step 5 exists

## 5. AWS account access for CI

- Terraform in `infra/aws/`: an S3 bucket for artifacts, an ECR repository,
  and an IAM role that GitHub Actions assumes through OIDC (no long-lived
  access keys stored in GitHub)
- Least privilege: the role can push to one ECR repo and one bucket
- Needs: an AWS account, the AWS CLI configured locally, a budget alert

## 6. Deploy a container to AWS

- Reusable workflow `deploy-aws.yml` (`workflow_call`): build the Docker
  image, push it to ECR, deploy it to App Runner, wait for the health check
- First user: the claim-triage FastAPI service
- Cost control: smallest App Runner size, pause when not in use

## 7. Reusable CI workflow

- `python-ci.yml` (`workflow_call`): checkout, uv, `uv lock --check`,
  pre-commit, pytest with a JUnit report
- Repos call it for their lint and test jobs; repo-specific jobs (dbt,
  PostgreSQL, Docker) stay local
- Check names change, so branch rulesets must be updated in the same change

## 8. Grow the SDK from real duplication

Move code here only when a second repo needs it. Candidates: data-contract
helpers (claim-triage, movie app), MLflow tracking helpers, DuckDB helpers.
