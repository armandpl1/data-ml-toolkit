# AGENTS.md

Standards for anyone changing this repository, human or AI agent. Other
repositories depend on it, so changes here affect all of them.

## Workflow

1. **Branch off the latest `main`** and open a pull request. CI must pass.
2. **Plan, build, verify, document, then commit**, in that order.
3. **Before committing:** `make ci` (ruff, strict mypy, tests).
4. **Commit messages:** a short summary line, a blank line, then bullet points
   saying what changed and why.
5. **No AI attribution.** No `Co-Authored-By` or "Generated with" lines in
   commits or pull requests. Commits appear as the repository owner only.
6. **Never rewrite published history,** and never move or delete a release
   tag: repos install the toolkit by tag.

## Releases and compatibility

- **Semantic versioning.** Below 1.0, a breaking SDK change bumps the minor
  version (0.1 → 0.2); anything else bumps the patch.
- **Every release:** bump `version` in `pyproject.toml`, add a `CHANGELOG.md`
  entry, merge, then push the tag `vX.Y.Z`. `release.yml` refuses a tag that
  doesn't match `pyproject.toml`.
- **Changing a dev pin is a release.** After tagging, update each repo on a
  branch and confirm with `make check-sync`.

## SDK standards

- **Only shared code.** Add a module when a second repo needs it, not before.
- **Small dependencies.** The base install needs only `packaging` and
  `pyyaml`. Anything heavier goes in an extra (like `aws`), imported lazily
  with an error message naming the extra.
- **Typed and tested.** Strict mypy on `src/`; every public function has a test.
- **No network in tests.** AWS calls use botocore's Stubber.
- **Secrets never in code.** AWS credentials come from the standard chain
  (environment, `~/.aws`, or an IAM role via OIDC in CI).

## Documentation standards

- **README.md** is the source of truth for how repos use the toolkit.
- **docs/STEPS.md** tracks the roadmap; update its status table as steps land.
- **Plain language** and aligned Markdown tables.

## CI/CD standards

- **Pin actions to full commit SHAs,** with the version in a comment.
- **Least privilege:** read-only token by default; only the release job may
  write (to create the GitHub Release).
- **Every job has a timeout.**
- **Validate workflow changes** with actionlint:
  `uvx --from actionlint-py actionlint .github/workflows/*.yml`
