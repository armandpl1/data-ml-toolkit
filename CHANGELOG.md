# Changelog

All notable changes, newest first. Versions follow semantic versioning.

## 0.2.0 (2026-11-22)

- `llm`: `Request`/`Response`, `AnthropicProvider` (official SDK, `llm` extra),
  `Recorded` record/replay keyed by request hash, `Ledger` with prices,
  latency, a JSONL log and a spending cap
- `evals`: `GoldenSet` (JSONL, hash-based frozen splits, content version),
  `bootstrap_ci`, `paired_diff`, `noise_floor`, `compare`, `Judge` (structured
  JSON labels) and `calibrate` (agreement, Cohen's kappa, confusion); `evals`
  extra (numpy)
- No breaking changes to 0.1.0 modules; dev pins unchanged

## 0.1.0 (2026-11-18)

- SDK: `logging` (structured JSON), `config` (strict YAML, repo root, paths),
  `s3` (upload, download, list; `aws` extra)
- `dev` extra pins the shared tools: ruff 0.17.5, mypy 2.4.1, pytest 9.2.0,
  pre-commit 4.7.0
- `dmt` command: `version`, `pins`, `check-sync`
