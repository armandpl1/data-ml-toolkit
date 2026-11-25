"""Evaluation helpers: golden datasets, statistics, regressions and LLM judges.

Needs the `evals` extra: `data-ml-toolkit[evals]` (numpy). The judge also
needs a provider from `data_ml_toolkit.llm` (replayed recordings work offline).

- **Golden sets** (`GoldenSet.load`): a JSONL file, one example per line with
  an `id`. Each example's split (dev or test) comes from a hash of its id,
  so adding examples never moves existing ones. `version` hashes the
  content: pin it in a test to notice any edit to the set.
- **Statistics:** `bootstrap_ci` for one score, `paired_diff` for two systems
  scored on the same examples.
- **Regressions** (`compare`): run each system several times; the noise
  floor is how much the average score moves between runs of the *same*
  system. A change counts only if its confidence interval excludes 0 and it
  is bigger than the noise floor.
- **LLM judge** (`Judge`): grades an answer into one of a fixed set of
  labels, with a short reason, using structured JSON output. `calibrate`
  compares its labels with human labels (agreement and Cohen's kappa) before
  anyone trusts it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import numpy as np
except ImportError as err:  # pragma: no cover - exercised only without the extra
    raise ImportError("data_ml_toolkit.evals needs numpy: install data-ml-toolkit[evals]") from err

from data_ml_toolkit.llm import LLM, Request

# --- Golden sets ---------------------------------------------------------------


@dataclass(frozen=True)
class Example:
    id: str
    data: dict[str, Any]
    split: str


@dataclass(frozen=True)
class GoldenSet:
    examples: tuple[Example, ...]
    version: str

    @classmethod
    def load(cls, path: str | Path, test_fraction: float = 0.3, salt: str = "") -> GoldenSet:
        """Read JSONL; an explicit `split` field wins over the hash-based one."""
        if not 0 <= test_fraction <= 1:
            raise ValueError("test_fraction must be between 0 and 1")
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not str(row.get("id", "")).strip():
                raise ValueError(f"{path}:{number}: every example needs a non-empty 'id'")
            rows.append(row)
        ids = [str(r["id"]) for r in rows]
        if duplicates := sorted(i for i, n in Counter(ids).items() if n > 1):
            raise ValueError(f"{path}: duplicate ids {duplicates}")

        examples = tuple(
            Example(
                id=str(r["id"]),
                data=r,
                split=str(r.get("split") or hash_split(str(r["id"]), test_fraction, salt)),
            )
            for r in rows
        )
        canonical = "\n".join(sorted(json.dumps(r, sort_keys=True) for r in rows))
        version = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return cls(examples, version)

    def split(self, name: str) -> list[Example]:
        return [e for e in self.examples if e.split == name]


def hash_split(example_id: str, test_fraction: float, salt: str = "") -> str:
    """'test' for a stable `test_fraction` of ids, else 'dev'."""
    digest = hashlib.sha256(f"{salt}{example_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if bucket < test_fraction else "dev"


# --- Statistics ----------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def bootstrap_ci(
    values: Sequence[float], samples: int = 2000, seed: int = 0, level: float = 0.95
) -> Interval:
    """Mean and percentile bootstrap interval (resampling examples)."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("need at least one value")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(samples, data.size))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return Interval(float(data.mean()), float(low), float(high))


def paired_diff(
    baseline: Sequence[float],
    candidate: Sequence[float],
    samples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Candidate minus baseline, scored on the same examples in the same order."""
    if len(baseline) != len(candidate):
        raise ValueError("paired scores need the same examples on both sides")
    diffs = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    return bootstrap_ci(diffs.tolist(), samples, seed, level)


def noise_floor(runs: Sequence[Sequence[float]]) -> float:
    """Standard deviation of the average score across repeated runs of one system."""
    if len(runs) < 2:
        raise ValueError("the noise floor needs at least two runs")
    means = [float(np.mean(run)) for run in runs]
    return float(np.std(means, ddof=1))


Verdict = Literal["improved", "regressed", "no_change"]


@dataclass(frozen=True)
class Comparison:
    verdict: Verdict
    diff: Interval
    noise: float


def compare(
    baseline_runs: Sequence[Sequence[float]],
    candidate_runs: Sequence[Sequence[float]],
    samples: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Is the candidate better or worse than the baseline, beyond run-to-run noise?

    Each run is a list of per-example scores, examples in the same order in
    every run. Scores are averaged over runs per example, then paired.
    """
    noise = max(noise_floor(baseline_runs), noise_floor(candidate_runs))
    base = np.mean(np.asarray(baseline_runs, dtype=float), axis=0)
    cand = np.mean(np.asarray(candidate_runs, dtype=float), axis=0)
    diff = paired_diff(base.tolist(), cand.tolist(), samples, seed)
    verdict: Verdict = "no_change"
    if diff.excludes_zero() and abs(diff.mean) > noise:
        verdict = "improved" if diff.mean > 0 else "regressed"
    return Comparison(verdict, diff, noise)


# --- LLM judge -----------------------------------------------------------------


@dataclass(frozen=True)
class Grade:
    label: str
    reason: str


class Judge:
    """Grade an answer into one of `labels` with a model, via structured JSON output."""

    def __init__(
        self,
        llm: LLM,
        model: str,
        rubric: str,
        labels: Sequence[str],
        effort: str | None = "low",
        max_tokens: int = 1024,
    ) -> None:
        if len(set(labels)) < 2:
            raise ValueError("a judge needs at least two distinct labels")
        self.llm, self.model, self.rubric = llm, model, rubric
        self.labels = list(labels)
        self.effort, self.max_tokens = effort, max_tokens

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "label": {"type": "string", "enum": self.labels},
            },
            "required": ["reason", "label"],
            "additionalProperties": False,
        }

    def request(self, task: str, answer: str, reference: str | None = None) -> Request:
        parts = [f"<task>\n{task}\n</task>", f"<answer>\n{answer}\n</answer>"]
        if reference is not None:
            parts.append(f"<reference>\n{reference}\n</reference>")
        return Request(
            model=self.model,
            system=(
                "You grade answers against a rubric. Give a one or two sentence reason, "
                f"then one label from: {', '.join(self.labels)}.\n\nRubric:\n{self.rubric}"
            ),
            prompt="\n\n".join(parts),
            max_tokens=self.max_tokens,
            effort=self.effort,
            json_schema=self.schema(),
        )

    def grade(self, task: str, answer: str, reference: str | None = None) -> Grade:
        response = self.llm.complete(self.request(task, answer, reference))
        if response.refused:
            raise RuntimeError("the judge model declined to grade this answer")
        data = response.json()
        if data.get("label") not in self.labels:
            raise ValueError(f"judge returned an unknown label: {data.get('label')!r}")
        return Grade(label=str(data["label"]), reason=str(data.get("reason", "")))


@dataclass(frozen=True)
class Calibration:
    n: int
    agreement: float
    kappa: float
    confusion: dict[str, dict[str, int]]  # human label -> judge label -> count

    def trusted(self, min_kappa: float = 0.6) -> bool:
        """Kappa 0.6+ is conventionally 'substantial' agreement."""
        return self.kappa >= min_kappa


def calibrate(judge_labels: Sequence[str], human_labels: Sequence[str]) -> Calibration:
    """How well the judge agrees with people, corrected for chance (Cohen's kappa)."""
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError("need the same, non-zero number of judge and human labels")
    n = len(judge_labels)
    agreement = sum(j == h for j, h in zip(judge_labels, human_labels, strict=True)) / n
    judge_freq, human_freq = Counter(judge_labels), Counter(human_labels)
    chance = sum(judge_freq[label] * human_freq[label] for label in human_freq) / n**2
    kappa = 1.0 if chance == 1 else (agreement - chance) / (1 - chance)
    labels = sorted(set(judge_labels) | set(human_labels))
    confusion = {h: {j: 0 for j in labels} for h in labels}
    for j, h in zip(judge_labels, human_labels, strict=True):
        confusion[h][j] += 1
    return Calibration(n, round(agreement, 4), round(kappa, 4), confusion)
