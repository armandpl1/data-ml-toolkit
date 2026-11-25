import json
from pathlib import Path
from typing import Any

import pytest

from data_ml_toolkit.evals import (
    GoldenSet,
    Judge,
    bootstrap_ci,
    calibrate,
    compare,
    hash_split,
    noise_floor,
    paired_diff,
)
from data_ml_toolkit.llm import LLM, Request, Response


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# --- Golden sets ---------------------------------------------------------------


def test_splits_are_stable_when_examples_are_added(tmp_path: Path) -> None:
    rows = [{"id": f"ex{i}", "q": i} for i in range(200)]
    small = GoldenSet.load(write_jsonl(tmp_path / "a.jsonl", rows[:100]))
    big = GoldenSet.load(write_jsonl(tmp_path / "b.jsonl", rows))
    before = {e.id: e.split for e in small.examples}
    assert all(before[e.id] == e.split for e in big.examples if e.id in before)
    share = len(big.split("test")) / len(big.examples)
    assert 0.2 < share < 0.4  # default test_fraction 0.3


def test_explicit_split_wins_and_version_tracks_content(tmp_path: Path) -> None:
    rows = [{"id": "a", "split": "test"}, {"id": "b"}]
    one = GoldenSet.load(write_jsonl(tmp_path / "x.jsonl", rows), test_fraction=0.0)
    assert [e.split for e in one.examples] == ["test", "dev"]
    reordered = GoldenSet.load(write_jsonl(tmp_path / "y.jsonl", rows[::-1]), test_fraction=0.0)
    assert reordered.version == one.version  # order doesn't matter
    edited = GoldenSet.load(write_jsonl(tmp_path / "z.jsonl", [rows[0], {"id": "b", "x": 1}]))
    assert edited.version != one.version


def test_golden_set_rejects_bad_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate ids"):
        GoldenSet.load(write_jsonl(tmp_path / "d.jsonl", [{"id": "a"}, {"id": "a"}]))
    with pytest.raises(ValueError, match="non-empty 'id'"):
        GoldenSet.load(write_jsonl(tmp_path / "e.jsonl", [{"q": 1}]))
    with pytest.raises(ValueError, match="test_fraction"):
        GoldenSet.load(tmp_path / "d.jsonl", test_fraction=2)


def test_hash_split_uses_the_salt() -> None:
    ids = [str(i) for i in range(100)]
    assert [hash_split(i, 0.5) for i in ids] != [hash_split(i, 0.5, "v2") for i in ids]


# --- Statistics ----------------------------------------------------------------


def test_bootstrap_and_paired_diff() -> None:
    ci = bootstrap_ci([1.0] * 50 + [0.0] * 50, seed=1)
    assert ci.mean == 0.5 and ci.low < 0.5 < ci.high
    assert bootstrap_ci([0.7, 0.7]).low == pytest.approx(0.7)
    diff = paired_diff([0.0] * 40, [1.0] * 30 + [0.0] * 10)
    assert diff.mean == 0.75 and diff.excludes_zero()
    with pytest.raises(ValueError):
        paired_diff([1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_noise_floor() -> None:
    assert noise_floor([[1.0, 0.0], [1.0, 0.0]]) == 0.0
    assert noise_floor([[1.0, 1.0], [0.0, 0.0]]) == pytest.approx(0.7071, abs=1e-4)
    with pytest.raises(ValueError):
        noise_floor([[1.0]])


def test_compare_needs_a_real_and_big_enough_change() -> None:
    base = [[0.0] * 30 + [1.0] * 30, [0.0] * 30 + [1.0] * 30]
    better = [[1.0] * 50 + [0.0] * 10, [1.0] * 50 + [0.0] * 10]
    assert compare(base, better).verdict == "improved"
    assert compare(better, base).verdict == "regressed"
    assert compare(base, base).verdict == "no_change"
    # A small real gain hidden inside large run-to-run noise doesn't count
    noisy = [[1.0] * 60, [0.0] * 60]
    slightly = [[1.0] * 60, [0.0] * 54 + [1.0] * 6]
    assert compare(slightly, noisy).verdict == "no_change"


# --- Judge ---------------------------------------------------------------------


class ScriptedProvider:
    """Returns pre-written judge replies and keeps the requests it saw."""

    def __init__(self, replies: list[str], stop_reason: str = "end_turn") -> None:
        self.replies, self.stop_reason = replies, stop_reason
        self.requests: list[Request] = []

    def complete(self, request: Request) -> Response:
        self.requests.append(request)
        return Response(self.replies.pop(0), request.model, self.stop_reason, 50, 20, 0.2)


def judge_with(replies: list[str], stop_reason: str = "end_turn") -> tuple[Judge, Any]:
    provider = ScriptedProvider(replies, stop_reason)
    judge = Judge(LLM(provider), "claude-haiku-5-5", "Faithful to the evidence?", ["yes", "no"])
    return judge, provider


def test_judge_grades_with_a_constrained_schema() -> None:
    judge, provider = judge_with([json.dumps({"reason": "matches", "label": "yes"})])
    grade = judge.grade("Explain why", "Because you rated Heat 4.5", reference="Heat: 4.5")
    assert (grade.label, grade.reason) == ("yes", "matches")
    request = provider.requests[0]
    assert request.json_schema["properties"]["label"]["enum"] == ["yes", "no"]
    assert "<reference>" in request.prompt and request.effort == "low"


def test_judge_rejects_refusals_and_unknown_labels() -> None:
    judge, _ = judge_with([""], stop_reason="refusal")
    with pytest.raises(RuntimeError, match="declined"):
        judge.grade("t", "a")
    judge, _ = judge_with([json.dumps({"reason": "r", "label": "maybe"})])
    with pytest.raises(ValueError, match="unknown label"):
        judge.grade("t", "a")
    with pytest.raises(ValueError, match="two distinct labels"):
        Judge(LLM(ScriptedProvider([])), "m", "r", ["yes", "yes"])


def test_calibration_agreement_and_kappa() -> None:
    perfect = calibrate(["yes", "no", "yes"], ["yes", "no", "yes"])
    assert perfect.agreement == 1.0 and perfect.kappa == 1.0 and perfect.trusted()
    # Always saying "yes" agrees 80% of the time on an 80/20 set, but kappa is 0
    lazy = calibrate(["yes"] * 10, ["yes"] * 8 + ["no"] * 2)
    assert lazy.agreement == 0.8 and lazy.kappa == 0.0 and not lazy.trusted()
    assert lazy.confusion["no"]["yes"] == 2
    with pytest.raises(ValueError):
        calibrate([], [])
