"""No network: the Anthropic client is a fake object with the SDK's response shape."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from data_ml_toolkit.llm import (
    LLM,
    AnthropicProvider,
    BudgetExceeded,
    Ledger,
    Price,
    Recorded,
    RecordingMissing,
    Request,
    Response,
)


class FakeMessages:
    def __init__(self, text: str = "hello", stop_reason: str = "end_turn") -> None:
        self.calls: list[dict[str, Any]] = []
        self.text, self.stop_reason = text, stop_reason

    def create(self, **params: Any) -> Any:
        self.calls.append(params)
        return SimpleNamespace(
            model=params["model"],
            stop_reason=self.stop_reason,
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.text),
            ],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )


def fake_provider(**kwargs: Any) -> tuple[AnthropicProvider, FakeMessages]:
    messages = FakeMessages(**kwargs)
    return AnthropicProvider(client=SimpleNamespace(messages=messages)), messages


REQ = Request(model="claude-haiku-5-5", prompt="Explain", system="Be brief", effort="low")


def test_request_key_is_stable_and_covers_every_field() -> None:
    assert REQ.key() == Request(**{**REQ.__dict__}).key()
    assert REQ.key() != Request(model="claude-haiku-5-5", prompt="Explain").key()
    assert len(REQ.key()) == 32


def test_anthropic_params_and_text_extraction() -> None:
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    provider, messages = fake_provider(text='{"ok": true}')
    reply = provider.complete(Request(model="claude-haiku-5-5", prompt="x", json_schema=schema))
    assert reply.json() == {"ok": True}  # thinking block skipped, text kept
    assert messages.calls[0]["output_config"] == {
        "format": {"type": "json_schema", "schema": schema}
    }
    params = AnthropicProvider.params(REQ)
    assert params["system"] == "Be brief"
    assert params["output_config"] == {"effort": "low"}
    assert params["messages"] == [{"role": "user", "content": "Explain"}]
    assert "output_config" not in AnthropicProvider.params(Request(model="m", prompt="p"))


def test_refusal_is_reported() -> None:
    provider, _ = fake_provider(text="", stop_reason="refusal")
    assert provider.complete(REQ).refused


def test_record_then_replay_without_the_model(tmp_path: Path) -> None:
    provider, messages = fake_provider()
    first = Recorded(provider, tmp_path, mode="auto").complete(REQ)
    assert not first.replayed
    again = Recorded(None, tmp_path, mode="replay").complete(REQ)
    assert again.replayed and again.text == first.text
    assert len(messages.calls) == 1
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["request"]["prompt"] == "Explain"


def test_replay_mode_never_calls_and_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(RecordingMissing, match="mode='record'"):
        Recorded(None, tmp_path, mode="replay").complete(REQ)
    with pytest.raises(ValueError, match="needs a provider"):
        Recorded(None, tmp_path, mode="auto")
    with pytest.raises(ValueError, match="mode must be"):
        Recorded(None, tmp_path, mode="live")  # type: ignore[arg-type]


def test_record_mode_overwrites(tmp_path: Path) -> None:
    provider, messages = fake_provider()
    Recorded(provider, tmp_path, mode="record").complete(REQ)
    Recorded(provider, tmp_path, mode="record").complete(REQ)
    assert len(messages.calls) == 2


def test_price_tiers() -> None:
    price = Price(0.10, 0.50, long_input=0.50, long_output=2.50, long_threshold=100_000)
    assert price.cost(1_000_000, 0) == pytest.approx(0.50)  # above the threshold
    assert price.cost(100_000, 1_000_000) == pytest.approx(0.01 + 0.50)
    assert Price(4.0, 20.0).cost(1000, 200) == pytest.approx(0.008)


def test_ledger_prices_logs_and_caps(tmp_path: Path) -> None:
    provider, messages = fake_provider()
    log = tmp_path / "ledger.jsonl"
    # Haiku 5.5: 1000 x $0.10 + 200 x $0.50 per million = $0.0002 per call
    llm = LLM(provider, Ledger(cap_usd=0.0004, path=log))
    llm.complete(REQ)
    llm.complete(REQ)
    with pytest.raises(BudgetExceeded):
        llm.complete(REQ)
    assert len(messages.calls) == 2
    summary = llm.ledger.summary()
    assert summary["live_calls"] == 2 and summary["spent_usd"] == pytest.approx(0.0004)
    assert len(log.read_text().splitlines()) == 2


def test_replays_are_free_and_allowed_over_the_cap(tmp_path: Path) -> None:
    provider, _ = fake_provider()
    Recorded(provider, tmp_path, mode="record").complete(REQ)
    llm = LLM(Recorded(None, tmp_path), Ledger(cap_usd=0.0))
    reply = llm.complete(REQ)
    assert reply.replayed
    assert llm.ledger.summary()["spent_usd"] == 0
    assert llm.ledger.summary()["latency_p50_s"] is None


def test_unknown_model_fails_before_calling() -> None:
    provider, messages = fake_provider()
    with pytest.raises(KeyError, match="No price"):
        LLM(provider).complete(Request(model="mystery-model", prompt="x"))
    assert messages.calls == []


def test_dated_model_names_use_the_base_price() -> None:
    ledger = Ledger()
    reply = Response("x", "claude-haiku-5-5-20261101", "end_turn", 10, 10, 0.1)
    entry = ledger.record(REQ, reply)
    assert entry["cost_usd"] > 0
