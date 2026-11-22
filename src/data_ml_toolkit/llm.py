"""One small interface for calling LLMs, with record/replay and a cost ledger.

Needs the `llm` extra for live calls: `data-ml-toolkit[llm]`. Replaying
recorded calls needs nothing extra, so tests and CI run offline and free.

    from data_ml_toolkit.llm import LLM, AnthropicProvider, Ledger, Recorded, Request

    llm = LLM(
        Recorded(AnthropicProvider(), "tests/recordings", mode="auto"),
        Ledger(cap_usd=1.00),
    )
    reply = llm.complete(Request(model="claude-haiku-5-5", prompt="Say hi", effort="low"))
    print(reply.text, llm.ledger.summary())

Pieces:

- `Request` / `Response`: plain data, the same for every provider.
- `Provider`: anything with `complete(request) -> Response`. `AnthropicProvider`
  calls the Messages API through the official SDK; credentials come from the
  environment (`ANTHROPIC_API_KEY` or an `ant auth login` profile), never code.
- `Recorded`: saves each response under a hash of its request and replays it
  later. Modes: `replay` (never calls the model; a missing recording is an
  error), `record` (always calls and overwrites), `auto` (replay if present).
- `Ledger`: tokens, cost and latency per call, and a spending cap checked
  before every live call. Replayed calls are logged but cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

Mode = Literal["replay", "record", "auto"]


@dataclass(frozen=True)
class Request:
    """One single-turn call: a system prompt and one user message."""

    model: str
    prompt: str
    system: str = ""
    max_tokens: int = 1024
    effort: str | None = None  # "low" ... "max"; None keeps the model's default
    json_schema: Mapping[str, Any] | None = None  # constrain the reply to this JSON schema

    def key(self) -> str:
        """Stable hash of every field: the recording's file name."""
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]


@dataclass(frozen=True)
class Response:
    text: str
    model: str
    stop_reason: str  # "end_turn", "max_tokens", "refusal", ...
    input_tokens: int
    output_tokens: int
    latency_s: float
    replayed: bool = False

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    def json(self) -> Any:
        """The reply parsed as JSON (for requests with a `json_schema`)."""
        return json.loads(self.text)


class Provider(Protocol):
    def complete(self, request: Request) -> Response: ...


class AnthropicProvider:
    """Claude through the official `anthropic` SDK (Messages API)."""

    def __init__(self, client: Any = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as err:
                raise ImportError(
                    "AnthropicProvider needs the anthropic SDK: install data-ml-toolkit[llm]"
                ) from err
            client = anthropic.Anthropic()
        self.client = client

    @staticmethod
    def params(request: Request) -> dict[str, Any]:
        """The Messages API arguments for a request."""
        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            params["system"] = request.system
        output_config: dict[str, Any] = {}
        if request.effort:
            output_config["effort"] = request.effort
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": dict(request.json_schema)}
        if output_config:
            params["output_config"] = output_config
        return params

    def complete(self, request: Request) -> Response:
        start = time.perf_counter()
        message = self.client.messages.create(**self.params(request))
        latency = time.perf_counter() - start
        text = "".join(block.text for block in message.content if block.type == "text")
        return Response(
            text=text,
            model=message.model,
            stop_reason=str(message.stop_reason),
            input_tokens=int(message.usage.input_tokens),
            output_tokens=int(message.usage.output_tokens),
            latency_s=round(latency, 3),
        )


class RecordingMissing(LookupError):
    pass


class Recorded:
    """Record responses to `directory` and replay them, keyed by the request hash."""

    def __init__(
        self, provider: Provider | None, directory: str | Path, mode: Mode = "replay"
    ) -> None:
        if mode not in ("replay", "record", "auto"):
            raise ValueError(f"mode must be replay, record or auto, not {mode!r}")
        if mode != "replay" and provider is None:
            raise ValueError(f"mode {mode!r} calls the model, so it needs a provider")
        self.provider = provider
        self.directory = Path(directory)
        self.mode = mode

    def path(self, request: Request) -> Path:
        return self.directory / f"{request.key()}.json"

    def complete(self, request: Request) -> Response:
        path = self.path(request)
        if self.mode != "record" and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            return replace(Response(**saved["response"]), replayed=True)
        if self.mode == "replay" or self.provider is None:
            raise RecordingMissing(
                f"No recording for this request ({path}). Record it once with "
                "mode='record' or 'auto' and commit the file."
            )
        response = self.provider.complete(request)
        self.directory.mkdir(parents=True, exist_ok=True)
        saved = {"request": asdict(request), "response": asdict(response)}
        path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return response


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens. `long_*` apply above `long_threshold` input tokens."""

    input: float
    output: float
    long_input: float | None = None
    long_output: float | None = None
    long_threshold: int | None = None

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        long = self.long_threshold is not None and input_tokens > self.long_threshold
        rate_in = self.long_input if long and self.long_input is not None else self.input
        rate_out = self.long_output if long and self.long_output is not None else self.output
        return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


# Anthropic first-party list prices (checked 2026-11-22). Add models as repos use them.
PRICES: dict[str, Price] = {
    "claude-fable-5-1": Price(10.00, 50.00),
    "claude-opus-5-5": Price(4.00, 20.00),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-haiku-5-5": Price(
        0.10, 0.50, long_input=0.50, long_output=2.50, long_threshold=100_000
    ),
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    """Cost and latency of every call; refuses live calls once `cap_usd` is spent."""

    cap_usd: float | None = None
    path: Path | None = None  # optional JSONL log, one line per call
    prices: Mapping[str, Price] = field(default_factory=lambda: dict(PRICES))
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def spent_usd(self) -> float:
        return float(sum(e["cost_usd"] for e in self.entries if not e["replayed"]))

    def check(self) -> None:
        if self.cap_usd is not None and self.spent_usd >= self.cap_usd:
            raise BudgetExceeded(f"spent ${self.spent_usd:.4f} of a ${self.cap_usd:.2f} cap")

    def price(self, model: str) -> Price:
        # Responses may name a dated snapshot of the requested model
        for name, price in self.prices.items():
            if model == name or model.startswith(name + "-"):
                return price
        raise KeyError(f"No price for model {model!r}: add it to the ledger's prices")

    def record(self, request: Request, response: Response) -> dict[str, Any]:
        cost = self.price(request.model).cost(response.input_tokens, response.output_tokens)
        entry = {
            "model": response.model,
            "request_key": request.key(),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": round(cost, 6),
            "latency_s": response.latency_s,
            "stop_reason": response.stop_reason,
            "replayed": response.replayed,
        }
        self.entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def summary(self) -> dict[str, Any]:
        live = [e for e in self.entries if not e["replayed"]]
        latencies = sorted(e["latency_s"] for e in live)
        return {
            "calls": len(self.entries),
            "live_calls": len(live),
            "input_tokens": sum(e["input_tokens"] for e in self.entries),
            "output_tokens": sum(e["output_tokens"] for e in self.entries),
            "spent_usd": round(self.spent_usd, 6),
            "latency_p50_s": statistics.median(latencies) if latencies else None,
            "latency_max_s": latencies[-1] if latencies else None,
        }


class LLM:
    """A provider plus a ledger: every call is priced, logged and checked against the cap."""

    def __init__(self, provider: Provider, ledger: Ledger | None = None) -> None:
        self.provider = provider
        self.ledger = ledger or Ledger()

    def complete(self, request: Request) -> Response:
        self.ledger.price(request.model)  # unknown model: fail before spending anything
        is_replay = isinstance(self.provider, Recorded) and (
            self.provider.mode != "record" and self.provider.path(request).exists()
        )
        if not is_replay:
            self.ledger.check()
        response = self.provider.complete(request)
        self.ledger.record(request, response)
        return response
