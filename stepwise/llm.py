"""LLM adapters and cost accounting.

The engine only sees `LLM.complete()`. `AnthropicLLM` calls Claude; `ScriptedLLM` replays a fixed
script so crash tests are deterministic and need no network or API key.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import anthropic

logger = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens (input, output), Claude API first-party rates.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
_MOST_EXPENSIVE = max(PRICING_PER_MTOK.values())

# Rough chars-per-token used only for the pre-call reservation. Deliberately low, so the
# estimate overshoots and the reservation errs on the side of stopping early.
_CHARS_PER_TOKEN_ESTIMATE = 3


class LLMError(Exception):
    def __init__(self, message: str, *, may_have_billed: bool = False) -> None:
        super().__init__(message)
        self.may_have_billed = may_have_billed


class TransientLLMError(LLMError):
    """Retryable failure (rate limit, overload, network). The run is requeued with backoff."""


class PermanentLLMError(LLMError):
    """Non-retryable failure (bad request, auth). The run fails."""


@dataclass(frozen=True)
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str
    model: str
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    model: str

    async def complete(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int
    ) -> LLMResponse: ...


def price_for(model: str) -> tuple[float, float]:
    price = PRICING_PER_MTOK.get(model)
    if price is None:
        logger.warning("no price for model %s; charging the most expensive known rate", model)
        return _MOST_EXPENSIVE
    return price


def cost_micros(model: str, input_tokens: int, output_tokens: int) -> int:
    """Exact cost in micro-dollars. $/MTok equals micro-dollars per token, so no scaling is needed."""
    input_price, output_price = price_for(model)
    return round(input_tokens * input_price + output_tokens * output_price)


def estimate_max_cost_micros(
    model: str, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int
) -> int:
    """Upper-bound cost of one call: estimated prompt tokens plus the full output allowance."""
    prompt_chars = len(system) + len(json.dumps(messages)) + len(json.dumps(tools))
    estimated_input_tokens = prompt_chars // _CHARS_PER_TOKEN_ESTIMATE + 1
    return cost_micros(model, estimated_input_tokens, max_tokens)


class AnthropicLLM:
    def __init__(self, *, model: str, effort: str, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.model = model
        self._effort = effort
        self._client = client or anthropic.AsyncAnthropic()

    async def complete(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int
    ) -> LLMResponse:
        try:
            response = await self._client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
                output_config={"effort": self._effort},
                # On a safety decline the API re-runs the request on a recommended fallback model.
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise TransientLLMError(f"rate limited: {exc.message}") from exc
        except anthropic.InternalServerError as exc:
            raise TransientLLMError(f"server error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            # Includes timeouts: the request may have reached the API and been billed.
            raise TransientLLMError(f"connection error: {exc}", may_have_billed=True) from exc
        except anthropic.APIStatusError as exc:
            raise PermanentLLMError(f"API error {exc.status_code}: {exc.message}") from exc

        usage = response.usage
        input_tokens = (
            usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
        )
        # Blocks are stored and sent back exactly as returned: thinking blocks must round-trip
        # unchanged on later turns, so nothing is stripped here.
        content = [block.model_dump(mode="json", exclude_none=True) for block in response.content]
        # Bill at the pricier of the requested and serving model, in case a fallback ran.
        billed_model = max((self.model, response.model), key=lambda m: sum(price_for(m)))
        return LLMResponse(
            content=content,
            stop_reason=response.stop_reason or "end_turn",
            model=billed_model,
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
        )


class ScriptedLLM:
    """Plays back a list of canned responses, picked by how many assistant turns already exist.

    Keying on conversation length (not call count) makes the script replay-safe: a worker that
    resumes a run after a crash gets the same response for the same point in the conversation.
    """

    def __init__(self, script: list[dict[str, Any]], *, model: str = "claude-opus-5-5") -> None:
        self.model = model
        self._script = script
        self.calls = 0

    @classmethod
    def from_file(cls, path: str | Path, *, model: str = "claude-opus-5-5") -> "ScriptedLLM":
        return cls(json.loads(Path(path).read_text()), model=model)

    async def complete(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], max_tokens: int
    ) -> LLMResponse:
        self.calls += 1
        turn = sum(1 for message in messages if message["role"] == "assistant")
        if turn >= len(self._script):
            raise PermanentLLMError(f"script has no response for turn {turn}")
        entry = self._script[turn]
        return LLMResponse(
            content=entry["content"],
            stop_reason=entry.get("stop_reason", "end_turn"),
            model=self.model,
            input_tokens=entry.get("input_tokens", 1000),
            output_tokens=entry.get("output_tokens", 200),
        )


def build_llm(spec: str, *, model: str, effort: str) -> LLM:
    if spec == "anthropic":
        return AnthropicLLM(model=model, effort=effort)
    if spec.startswith("scripted:"):
        return ScriptedLLM.from_file(spec.removeprefix("scripted:"), model=model)
    raise ValueError(f"unknown LLM spec {spec!r}; expected 'anthropic' or 'scripted:<path>'")
