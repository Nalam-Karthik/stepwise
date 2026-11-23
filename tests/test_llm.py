from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from stepwise.llm import (
    FALLBACK_BETA,
    AnthropicLLM,
    PermanentLLMError,
    ScriptedLLM,
    TransientLLMError,
    build_llm,
    cost_micros,
    estimate_max_cost_micros,
)

REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class Block:
    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(self._fields)


class FakeMessages:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def fake_client(outcome: Any) -> tuple[Any, FakeMessages]:
    messages = FakeMessages(outcome)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def response(model: str = "claude-opus-5-5") -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=1000, output_tokens=500, cache_creation_input_tokens=None, cache_read_input_tokens=200
    )
    content = [Block(type="thinking", thinking="", signature="abc"), Block(type="text", text="hi")]
    return SimpleNamespace(content=content, stop_reason="end_turn", model=model, usage=usage)


async def complete(llm: AnthropicLLM):
    return await llm.complete(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_tokens=100)


async def test_request_enables_default_fallbacks_and_effort_and_keeps_blocks_verbatim():
    client, messages = fake_client(response())
    result = await complete(AnthropicLLM(model="claude-opus-5-5", effort="medium", client=client))

    assert messages.kwargs["fallbacks"] == "default"
    assert messages.kwargs["betas"] == [FALLBACK_BETA]
    assert messages.kwargs["output_config"] == {"effort": "medium"}
    assert result.content[0] == {"type": "thinking", "thinking": "", "signature": "abc"}
    assert result.input_tokens == 1200  # cache reads count as input


async def test_a_fallback_to_a_pricier_model_is_billed_at_the_pricier_rate():
    client, _ = fake_client(response(model="claude-opus-4-8"))
    result = await complete(AnthropicLLM(model="claude-opus-5-5", effort="medium", client=client))
    assert result.model == "claude-opus-4-8"


@pytest.mark.parametrize(
    ("error", "expected", "may_have_billed"),
    [
        (
            anthropic.RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None),
            TransientLLMError,
            False,
        ),
        (
            anthropic.InternalServerError("oops", response=httpx.Response(529, request=REQUEST), body=None),
            TransientLLMError,
            False,
        ),
        (anthropic.APITimeoutError(request=REQUEST), TransientLLMError, True),
        (
            anthropic.BadRequestError("bad", response=httpx.Response(400, request=REQUEST), body=None),
            PermanentLLMError,
            False,
        ),
    ],
)
async def test_sdk_errors_map_to_retry_policy(error, expected, may_have_billed):
    client, _ = fake_client(error)
    with pytest.raises(expected) as raised:
        await complete(AnthropicLLM(model="claude-opus-5-5", effort="medium", client=client))
    assert raised.value.may_have_billed is may_have_billed


def test_costs_are_exact_micro_dollars():
    assert cost_micros("claude-opus-5-5", 1_000_000, 0) == 4_000_000
    assert cost_micros("claude-haiku-5-5", 1000, 1000) == 600
    assert cost_micros("some-future-model", 1, 0) == 10  # unknown models get the top rate


def test_estimate_covers_the_whole_output_allowance():
    estimate = estimate_max_cost_micros("claude-opus-5-5", system="", messages=[], tools=[], max_tokens=16_000)
    assert estimate >= 16_000 * 20


def test_build_llm(tmp_path):
    script = tmp_path / "s.json"
    script.write_text("[]")
    assert isinstance(build_llm(f"scripted:{script}", model="m", effort="low"), ScriptedLLM)
    with pytest.raises(ValueError):
        build_llm("openai", model="m", effort="low")
