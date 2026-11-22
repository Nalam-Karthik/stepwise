"""Pure functions that turn a run's journal into a conversation and the next action.

No I/O here: given the same journal, a fresh worker after a crash computes exactly the same next
action as the worker that died. That determinism is what makes resume-from-journal correct.
"""

import json
from dataclasses import dataclass
from typing import Any

from stepwise.models import Step, StepKind, StepStatus


@dataclass(frozen=True)
class CallLLM:
    messages: list[dict[str, Any]]


@dataclass(frozen=True)
class RunTool:
    tool_use_id: str
    name: str
    input: dict[str, Any]
    started_seq: int | None  # set when an external call was started but never journaled as done


@dataclass(frozen=True)
class Finish:
    result: str


@dataclass(frozen=True)
class Fail:
    error: str


Action = CallLLM | RunTool | Finish | Fail


def tool_result_block(step: Step) -> dict[str, Any]:
    output = step.payload["output"]
    return {
        "type": "tool_result",
        "tool_use_id": step.tool_use_id,
        "content": output if isinstance(output, str) else json.dumps(output, sort_keys=True),
        "is_error": step.payload.get("is_error", False),
    }


def build_messages(goal: str, steps: list[Step]) -> list[dict[str, Any]]:
    """Rebuild the exact message list the model saw, from the journal alone."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": goal}]
    results: list[dict[str, Any]] = []
    for step in steps:
        if step.kind == StepKind.LLM:
            if results:
                messages.append({"role": "user", "content": results})
                results = []
            messages.append({"role": "assistant", "content": step.payload["content"]})
        elif step.status == StepStatus.COMPLETED:
            results.append(tool_result_block(step))
    if results:
        messages.append({"role": "user", "content": results})
    return messages


def final_text(content: list[dict[str, Any]]) -> str:
    return "\n".join(block["text"] for block in content if block.get("type") == "text").strip()


def plan_next(goal: str, steps: list[Step], *, max_steps: int) -> Action:
    llm_steps = [step for step in steps if step.kind == StepKind.LLM]
    if not llm_steps:
        return CallLLM(build_messages(goal, steps))

    last_llm = llm_steps[-1]
    tool_steps = {s.tool_use_id: s for s in steps if s.kind == StepKind.TOOL and s.seq > last_llm.seq}
    content = last_llm.payload["content"]

    # Tool calls from the latest turn run in the order the model listed them.
    for block in content:
        if block.get("type") != "tool_use":
            continue
        step = tool_steps.get(block["id"])
        if step is None:
            return RunTool(block["id"], block["name"], block.get("input") or {}, started_seq=None)
        if step.status == StepStatus.STARTED:
            return RunTool(block["id"], block["name"], block.get("input") or {}, started_seq=step.seq)

    stop_reason = last_llm.payload["stop_reason"]
    if stop_reason in ("tool_use", "pause_turn"):
        if len(llm_steps) >= max_steps:
            return Fail(f"step limit reached ({max_steps} model calls)")
        return CallLLM(build_messages(goal, steps))
    if stop_reason in ("end_turn", "stop_sequence"):
        return Finish(final_text(content))
    if stop_reason == "refusal":
        return Fail("the model declined the request")
    return Fail(f"model stopped with {stop_reason!r} before finishing")
