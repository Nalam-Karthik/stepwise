from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from stepwise.models import Step, StepKind, StepStatus
from stepwise.planner import CallLLM, Fail, Finish, RunTool, build_messages, plan_next

RUN_ID = uuid4()
GOAL = "Refund ord_1001"


def llm_step(seq: int, content: list[dict[str, Any]], stop_reason: str = "tool_use") -> Step:
    return Step(
        RUN_ID,
        seq,
        StepKind.LLM,
        StepStatus.COMPLETED,
        None,
        None,
        {"content": content, "stop_reason": stop_reason},
        0,
        1,
        datetime.now(UTC),
    )


def tool_step(seq: int, tool_use_id: str, *, status: StepStatus = StepStatus.COMPLETED, output: Any = None) -> Step:
    payload = {"input": {}, "output": output or {"ok": True}, "is_error": False}
    return Step(RUN_ID, seq, StepKind.TOOL, status, tool_use_id, "get_order", payload, 0, 1, datetime.now(UTC))


def tool_use(tool_use_id: str, name: str = "get_order") -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_use_id, "name": name, "input": {"order_id": "ord_1001"}}


def test_empty_journal_calls_the_model_with_the_goal():
    action = plan_next(GOAL, [], max_steps=5)
    assert action == CallLLM([{"role": "user", "content": GOAL}])


def test_pending_tool_use_runs_next():
    action = plan_next(GOAL, [llm_step(1, [tool_use("a")])], max_steps=5)
    assert action == RunTool("a", "get_order", {"order_id": "ord_1001"}, started_seq=None)


def test_parallel_tool_uses_run_in_order_and_skip_finished_ones():
    steps = [llm_step(1, [tool_use("a"), tool_use("b")]), tool_step(2, "a")]
    action = plan_next(GOAL, steps, max_steps=5)
    assert isinstance(action, RunTool) and action.tool_use_id == "b"


def test_started_external_call_is_resumed_with_its_seq():
    steps = [llm_step(1, [tool_use("a")]), tool_step(2, "a", status=StepStatus.STARTED)]
    action = plan_next(GOAL, steps, max_steps=5)
    assert action == RunTool("a", "get_order", {"order_id": "ord_1001"}, started_seq=2)


def test_all_results_in_calls_the_model_with_tool_results_grouped_in_one_message():
    steps = [llm_step(1, [tool_use("a"), tool_use("b")]), tool_step(2, "a"), tool_step(3, "b")]
    action = plan_next(GOAL, steps, max_steps=5)
    assert isinstance(action, CallLLM)
    last = action.messages[-1]
    assert last["role"] == "user"
    assert [block["tool_use_id"] for block in last["content"]] == ["a", "b"]


def test_end_turn_finishes_with_the_text():
    steps = [llm_step(1, [{"type": "text", "text": "All done."}], stop_reason="end_turn")]
    assert plan_next(GOAL, steps, max_steps=5) == Finish("All done.")


def test_step_limit_fails_the_run():
    steps = [llm_step(1, [tool_use("a")]), tool_step(2, "a")]
    action = plan_next(GOAL, steps, max_steps=1)
    assert isinstance(action, Fail) and "step limit" in action.error


def test_refusal_and_max_tokens_fail():
    assert isinstance(plan_next(GOAL, [llm_step(1, [], stop_reason="refusal")], max_steps=5), Fail)
    assert isinstance(plan_next(GOAL, [llm_step(1, [], stop_reason="max_tokens")], max_steps=5), Fail)


def test_build_messages_round_trips_assistant_content_unchanged():
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}, tool_use("a")]
    messages = build_messages(GOAL, [llm_step(1, content), tool_step(2, "a", output={"x": 1})])
    assert messages[1] == {"role": "assistant", "content": content}
    assert messages[2]["content"][0]["content"] == '{"x": 1}'


def test_started_tool_steps_are_not_sent_as_results():
    messages = build_messages(GOAL, [llm_step(1, [tool_use("a")]), tool_step(2, "a", status=StepStatus.STARTED)])
    assert [m["role"] for m in messages] == ["user", "assistant"]
