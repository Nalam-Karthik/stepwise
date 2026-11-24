from stepwise import store
from stepwise.models import RunStatus, StepKind
from tests.scripts import final_turn, refund_script, tool_turn


async def test_happy_path_refunds_emails_and_journals_every_step(pool, make_worker, create_run):
    run = await create_run()
    assert await make_worker(refund_script()).run_once()

    done = await store.get_run(pool, run.id)
    assert done.status == RunStatus.SUCCEEDED
    assert "Refunded 2900" in done.result
    steps = await store.list_steps(pool, run.id)
    assert [(s.kind, s.tool_name) for s in steps] == [
        (StepKind.LLM, None),
        (StepKind.TOOL, "get_order"),
        (StepKind.LLM, None),
        (StepKind.TOOL, "issue_refund"),
        (StepKind.LLM, None),
        (StepKind.TOOL, "send_customer_email"),
        (StepKind.LLM, None),
    ]
    assert await pool.fetchval("SELECT refunded_cents FROM orders WHERE id = 'ord_1001'") == 2900
    assert await pool.fetchval("SELECT count(*) FROM email_deliveries") == 1
    # 4 calls x (input x $4 + output x $20 per MTok), with no reservation left behind.
    assert done.spent_micros == sum(s.cost_micros for s in steps) > 0
    assert done.reserved_micros == 0


async def test_tool_errors_are_returned_to_the_model_and_the_run_carries_on(pool, make_worker, create_run):
    run = await create_run()
    script = [
        tool_turn("tu_1", "issue_refund", {"order_id": "ord_1003", "amount_cents": 1000, "reason": "never arrived"}),
        final_turn("That order was cancelled, so there is nothing to refund."),
    ]
    await make_worker(script).run_once()

    steps = await store.list_steps(pool, run.id)
    assert steps[1].payload["is_error"] is True
    assert "cancelled" in steps[1].payload["output"]["error"]
    assert await pool.fetchval("SELECT count(*) FROM refunds") == 0
    assert (await store.get_run(pool, run.id)).status == RunStatus.SUCCEEDED


async def test_invalid_arguments_and_unknown_tools_become_error_results(pool, make_worker, create_run):
    run = await create_run()
    script = [
        {
            "content": [
                {"type": "tool_use", "id": "tu_bad", "name": "issue_refund", "input": {"order_id": "ord_1001"}},
                {"type": "tool_use", "id": "tu_nope", "name": "delete_database", "input": {}},
            ],
            "stop_reason": "tool_use",
        },
        final_turn("ok"),
    ]
    await make_worker(script).run_once()

    steps = await store.list_steps(pool, run.id)
    errors = {s.tool_use_id: s.payload["output"]["error"] for s in steps if s.kind == StepKind.TOOL}
    assert "invalid arguments" in errors["tu_bad"] and "amount_cents" in errors["tu_bad"]
    assert "unknown tool" in errors["tu_nope"]


async def test_budget_stops_the_run_before_a_call_that_could_overspend(pool, make_worker, create_run):
    # Each call reserves its worst case first: about $0.33 (16k output tokens at $20/MTok). The first
    # call fits a $0.40 budget and really costs $0.20 (50k input tokens), so the second can't fit.
    run = await create_run(max_cost_micros=400_000)
    script = refund_script()
    script[0] = script[0] | {"input_tokens": 50_000}
    await make_worker(script).run_once()

    done = await store.get_run(pool, run.id)
    assert done.status == RunStatus.BUDGET_EXCEEDED
    assert "could cost up to" in done.error
    assert len([s for s in await store.list_steps(pool, run.id) if s.kind == StepKind.LLM]) == 1
    assert done.spent_micros <= done.max_cost_micros


async def test_step_limit_ends_a_looping_agent(pool, make_worker, create_run):
    run = await create_run(max_steps=2)
    loop = [tool_turn(f"tu_{i}", "get_order", {"order_id": "ord_1001"}) for i in range(5)]
    await make_worker(loop).run_once()

    done = await store.get_run(pool, run.id)
    assert done.status == RunStatus.BUDGET_EXCEEDED
    assert "step limit" in done.error


async def test_permanent_llm_error_fails_the_run(pool, make_worker, create_run):
    run = await create_run()
    await make_worker([]).run_once()  # empty script: the scripted model raises PermanentLLMError

    done = await store.get_run(pool, run.id)
    assert done.status == RunStatus.FAILED
    assert "no response for turn 0" in done.error
    assert done.reserved_micros == 0 and done.spent_micros == 0


async def test_idle_worker_reports_no_work(make_worker):
    assert await make_worker([]).run_once() is False
