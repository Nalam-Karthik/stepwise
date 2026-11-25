import pytest

from stepwise import store
from stepwise.models import ApprovalStatus, RunStatus, StepKind
from tests.scripts import refund_script

BIG_REFUND = refund_script(order_id="ord_1002", amount_cents=10_000, email="ravi@example.com")


async def test_large_refund_parks_the_run_until_a_human_approves(pool, make_worker, create_run):
    run = await create_run("Refund 100 dollars on ord_1002.")
    worker = make_worker(BIG_REFUND)
    await worker.run_once()

    parked = await store.get_run(pool, run.id)
    assert parked.status == RunStatus.AWAITING_APPROVAL
    assert parked.lease_owner is None
    assert parked.attempts == 0  # pausing for a human is not a failed attempt
    [approval] = await store.list_approvals(pool, status=ApprovalStatus.PENDING, limit=10, offset=0)
    assert approval.tool_name == "issue_refund" and approval.tool_input["amount_cents"] == 10_000
    assert await pool.fetchval("SELECT count(*) FROM refunds") == 0
    assert await worker.run_once() is False  # nothing claimable while it waits

    await store.decide_approval(pool, approval.id, approve=True, decided_by="ops@example.com", reason=None)
    assert (await store.get_run(pool, run.id)).status == RunStatus.QUEUED
    await worker.run_once()

    assert (await store.get_run(pool, run.id)).status == RunStatus.SUCCEEDED
    assert await pool.fetchval("SELECT refunded_cents FROM orders WHERE id = 'ord_1002'") == 10_000
    assert await pool.fetchval("SELECT count(*) FROM refunds") == 1


async def test_rejection_is_reported_to_the_model_and_nothing_is_refunded(pool, make_worker, create_run):
    run = await create_run("Refund 100 dollars on ord_1002.")
    worker = make_worker(BIG_REFUND)
    await worker.run_once()
    [approval] = await store.list_approvals(pool, status=ApprovalStatus.PENDING, limit=10, offset=0)

    await store.decide_approval(pool, approval.id, approve=False, decided_by="ops@example.com", reason="fraud risk")
    await worker.run_once()

    assert (await store.get_run(pool, run.id)).status == RunStatus.SUCCEEDED
    assert await pool.fetchval("SELECT count(*) FROM refunds") == 0
    refund_step = next(s for s in await store.list_steps(pool, run.id) if s.tool_name == "issue_refund")
    assert refund_step.kind == StepKind.TOOL and refund_step.payload["is_error"] is True
    assert refund_step.payload["output"]["error"] == "a human rejected by ops@example.com: fraud risk"


async def test_an_approval_can_only_be_decided_once(pool, make_worker, create_run):
    await create_run()
    await make_worker(BIG_REFUND).run_once()
    [approval] = await store.list_approvals(pool, status=ApprovalStatus.PENDING, limit=10, offset=0)

    await store.decide_approval(pool, approval.id, approve=True, decided_by="a", reason=None)
    with pytest.raises(store.ApprovalConflictError):
        await store.decide_approval(pool, approval.id, approve=False, decided_by="b", reason=None)


async def test_small_refunds_skip_approval(pool, make_worker, create_run):
    run = await create_run()
    await make_worker(refund_script(amount_cents=4_000)).run_once()
    assert (await store.get_run(pool, run.id)).status == RunStatus.SUCCEEDED
    assert await store.list_approvals(pool, status=None, limit=10, offset=0) == []
