import asyncio

import pytest

from stepwise import store
from stepwise.engine import RunExecutor
from stepwise.llm import ScriptedLLM
from stepwise.models import RunStatus
from stepwise.planner import RunTool
from stepwise.tools import IssueRefundArgs, ToolContext, default_registry
from tests.conftest import claim
from tests.scripts import refund_script


async def test_concurrent_workers_never_claim_the_same_run(pool, create_run):
    runs = [await create_run() for _ in range(5)]
    leases = await asyncio.gather(*(store.claim_run(pool, owner=f"w{i}", lease_seconds=5) for i in range(20)))
    claimed = [lease.run_id for lease in leases if lease is not None]
    assert sorted(claimed) == sorted(run.id for run in runs)


async def test_expired_lease_is_taken_over_with_a_new_epoch(pool, create_run):
    await create_run()
    first = await claim(pool, owner="a", lease_seconds=0.1)
    assert await store.claim_run(pool, owner="b", lease_seconds=5) is None
    await asyncio.sleep(0.2)
    second = await claim(pool, owner="b")
    assert second.run_id == first.run_id and second.epoch == first.epoch + 1

    with pytest.raises(store.LeaseLostError):
        await store.heartbeat(pool, first, lease_seconds=5)
    await store.heartbeat(pool, second, lease_seconds=5)


async def test_zombie_worker_cannot_commit_a_side_effect_after_takeover(pool, settings, create_run):
    """A worker that stalls past its lease wakes up and tries to refund: the fence rolls it back."""
    run = await create_run()
    zombie = await claim(pool, owner="zombie", lease_seconds=0.1)
    await asyncio.sleep(0.2)
    await claim(pool, owner="healthy")

    executor = RunExecutor(pool, ScriptedLLM(refund_script()), default_registry(), settings)
    args = IssueRefundArgs(order_id="ord_1001", amount_cents=2900, reason="damaged item")
    action = RunTool("tu_refund", "issue_refund", args.model_dump(), started_seq=None)
    ctx = ToolContext(run_id=run.id, tool_use_id="tu_refund", settings=settings)
    tool = default_registry().get("issue_refund")
    with pytest.raises(store.LeaseLostError):
        await executor._run_transactional(zombie, action, tool, args, ctx)

    assert await pool.fetchval("SELECT count(*) FROM refunds") == 0
    assert await store.list_steps(pool, run.id) == []


async def test_cancelling_a_run_stops_the_worker_holding_it(pool, settings, create_run):
    run = await create_run()
    lease = await claim(pool)
    await store.cancel_run(pool, run.id)

    executor = RunExecutor(pool, ScriptedLLM(refund_script()), default_registry(), settings)
    with pytest.raises(store.LeaseLostError):
        await executor.execute(lease)
    assert (await store.get_run(pool, run.id)).status == RunStatus.CANCELLED


async def test_runs_that_keep_crashing_are_failed_after_max_attempts(pool, create_run):
    run = await create_run()
    for _ in range(3):
        await claim(pool, lease_seconds=0.01)
        await asyncio.sleep(0.03)

    assert await store.fail_exhausted_runs(pool, max_attempts=3) == 1
    failed = await store.get_run(pool, run.id)
    assert failed.status == RunStatus.FAILED and "gave up after 3 attempts" in failed.error


async def test_transient_errors_requeue_with_backoff_then_give_up(pool, make_worker, settings, create_run):
    from stepwise.llm import TransientLLMError

    class Flaky(ScriptedLLM):
        async def complete(self, **kwargs):
            raise TransientLLMError("overloaded")

    run = await create_run()
    worker = make_worker([])
    worker._executor._llm = Flaky([])
    for _ in range(settings.max_attempts):
        await worker.run_once()

    done = await store.get_run(pool, run.id)
    assert done.status == RunStatus.FAILED
    assert done.attempts == settings.max_attempts and "gave up" in done.error
    assert done.reserved_micros == 0 and done.spent_micros == 0  # an answered error is never billed
