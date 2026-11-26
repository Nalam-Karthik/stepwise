"""Kill a real worker process at each dangerous point, then let a fresh worker finish the run.

The crashed process exits through os._exit: no exception handlers, no finally blocks, no lease
release, the same as SIGKILL or a lost VM. The second worker finds the run by its expired lease.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from stepwise import store
from stepwise.faults import CRASH_EXIT_CODE
from stepwise.models import RunStatus, StepKind, StepStatus
from tests.conftest import TEST_DSN
from tests.scripts import final_turn, refund_script, tool_turn

LEASE_SECONDS = 1.0
REPO_ROOT = Path(__file__).resolve().parent.parent


async def run_worker(script_path: Path, *, fault: str | None = None) -> int:
    env = os.environ | {
        "STEPWISE_DATABASE_URL": TEST_DSN,
        "STEPWISE_LLM": f"scripted:{script_path}",
        "STEPWISE_LEASE_SECONDS": str(LEASE_SECONDS),
        "STEPWISE_HEARTBEAT_SECONDS": "0.3",
        "STEPWISE_POLL_SECONDS": "0.05",
        "STEPWISE_RETRY_BACKOFF_SECONDS": "0",
    }
    env.pop("STEPWISE_FAULT", None)
    if fault:
        env["STEPWISE_FAULT"] = fault
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "stepwise.worker",
        "--exit-when-idle",
        cwd=REPO_ROOT,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    if proc.returncode not in (0, CRASH_EXIT_CODE):
        pytest.fail(f"worker exited {proc.returncode}:\n{output.decode()}")
    return proc.returncode


async def crash_then_recover(script: list, tmp_path: Path, fault: str) -> None:
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script))
    assert await run_worker(script_path, fault=fault) == CRASH_EXIT_CODE
    await asyncio.sleep(LEASE_SECONDS + 0.3)  # let the dead worker's lease expire
    assert await run_worker(script_path) == 0


async def test_crash_after_external_call_does_not_send_the_email_twice(pool, create_run, tmp_path):
    run = await create_run()
    await crash_then_recover(refund_script(), tmp_path, "after_external_call")

    done = await store.get_run(pool, run.id)
    assert done.status == RunStatus.SUCCEEDED and done.attempts == 2
    assert await pool.fetchval("SELECT count(*) FROM email_deliveries") == 1
    email_step = next(s for s in await store.list_steps(pool, run.id) if s.tool_name == "send_customer_email")
    assert email_step.status == StepStatus.COMPLETED
    assert email_step.payload["output"]["deduplicated"] is True  # the retry hit the provider's dedup
    assert email_step.lease_epoch == 2


async def test_crash_before_external_call_sends_it_exactly_once(pool, create_run, tmp_path):
    run = await create_run()
    await crash_then_recover(refund_script(), tmp_path, "before_external_call")

    assert (await store.get_run(pool, run.id)).status == RunStatus.SUCCEEDED
    assert await pool.fetchval("SELECT count(*) FROM email_deliveries") == 1
    email_step = next(s for s in await store.list_steps(pool, run.id) if s.tool_name == "send_customer_email")
    assert email_step.payload["output"]["deduplicated"] is False


async def test_crash_inside_a_transactional_tool_rolls_back_and_refunds_once(pool, create_run, tmp_path):
    run = await create_run()
    script = [
        tool_turn("tu_refund", "issue_refund", {"order_id": "ord_1001", "amount_cents": 2900, "reason": "damaged"}),
        final_turn("Refunded."),
    ]
    await crash_then_recover(script, tmp_path, "before_tool_commit")

    assert (await store.get_run(pool, run.id)).status == RunStatus.SUCCEEDED
    assert await pool.fetchval("SELECT count(*) FROM refunds") == 1
    assert await pool.fetchval("SELECT refunded_cents FROM orders WHERE id = 'ord_1001'") == 2900


async def test_crash_after_model_call_charges_the_lost_call_and_still_finishes(pool, create_run, tmp_path):
    run = await create_run()
    await crash_then_recover(refund_script(), tmp_path, "after_llm_call")

    done = await store.get_run(pool, run.id)
    assert done.status == RunStatus.SUCCEEDED
    steps = await store.list_steps(pool, run.id)
    journaled = sum(s.cost_micros for s in steps)
    # The crashed call's bill is unknown, so its worst-case reservation was charged on takeover.
    assert done.spent_micros > journaled
    assert done.reserved_micros == 0
    assert len([s for s in steps if s.kind == StepKind.LLM]) == 4
    assert await pool.fetchval("SELECT count(*) FROM refunds") == 1
