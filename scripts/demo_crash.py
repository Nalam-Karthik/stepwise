"""Crash a worker right after it emails a customer, then watch a second worker finish the run.

Uses a scripted model, so it needs Postgres (`make db`) but no API key:

    .venv/bin/python scripts/demo_crash.py
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stepwise import store  # noqa: E402
from stepwise.config import get_settings  # noqa: E402
from stepwise.db import create_pool, migrate  # noqa: E402
from tests.scripts import refund_script  # noqa: E402

LEASE_SECONDS = 2


def start_worker(script: Path, fault: str | None) -> int:
    env = os.environ | {
        "STEPWISE_LLM": f"scripted:{script}",
        "STEPWISE_LEASE_SECONDS": str(LEASE_SECONDS),
        "STEPWISE_HEARTBEAT_SECONDS": "0.5",
    }
    env.pop("STEPWISE_FAULT", None)
    if fault:
        env["STEPWISE_FAULT"] = fault
    return subprocess.run([sys.executable, "-m", "stepwise.worker", "--exit-when-idle"], cwd=ROOT, env=env).returncode


async def main() -> None:
    pool = await create_pool(get_settings().database_url)
    await migrate(pool)
    await pool.execute("DELETE FROM email_deliveries; DELETE FROM refunds; UPDATE orders SET refunded_cents = 0")
    run = await store.create_run(
        pool, goal="Order ord_1001 arrived damaged. Refund $29.", max_cost_micros=2_000_000, max_steps=10
    )

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "script.json"
        script.write_text(json.dumps(refund_script()))

        print(f"\n== worker 1: crashes right after the email provider accepts the message (run {run.id})")
        print(f"   exit code {start_worker(script, 'after_external_call')}")
        print(f"\n== waiting {LEASE_SECONDS}s for its lease to expire")
        await asyncio.sleep(LEASE_SECONDS + 0.5)
        print("\n== worker 2: claims the run with a new lease epoch and resumes from the journal")
        print(f"   exit code {start_worker(script, None)}")

    done = await store.get_run(pool, run.id)
    print(f"\nrun {done.status} after {done.attempts} attempts, spent ${done.spent_micros / 1e6:.4f}")
    print(f"result: {done.result}\n")
    print(f"{'seq':>3}  {'kind':<4}  {'tool':<20}  {'epoch':>5}  output")
    for step in await store.list_steps(pool, run.id):
        output = step.payload.get("output", step.payload.get("stop_reason", ""))
        print(f"{step.seq:>3}  {step.kind:<4}  {step.tool_name or '':<20}  {step.lease_epoch:>5}  {output}")
    emails = await pool.fetchval("SELECT count(*) FROM email_deliveries")
    refunds = await pool.fetchval("SELECT count(*) FROM refunds")
    print(f"\nemails delivered: {emails}   refunds issued: {refunds}")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
