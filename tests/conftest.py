import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import asyncpg
import pytest

from stepwise import store
from stepwise.config import Settings
from stepwise.db import create_pool, migrate
from stepwise.engine import RunExecutor
from stepwise.faults import FaultInjector
from stepwise.llm import ScriptedLLM
from stepwise.models import Lease, Run
from stepwise.tools import default_registry
from stepwise.worker import Worker

ADMIN_DSN = os.environ.get("STEPWISE_TEST_ADMIN_DSN", "postgresql://stepwise:stepwise@localhost:5546/stepwise")
TEST_DB = "stepwise_test"
TEST_DSN = ADMIN_DSN.rsplit("/", 1)[0] + f"/{TEST_DB}"


@pytest.fixture(scope="session")
async def pool() -> AsyncIterator[asyncpg.Pool]:
    try:
        admin = await asyncpg.connect(ADMIN_DSN)
    except OSError as exc:
        pytest.exit(f"Postgres is not reachable at {ADMIN_DSN} ({exc}). Run `make db` first.", returncode=2)
    try:
        await admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        await admin.execute(f"CREATE DATABASE {TEST_DB}")
    finally:
        await admin.close()
    test_pool = await create_pool(TEST_DSN, max_size=20)
    await migrate(test_pool)
    yield test_pool
    await test_pool.close()


@pytest.fixture(autouse=True)
async def clean_db(request: pytest.FixtureRequest) -> AsyncIterator[None]:
    if "pool" not in request.fixturenames:
        yield
        return
    db: asyncpg.Pool = request.getfixturevalue("pool")
    await db.execute("TRUNCATE runs, steps, approvals, refunds, email_deliveries")
    await db.execute("UPDATE orders SET refunded_cents = 0")
    yield


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url=TEST_DSN,
        lease_seconds=5,
        heartbeat_seconds=1,
        poll_seconds=0.05,
        retry_backoff_seconds=0,
        max_attempts=3,
        llm="scripted:unused",
    )


@pytest.fixture
def make_worker(pool: asyncpg.Pool, settings: Settings) -> Callable[..., Worker]:
    def _make(script: list[dict[str, Any]], *, worker_id: str = "test-worker", fault: str | None = None) -> Worker:
        executor = RunExecutor(pool, ScriptedLLM(script), default_registry(), settings, FaultInjector(fault))
        return Worker(pool, executor, settings, worker_id=worker_id)

    return _make


@pytest.fixture
def create_run(pool: asyncpg.Pool) -> Callable[..., Any]:
    async def _create(goal: str = "Refund order ord_1001, it arrived damaged.", **kwargs: Any) -> Run:
        params = {"max_cost_micros": 2_000_000, "max_steps": 10} | kwargs
        return await store.create_run(pool, goal=goal, **params)

    return _create


async def claim(pool: asyncpg.Pool, owner: str = "test", lease_seconds: float = 5) -> Lease:
    lease = await store.claim_run(pool, owner=owner, lease_seconds=lease_seconds)
    assert lease is not None
    return lease
