"""Worker process: claim a run, drive it while heartbeating its lease, repeat.

Run as `python -m stepwise.worker`. Start as many as you like, on any number of machines.
"""

import argparse
import asyncio
import logging
import os
import signal
import socket
from uuid import uuid4

import asyncpg

from stepwise import store
from stepwise.config import Settings, get_settings
from stepwise.db import create_pool, migrate
from stepwise.engine import RunExecutor
from stepwise.faults import FaultInjector
from stepwise.llm import PermanentLLMError, TransientLLMError, build_llm
from stepwise.models import Lease
from stepwise.tools import default_registry

logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, pool: asyncpg.Pool, executor: RunExecutor, settings: Settings, *, worker_id: str) -> None:
        self._pool = pool
        self._executor = executor
        self._settings = settings
        self.worker_id = worker_id

    async def run_forever(self, stop: asyncio.Event, *, exit_when_idle: bool = False) -> None:
        logger.info("worker %s started", self.worker_id)
        while not stop.is_set():
            worked = await self.run_once()
            if worked:
                continue
            if exit_when_idle:
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._settings.poll_seconds)
            except TimeoutError:
                pass
        logger.info("worker %s stopped", self.worker_id)

    async def run_once(self) -> bool:
        """Claim and drive one run. Returns False when there was nothing to claim."""
        await store.fail_exhausted_runs(self._pool, max_attempts=self._settings.max_attempts)
        lease = await store.claim_run(self._pool, owner=self.worker_id, lease_seconds=self._settings.lease_seconds)
        if lease is None:
            return False
        logger.info("worker %s claimed run %s (epoch %d)", self.worker_id, lease.run_id, lease.epoch)
        await self._drive(lease)
        return True

    async def _drive(self, lease: Lease) -> None:
        execution = asyncio.create_task(self._executor.execute(lease))
        beat = asyncio.create_task(self._heartbeat(lease, execution))
        try:
            await execution
        except store.LeaseLostError:
            logger.warning("run %s: lease lost (cancelled or taken over); abandoning", lease.run_id)
        except asyncio.CancelledError:
            if not beat.done():
                raise
            logger.warning("run %s: heartbeat failed; abandoning", lease.run_id)
        except TransientLLMError as exc:
            await self._retry_later(lease, str(exc))
        except PermanentLLMError as exc:
            await store.fail_run(self._pool, lease, error=str(exc))
        except Exception as exc:
            logger.exception("run %s: unexpected error", lease.run_id)
            await self._retry_later(lease, f"{type(exc).__name__}: {exc}")
        finally:
            beat.cancel()

    async def _retry_later(self, lease: Lease, error: str) -> None:
        run = await store.get_run(self._pool, lease.run_id)
        attempts = run.attempts if run else self._settings.max_attempts
        if attempts >= self._settings.max_attempts:
            await store.fail_run(self._pool, lease, error=f"{error} (gave up after {attempts} attempts)")
            return
        delay = self._settings.retry_backoff_seconds * 2 ** (attempts - 1)
        logger.warning("run %s: %s; retrying in %.1fs", lease.run_id, error, delay)
        await store.requeue_run(self._pool, lease, error=error, delay_seconds=delay)

    async def _heartbeat(self, lease: Lease, execution: asyncio.Task) -> None:
        while True:
            await asyncio.sleep(self._settings.heartbeat_seconds)
            try:
                await store.heartbeat(self._pool, lease, lease_seconds=self._settings.lease_seconds)
            except store.LeaseLostError:
                # Every journal write is fenced, so a late write would fail anyway; stopping
                # early just saves a wasted model call.
                execution.cancel()
                return
            except Exception:
                logger.exception("run %s: heartbeat error; will retry", lease.run_id)


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:6]}"


async def main(exit_when_idle: bool) -> None:
    settings = get_settings()
    pool = await create_pool(settings.database_url)
    try:
        await migrate(pool)
        llm = build_llm(settings.llm, model=settings.model, effort=settings.effort)
        executor = RunExecutor(pool, llm, default_registry(), settings, FaultInjector(settings.fault))
        worker = Worker(pool, executor, settings, worker_id=_worker_id())

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await worker.run_forever(stop, exit_when_idle=exit_when_idle)
    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a stepwise worker.")
    parser.add_argument("--exit-when-idle", action="store_true", help="exit once there is nothing to claim")
    cli_args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main(cli_args.exit_when_idle))
