"""SQL for runs, the step journal and approvals.

Every write that a worker makes on behalf of a run goes through `fence()` first, inside the
same transaction. `fence()` locks the run row and checks the caller's lease epoch, so a worker
that lost its lease (paused by GC, partitioned, presumed dead) can never append to the journal
or commit a tool's side effects after another worker has taken over.
"""

from typing import Any
from uuid import UUID, uuid4

import asyncpg

from stepwise.models import Approval, ApprovalStatus, Lease, Run, RunStatus, Step, StepKind, StepStatus

Executor = asyncpg.Connection | asyncpg.Pool


class LeaseLostError(Exception):
    """The run was claimed by another worker, cancelled or finished since this lease was issued."""


class ApprovalConflictError(Exception):
    """The approval was already decided."""


# ---------------------------------------------------------------- runs


async def create_run(db: Executor, *, goal: str, max_cost_micros: int, max_steps: int) -> Run:
    row = await db.fetchrow(
        "INSERT INTO runs (id, goal, status, max_cost_micros, max_steps) VALUES ($1, $2, 'queued', $3, $4) RETURNING *",
        uuid4(),
        goal,
        max_cost_micros,
        max_steps,
    )
    return Run.from_row(row)


async def get_run(db: Executor, run_id: UUID) -> Run | None:
    row = await db.fetchrow("SELECT * FROM runs WHERE id = $1", run_id)
    return Run.from_row(row) if row else None


async def fail_exhausted_runs(db: Executor, *, max_attempts: int) -> int:
    """Poison-run guard: an expired lease on a run that has used every attempt fails the run."""
    result = await db.execute(
        "UPDATE runs SET status = 'failed', lease_owner = NULL, lease_expires_at = NULL,"
        " error = coalesce(error, 'worker crashed') || ' (gave up after ' || attempts || ' attempts)',"
        " updated_at = now()"
        " WHERE status = 'running' AND lease_expires_at < now() AND attempts >= $1",
        max_attempts,
    )
    return int(result.split()[-1])


async def claim_run(db: Executor, *, owner: str, lease_seconds: float) -> Lease | None:
    """Claim the oldest runnable run: a queued one, or a running one whose worker stopped heartbeating.

    SKIP LOCKED lets many workers poll the same table without blocking each other. Any budget
    reservation still on the row belongs to a dead attempt whose LLM call may have been billed,
    so it is charged as spent rather than released.
    """
    row = await db.fetchrow(
        """
        UPDATE runs SET
            status = 'running',
            lease_owner = $1,
            lease_expires_at = now() + make_interval(secs => $2),
            lease_epoch = lease_epoch + 1,
            attempts = attempts + 1,
            spent_micros = spent_micros + reserved_micros,
            reserved_micros = 0,
            updated_at = now()
        WHERE id = (
            SELECT id FROM runs
            WHERE (status = 'queued' AND available_at <= now())
               OR (status = 'running' AND lease_expires_at < now())
            ORDER BY available_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, lease_epoch
        """,
        owner,
        lease_seconds,
    )
    if row is None:
        return None
    return Lease(run_id=row["id"], epoch=row["lease_epoch"], owner=owner)


async def heartbeat(db: Executor, lease: Lease, *, lease_seconds: float) -> None:
    result = await db.execute(
        "UPDATE runs SET lease_expires_at = now() + make_interval(secs => $3)"
        " WHERE id = $1 AND lease_epoch = $2 AND status = 'running'",
        lease.run_id,
        lease.epoch,
        lease_seconds,
    )
    if result == "UPDATE 0":
        raise LeaseLostError(str(lease.run_id))


async def fence(conn: asyncpg.Connection, lease: Lease) -> Run:
    """Lock the run row and verify the lease. Must be called inside a transaction."""
    row = await conn.fetchrow("SELECT * FROM runs WHERE id = $1 FOR UPDATE", lease.run_id)
    if row is None or row["lease_epoch"] != lease.epoch or row["status"] != RunStatus.RUNNING:
        raise LeaseLostError(str(lease.run_id))
    return Run.from_row(row)


async def finish_run(
    conn: asyncpg.Connection,
    lease: Lease,
    status: RunStatus,
    *,
    result: str | None = None,
    error: str | None = None,
) -> None:
    """Move a fenced run to a status that needs no worker (terminal or awaiting approval).

    `attempts` counts consecutive failed attempts, so a deliberate pause for approval resets it;
    otherwise every approval round-trip would eat into the crash-retry allowance.
    """
    await conn.execute(
        "UPDATE runs SET status = $2, result = $3, error = $4, lease_owner = NULL, lease_expires_at = NULL,"
        " attempts = CASE WHEN $2 = 'awaiting_approval' THEN 0 ELSE attempts END, updated_at = now()"
        " WHERE id = $1",
        lease.run_id,
        status,
        result,
        error,
    )


async def requeue_run(db: Executor, lease: Lease, *, error: str, delay_seconds: float) -> None:
    """Give the run back after a transient failure; another worker retries it after the delay.

    A leftover reservation means an LLM call whose billing is unknown, so it is charged."""
    await db.execute(
        "UPDATE runs SET status = 'queued', lease_owner = NULL, lease_expires_at = NULL,"
        " spent_micros = spent_micros + reserved_micros, reserved_micros = 0,"
        " error = $3, available_at = now() + make_interval(secs => $4), updated_at = now()"
        " WHERE id = $1 AND lease_epoch = $2 AND status = 'running'",
        lease.run_id,
        lease.epoch,
        error,
        delay_seconds,
    )


async def fail_run(db: Executor, lease: Lease, *, error: str) -> None:
    await db.execute(
        "UPDATE runs SET status = 'failed', error = $3, lease_owner = NULL, lease_expires_at = NULL,"
        " spent_micros = spent_micros + reserved_micros, reserved_micros = 0, updated_at = now()"
        " WHERE id = $1 AND lease_epoch = $2 AND status = 'running'",
        lease.run_id,
        lease.epoch,
        error,
    )


async def cancel_run(db: Executor, run_id: UUID) -> Run | None:
    """Cancel a non-terminal run. A worker holding it fails its next fence and stops."""
    row = await db.fetchrow(
        "UPDATE runs SET status = 'cancelled', lease_owner = NULL, lease_expires_at = NULL, updated_at = now()"
        " WHERE id = $1 AND status IN ('queued', 'running', 'awaiting_approval') RETURNING *",
        run_id,
    )
    return Run.from_row(row) if row else None


# ---------------------------------------------------------------- budget


async def reserve_budget(conn: asyncpg.Connection, run_id: UUID, micros: int) -> None:
    await conn.execute(
        "UPDATE runs SET reserved_micros = reserved_micros + $2, updated_at = now() WHERE id = $1",
        run_id,
        micros,
    )


async def release_reservation(db: Executor, lease: Lease, micros: int) -> None:
    """Drop a reservation for an LLM call that is known not to have been billed."""
    await db.execute(
        "UPDATE runs SET reserved_micros = greatest(reserved_micros - $3, 0) WHERE id = $1 AND lease_epoch = $2",
        lease.run_id,
        lease.epoch,
        micros,
    )


# ---------------------------------------------------------------- steps


async def list_steps(db: Executor, run_id: UUID, *, limit: int | None = None, offset: int = 0) -> list[Step]:
    rows = await db.fetch(
        "SELECT * FROM steps WHERE run_id = $1 ORDER BY seq LIMIT $2 OFFSET $3",
        run_id,
        limit,
        offset,
    )
    return [Step.from_row(row) for row in rows]


async def _next_seq(conn: asyncpg.Connection, run_id: UUID) -> int:
    return await conn.fetchval("SELECT coalesce(max(seq), 0) + 1 FROM steps WHERE run_id = $1", run_id)


async def append_llm_step(
    conn: asyncpg.Connection,
    lease: Lease,
    *,
    payload: dict[str, Any],
    cost_micros: int,
    reserved_micros: int,
    input_tokens: int,
    output_tokens: int,
) -> int:
    """Journal a completed LLM call and settle its reservation. Caller must hold the fence."""
    seq = await _next_seq(conn, lease.run_id)
    await conn.execute(
        "INSERT INTO steps (run_id, seq, kind, status, payload, cost_micros, lease_epoch, completed_at)"
        " VALUES ($1, $2, 'llm', 'completed', $3, $4, $5, now())",
        lease.run_id,
        seq,
        payload,
        cost_micros,
        lease.epoch,
    )
    await conn.execute(
        "UPDATE runs SET spent_micros = spent_micros + $2,"
        " reserved_micros = greatest(reserved_micros - $3, 0),"
        " input_tokens = input_tokens + $4, output_tokens = output_tokens + $5, updated_at = now()"
        " WHERE id = $1",
        lease.run_id,
        cost_micros,
        reserved_micros,
        input_tokens,
        output_tokens,
    )
    return seq


async def append_tool_step(
    conn: asyncpg.Connection,
    lease: Lease,
    *,
    tool_use_id: str,
    tool_name: str,
    status: StepStatus,
    payload: dict[str, Any],
) -> int:
    seq = await _next_seq(conn, lease.run_id)
    await conn.execute(
        "INSERT INTO steps (run_id, seq, kind, status, tool_use_id, tool_name, payload, lease_epoch, completed_at)"
        " VALUES ($1, $2, 'tool', $3, $4, $5, $6, $7, CASE WHEN $3 = 'completed' THEN now() END)",
        lease.run_id,
        seq,
        status,
        tool_use_id,
        tool_name,
        payload,
        lease.epoch,
    )
    return seq


async def complete_tool_step(conn: asyncpg.Connection, lease: Lease, *, seq: int, payload: dict[str, Any]) -> None:
    await conn.execute(
        "UPDATE steps SET status = 'completed', payload = $3, completed_at = now(), lease_epoch = $4"
        " WHERE run_id = $1 AND seq = $2 AND kind = $5",
        lease.run_id,
        seq,
        payload,
        lease.epoch,
        StepKind.TOOL,
    )


# ---------------------------------------------------------------- approvals


async def get_approval_for_tool(db: Executor, run_id: UUID, tool_use_id: str) -> Approval | None:
    row = await db.fetchrow("SELECT * FROM approvals WHERE run_id = $1 AND tool_use_id = $2", run_id, tool_use_id)
    return Approval.from_row(row) if row else None


async def get_approval(db: Executor, approval_id: UUID) -> Approval | None:
    row = await db.fetchrow("SELECT * FROM approvals WHERE id = $1", approval_id)
    return Approval.from_row(row) if row else None


async def create_approval(
    conn: asyncpg.Connection, *, run_id: UUID, tool_use_id: str, tool_name: str, tool_input: dict[str, Any]
) -> Approval:
    row = await conn.fetchrow(
        "INSERT INTO approvals (id, run_id, tool_use_id, tool_name, tool_input, status)"
        " VALUES ($1, $2, $3, $4, $5, 'pending')"
        " ON CONFLICT (run_id, tool_use_id) DO UPDATE SET tool_name = EXCLUDED.tool_name RETURNING *",
        uuid4(),
        run_id,
        tool_use_id,
        tool_name,
        tool_input,
    )
    return Approval.from_row(row)


async def list_approvals(db: Executor, *, status: ApprovalStatus | None, limit: int, offset: int) -> list[Approval]:
    rows = await db.fetch(
        "SELECT * FROM approvals WHERE ($1::text IS NULL OR status = $1) ORDER BY created_at LIMIT $2 OFFSET $3",
        status,
        limit,
        offset,
    )
    return [Approval.from_row(row) for row in rows]


async def decide_approval(
    pool: asyncpg.Pool, approval_id: UUID, *, approve: bool, decided_by: str, reason: str | None
) -> Approval | None:
    """Record a human decision and wake the run. Returns None if the approval does not exist."""
    async with pool.acquire() as conn, conn.transaction():
        row = await conn.fetchrow("SELECT * FROM approvals WHERE id = $1 FOR UPDATE", approval_id)
        if row is None:
            return None
        if row["status"] != ApprovalStatus.PENDING:
            raise ApprovalConflictError(str(approval_id))
        decided = await conn.fetchrow(
            "UPDATE approvals SET status = $2, decided_by = $3, reason = $4, decided_at = now()"
            " WHERE id = $1 RETURNING *",
            approval_id,
            ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED,
            decided_by,
            reason,
        )
        await conn.execute(
            "UPDATE runs SET status = 'queued', available_at = now(), updated_at = now()"
            " WHERE id = $1 AND status = 'awaiting_approval'",
            row["run_id"],
        )
        return Approval.from_row(decided)
