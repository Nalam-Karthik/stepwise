"""Drives one run forward, one journaled step at a time, while holding its lease."""

import logging
from typing import Any

import asyncpg
from pydantic import BaseModel

from stepwise import store
from stepwise.config import Settings
from stepwise.faults import NO_FAULTS, FaultInjector
from stepwise.llm import LLM, LLMError, cost_micros, estimate_max_cost_micros
from stepwise.models import ApprovalStatus, Lease, RunStatus, StepStatus
from stepwise.planner import CallLLM, Fail, Finish, RunTool, plan_next
from stepwise.tools import Tool, ToolContext, ToolError, ToolRegistry, parse_args

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a customer support agent for an online store. You can look up orders, \
issue refunds and email customers.

Check the order before refunding it. Never refund more than is left unrefunded on an order. A large \
refund may be sent to a human for approval; if a tool result says a human rejected it, do not retry \
it, and tell the customer politely instead. After acting, email the customer a short confirmation. \
Finish with a one-paragraph summary of what you did."""


class RunExecutor:
    def __init__(
        self,
        pool: asyncpg.Pool,
        llm: LLM,
        tools: ToolRegistry,
        settings: Settings,
        faults: FaultInjector = NO_FAULTS,
    ) -> None:
        self._pool = pool
        self._llm = llm
        self._tools = tools
        self._settings = settings
        self._faults = faults

    async def execute(self, lease: Lease) -> RunStatus:
        """Advance the run until it needs no worker. Returns the status it was left in."""
        while True:
            status = await self._advance(lease)
            if status is not None:
                return status

    async def _advance(self, lease: Lease) -> RunStatus | None:
        run = await store.get_run(self._pool, lease.run_id)
        if run is None or run.lease_epoch != lease.epoch or run.status != RunStatus.RUNNING:
            raise store.LeaseLostError(str(lease.run_id))
        steps = await store.list_steps(self._pool, lease.run_id)
        action = plan_next(run.goal, steps, max_steps=run.max_steps)

        match action:
            case Finish(result=result):
                return await self._finish(lease, RunStatus.SUCCEEDED, result=result)
            case Fail(error=error):
                status = RunStatus.BUDGET_EXCEEDED if error.startswith("step limit") else RunStatus.FAILED
                return await self._finish(lease, status, error=error)
            case CallLLM(messages=messages):
                return await self._call_llm(lease, messages)
            case RunTool():
                return await self._run_tool(lease, action)
        raise AssertionError(f"unhandled action {action!r}")

    async def _finish(
        self, lease: Lease, status: RunStatus, *, result: str | None = None, error: str | None = None
    ) -> RunStatus:
        async with self._pool.acquire() as conn, conn.transaction():
            await store.fence(conn, lease)
            await store.finish_run(conn, lease, status, result=result, error=error)
        logger.info("run %s finished: %s", lease.run_id, status)
        return status

    # ------------------------------------------------------------ LLM steps

    async def _call_llm(self, lease: Lease, messages: list[dict[str, Any]]) -> RunStatus | None:
        tools = self._tools.definitions()
        max_tokens = self._settings.max_output_tokens
        reservation = estimate_max_cost_micros(
            self._llm.model, system=SYSTEM_PROMPT, messages=messages, tools=tools, max_tokens=max_tokens
        )

        async with self._pool.acquire() as conn, conn.transaction():
            run = await store.fence(conn, lease)
            committed = run.spent_micros + run.reserved_micros
            if committed + reservation > run.max_cost_micros:
                await store.finish_run(
                    conn,
                    lease,
                    RunStatus.BUDGET_EXCEEDED,
                    error=(
                        f"next model call could cost up to ${reservation / 1e6:.4f}; "
                        f"${committed / 1e6:.4f} of ${run.max_cost_micros / 1e6:.4f} already spent"
                    ),
                )
                return RunStatus.BUDGET_EXCEEDED
            await store.reserve_budget(conn, lease.run_id, reservation)

        try:
            response = await self._llm.complete(
                system=SYSTEM_PROMPT, messages=messages, tools=tools, max_tokens=max_tokens
            )
        except LLMError as exc:
            # Only an error the API answered (429, 4xx, 5xx) proves the call was not billed. After
            # a timeout or dropped connection the reservation stays, and is charged as spent when
            # the run is requeued: the budget may under-use, but it never overspends.
            if not exc.may_have_billed:
                await store.release_reservation(self._pool, lease, reservation)
            raise

        self._faults.hit("after_llm_call")
        cost = cost_micros(response.model, response.input_tokens, response.output_tokens)
        async with self._pool.acquire() as conn, conn.transaction():
            await store.fence(conn, lease)
            await store.append_llm_step(
                conn,
                lease,
                payload={"content": response.content, "stop_reason": response.stop_reason, "model": response.model},
                cost_micros=cost,
                reserved_micros=reservation,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        return None

    # ------------------------------------------------------------ tool steps

    async def _run_tool(self, lease: Lease, action: RunTool) -> RunStatus | None:
        tool = self._tools.get(action.name)
        if tool is None:
            await self._journal_tool_result(lease, action, {"error": f"unknown tool {action.name!r}"}, is_error=True)
            return None
        try:
            args = parse_args(tool, action.input)
        except ToolError as exc:
            await self._journal_tool_result(lease, action, {"error": str(exc)}, is_error=True)
            return None

        if tool.needs_approval(args, self._settings):
            gate = await self._approval_gate(lease, action)
            if gate == "wait":
                return RunStatus.AWAITING_APPROVAL
            if gate == "rejected":
                return None

        ctx = ToolContext(run_id=lease.run_id, tool_use_id=action.tool_use_id, settings=self._settings)
        if tool.transactional:
            await self._run_transactional(lease, action, tool, args, ctx)
        else:
            await self._run_external(lease, action, tool, args, ctx)
        return None

    async def _approval_gate(self, lease: Lease, action: RunTool) -> str:
        """Returns "approved", "rejected" (result already journaled) or "wait" (run parked)."""
        approval = await store.get_approval_for_tool(self._pool, lease.run_id, action.tool_use_id)
        if approval is not None and approval.status == ApprovalStatus.APPROVED:
            return "approved"
        if approval is not None and approval.status == ApprovalStatus.REJECTED:
            note = f"rejected by {approval.decided_by}" + (f": {approval.reason}" if approval.reason else "")
            await self._journal_tool_result(lease, action, {"error": f"a human {note}"}, is_error=True)
            return "rejected"

        async with self._pool.acquire() as conn, conn.transaction():
            await store.fence(conn, lease)
            await store.create_approval(
                conn,
                run_id=lease.run_id,
                tool_use_id=action.tool_use_id,
                tool_name=action.name,
                tool_input=action.input,
            )
            await store.finish_run(conn, lease, RunStatus.AWAITING_APPROVAL)
        logger.info("run %s waiting for approval of %s", lease.run_id, action.name)
        return "wait"

    async def _run_transactional(
        self, lease: Lease, action: RunTool, tool: Tool, args: BaseModel, ctx: ToolContext
    ) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await store.fence(conn, lease)
            try:
                async with conn.transaction():  # savepoint: a ToolError undoes the tool's partial writes only
                    output, is_error = await tool.execute(conn, args, ctx), False
            except ToolError as exc:
                output, is_error = {"error": str(exc)}, True
            self._faults.hit("before_tool_commit")
            await store.append_tool_step(
                conn,
                lease,
                tool_use_id=action.tool_use_id,
                tool_name=action.name,
                status=StepStatus.COMPLETED,
                payload={"input": action.input, "output": output, "is_error": is_error},
            )

    async def _run_external(self, lease: Lease, action: RunTool, tool: Tool, args: BaseModel, ctx: ToolContext) -> None:
        seq = action.started_seq
        if seq is None:
            async with self._pool.acquire() as conn, conn.transaction():
                await store.fence(conn, lease)
                seq = await store.append_tool_step(
                    conn,
                    lease,
                    tool_use_id=action.tool_use_id,
                    tool_name=action.name,
                    status=StepStatus.STARTED,
                    payload={"input": action.input, "idempotency_key": ctx.idempotency_key},
                )
        else:
            logger.info("run %s: retrying %s with key %s", lease.run_id, action.name, ctx.idempotency_key)

        self._faults.hit("before_external_call")
        async with self._pool.acquire() as ext_conn:
            try:
                output, is_error = await tool.execute(ext_conn, args, ctx), False
            except ToolError as exc:
                output, is_error = {"error": str(exc)}, True
        self._faults.hit("after_external_call")

        async with self._pool.acquire() as conn, conn.transaction():
            await store.fence(conn, lease)
            await store.complete_tool_step(
                conn,
                lease,
                seq=seq,
                payload={
                    "input": action.input,
                    "idempotency_key": ctx.idempotency_key,
                    "output": output,
                    "is_error": is_error,
                },
            )

    async def _journal_tool_result(
        self, lease: Lease, action: RunTool, output: dict[str, Any], *, is_error: bool
    ) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await store.fence(conn, lease)
            payload = {"input": action.input, "output": output, "is_error": is_error}
            if action.started_seq is not None:
                await store.complete_tool_step(conn, lease, seq=action.started_seq, payload=payload)
            else:
                await store.append_tool_step(
                    conn,
                    lease,
                    tool_use_id=action.tool_use_id,
                    tool_name=action.name,
                    status=StepStatus.COMPLETED,
                    payload=payload,
                )
