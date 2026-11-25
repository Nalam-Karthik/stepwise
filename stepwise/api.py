"""HTTP API: submit runs, inspect their journal, cancel them, and decide pending approvals."""

import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

import asyncpg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from stepwise import store
from stepwise.config import Settings, get_settings
from stepwise.db import create_pool, migrate
from stepwise.models import Approval, ApprovalStatus, Run, Step

MAX_PAGE_SIZE = 200


# ---------------------------------------------------------------- schemas


class Envelope[T](BaseModel):
    success: bool = True
    data: T | None = None
    error: str | None = None


class CreateRunRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=10_000)
    max_cost_usd: float = Field(gt=0, le=100)
    max_steps: int = Field(default=20, gt=0, le=200)


class RunOut(BaseModel):
    id: UUID
    goal: str
    status: str
    max_cost_usd: float
    spent_usd: float
    reserved_usd: float
    input_tokens: int
    output_tokens: int
    max_steps: int
    attempts: int
    result: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, run: Run) -> "RunOut":
        return cls(
            id=run.id,
            goal=run.goal,
            status=run.status,
            max_cost_usd=run.max_cost_micros / 1e6,
            spent_usd=run.spent_micros / 1e6,
            reserved_usd=run.reserved_micros / 1e6,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            max_steps=run.max_steps,
            attempts=run.attempts,
            result=run.result,
            error=run.error,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )


class StepOut(BaseModel):
    seq: int
    kind: str
    status: str
    tool_name: str | None
    tool_use_id: str | None
    payload: dict[str, Any]
    cost_usd: float
    lease_epoch: int
    created_at: datetime

    @classmethod
    def of(cls, step: Step) -> "StepOut":
        return cls(
            seq=step.seq,
            kind=step.kind,
            status=step.status,
            tool_name=step.tool_name,
            tool_use_id=step.tool_use_id,
            payload=step.payload,
            cost_usd=step.cost_micros / 1e6,
            lease_epoch=step.lease_epoch,
            created_at=step.created_at,
        )


class ApprovalOut(BaseModel):
    id: UUID
    run_id: UUID
    tool_name: str
    tool_input: dict[str, Any]
    status: str
    decided_by: str | None
    reason: str | None
    created_at: datetime
    decided_at: datetime | None

    @classmethod
    def of(cls, approval: Approval) -> "ApprovalOut":
        return cls(
            id=approval.id,
            run_id=approval.run_id,
            tool_name=approval.tool_name,
            tool_input=approval.tool_input,
            status=approval.status,
            decided_by=approval.decided_by,
            reason=approval.reason,
            created_at=approval.created_at,
            decided_at=approval.decided_at,
        )


class DecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    decided_by: str = Field(min_length=1, max_length=200)
    reason: str | None = Field(default=None, max_length=1000)


# ---------------------------------------------------------------- app


def create_app(settings: Settings | None = None, pool: asyncpg.Pool | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = pool is None
        app.state.pool = pool or await create_pool(settings.database_url)
        try:
            await migrate(app.state.pool)
            yield
        finally:
            if owned:
                await app.state.pool.close()

    app = FastAPI(title="stepwise", version="0.1.0", lifespan=lifespan)

    def get_pool(request: Request) -> asyncpg.Pool:
        return request.app.state.pool

    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid X-API-Key")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        body = Envelope[None](success=False, error=str(exc.detail))
        return JSONResponse(body.model_dump(), status_code=exc.status_code)

    @app.get("/healthz")
    async def healthz(pool: asyncpg.Pool = Depends(get_pool)) -> Envelope[dict[str, str]]:
        await pool.fetchval("SELECT 1")
        return Envelope(data={"status": "ok"})

    authed = [Depends(require_api_key)]

    @app.post("/v1/runs", status_code=status.HTTP_201_CREATED, dependencies=authed)
    async def create_run(body: CreateRunRequest, pool: asyncpg.Pool = Depends(get_pool)) -> Envelope[RunOut]:
        run = await store.create_run(
            pool, goal=body.goal, max_cost_micros=round(body.max_cost_usd * 1e6), max_steps=body.max_steps
        )
        return Envelope(data=RunOut.of(run))

    @app.get("/v1/runs/{run_id}", dependencies=authed)
    async def get_run(run_id: UUID, pool: asyncpg.Pool = Depends(get_pool)) -> Envelope[RunOut]:
        run = await store.get_run(pool, run_id)
        if run is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
        return Envelope(data=RunOut.of(run))

    @app.get("/v1/runs/{run_id}/steps", dependencies=authed)
    async def list_steps(
        run_id: UUID,
        limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(default=0, ge=0),
        pool: asyncpg.Pool = Depends(get_pool),
    ) -> Envelope[list[StepOut]]:
        if await store.get_run(pool, run_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
        steps = await store.list_steps(pool, run_id, limit=limit, offset=offset)
        return Envelope(data=[StepOut.of(step) for step in steps])

    @app.post("/v1/runs/{run_id}/cancel", dependencies=authed)
    async def cancel_run(run_id: UUID, pool: asyncpg.Pool = Depends(get_pool)) -> Envelope[RunOut]:
        run = await store.cancel_run(pool, run_id)
        if run is None:
            existing = await store.get_run(pool, run_id)
            if existing is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "run not found")
            raise HTTPException(status.HTTP_409_CONFLICT, f"run already {existing.status}")
        return Envelope(data=RunOut.of(run))

    @app.get("/v1/approvals", dependencies=authed)
    async def list_approvals(
        approval_status: ApprovalStatus | None = Query(default=ApprovalStatus.PENDING, alias="status"),
        limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
        offset: int = Query(default=0, ge=0),
        pool: asyncpg.Pool = Depends(get_pool),
    ) -> Envelope[list[ApprovalOut]]:
        approvals = await store.list_approvals(pool, status=approval_status, limit=limit, offset=offset)
        return Envelope(data=[ApprovalOut.of(a) for a in approvals])

    @app.post("/v1/approvals/{approval_id}/decision", dependencies=authed)
    async def decide(
        approval_id: UUID, body: DecisionRequest, pool: asyncpg.Pool = Depends(get_pool)
    ) -> Envelope[ApprovalOut]:
        try:
            approval = await store.decide_approval(
                pool,
                approval_id,
                approve=body.decision == "approve",
                decided_by=body.decided_by,
                reason=body.reason,
            )
        except store.ApprovalConflictError:
            raise HTTPException(status.HTTP_409_CONFLICT, "approval already decided") from None
        if approval is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "approval not found")
        return Envelope(data=ApprovalOut.of(approval))

    return app


def __getattr__(name: str) -> Any:
    # `uvicorn stepwise.api:app` builds the app lazily so importing this module has no side effects.
    if name == "app":
        return create_app()
    raise AttributeError(name)
