from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

import asyncpg


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"


class StepKind(StrEnum):
    LLM = "llm"
    TOOL = "tool"


class StepStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Run:
    id: UUID
    goal: str
    status: RunStatus
    max_cost_micros: int
    max_steps: int
    spent_micros: int
    reserved_micros: int
    input_tokens: int
    output_tokens: int
    lease_owner: str | None
    lease_epoch: int
    attempts: int
    result: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> "Run":
        return cls(
            id=row["id"],
            goal=row["goal"],
            status=RunStatus(row["status"]),
            max_cost_micros=row["max_cost_micros"],
            max_steps=row["max_steps"],
            spent_micros=row["spent_micros"],
            reserved_micros=row["reserved_micros"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            lease_owner=row["lease_owner"],
            lease_epoch=row["lease_epoch"],
            attempts=row["attempts"],
            result=row["result"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class Step:
    run_id: UUID
    seq: int
    kind: StepKind
    status: StepStatus
    tool_use_id: str | None
    tool_name: str | None
    payload: dict[str, Any]
    cost_micros: int
    lease_epoch: int
    created_at: datetime

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> "Step":
        return cls(
            run_id=row["run_id"],
            seq=row["seq"],
            kind=StepKind(row["kind"]),
            status=StepStatus(row["status"]),
            tool_use_id=row["tool_use_id"],
            tool_name=row["tool_name"],
            payload=row["payload"],
            cost_micros=row["cost_micros"],
            lease_epoch=row["lease_epoch"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class Approval:
    id: UUID
    run_id: UUID
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    status: ApprovalStatus
    decided_by: str | None
    reason: str | None
    created_at: datetime
    decided_at: datetime | None

    @classmethod
    def from_row(cls, row: asyncpg.Record) -> "Approval":
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            tool_use_id=row["tool_use_id"],
            tool_name=row["tool_name"],
            tool_input=row["tool_input"],
            status=ApprovalStatus(row["status"]),
            decided_by=row["decided_by"],
            reason=row["reason"],
            created_at=row["created_at"],
            decided_at=row["decided_at"],
        )


@dataclass(frozen=True)
class Lease:
    run_id: UUID
    epoch: int
    owner: str
