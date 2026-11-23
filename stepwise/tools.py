"""Tool contract and the bundled support-agent tools.

Two kinds of side effect, two ways to make them happen exactly once:

* Transactional tools write to this Postgres. The engine runs them inside the same transaction
  that journals their result, so the effect and the journal entry commit or roll back together.
* External tools call a system we don't control. The engine journals a `started` step first,
  then calls the tool with a stable idempotency key (`<run id>:<tool_use id>`). After a crash the
  call is repeated with the same key and the provider returns the original result.
"""

import hashlib
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from stepwise.config import Settings


class ToolError(Exception):
    """An expected failure the model should see and react to (bad order id, over-refund...)."""


@dataclass(frozen=True)
class ToolContext:
    run_id: UUID
    tool_use_id: str
    settings: Settings

    @property
    def idempotency_key(self) -> str:
        return f"{self.run_id}:{self.tool_use_id}"


class Tool(Protocol):
    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]
    transactional: ClassVar[bool]

    def needs_approval(self, args: BaseModel, settings: Settings) -> bool: ...

    async def execute(self, conn: asyncpg.Connection, args: BaseModel, ctx: ToolContext) -> dict[str, Any]: ...


def tool_definition(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        # Not `strict`: the schemas use length/range constraints strict mode may not accept, and
        # every input is validated by the args model before it runs anyway.
        "input_schema": tool.args_model.model_json_schema(),
    }


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def definitions(self) -> list[dict[str, Any]]:
        return [tool_definition(tool) for tool in self._tools.values()]


def parse_args(tool: Tool, raw: dict[str, Any]) -> BaseModel:
    try:
        return tool.args_model.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors())
        raise ToolError(f"invalid arguments: {problems}") from exc


# ---------------------------------------------------------------- bundled tools


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetOrderArgs(_Args):
    order_id: str = Field(min_length=1, max_length=64)


class IssueRefundArgs(_Args):
    order_id: str = Field(min_length=1, max_length=64)
    amount_cents: int = Field(gt=0, le=10_000_000)
    reason: str = Field(min_length=3, max_length=500)


class SendEmailArgs(_Args):
    to: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=254)
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=5000)


class GetOrder:
    name = "get_order"
    description = "Look up an order: customer email, amount paid, amount already refunded and status."
    args_model = GetOrderArgs
    transactional = True

    def needs_approval(self, args: BaseModel, settings: Settings) -> bool:
        return False

    async def execute(self, conn: asyncpg.Connection, args: GetOrderArgs, ctx: ToolContext) -> dict[str, Any]:
        row = await conn.fetchrow("SELECT * FROM orders WHERE id = $1", args.order_id)
        if row is None:
            raise ToolError(f"order {args.order_id} not found")
        return {
            "order_id": row["id"],
            "customer_email": row["customer_email"],
            "amount_cents": row["amount_cents"],
            "refunded_cents": row["refunded_cents"],
            "status": row["status"],
        }


class IssueRefund:
    name = "issue_refund"
    description = (
        "Refund part or all of a delivered order. Amounts are in cents and cannot exceed what is "
        "left unrefunded. Large refunds wait for a human to approve them."
    )
    args_model = IssueRefundArgs
    transactional = True

    def needs_approval(self, args: IssueRefundArgs, settings: Settings) -> bool:
        return args.amount_cents > settings.refund_approval_threshold_cents

    async def execute(self, conn: asyncpg.Connection, args: IssueRefundArgs, ctx: ToolContext) -> dict[str, Any]:
        order = await conn.fetchrow("SELECT * FROM orders WHERE id = $1 FOR UPDATE", args.order_id)
        if order is None:
            raise ToolError(f"order {args.order_id} not found")
        if order["status"] != "delivered":
            raise ToolError(f"order {args.order_id} is {order['status']}; only delivered orders can be refunded")
        remaining = order["amount_cents"] - order["refunded_cents"]
        if args.amount_cents > remaining:
            raise ToolError(f"refund of {args.amount_cents} exceeds the {remaining} cents left on the order")

        refund_id = uuid4()
        await conn.execute(
            "INSERT INTO refunds (id, order_id, amount_cents, reason, idempotency_key) VALUES ($1, $2, $3, $4, $5)",
            refund_id,
            args.order_id,
            args.amount_cents,
            args.reason,
            ctx.idempotency_key,
        )
        await conn.execute(
            "UPDATE orders SET refunded_cents = refunded_cents + $2 WHERE id = $1", args.order_id, args.amount_cents
        )
        return {
            "refund_id": str(refund_id),
            "amount_cents": args.amount_cents,
            "remaining_cents": remaining - args.amount_cents,
        }


class SendCustomerEmail:
    """Stands in for an email provider API that dedupes on an Idempotency-Key header.

    It writes on its own connection, committed independently of the journal, which is exactly how
    a real external call behaves: the effect can land even if the worker dies before journaling it.
    """

    name = "send_customer_email"
    description = "Email a customer. Use it to confirm a refund or explain why one was declined."
    args_model = SendEmailArgs
    transactional = False

    def needs_approval(self, args: BaseModel, settings: Settings) -> bool:
        return False

    async def execute(self, conn: asyncpg.Connection, args: SendEmailArgs, ctx: ToolContext) -> dict[str, Any]:
        message_id = "msg_" + hashlib.sha256(ctx.idempotency_key.encode()).hexdigest()[:16]
        inserted = await conn.fetchval(
            "INSERT INTO email_deliveries (idempotency_key, recipient, subject, body, provider_message_id)"
            " VALUES ($1, $2, $3, $4, $5) ON CONFLICT (idempotency_key) DO NOTHING RETURNING provider_message_id",
            ctx.idempotency_key,
            args.to,
            args.subject,
            args.body,
            message_id,
        )
        if inserted is None:
            existing = await conn.fetchval(
                "SELECT provider_message_id FROM email_deliveries WHERE idempotency_key = $1", ctx.idempotency_key
            )
            return {"message_id": existing, "deduplicated": True}
        return {"message_id": inserted, "deduplicated": False}


def default_registry() -> ToolRegistry:
    return ToolRegistry([GetOrder(), IssueRefund(), SendCustomerEmail()])
