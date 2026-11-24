"""Canned model turns for deterministic runs."""

from typing import Any


def tool_turn(tool_use_id: str, name: str, tool_input: dict[str, Any], text: str = "") -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    content.append({"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input})
    return {"content": content, "stop_reason": "tool_use", "input_tokens": 1200, "output_tokens": 150}


def final_turn(text: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "input_tokens": 1500,
        "output_tokens": 120,
    }


def refund_script(*, order_id: str = "ord_1001", amount_cents: int = 2900, email: str = "asha@example.com") -> list:
    return [
        tool_turn("tu_lookup", "get_order", {"order_id": order_id}, text="Let me check the order."),
        tool_turn(
            "tu_refund",
            "issue_refund",
            {"order_id": order_id, "amount_cents": amount_cents, "reason": "item arrived damaged"},
        ),
        tool_turn(
            "tu_email",
            "send_customer_email",
            {"to": email, "subject": "Your refund", "body": f"We refunded {amount_cents} cents."},
        ),
        final_turn(f"Refunded {amount_cents} cents on {order_id} and emailed the customer."),
    ]
