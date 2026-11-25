from collections.abc import AsyncIterator

import httpx
import pytest

from stepwise.api import create_app
from tests.scripts import refund_script

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture
async def client(pool, settings) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings.model_copy(update={"api_key": "test-key"}), pool=pool)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def test_requests_without_the_api_key_are_rejected(client):
    response = await client.post("/v1/runs", json={"goal": "x", "max_cost_usd": 1})
    assert response.status_code == 401
    assert response.json() == {"success": False, "data": None, "error": "missing or invalid X-API-Key"}


async def test_healthz_needs_no_key(client):
    assert (await client.get("/healthz")).json()["data"] == {"status": "ok"}


async def test_create_and_follow_a_run_through_approval(client, make_worker):
    created = await client.post(
        "/v1/runs", headers=HEADERS, json={"goal": "Refund $100 on ord_1002", "max_cost_usd": 2, "max_steps": 10}
    )
    assert created.status_code == 201
    run_id = created.json()["data"]["id"]
    worker = make_worker(refund_script(order_id="ord_1002", amount_cents=10_000, email="ravi@example.com"))
    await worker.run_once()

    run = (await client.get(f"/v1/runs/{run_id}", headers=HEADERS)).json()["data"]
    assert run["status"] == "awaiting_approval"
    [approval] = (await client.get("/v1/approvals", headers=HEADERS)).json()["data"]

    decision = {"decision": "approve", "decided_by": "ops@example.com"}
    ok = await client.post(f"/v1/approvals/{approval['id']}/decision", headers=HEADERS, json=decision)
    assert ok.status_code == 200 and ok.json()["data"]["status"] == "approved"
    again = await client.post(f"/v1/approvals/{approval['id']}/decision", headers=HEADERS, json=decision)
    assert again.status_code == 409

    await worker.run_once()
    run = (await client.get(f"/v1/runs/{run_id}", headers=HEADERS)).json()["data"]
    assert run["status"] == "succeeded" and run["spent_usd"] > 0
    steps = (await client.get(f"/v1/runs/{run_id}/steps", headers=HEADERS, params={"limit": 3})).json()["data"]
    assert [s["seq"] for s in steps] == [1, 2, 3]


async def test_cancel_and_not_found(client):
    run_id = (await client.post("/v1/runs", headers=HEADERS, json={"goal": "x", "max_cost_usd": 1})).json()["data"][
        "id"
    ]
    assert (await client.post(f"/v1/runs/{run_id}/cancel", headers=HEADERS)).json()["data"]["status"] == "cancelled"
    assert (await client.post(f"/v1/runs/{run_id}/cancel", headers=HEADERS)).status_code == 409
    missing = "00000000-0000-0000-0000-000000000000"
    assert (await client.get(f"/v1/runs/{missing}", headers=HEADERS)).status_code == 404
    assert (await client.get(f"/v1/runs/{missing}/steps", headers=HEADERS)).status_code == 404


async def test_request_validation(client):
    bad = await client.post("/v1/runs", headers=HEADERS, json={"goal": "", "max_cost_usd": -1})
    assert bad.status_code == 422
