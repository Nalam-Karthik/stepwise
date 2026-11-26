# stepwise

A durable runtime for LLM agents. Workers can crash at any point and a run still finishes,
each tool side effect happens exactly once, risky actions wait for a human, and no run spends
past its cost budget.

Most agent frameworks keep the conversation in process memory. If the process dies after the
agent has refunded a customer but before it records that, a retry refunds them again. stepwise
treats an agent run the way a payments system treats a transaction: every step is journaled in
Postgres, every write is fenced by a lease epoch, and recovery replays the journal rather
than re-running the agent.

```
  POST /v1/runs ──▶ runs (queued)
                       │  claim: UPDATE … FOR UPDATE SKIP LOCKED, lease_epoch += 1
                       ▼
                 worker ── heartbeat every 10s ──▶ lease_expires_at
                       │
          ┌────────────┴──────────── loop ─────────────────────────┐
          │ steps = journal(run)                                   │
          │ action = plan_next(goal, steps)     # pure function    │
          │   CallLLM  → reserve budget → Claude → journal (fenced)│
          │   RunTool  → approval gate → execute → journal (fenced)│
          │   Finish / Fail                                        │
          └────────────────────────────────────────────────────────┘
```

## Guarantees and how they hold

| Guarantee | Mechanism |
|---|---|
| **A crashed run resumes where it stopped** | The journal (`steps`) is the only state. `plan_next()` is a pure function of the journal, so a new worker computes the same next action the dead one would have. The conversation, including thinking blocks, is rebuilt byte for byte from stored content. |
| **One worker per run** | Claims use `FOR UPDATE SKIP LOCKED` and bump `lease_epoch`. A worker that stops heartbeating loses the run when its lease expires. |
| **A zombie worker can't write** | Every journal write runs `fence()` first, in the same transaction: lock the run row and check the epoch. A worker paused past its lease (GC, partition) fails the fence, and its transaction, side effects included, rolls back. |
| **Exactly-once effects in our own database** | *Transactional* tools (e.g. `issue_refund`) run inside the transaction that journals their result. The effect and the journal entry commit together or not at all. |
| **Exactly-once effects in external systems** | *External* tools (e.g. `send_customer_email`) journal a `started` step, then call the provider with `Idempotency-Key: <run>:<tool_use_id>`. After a crash the call repeats with the same key and the provider returns the first result. |
| **Never overspend** | Before each model call, the worst-case cost (estimated prompt plus the full `max_tokens` output) is reserved against the budget. A crash leaves the reservation on the row, and the next claim charges it as spent, because that call may have been billed. The budget can be under-used, but not overspent. |
| **Humans gate risky actions** | A tool can require approval for some inputs (refunds over $50). The run parks as `awaiting_approval` and holds no worker. A decision through the API requeues it. A rejection goes back to the model as a tool error. |
| **Runaway loops stop** | `max_steps` caps model calls. A run that crashes `max_attempts` times in a row is failed, not retried forever. |

## Proven by fault injection

`tests/test_crash_recovery.py` starts a real worker process and kills it with `os._exit(137)`
at each dangerous point: no exception handlers run and the lease is not released. A second
process then finishes the run.

| Crash point | What the test asserts |
|---|---|
| `after_llm_call` (response received, not journaled) | Run succeeds; the lost call's reservation is charged; four model steps in the journal |
| `before_tool_commit` (refund written, transaction open) | Refund rolled back, then issued exactly once |
| `before_external_call` (`started` journaled, email not sent) | Exactly one email; not marked as a duplicate |
| `after_external_call` (email sent, result not journaled) | Exactly one email; the retry was de-duplicated by the provider |

To watch it happen:

```
$ make demo
== worker 1: crashes right after the email provider accepts the message
   exit code 137
== worker 2: claims the run with a new lease epoch and resumes from the journal
seq  kind  tool                  epoch  output
  5  llm                             1  tool_use
  6  tool  send_customer_email       2  {'message_id': 'msg_a744…', 'deduplicated': True}
  7  llm                             2  end_turn
emails delivered: 1   refunds issued: 1
```

## Running it

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
make install     # Python 3.13 venv in .venv with dev dependencies
make db          # Postgres 16 on localhost:5546
make test        # 45 tests, about 12s, 92% coverage
make demo        # the crash demo above; no API key needed
```

Against the real model (Claude Opus 5.5 by default), set `ANTHROPIC_API_KEY` and run:

```bash
make api                         # terminal 1, port 8080
make worker                      # terminal 2; start more workers as you like
curl -s localhost:8080/v1/runs -H 'X-API-Key: dev-key' -H 'content-type: application/json' \
  -d '{"goal": "Order ord_1002 arrived broken. Refund $100 and let the customer know.", "max_cost_usd": 1}'
curl -s localhost:8080/v1/approvals -H 'X-API-Key: dev-key'      # the $100 refund is waiting
curl -s localhost:8080/v1/approvals/<id>/decision -H 'X-API-Key: dev-key' \
  -H 'content-type: application/json' -d '{"decision": "approve", "decided_by": "you@example.com"}'
```

## API

All endpoints except `/healthz` need `X-API-Key`. Responses use `{success, data, error}`.

| Method | Path | |
|---|---|---|
| `POST` | `/v1/runs` | `{goal, max_cost_usd, max_steps}` → queued run |
| `GET` | `/v1/runs/{id}` | status, spend, tokens, result |
| `GET` | `/v1/runs/{id}/steps` | the journal, paginated |
| `POST` | `/v1/runs/{id}/cancel` | the worker holding the run stops at its next fenced write |
| `GET` | `/v1/approvals?status=pending` | approval queue |
| `POST` | `/v1/approvals/{id}/decision` | `{decision: approve\|reject, decided_by, reason}`; 409 if already decided |

## Layout

```
stepwise/
  planner.py    pure: journal → messages, journal → next action
  engine.py     executes one action at a time under the lease
  store.py      all SQL: claim, heartbeat, fence, journal, budget, approvals
  tools.py      tool contract + demo tools (get_order, issue_refund, send_customer_email)
  llm.py        Claude adapter (server-side refusal fallback on), scripted model, pricing
  worker.py     claim loop, heartbeat task, retry with exponential backoff
  api.py        FastAPI
  faults.py     crash injection points
  migrations/   001 runtime tables, 002 demo support domain
```

## Limitations and next steps

- **Live-model coverage.** The Claude adapter is tested against a fake SDK client (request
  shape, error-to-retry mapping, fallback billing). End-to-end runs in the test suite use the
  scripted model.
- **Cost estimate.** The reservation estimates prompt tokens as chars/3. An unusual prompt
  could tokenize denser than that, letting one call exceed its reservation; the next call's
  budget check then stops the run. Switching to `count_tokens` would make the bound exact.
- **Graceful shutdown.** SIGTERM stops a worker between runs, not mid-run; a run in progress
  is picked up by another worker once its lease expires.
- **Prompt caching** is off, so cost math uses only the input and output rates.
- **Next:** OpenTelemetry spans per step, a `LISTEN/NOTIFY` wake-up to replace polling, and
  per-tenant budgets.
