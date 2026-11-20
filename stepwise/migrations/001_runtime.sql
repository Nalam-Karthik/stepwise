-- Core runtime tables. Money is stored as integer micro-dollars to keep budget math exact.

CREATE TABLE runs (
    id                uuid PRIMARY KEY,
    goal              text        NOT NULL,
    status            text        NOT NULL CHECK (status IN (
                          'queued', 'running', 'awaiting_approval',
                          'succeeded', 'failed', 'budget_exceeded', 'cancelled')),
    max_cost_micros   bigint      NOT NULL CHECK (max_cost_micros > 0),
    max_steps         int         NOT NULL CHECK (max_steps > 0),
    spent_micros      bigint      NOT NULL DEFAULT 0,
    -- Worst-case cost of an LLM call that is in flight. A crash leaves it here, and the next
    -- claim moves it into spent_micros, so a crash can never let a run overspend.
    reserved_micros   bigint      NOT NULL DEFAULT 0,
    input_tokens      bigint      NOT NULL DEFAULT 0,
    output_tokens     bigint      NOT NULL DEFAULT 0,
    lease_owner       text,
    lease_expires_at  timestamptz,
    -- Fencing token: bumped on every claim. Every journal write checks it.
    lease_epoch       bigint      NOT NULL DEFAULT 0,
    attempts          int         NOT NULL DEFAULT 0,
    available_at      timestamptz NOT NULL DEFAULT now(),
    result            text,
    error             text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX runs_claimable_idx ON runs (available_at) WHERE status IN ('queued', 'running');

-- Append-only journal. A run's conversation is rebuilt from it on every resume.
CREATE TABLE steps (
    run_id        uuid        NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    seq           int         NOT NULL,
    kind          text        NOT NULL CHECK (kind IN ('llm', 'tool')),
    status        text        NOT NULL CHECK (status IN ('started', 'completed')),
    tool_use_id   text,
    tool_name     text,
    payload       jsonb       NOT NULL,
    cost_micros   bigint      NOT NULL DEFAULT 0,
    lease_epoch   bigint      NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    completed_at  timestamptz,
    PRIMARY KEY (run_id, seq),
    UNIQUE (run_id, tool_use_id)
);

CREATE TABLE approvals (
    id           uuid PRIMARY KEY,
    run_id       uuid        NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    tool_use_id  text        NOT NULL,
    tool_name    text        NOT NULL,
    tool_input   jsonb       NOT NULL,
    status       text        NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
    decided_by   text,
    reason       text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    decided_at   timestamptz,
    UNIQUE (run_id, tool_use_id)
);

CREATE INDEX approvals_pending_idx ON approvals (created_at) WHERE status = 'pending';
