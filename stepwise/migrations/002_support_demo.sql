-- Demo domain for the bundled tools: a support agent that can look up orders, issue refunds
-- and email customers. Refunds live in this database (so they commit atomically with the
-- journal); email_deliveries stands in for an external provider that dedupes on an
-- Idempotency-Key, the way Stripe or SES do.

CREATE TABLE orders (
    id              text PRIMARY KEY,
    customer_email  text NOT NULL,
    amount_cents    int  NOT NULL CHECK (amount_cents > 0),
    refunded_cents  int  NOT NULL DEFAULT 0 CHECK (refunded_cents >= 0),
    status          text NOT NULL CHECK (status IN ('delivered', 'cancelled')),
    CHECK (refunded_cents <= amount_cents)
);

CREATE TABLE refunds (
    id               uuid PRIMARY KEY,
    order_id         text        NOT NULL REFERENCES orders (id),
    amount_cents     int         NOT NULL CHECK (amount_cents > 0),
    reason           text        NOT NULL,
    idempotency_key  text        NOT NULL UNIQUE,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE email_deliveries (
    idempotency_key      text PRIMARY KEY,
    recipient            text        NOT NULL,
    subject              text        NOT NULL,
    body                 text        NOT NULL,
    provider_message_id  text        NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT now()
);

INSERT INTO orders (id, customer_email, amount_cents, status) VALUES
    ('ord_1001', 'asha@example.com', 12900, 'delivered'),
    ('ord_1002', 'ravi@example.com', 249900, 'delivered'),
    ('ord_1003', 'meera@example.com', 4500, 'cancelled');
