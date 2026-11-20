import json
from pathlib import Path

import asyncpg

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_MIGRATION_LOCK_ID = 7_431_001


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def create_pool(dsn: str, *, min_size: int = 1, max_size: int = 10) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, init=_init_connection)


async def migrate(pool: asyncpg.Pool) -> list[str]:
    """Apply pending SQL migrations in filename order. Safe to call from many processes at once."""
    applied_now: list[str] = []
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock($1)", _MIGRATION_LOCK_ID)
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            applied = {row["version"] for row in await conn.fetch("SELECT version FROM schema_migrations")}
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if path.stem in applied:
                    continue
                async with conn.transaction():
                    await conn.execute(path.read_text())
                    await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", path.stem)
                applied_now.append(path.stem)
        finally:
            await conn.execute("SELECT pg_advisory_unlock($1)", _MIGRATION_LOCK_ID)
    return applied_now
