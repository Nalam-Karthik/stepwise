from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STEPWISE_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://stepwise:stepwise@localhost:5546/stepwise"
    api_key: str = "dev-key"

    # "anthropic" or "scripted:<path to JSON script>"
    llm: str = "anthropic"
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_output_tokens: int = 16000

    lease_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    poll_seconds: float = 0.5
    max_attempts: int = 5
    retry_backoff_seconds: float = 2.0

    refund_approval_threshold_cents: int = 5000

    # Fault injection for crash tests, e.g. "after_external_call". Never set in production.
    fault: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
