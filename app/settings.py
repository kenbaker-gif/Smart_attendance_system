from pydantic import BaseSettings, Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables (and .env).

    Use `from app.settings import settings` to access validated configuration.
    """

    ENV: str = Field("production", description="Environment: development|staging|production")
    LOG_LEVEL: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")

    DB_USER: Optional[str]
    DB_PASSWORD: Optional[str]
    DB_HOST: Optional[str]
    DB_PORT: Optional[str]
    DB_NAME: Optional[str]
    DB_OPTIONS: Optional[str] = None

    SUPABASE_URL: Optional[str]
    SUPABASE_KEY: Optional[str]
    SUPABASE_BUCKET: Optional[str]
    USE_SUPABASE: bool = False

    ADMIN_SECRET: Optional[str]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()