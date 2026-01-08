# Try to use pydantic BaseSettings if available; otherwise fall back to a lightweight dataclass-based settings.
from typing import Optional
import os

try:
    from pydantic import BaseSettings, Field

    class Settings(BaseSettings):
        """Application settings loaded from environment variables (and .env)."""

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

except Exception:
    # Lightweight fallback for environments where pydantic-settings isn't installed
    from dataclasses import dataclass

    @dataclass
    class Settings:
        ENV: str = os.getenv("ENV", "production")
        LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        DB_USER: Optional[str] = os.getenv("DB_USER")
        DB_PASSWORD: Optional[str] = os.getenv("DB_PASSWORD")
        DB_HOST: Optional[str] = os.getenv("DB_HOST")
        DB_PORT: Optional[str] = os.getenv("DB_PORT")
        DB_NAME: Optional[str] = os.getenv("DB_NAME")
        DB_OPTIONS: Optional[str] = os.getenv("DB_OPTIONS")

        SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
        SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
        SUPABASE_BUCKET: Optional[str] = os.getenv("SUPABASE_BUCKET")
        USE_SUPABASE: bool = os.getenv("USE_SUPABASE", "false").lower() == "true"

        ADMIN_SECRET: Optional[str] = os.getenv("ADMIN_SECRET")

    settings = Settings()