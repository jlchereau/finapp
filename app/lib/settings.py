"""
Application-wide settings management.

This module uses pydantic-settings to manage configuration.
It allows loading settings from environment variables and .env files,
providing a centralized, type-safe configuration system.
"""

# import multiprocessing
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Defines the application's global settings.

    Settings are loaded from the following sources, in order of precedence:
    1. Environment variables.
    2. .env file in the project root.
    3. Default values defined in this class.
    """

    # A sensible default for I/O-bound tasks. Using cpu_count * 2.
    # This can be overridden by setting the variable in a .env file
    # or as an environment variable.
    # PROVIDER_CONCURRENCY_LIMIT: int = (multiprocessing.cpu_count() or 1) * 2
    # Flow Cache Settings (aiocache in-memory)
    FLOW_CACHE_ENABLED: bool = True
    FLOW_CACHE_TTL: int = 300  # 5 minutes default

    # Provider Cache Settings (file-based)
    PROVIDER_CACHE_ENABLED: bool = True

    # Logging Settings
    DEBUG_LEVEL: str = "debug"  # debug, info, warning, error

    @field_validator("DEBUG_LEVEL")
    @classmethod
    def validate_debug_level(cls, v: str) -> str:
        """Validate DEBUG_LEVEL is a supported log level (case-insensitive)."""
        valid_levels = {"debug", "info", "warning", "error"}
        if v.lower() not in valid_levels:
            raise ValueError(f"DEBUG_LEVEL must be one of: {', '.join(valid_levels)}")
        return v.lower()  # Store as lowercase for consistent comparison

    # This tells Pydantic to look for a .env file.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Create a single, importable instance of the settings for use across the app.
settings = Settings()
