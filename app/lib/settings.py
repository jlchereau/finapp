"""
Application-wide settings management.

This module uses pydantic-settings to manage configuration.
It allows loading settings from environment variables and .env files,
providing a centralized, type-safe configuration system.
"""

# import multiprocessing
from pathlib import Path
from pydantic import Field, field_validator
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

    # Storage Settings
    PROVIDER_CACHE_ROOT: str = "data"  # Base directory for all cache and data storage

    # API Keys
    FRED_API_KEY: str = Field(
        default="", description="FRED (Federal Reserve Economic Data) API key"
    )
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    PERPLEXITY_API_KEY: str = Field(default="", description="Perplexity API key")
    SERPER_API_KEY: str = Field(default="", description="Serper API key")

    # IBKR Settings
    IB_GATEWAY_CLIENT_ID: int = Field(
        default=1, description="Interactive Brokers Gateway client ID"
    )
    IB_GATEWAY_PORT: int = Field(
        default=4001,
        description="Interactive Brokers Gateway port (4001 for live, 4002 for paper)",
    )

    @field_validator("DEBUG_LEVEL")
    @classmethod
    def validate_debug_level(cls, v: str) -> str:
        """Validate DEBUG_LEVEL is a supported log level (case-insensitive)."""
        valid_levels = {"debug", "info", "warning", "error"}
        if v.lower() not in valid_levels:
            raise ValueError(f"DEBUG_LEVEL must be one of: {', '.join(valid_levels)}")
        return v.lower()  # Store as lowercase for consistent comparison

    @field_validator("PROVIDER_CACHE_ROOT")
    @classmethod
    def validate_provider_cache_root(cls, v: str) -> str:
        """Validate and normalize PROVIDER_CACHE_ROOT path."""
        if not v:
            raise ValueError("PROVIDER_CACHE_ROOT cannot be empty")

        # Convert to Path and resolve relative to project root if not absolute
        path = Path(v)
        if not path.is_absolute():
            # Find project root by looking for rxconfig.py
            current_path = Path(__file__).resolve()
            for parent in current_path.parents:
                if (parent / "rxconfig.py").exists():
                    path = parent / path
                    break
            else:
                # Fallback to relative to current file's parent
                path = current_path.parent.parent.parent / path

        return str(path)

    @field_validator("FRED_API_KEY")
    @classmethod
    def validate_fred_api_key(cls, v: str) -> str:
        """Validate FRED API key format (basic length check)."""
        if v and len(v) < 10:  # Basic sanity check - FRED keys are typically 32 chars
            raise ValueError("FRED_API_KEY appears to be invalid (too short)")
        return v

    # This tells Pydantic to look for a .env file.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Create a single, importable instance of the settings for use across the app.
settings = Settings()
