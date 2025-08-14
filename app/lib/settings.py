"""
Application-wide settings management.

This module uses pydantic-settings to manage configuration.
It allows loading settings from environment variables and .env files,
providing a centralized, type-safe configuration system.
"""

# import multiprocessing
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
    # Enable or disable caching for providers (_fetch_data methods)
    CACHE_ENABLED: bool = True

    # This tells Pydantic to look for a .env file.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Create a single, importable instance of the settings for use across the app.
settings = Settings()
