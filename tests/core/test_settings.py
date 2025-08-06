"""
Unit tests for the application's core configuration.
"""

import os
from unittest.mock import patch
import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict


# This is a simplified version of the actual Settings class for testing
class TestSettings(BaseSettings):
    """Test settings class."""

    PROVIDER_CONCURRENCY_LIMIT: int = 1
    model_config = SettingsConfigDict(
        env_file=".env.test", env_file_encoding="utf-8", extra="ignore"
    )


@pytest.fixture(scope="function")
def test_env_file():
    """Fixture to create and clean up a temporary .env.test file."""
    env_content = "PROVIDER_CONCURRENCY_LIMIT=50"
    env_file = ".env.test"
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(env_content)
    yield
    os.remove(env_file)


@pytest.fixture(scope="function")
def mock_cpu_count():
    """Fixture to mock multiprocessing.cpu_count()."""
    with patch("multiprocessing.cpu_count", return_value=4) as mock:
        yield mock


def test_settings_default_value(mock_cpu_count):
    """Test that the default value is calculated correctly."""
    # We need to reload the module to re-evaluate the default value
    from app.core import settings
    import importlib

    importlib.reload(settings)
    expected_default = (mock_cpu_count.return_value or 1) * 2
    assert settings.settings.PROVIDER_CONCURRENCY_LIMIT == expected_default


def test_settings_from_env_file(test_env_file):
    """Test that settings are correctly loaded from an .env file."""
    settings = TestSettings()
    assert settings.PROVIDER_CONCURRENCY_LIMIT == 50


@patch.dict(os.environ, {"PROVIDER_CONCURRENCY_LIMIT": "100"})
def test_settings_from_environment_variable():
    """Test that settings are correctly loaded from environment variables."""
    settings = TestSettings()
    assert settings.PROVIDER_CONCURRENCY_LIMIT == 100


@patch.dict(os.environ, {"PROVIDER_CONCURRENCY_LIMIT": "100"})
def test_settings_priority(test_env_file):
    """Test that environment variables have priority over .env files."""
    settings = TestSettings()
    # The value from the environment variable (100) should be used,
    # not the .env file (50)
    assert settings.PROVIDER_CONCURRENCY_LIMIT == 100
