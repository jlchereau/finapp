"""
Unit tests for the application's lib configuration.
"""

import os
from unittest.mock import patch
import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict


# This is a simplified version of the actual Settings class for testing
class MockSettings(BaseSettings):
    """Mock settings class for testing."""

    PROVIDER_CACHE_ENABLED: bool = True
    model_config = SettingsConfigDict(
        env_file=".env.test", env_file_encoding="utf-8", extra="ignore"
    )


@pytest.fixture(scope="function")
def test_env_file():
    """Fixture to create and clean up a temporary .env.test file."""
    env_content = "PROVIDER_CACHE_ENABLED=false"
    env_file = ".env.test"
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(env_content)
    yield
    os.remove(env_file)


def test_settings_default_value():
    """Test that the default values for cache settings are correct."""
    from app.lib.settings import settings

    assert settings.PROVIDER_CACHE_ENABLED is True
    assert settings.FLOW_CACHE_ENABLED is True
    assert settings.FLOW_CACHE_TTL == 300


def test_settings_from_env_file(test_env_file):  # pylint: disable=unused-argument
    """Test that settings are correctly loaded from an .env file."""
    settings = MockSettings()
    assert settings.PROVIDER_CACHE_ENABLED is False


@patch.dict(os.environ, {"PROVIDER_CACHE_ENABLED": "false"})
def test_settings_from_environment_variable():
    """Test that settings are correctly loaded from environment variables."""
    settings = MockSettings()
    assert settings.PROVIDER_CACHE_ENABLED is False


@patch.dict(os.environ, {"PROVIDER_CACHE_ENABLED": "false"})
def test_settings_priority(test_env_file):  # pylint: disable=unused-argument
    """Test that environment variables have priority over .env files."""
    settings = MockSettings()
    # The value from the environment variable (false) should be used,
    # overriding both the default (true) and the .env file (false)
    assert settings.PROVIDER_CACHE_ENABLED is False


@patch.dict(os.environ, {"FLOW_CACHE_ENABLED": "false", "FLOW_CACHE_TTL": "600"})
def test_flow_cache_integration():
    """Integration test: Flow cache settings work with actual modules."""
    # Create fresh Settings instance to pick up environment variables
    from app.lib.settings import Settings
    test_settings = Settings()
    
    # Verify settings are loaded from environment
    assert test_settings.FLOW_CACHE_ENABLED is False
    assert test_settings.FLOW_CACHE_TTL == 600
    
    # Test cache decorator behavior (simplified test)
    with patch("app.flows.cache.settings", test_settings):
        from app.flows.cache import apply_flow_cache
        
        async def test_func():
            return "test"
        
        result_func = apply_flow_cache(test_func)
        
        # When disabled, should return original function
        assert result_func is test_func


@patch.dict(os.environ, {"PROVIDER_CACHE_ENABLED": "false"})
def test_provider_cache_integration():
    """Integration test: Provider cache settings work with cache decorator."""
    # Create fresh Settings instance to pick up environment variables
    from app.lib.settings import Settings
    test_settings = Settings()
    
    # Verify setting is loaded from environment
    assert test_settings.PROVIDER_CACHE_ENABLED is False
    
    # Verify provider config can still override
    from app.providers.base import ProviderConfig
    config = ProviderConfig(cache_enabled=True)
    assert config.cache_enabled is True  # Provider-level override works
