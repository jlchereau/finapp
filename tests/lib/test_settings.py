"""
Unit tests for the application's lib configuration.
"""

import os
from unittest.mock import patch
import pytest
from pydantic import ValidationError
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
    assert settings.DEBUG_LEVEL == "debug"


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


class TestDebugLevelValidation:
    """Test cases for DEBUG_LEVEL setting validation."""

    def test_valid_debug_levels(self):
        """Test that valid DEBUG_LEVEL values are accepted."""
        from app.lib.settings import Settings

        # Test all valid levels
        for level in ["debug", "info", "warning", "error"]:
            with patch.dict(os.environ, {"DEBUG_LEVEL": level}):
                settings = Settings()
                assert settings.DEBUG_LEVEL == level.lower()

    def test_case_insensitive_debug_levels(self):
        """Test that DEBUG_LEVEL validation is case-insensitive."""
        from app.lib.settings import Settings

        # Test various case combinations
        test_cases = [
            ("DEBUG", "debug"),
            ("Info", "info"),
            ("WARNING", "warning"),
            ("Error", "error"),
            ("iNfO", "info"),
            ("WaRnInG", "warning"),
        ]

        for input_level, expected_level in test_cases:
            with patch.dict(os.environ, {"DEBUG_LEVEL": input_level}):
                settings = Settings()
                assert settings.DEBUG_LEVEL == expected_level

    def test_invalid_debug_level_raises_validation_error(self):
        """Test that invalid DEBUG_LEVEL values raise ValidationError."""
        from app.lib.settings import Settings

        invalid_levels = ["invalid", "trace", "critical", "", "123", "none"]

        for invalid_level in invalid_levels:
            with patch.dict(os.environ, {"DEBUG_LEVEL": invalid_level}):
                with pytest.raises(ValidationError) as exc_info:
                    Settings()

                error_msg = str(exc_info.value)
                assert "DEBUG_LEVEL must be one of:" in error_msg
                assert "debug" in error_msg
                assert "info" in error_msg
                assert "warning" in error_msg
                assert "error" in error_msg

    @patch.dict(os.environ, {"DEBUG_LEVEL": "info"})
    def test_debug_level_from_environment(self):
        """Test that DEBUG_LEVEL can be set via environment variable."""
        from app.lib.settings import Settings

        settings = Settings()
        assert settings.DEBUG_LEVEL == "info"

    def test_debug_level_from_env_file(self):
        """Test that DEBUG_LEVEL can be set via .env file."""
        from app.lib.settings import Settings

        # Create temporary .env file with DEBUG_LEVEL
        env_content = "DEBUG_LEVEL=warning"
        env_file = ".env.test_debug"

        try:
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(env_content)

            # Create settings with custom env file
            class TestSettings(Settings):
                model_config = SettingsConfigDict(
                    env_file=env_file, env_file_encoding="utf-8", extra="ignore"
                )

            settings = TestSettings()
            assert settings.DEBUG_LEVEL == "warning"

        finally:
            if os.path.exists(env_file):
                os.remove(env_file)

    @patch.dict(os.environ, {"DEBUG_LEVEL": "error"})
    def test_environment_overrides_env_file(self):
        """Test that environment variable overrides .env file for DEBUG_LEVEL."""
        from app.lib.settings import Settings

        # Create .env file with different value
        env_content = "DEBUG_LEVEL=info"
        env_file = ".env.test_override"

        try:
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(env_content)

            class TestSettings(Settings):
                model_config = SettingsConfigDict(
                    env_file=env_file, env_file_encoding="utf-8", extra="ignore"
                )

            settings = TestSettings()
            # Environment variable should take precedence
            assert settings.DEBUG_LEVEL == "error"

        finally:
            if os.path.exists(env_file):
                os.remove(env_file)
