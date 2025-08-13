"""
Unit tests for the headers module.
Tests user agent utilities and HTTP headers functionality.
"""

import pytest

from app.models.headers import (
    USER_AGENTS,
    get_random_user_agent,
    get_default_headers,
)


class TestUserAgents:
    """Test cases for user agent functionality."""

    def test_user_agents_list_not_empty(self):
        """Test that USER_AGENTS list is not empty."""
        assert len(USER_AGENTS) > 0

    def test_user_agents_are_strings(self):
        """Test that all user agents are strings."""
        for user_agent in USER_AGENTS:
            assert isinstance(user_agent, str)
            assert len(user_agent) > 0

    def test_user_agents_contain_browser_info(self):
        """Test that user agents contain expected browser information."""
        # Check that we have various browsers represented
        has_chrome = any("Chrome" in ua for ua in USER_AGENTS)
        has_firefox = any("Firefox" in ua for ua in USER_AGENTS)
        has_safari = any("Safari" in ua for ua in USER_AGENTS)

        assert has_chrome, "Should have Chrome user agents"
        assert has_firefox, "Should have Firefox user agents"
        assert has_safari, "Should have Safari user agents"

    def test_get_random_user_agent_returns_valid_agent(self):
        """Test that get_random_user_agent returns a valid user agent."""
        user_agent = get_random_user_agent()

        assert isinstance(user_agent, str)
        assert len(user_agent) > 0
        assert user_agent in USER_AGENTS

    def test_get_random_user_agent_varies(self):
        """Test that get_random_user_agent returns different values."""
        # Generate multiple user agents and check for variation
        user_agents = [get_random_user_agent() for _ in range(20)]
        unique_agents = set(user_agents)

        # With 10 different user agents, we should see some variation
        # (though randomness means this could occasionally fail)
        assert len(unique_agents) > 1, "Should see variation in user agents"


class TestDefaultHeaders:
    """Test cases for default headers functionality."""

    def test_get_default_headers_structure(self):
        """Test that get_default_headers returns proper structure."""
        headers = get_default_headers()

        assert isinstance(headers, dict)
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert "Accept-Language" in headers
        assert "Accept-Encoding" in headers

    def test_get_default_headers_user_agent_valid(self):
        """Test that User-Agent in default headers is valid."""
        headers = get_default_headers()
        user_agent = headers["User-Agent"]

        assert isinstance(user_agent, str)
        assert len(user_agent) > 0
        assert user_agent in USER_AGENTS

    def test_get_default_headers_values(self):
        """Test that default headers have expected values."""
        headers = get_default_headers()

        # Check specific header values
        assert headers["Accept"] == "application/json, text/plain, */*"
        assert headers["Accept-Language"] == "en-US,en;q=0.9"
        assert headers["Accept-Encoding"] == "gzip, deflate, br"
        assert headers["DNT"] == "1"
        assert headers["Connection"] == "keep-alive"
        assert headers["Upgrade-Insecure-Requests"] == "1"

    def test_get_default_headers_user_agent_varies(self):
        """Test that User-Agent varies between calls."""
        # Generate multiple header sets and check User-Agent variation
        header_sets = [get_default_headers() for _ in range(20)]
        user_agents = [headers["User-Agent"] for headers in header_sets]
        unique_agents = set(user_agents)

        # Should see some variation in user agents
        assert len(unique_agents) > 1, "Should see variation in User-Agent headers"

    def test_get_default_headers_immutable(self):
        """Test that modifying returned headers doesn't affect future calls."""
        headers1 = get_default_headers()
        original_user_agent = headers1["User-Agent"]

        # Modify the returned headers
        headers1["User-Agent"] = "Modified"
        headers1["Custom-Header"] = "Added"

        # Get new headers
        headers2 = get_default_headers()

        # Should not be affected by previous modifications
        assert headers2["User-Agent"] != "Modified"
        assert headers2["User-Agent"] in USER_AGENTS
        assert "Custom-Header" not in headers2
