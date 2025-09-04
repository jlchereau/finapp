"""
Unit tests for the web page scrapper module.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.lib.scrapper import WebPageScrapper


# Sample HTML for testing
SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Page</title>
</head>
<body>
    <div class="content">
        <h1>Main Title</h1>
        <p class="description">This is a test page</p>
        <p class="description">Second paragraph</p>
    </div>
    <nav>
        <a href="/home" class="nav-link">Home</a>
        <a href="/about" class="nav-link">About</a>
        <a href="https://example.com" class="external-link">External</a>
        <span class="no-href">No Link</span>
    </nav>
    <ul class="list">
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
</body>
</html>
"""


class TestWebPageScrapper:
    """Test cases for WebPageScrapper class."""

    def test_initialization_default_timeout(self):
        """Test scrapper initialization with default timeout."""
        scrapper = WebPageScrapper("https://example.com")

        assert scrapper.url == "https://example.com"
        assert scrapper.timeout == 30.0
        assert scrapper._soup is None
        assert scrapper._client is None

    def test_initialization_custom_timeout(self):
        """Test scrapper initialization with custom timeout."""
        scrapper = WebPageScrapper("https://example.com", timeout=60.0)

        assert scrapper.url == "https://example.com"
        assert scrapper.timeout == 60.0
        assert scrapper._soup is None
        assert scrapper._client is None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_private_fetch_page_success(self, mock_client_class):
        """Test successful page fetching and parsing."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test fetch
        scrapper = WebPageScrapper("https://example.com")
        await scrapper._fetch_page()

        # Verify HTTP client was called correctly
        mock_client_class.assert_called_once_with(timeout=30.0)
        mock_client.get.assert_called_once_with("https://example.com")
        mock_response.raise_for_status.assert_called_once()

        # Verify BeautifulSoup object was created
        assert scrapper._soup is not None
        assert scrapper._client == mock_client

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_private_fetch_page_only_once(self, mock_client_class):
        """Test that page is only fetched once (lazy loading)."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test multiple fetch calls
        scrapper = WebPageScrapper("https://example.com")
        await scrapper._fetch_page()
        await scrapper._fetch_page()  # Second call should not make HTTP request

        # Verify HTTP client was called only once
        mock_client_class.assert_called_once()
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_extract_text_single_element(self, mock_client_class):
        """Test extracting text from a single element."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test text extraction
        scrapper = WebPageScrapper("https://example.com")
        result = await scrapper.extract_text("h1")

        assert result == ["Main Title"]

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_extract_text_multiple_elements(self, mock_client_class):
        """Test extracting text from multiple elements."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test text extraction
        scrapper = WebPageScrapper("https://example.com")
        result = await scrapper.extract_text(".description")

        assert result == ["This is a test page", "Second paragraph"]

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_extract_text_list_items(self, mock_client_class):
        """Test extracting text from list items."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test text extraction
        scrapper = WebPageScrapper("https://example.com")
        result = await scrapper.extract_text("li")

        assert result == ["Item 1", "Item 2", "Item 3"]

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_extract_text_no_matches(self, mock_client_class):
        """Test extracting text when selector matches no elements."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test text extraction
        scrapper = WebPageScrapper("https://example.com")
        result = await scrapper.extract_text(".nonexistent")

        assert result == []

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_extract_href_links(self, mock_client_class):
        """Test extracting href attributes from links."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test href extraction
        scrapper = WebPageScrapper("https://example.com")
        result = await scrapper.extract_href(".nav-link")

        assert result == ["/home", "/about"]

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_extract_href_all_links(self, mock_client_class):
        """Test extracting href attributes from all anchor tags."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test href extraction
        scrapper = WebPageScrapper("https://example.com")
        result = await scrapper.extract_href("a")

        assert result == ["/home", "/about", "https://example.com"]

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_extract_href_elements_without_href(self, mock_client_class):
        """Test extracting href from elements that don't have href attributes."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test href extraction from span (which has no href)
        scrapper = WebPageScrapper("https://example.com")
        result = await scrapper.extract_href(".no-href")

        assert result == []

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_extract_href_no_matches(self, mock_client_class):
        """Test extracting href when selector matches no elements."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test href extraction
        scrapper = WebPageScrapper("https://example.com")
        result = await scrapper.extract_href(".nonexistent")

        assert result == []

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_lazy_loading_on_extract_text(self, mock_client_class):
        """Test that page is fetched automatically on first extract_text call."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test that fetch is called automatically
        scrapper = WebPageScrapper("https://example.com")
        assert scrapper._soup is None  # Not fetched yet

        result = await scrapper.extract_text("h1")

        assert scrapper._soup is not None  # Fetched automatically
        assert result == ["Main Title"]

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_lazy_loading_on_extract_href(self, mock_client_class):
        """Test that page is fetched automatically on first extract_href call."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test that fetch is called automatically
        scrapper = WebPageScrapper("https://example.com")
        assert scrapper._soup is None  # Not fetched yet

        result = await scrapper.extract_href("a")

        assert scrapper._soup is not None  # Fetched automatically
        assert result == ["/home", "/about", "https://example.com"]

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        """Test closing when no client was created."""
        scrapper = WebPageScrapper("https://example.com")

        # Should not raise an error
        await scrapper.close()

        assert scrapper._client is None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_close_with_client(self, mock_client_class):
        """Test closing when client was created."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create client by fetching page
        scrapper = WebPageScrapper("https://example.com")
        await scrapper._fetch_page()

        assert scrapper._client is not None

        # Test close
        await scrapper.close()

        mock_client.aclose.assert_called_once()
        assert scrapper._client is None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    async def test_context_manager(self, mock_client_class):
        """Test using scrapper as async context manager."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Test context manager
        async with WebPageScrapper("https://example.com") as scrapper:
            result = await scrapper.extract_text("h1")
            assert result == ["Main Title"]
            assert scrapper._client is not None

        # Verify client was closed
        mock_client.aclose.assert_called_once()
