"""Web page scrapper module."""

from typing import Optional
import httpx
from bs4 import BeautifulSoup


class WebPageScrapper:
    """
    A simple web page scrapper that fetches and extracts data from HTML pages.

    Uses async/await pattern for HTTP requests and BeautifulSoup for HTML parsing.
    Implements lazy loading - page is fetched only when extraction methods are called.
    """

    def __init__(self, url: str, timeout: float = 30.0):
        """
        Initialize the scrapper with a URL.

        Args:
            url: The URL to scrape
            timeout: HTTP request timeout in seconds (default: 30.0)
        """
        self.url = url
        self.timeout = timeout
        self._soup: Optional[BeautifulSoup] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def _fetch_page(self) -> None:
        """
        Fetch the web page and parse it with BeautifulSoup.

        This method is called automatically by extract methods if the page
        hasn't been fetched yet.
        """
        if self._soup is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            response = await self._client.get(self.url)
            response.raise_for_status()
            self._soup = BeautifulSoup(response.text, "html.parser")

    async def extract_text(self, css_selector: str) -> list[str]:
        """
        Extract text content from elements matching the CSS selector.

        Args:
            css_selector: CSS selector string to find elements

        Returns:
            List of text content from matching elements
        """
        await self._fetch_page()
        assert self._soup is not None  # _fetch_page ensures this
        elements = self._soup.select(css_selector)
        return [element.get_text(strip=True) for element in elements]

    async def extract_href(self, css_selector: str) -> list[str]:
        """
        Extract href attributes from elements matching the CSS selector.

        Args:
            css_selector: CSS selector string to find elements

        Returns:
            List of href attribute values from matching elements
        """
        await self._fetch_page()
        assert self._soup is not None  # _fetch_page ensures this
        elements = self._soup.select(css_selector)
        hrefs = []
        for element in elements:
            href = element.get("href")
            if href and isinstance(href, str):
                hrefs.append(href)
        return hrefs

    async def close(self) -> None:
        """Close the HTTP client if it was created."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes HTTP client."""
        await self.close()
