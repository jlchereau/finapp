"""
Test page demonstrating the actual financial data workflow.
"""

from typing import List, Dict, Any
import reflex as rx
from ..templates.template import template
from ..flows.test import fetch_financial_data, fetch_multiple_tickers


class TestState(rx.State):  # pylint: disable=inherit-non-class
    """State for the test page using the actual financial data workflow."""

    # Single ticker inputs
    single_ticker: str = ""

    # Multiple ticker inputs
    multi_tickers: str = ""

    # Loading states
    is_loading_single: bool = False
    is_loading_multi: bool = False

    # Results
    single_result: Dict[str, Any] = {}
    multi_results: Dict[str, Any] = {}

    # Status and error handling
    status_message: str = ""
    errors: List[str] = []
    warnings: List[str] = []

    # Multi-ticker specific status and errors
    multi_status_message: str = ""
    multi_errors: List[str] = []
    multi_warnings: List[str] = []

    @rx.var
    def multi_results_display(self) -> List[Dict[str, Any]]:
        """Process multi-ticker results for display."""
        if not self.multi_results:
            return []

        display_results = []
        for ticker, result in self.multi_results.items():
            # Check for explicit error or no successful data sources
            has_explicit_error = result.get("error")
            quality = result.get("data_quality", {})
            completeness_score = quality.get("completeness_score", 0)
            successful_sources = quality.get("successful_sources", 0)
            has_provider_errors = result.get("errors") and len(result["errors"]) > 0

            # Consider failed if:
            # 1. Explicit top-level error, OR
            # 2. No successful data sources, OR
            # 3. Has provider errors (indicating some providers failed)
            if has_explicit_error or successful_sources == 0 or has_provider_errors:
                # Failed: either explicit error, no providers succeeded,
                # or provider errors
                error_msg = has_explicit_error or "No data found for this ticker"
                if result.get("errors"):
                    # Use the first specific error if available
                    error_msg = result["errors"][0]

                display_results.append(
                    {
                        "ticker": ticker,
                        "status": "failed",
                        "status_text": "Failed",
                        "details": error_msg,
                        "color_scheme": "red",
                    }
                )
            else:
                # Success: all providers succeeded without errors
                display_results.append(
                    {
                        "ticker": ticker,
                        "status": "success",
                        "status_text": "Success",
                        "details": f"Quality: {completeness_score:.0%}",
                        "color_scheme": "green",
                    }
                )

        return display_results

    @rx.event(background=True)  # pylint: disable=not-callable
    async def fetch_single_ticker(self):
        """Fetch data for a single ticker using the actual workflow."""
        if not self.single_ticker:
            return

        async with self:
            self.is_loading_single = True
            self.errors = []
            self.warnings = []
            self.status_message = f"Fetching data for {self.single_ticker}..."

        try:
            # Use the actual workflow
            result = await fetch_financial_data(self.single_ticker)

            async with self:
                self.single_result = result
                if result.get("errors"):
                    self.errors = result["errors"]
                if result.get("warnings"):
                    self.warnings = result["warnings"]

                # Set appropriate status message
                if result.get("error"):
                    self.status_message = f"Failed: {result['error']}"
                else:
                    quality = result.get("data_quality", {})
                    score = quality.get("completeness_score", 0)
                    self.status_message = (
                        f"Success for {self.single_ticker} " f"(Quality: {score:.0%})"
                    )

        except Exception as e:
            async with self:
                self.errors = [f"Error: {str(e)}"]
                self.status_message = "Error occurred during fetch"
                self.single_result = {}
        finally:
            async with self:
                self.is_loading_single = False

    @rx.event(background=True)  # pylint: disable=not-callable
    async def fetch_multi_tickers(self):
        """Fetch data for multiple tickers using the actual workflow."""
        if not self.multi_tickers:
            return

        async with self:
            self.is_loading_multi = True
            self.multi_errors = []
            self.multi_warnings = []
            self.multi_status_message = "Fetching multiple tickers..."

        try:
            tickers = [
                t.strip().upper() for t in self.multi_tickers.split(",") if t.strip()
            ]

            # Use the actual workflow for multiple tickers
            results = await fetch_multiple_tickers(tickers)

            async with self:
                self.multi_results = results

                # Collect errors and warnings from all results
                all_errors = []
                all_warnings = []
                for ticker_result in results.values():
                    if ticker_result.get("errors"):
                        all_errors.extend(ticker_result["errors"])
                    if ticker_result.get("warnings"):
                        all_warnings.extend(ticker_result["warnings"])

                self.multi_errors = all_errors
                self.multi_warnings = all_warnings

                # Calculate summary statistics
                successful_count = sum(
                    1 for result in results.values() if not result.get("error")
                )
                self.multi_status_message = (
                    f"Processed {len(tickers)} tickers: "
                    f"{successful_count} successful, "
                    f"{len(tickers) - successful_count} failed"
                )

        except Exception as e:
            async with self:
                self.multi_errors = [f"Error processing tickers: {str(e)}"]
                self.multi_status_message = "Error occurred during processing"
                self.multi_results = {}
        finally:
            async with self:
                self.is_loading_multi = False

    def clear_results(self):
        """Clear all results and reset state."""
        self.single_result = {}
        self.multi_results = {}
        self.errors = []
        self.warnings = []
        self.status_message = ""
        self.multi_errors = []
        self.multi_warnings = []
        self.multi_status_message = ""


# pylint: disable=not-callable
@rx.page(route="/test")  # pyright: ignore[reportArgumentType]
@template
def page():
    """Test page demonstrating the actual financial data workflow."""
    return rx.vstack(
        rx.heading("Financial Data Workflow Test", size="8"),
        rx.text(
            "This page demonstrates the actual financial data workflow "
            "with real provider integration.",
            size="4",
            color="gray",
        ),
        # Single ticker section
        rx.vstack(
            rx.heading("Single Ticker Analysis", size="6"),
            rx.hstack(
                rx.input(
                    placeholder="Enter ticker (e.g., AAPL)",
                    value=TestState.single_ticker,
                    on_change=TestState.set_single_ticker,
                    size="3",
                ),
                rx.button(
                    "Fetch Data",
                    on_click=TestState.fetch_single_ticker,
                    disabled=TestState.is_loading_single,
                    loading=TestState.is_loading_single,
                    size="3",
                ),
                rx.button(
                    "Clear",
                    on_click=TestState.clear_results,
                    variant="outline",
                    size="3",
                ),
                spacing="3",
            ),
            # Status message
            rx.cond(
                TestState.status_message,
                rx.callout(TestState.status_message, icon="info", size="2"),
            ),
            # Errors
            rx.cond(
                TestState.errors,
                rx.callout(
                    rx.vstack(rx.foreach(TestState.errors, rx.text)),
                    icon="triangle-alert",
                    color_scheme="red",
                    size="2",
                ),
            ),
            # Warnings
            rx.cond(
                TestState.warnings,
                rx.callout(
                    rx.vstack(rx.foreach(TestState.warnings, rx.text)),
                    icon="triangle-alert",
                    color_scheme="yellow",
                    size="2",
                ),
            ),
            # Single ticker results
            rx.cond(
                TestState.single_result,
                rx.vstack(
                    rx.heading("Results", size="4"),
                    # Data quality summary
                    rx.cond(
                        TestState.single_result["data_quality"],
                        rx.card(
                            rx.vstack(
                                rx.text("Data Quality", weight="bold"),
                                rx.hstack(
                                    rx.badge("Analysis Complete", color_scheme="green"),
                                    rx.text("Quality assessment available"),
                                    spacing="2",
                                ),
                                spacing="2",
                            )
                        ),
                    ),
                    spacing="4",
                ),
            ),
            spacing="4",
        ),
        # Multiple ticker section
        rx.vstack(
            rx.heading("Multiple Ticker Analysis", size="6"),
            rx.hstack(
                rx.input(
                    placeholder="Enter tickers: AAPL,GOOGL,MSFT",
                    value=TestState.multi_tickers,
                    on_change=TestState.set_multi_tickers,
                    size="3",
                    width="400px",
                ),
                rx.button(
                    "Fetch All",
                    on_click=TestState.fetch_multi_tickers,
                    disabled=TestState.is_loading_multi,
                    loading=TestState.is_loading_multi,
                    size="3",
                ),
                spacing="3",
            ),
            # Multi-ticker status message
            rx.cond(
                TestState.multi_status_message,
                rx.callout(TestState.multi_status_message, icon="info", size="2"),
            ),
            # Multi-ticker errors
            rx.cond(
                TestState.multi_errors,
                rx.callout(
                    rx.vstack(rx.foreach(TestState.multi_errors, rx.text)),
                    icon="triangle-alert",
                    color_scheme="red",
                    size="2",
                ),
            ),
            # Multi-ticker warnings
            rx.cond(
                TestState.multi_warnings,
                rx.callout(
                    rx.vstack(rx.foreach(TestState.multi_warnings, rx.text)),
                    icon="triangle-alert",
                    color_scheme="yellow",
                    size="2",
                ),
            ),
            # Multi ticker results
            rx.cond(
                TestState.multi_results_display,
                rx.vstack(
                    rx.heading("Batch Results", size="4"),
                    rx.grid(
                        rx.foreach(
                            TestState.multi_results_display,
                            lambda item: rx.card(
                                rx.vstack(
                                    rx.heading(item["ticker"], size="4"),
                                    rx.badge(
                                        item["status_text"],
                                        color_scheme=item["color_scheme"],
                                    ),
                                    rx.text(
                                        item["details"],
                                        size="2",
                                        color="gray",
                                    ),
                                    spacing="2",
                                )
                            ),
                        ),
                        columns="3",
                        spacing="4",
                    ),
                    spacing="4",
                ),
            ),
            spacing="4",
        ),
        spacing="6",
        width="100%",
        max_width="1200px",
        margin="auto",
        padding="4",
    )
