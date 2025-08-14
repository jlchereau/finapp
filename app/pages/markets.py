"""
Markets page
"""

import asyncio
from typing import Dict, Any, List

import reflex as rx
from ..templates.template import template
from ..flows.markets import process_ticker


class State(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""

    # Sample market data for combobox testing
    markets: list[str] = [
        "SPY - SPDR S&P 500 ETF",
        "QQQ - Invesco QQQ Trust",
        "VTI - Vanguard Total Stock Market",
        "AAPL - Apple Inc.",
        "MSFT - Microsoft Corporation",
        "GOOGL - Alphabet Inc.",
        "AMZN - Amazon.com Inc.",
        "TSLA - Tesla Inc.",
    ]

    selected_market: str = ""

    # Workflow execution state
    is_running: bool = False
    progress: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    @rx.event(background=True)
    async def run_workflows(self):
        """Run workflows for AAPL, GOOG, and MSFT in parallel."""
        async with self:
            if self.is_running:
                return

            # Reset state
            self.is_running = True
            self.progress = {
                "AAPL": "Starting...",
                "GOOG": "Starting...",
                "MSFT": "Starting...",
            }
            self.results = []
            self.errors = []

        tickers = ["AAPL", "GOOG", "MSFT"]

        async def process_single_ticker(ticker: str):
            """Process a single ticker and update progress."""
            try:
                async with self:
                    self.progress[ticker] = "Generating seed..."

                await asyncio.sleep(0.1)  # Small delay to ensure UI updates

                # After 3 seconds, update to second step
                await asyncio.sleep(3)
                async with self:
                    self.progress[ticker] = "Generating stock price..."

                # Process the ticker
                result = await process_ticker(ticker)

                async with self:
                    if "error" in result:
                        self.errors.append(
                            f"Error processing {ticker}: {result['error']}"
                        )
                        self.progress[ticker] = "Error"
                    else:
                        self.results.append(result)
                        self.progress[ticker] = "Completed"

            except Exception as e:
                async with self:
                    self.errors.append(
                        f"Unexpected error processing {ticker}: {str(e)}"
                    )
                    self.progress[ticker] = "Error"

        # Run all workflows in parallel
        await asyncio.gather(*[process_single_ticker(ticker) for ticker in tickers])

        async with self:
            self.is_running = False


# pylint: disable=not-callable
@rx.page(route="/markets")  # pyright: ignore[reportArgumentType]
@template
def page():
    """The markets page."""
    return rx.vstack(
        rx.heading("Markets", size="9"),
        # Button to trigger workflows
        rx.button(
            "Get Stock Prices (AAPL, GOOG, MSFT)",
            on_click=State.run_workflows,
            disabled=State.is_running,
            size="3",
            variant="solid",
        ),
        # Progress display
        rx.cond(
            State.is_running | (State.progress != {}),
            rx.vstack(
                rx.heading("Progress", size="6"),
                rx.foreach(
                    State.progress,
                    lambda item: rx.hstack(
                        rx.text(item[0], weight="bold", width="80px"),
                        rx.text(item[1]),
                        spacing="3",
                    ),
                ),
                spacing="2",
                width="100%",
            ),
        ),
        # Error display
        rx.cond(
            State.errors != [],
            rx.vstack(
                rx.heading("Errors", size="6", color="red"),
                rx.foreach(
                    State.errors,
                    lambda error: rx.text(error, color="red"),
                ),
                spacing="2",
                width="100%",
            ),
        ),
        # Results table
        rx.cond(
            State.results != [],
            rx.vstack(
                rx.heading("Stock Prices", size="6"),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Ticker"),
                            rx.table.column_header_cell("Price"),
                        ),
                    ),
                    rx.table.body(
                        rx.foreach(
                            State.results,
                            lambda result: rx.table.row(
                                rx.table.cell(result["ticker"]),
                                rx.table.cell(f"${result['price']:.2f}"),
                            ),
                        ),
                    ),
                    width="100%",
                ),
                spacing="3",
                width="100%",
            ),
        ),
        spacing="5",
        align="center",
        min_height="85vh",
        width="100%",
        max_width="800px",
        margin="0 auto",
        padding="4",
    )
